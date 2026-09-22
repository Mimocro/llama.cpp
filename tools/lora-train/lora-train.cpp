#include "arg.h"
#include "chat.h"
#include "common.h"
#include "json.h"
#include "log.h"
#include "llama.h"

#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// trains a LoRA adapter on top of a frozen base model
//
// the dataset is either a JSONL file with one {"messages": [...]} object per line, in which case
// only the assistant turns are supervised, or plain text, in which case every token is

struct train_seq {
    std::vector<llama_token> tokens;
    std::vector<llama_token> labels; // -1 marks a position that does not contribute to the loss
};

static std::string render_prompt(
        const common_chat_templates * tmpls,
        const std::vector<common_chat_msg> & messages,
        size_t n_msg,
        bool add_generation_prompt) {
    common_chat_templates_inputs inputs;
    inputs.messages.assign(messages.begin(), messages.begin() + n_msg);
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.use_jinja             = true;

    return common_chat_templates_apply(tmpls, inputs).prompt;
}

static bool seq_from_messages(
        llama_context * ctx,
        const common_chat_templates * tmpls,
        const std::vector<common_chat_msg> & messages,
        train_seq & out) {
    const std::string full = render_prompt(tmpls, messages, messages.size(), false);

    out.tokens = common_tokenize(ctx, full, true, true);
    out.labels.assign(out.tokens.size(), -1);

    // an assistant turn owns the tokens it adds on top of its own generation prompt
    for (size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].role != "assistant") {
            continue;
        }

        const size_t n_prefix = common_tokenize(ctx, render_prompt(tmpls, messages, i,     true),  true, true).size();
        const size_t n_turn   = common_tokenize(ctx, render_prompt(tmpls, messages, i + 1, false), true, true).size();

        for (size_t pos = n_prefix; pos < n_turn && pos < out.tokens.size(); ++pos) {
            out.labels[pos - 1] = out.tokens[pos];
        }
    }

    return out.tokens.size() > 1;
}

static std::vector<train_seq> load_dataset(
        llama_context * ctx,
        const common_chat_templates * tmpls,
        const std::string & text,
        int64_t n_ctx_data) {
    std::vector<train_seq> seqs;

    bool as_chat = false;
    {
        std::string first;
        for (char c : text) {
            if (c == '\n') break;
            first += c;
        }
        as_chat = first.find("\"messages\"") != std::string::npos;
    }

    if (as_chat) {
        size_t pos = 0;
        int64_t n_bad = 0;
        while (pos < text.size()) {
            const size_t end  = text.find('\n', pos);
            const std::string line = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
            pos = end == std::string::npos ? text.size() : end + 1;
            if (line.find_first_not_of(" \t\r") == std::string::npos) {
                continue;
            }

            std::vector<common_chat_msg> messages;
            {
                const common_json j = common_json::parse_no_throw(line);
                if (j.is_discarded() || !j.contains("messages")) {
                    n_bad++;
                    continue;
                }
                const common_json & msgs = j.at("messages");
                for (size_t k = 0; k < msgs.size(); ++k) {
                    const common_json & m = msgs.at(k);
                    common_chat_msg msg;
                    msg.role    = m.value("role",    std::string());
                    msg.content = m.value("content", std::string());

                    // OpenAI shaped tool calls, so agent traces can be used as they are
                    if (m.contains("tool_calls")) {
                        const common_json & tcs = m.at("tool_calls");
                        for (size_t t = 0; t < tcs.size(); ++t) {
                            const common_json & tc = tcs.at(t);
                            const common_json & fn = tc.contains("function") ? tc.at("function") : tc;

                            common_chat_tool_call call;
                            call.name      = fn.value("name", std::string());
                            call.arguments = fn.value("arguments", std::string());
                            call.id        = tc.value("id", std::string());

                            if (!call.name.empty()) {
                                msg.tool_calls.push_back(std::move(call));
                            }
                        }
                    }

                    if (msg.role == "tool") {
                        msg.tool_name    = m.value("name", std::string());
                        msg.tool_call_id = m.value("tool_call_id", std::string());
                    }

                    messages.push_back(std::move(msg));
                }
            }

            train_seq seq;
            if (seq_from_messages(ctx, tmpls, messages, seq)) {
                seqs.push_back(std::move(seq));
            }
        }
        if (n_bad > 0) {
            LOG_WRN("%s: skipped %" PRId64 " malformed lines\n", __func__, n_bad);
        }
    } else {
        // plain text: chop into chunks and supervise everything
        const std::vector<llama_token> tokens = common_tokenize(ctx, text, true, true);
        for (size_t off = 0; off + 1 < tokens.size(); off += n_ctx_data) {
            train_seq seq;
            const size_t n = std::min<size_t>(n_ctx_data, tokens.size() - off);
            seq.tokens.assign(tokens.begin() + off, tokens.begin() + off + n);
            seq.labels.assign(n, -1);
            for (size_t i = 0; i + 1 < n; ++i) {
                seq.labels[i] = seq.tokens[i + 1];
            }
            if (seq.tokens.size() > 1) {
                seqs.push_back(std::move(seq));
            }
        }
    }

    return seqs;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape   = false;
    params.out_file = "lora-trained.gguf";

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (params.lora_adapters.empty()) {
        LOG_ERR("%s: no adapter given, create one with llama-lora-init and pass it with --lora\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL || ctx == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    auto & adapters = llama_init->lora();
    if (adapters.empty()) {
        LOG_ERR("%s: adapter was not loaded\n", __func__);
        return 1;
    }

    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    const int64_t n_ctx_data = llama_n_ctx(ctx);

    common_chat_templates_ptr tmpls = common_chat_templates_init(model, params.chat_template);

    std::vector<train_seq> seqs = load_dataset(ctx, tmpls.get(), params.prompt, n_ctx_data);
    if (seqs.empty()) {
        LOG_ERR("%s: dataset is empty\n", __func__);
        return 1;
    }

    // pack every sequence into one fixed size datapoint
    const llama_token pad = llama_vocab_pad(llama_model_get_vocab(model));

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
            GGML_TYPE_I32, GGML_TYPE_I32, n_ctx_data, n_ctx_data, (int64_t) seqs.size(), /*ndata_shard =*/ 1);

    llama_token * data   = (llama_token *) ggml_opt_dataset_data  (dataset)->data;
    llama_token * labels = (llama_token *) ggml_opt_dataset_labels(dataset)->data;

    int64_t n_supervised = 0;
    for (size_t i = 0; i < seqs.size(); ++i) {
        llama_token * d = data   + i*n_ctx_data;
        llama_token * l = labels + i*n_ctx_data;

        for (int64_t p = 0; p < n_ctx_data; ++p) {
            const bool inside = p < (int64_t) seqs[i].tokens.size();
            d[p] = inside ? seqs[i].tokens[p] : pad;
            l[p] = inside ? seqs[i].labels[p] : -1;
            n_supervised += l[p] >= 0;
        }
    }

    const double supervised_ratio = (double) n_supervised / (double) (seqs.size()*n_ctx_data);

    LOG_INF("%s: %zu sequences of %" PRId64 " tokens, %.1f%% of the positions are supervised\n",
            __func__, seqs.size(), n_ctx_data, 100.0*supervised_ratio);

    // the loss averages over every position, so masking scales the gradient down by that ratio
    struct lr_opt & lr = params.lr;
    if (supervised_ratio > 0.0 && supervised_ratio < 1.0) {
        lr.lr0 /= (float) supervised_ratio;
        LOG_INF("%s: scaling lr to %.2g to compensate for the masked positions\n", __func__, (double) lr.lr0);
    }

    LOG_INF("%s: -optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -period %.2g -val %.2g\n",
            __func__, ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min,
            (double) lr.decay_epochs, (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch,
            (double) params.val_split);

    struct llama_opt_params lopt_params {
        /*n_ctx_train     =*/ 0,
        /*param_filter    =*/ llama_opt_param_filter_all,
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ common_opt_lr_pars,
        /*get_opt_pars_ud =*/ &params.lr,
        /*optimizer_type  =*/ params.optimizer,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

        // checkpoint after every epoch so a long run survives a restart
        const std::string path = lr.epoch + 1 == lr.epochs
            ? params.out_file
            : params.out_file + ".epoch" + std::to_string(lr.epoch);

        if (!llama_adapter_lora_save_to_file(adapters[0].get(), path.c_str())) {
            LOG_ERR("%s: failed to save the adapter\n", __func__);
            return 1;
        }
        LOG_INF("%s: saved %s\n", __func__, path.c_str());

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    ggml_opt_dataset_free(dataset);

    llama_backend_free();

    return 0;
}
