#include "arg.h"
#include "chat.h"
#include "common.h"
#include "json.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <clocale>
#include <cstdio>
#include <string>
#include <vector>

// trains a LoRA adapter on top of a frozen base model
//

struct train_seq {
    std::vector<llama_token> tokens;
    std::vector<llama_token> labels;  // -1 marks a position that does not contribute to the loss
    std::vector<llama_token> context; // decoded without gradients in front of tokens, may be empty
};

static std::string render(
        const common_chat_templates * tmpls,
        const std::vector<common_chat_msg> & messages,
        const std::vector<common_chat_tool> & tools,
        size_t n_msg,
        bool add_generation_prompt) {
    common_chat_templates_inputs inputs;
    inputs.messages.assign(messages.begin(), messages.begin() + n_msg);
    inputs.tools                 = tools;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.use_jinja             = true;

    return common_chat_templates_apply(tmpls, inputs).prompt;
}

static train_seq chat_seq(
        llama_context * ctx,
        const common_chat_templates * tmpls,
        const std::vector<common_chat_msg> & messages,
        const std::vector<common_chat_tool> & tools) {
    train_seq seq;
    seq.tokens = common_tokenize(ctx, render(tmpls, messages, tools, messages.size(), false), true, true);
    seq.labels.assign(seq.tokens.size(), -1);

    // an assistant turn owns the tokens it adds on top of its own generation prompt
    for (size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].role != "assistant") {
            continue;
        }

        const size_t begin = common_tokenize(ctx, render(tmpls, messages, tools, i,     true),  true, true).size();
        const size_t end   = common_tokenize(ctx, render(tmpls, messages, tools, i + 1, false), true, true).size();

        for (size_t pos = std::max<size_t>(begin, 1); pos < end && pos < seq.tokens.size(); ++pos) {
            seq.labels[pos - 1] = seq.tokens[pos];
        }
    }

    return seq;
}

static std::vector<train_seq> load_chat(
        llama_context * ctx,
        const common_chat_templates * tmpls,
        const std::string & text) {
    std::vector<train_seq> seqs;
    int64_t n_bad   = 0;
    int64_t n_empty = 0;

    size_t pos    = 0;
    size_t n_line = 0;
    while (pos < text.size()) {
        const size_t end = text.find('\n', pos);
        const std::string line = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        pos = end == std::string::npos ? text.size() : end + 1;
        n_line++;

        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }

        std::vector<common_chat_msg>  messages;
        std::vector<common_chat_tool> tools;
        try {
            const common_json j = common_json::parse(line);
            messages = common_chat_msgs_parse_oaicompat(j.at("messages"));
            if (j.contains("tools")) {
                tools = common_chat_tools_parse_oaicompat(j.at("tools"));
            }
        } catch (const std::exception & e) {
            if (n_bad++ == 0) {
                LOG_WRN("%s: line %zu: %s\n", __func__, n_line, e.what());
            }
            continue;
        }

        train_seq seq = chat_seq(ctx, tmpls, messages, tools);

        bool supervised = false;
        for (llama_token l : seq.labels) {
            supervised |= l >= 0;
        }
        if (!supervised) {
            n_empty++;
            continue;
        }

        seqs.push_back(std::move(seq));
    }

    if (n_bad > 0) {
        LOG_WRN("%s: skipped %" PRId64 " malformed lines, the first one is shown above\n", __func__, n_bad);
    }
    if (n_empty > 0) {
        LOG_WRN("%s: skipped %" PRId64 " lines without an assistant turn\n", __func__, n_empty);
    }

    return seqs;
}

static bool load_tokens(const std::string & text, int32_t n_vocab, std::vector<train_seq> & seqs) {
    int64_t n_bad = 0;

    size_t pos    = 0;
    size_t n_line = 0;
    while (pos < text.size()) {
        const size_t end = text.find('\n', pos);
        const std::string line = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        pos = end == std::string::npos ? text.size() : end + 1;
        n_line++;

        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }

        train_seq seq;
        try {
            const common_json j = common_json::parse(line);
            seq.tokens = j.at("tokens").get<std::vector<int>>();
            if (j.contains("context")) {
                seq.context = j.at("context").get<std::vector<int>>();
            }
        } catch (const std::exception & e) {
            if (n_bad++ == 0) {
                LOG_WRN("%s: line %zu: %s\n", __func__, n_line, e.what());
            }
            continue;
        }
        if (seq.tokens.size() < 2) {
            if (n_bad++ == 0) {
                LOG_WRN("%s: line %zu: fewer than two tokens\n", __func__, n_line);
            }
            continue;
        }

        for (const auto * part : { &seq.context, &seq.tokens }) {
            for (llama_token t : *part) {
                if (t < 0 || t >= n_vocab) {
                    LOG_ERR("%s: line %zu: token %d is outside the vocabulary of %d\n",
                            __func__, n_line, t, n_vocab);
                    return false;
                }
            }
        }

        seq.labels.assign(seq.tokens.size(), -1);
        for (size_t i = 0; i + 1 < seq.tokens.size(); ++i) {
            seq.labels[i] = seq.tokens[i + 1];
        }
        seqs.push_back(std::move(seq));
    }

    if (n_bad > 0) {
        LOG_WRN("%s: skipped %" PRId64 " bad lines, the first one is shown above\n", __func__, n_bad);
    }

    return true;
}

static std::vector<train_seq> load_text(llama_context * ctx, const std::string & text, int64_t n_ctx) {
    std::vector<train_seq> seqs;

    const std::vector<llama_token> tokens = common_tokenize(ctx, text, true, true);
    for (size_t off = 0; off + 1 < tokens.size(); off += n_ctx) {
        const size_t n = std::min<size_t>(n_ctx, tokens.size() - off);

        train_seq seq;
        seq.tokens.assign(tokens.begin() + off, tokens.begin() + off + n);
        seq.labels.assign(n, -1);
        for (size_t i = 0; i + 1 < n; ++i) {
            seq.labels[i] = seq.tokens[i + 1];
        }
        seqs.push_back(std::move(seq));
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

    const int64_t n_ctx = llama_n_ctx(ctx);

    std::vector<train_seq> seqs;
    if (params.train_format == "tokens") {
        if (!load_tokens(params.prompt, llama_vocab_n_tokens(llama_model_get_vocab(model)), seqs)) {
            return 1;
        }
    } else if (params.train_format == "text") {
        seqs = load_text(ctx, params.prompt, n_ctx);
    } else {
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, params.chat_template);
        seqs = load_chat(ctx, tmpls.get(), params.prompt);
    }
    if (seqs.empty()) {
        LOG_ERR("%s: dataset is empty\n", __func__);
        return 1;
    }

    // with a context in front of the sequences the window is -ub, and -c holds the context plus the window
    const bool prefixed = std::any_of(seqs.begin(), seqs.end(), [](const train_seq & s) { return !s.context.empty(); });
    const int64_t n_win = prefixed ? (int64_t) llama_n_ubatch(ctx) : n_ctx;
    if (prefixed && n_win >= n_ctx) {
        LOG_ERR("%s: the sequences have a context, so -c %" PRId64 " must hold it and the -ub %" PRId64 " window\n",
                __func__, n_ctx, n_win);
        return 1;
    }

    // the oldest context tokens go first when the context does not fit
    std::vector<const llama_token *> prefix  (seqs.size());
    std::vector<int32_t>             n_prefix(seqs.size());
    int64_t n_short = 0;
    int64_t n_long  = 0;
    for (size_t i = 0; i < seqs.size(); ++i) {
        const auto & c = seqs[i].context;
        const size_t keep = std::min<size_t>(c.size(), n_ctx - n_win);
        n_short += c.size() > keep;
        n_long   = std::max<int64_t>(n_long, keep);
        prefix[i]   = c.data() + c.size() - keep;
        n_prefix[i] = (int32_t) keep;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token pad = llama_vocab_pad(vocab);
    if (pad == LLAMA_TOKEN_NULL) {
        pad = llama_vocab_eos(vocab);
    }

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
            GGML_TYPE_I32, GGML_TYPE_I32, n_win, n_win, (int64_t) seqs.size(), /*ndata_shard =*/ 1);

    llama_token * data   = (llama_token *) ggml_opt_dataset_data  (dataset)->data;
    llama_token * labels = (llama_token *) ggml_opt_dataset_labels(dataset)->data;

    int64_t n_supervised = 0;
    int64_t n_padded     = 0;
    int64_t n_cut        = 0;
    int64_t n_lost       = 0;
    for (size_t i = 0; i < seqs.size(); ++i) {
        const train_seq & seq = seqs[i];
        llama_token * d = data   + i*n_win;
        llama_token * l = labels + i*n_win;

        for (int64_t p = 0; p < n_win; ++p) {
            const bool inside = p < (int64_t) seq.tokens.size();
            d[p] = inside ? seq.tokens[p] : pad;
            l[p] = inside ? seq.labels[p] : -1;
            n_supervised += l[p] >= 0;
        }

        n_padded += (int64_t) seq.tokens.size() < n_win;
        if ((int64_t) seq.tokens.size() > n_win) {
            n_cut++;
            for (size_t p = n_win; p < seq.labels.size(); ++p) {
                n_lost += seq.labels[p] >= 0;
            }
        }
    }

    LOG_INF("%s: %zu sequences of %" PRId64 " tokens, %.1f%% of the positions are trained on\n",
            __func__, seqs.size(), n_win, 100.0*n_supervised/(seqs.size()*n_win));
    LOG_INF("%s: %" PRId64 " sequences are shorter than %" PRId64 " and were padded\n", __func__, n_padded, n_win);
    if (n_cut > 0) {
        LOG_WRN("%s: %" PRId64 " sequences are longer than %" PRId64 " and were cut, losing %" PRId64 " trained tokens\n",
                __func__, n_cut, n_win, n_lost);
    }
    if (prefixed) {
        LOG_INF("%s: the sequences continue a context of up to %" PRId64 " tokens, decoded without gradients\n", __func__, n_long);
    }
    if (n_short > 0) {
        LOG_WRN("%s: %" PRId64 " contexts are longer than -c minus the window and lost their oldest tokens\n", __func__, n_short);
    }

    struct lr_opt & lr = params.lr;

    LOG_INF("%s: -optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -period %.2g -val %.2g\n",
            __func__, ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min,
            (double) lr.decay_epochs, (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch,
            (double) params.val_split);

    struct llama_opt_params lopt_params {
        /*n_ctx_train     =*/ prefixed ? (uint32_t) n_win : 0,
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
        if (prefixed) {
            llama_opt_epoch_prefix(ctx, dataset, prefix.data(), n_prefix.data(), result_train, result_eval, idata_split,
                                   ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        } else {
            llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                            ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        }
        fprintf(stderr, "\n");

        // every epoch is saved, the last one under the -o name
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
