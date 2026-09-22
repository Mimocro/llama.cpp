#include "ggml.h"
#include "gguf.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// creates a fresh LoRA adapter for a base model: A is random, B is zero, so the
// adapter is a no-op until it is trained

struct init_params {
    std::string model;
    std::string outfile = "lora-init.gguf";
    std::string targets =
        "attn_q.weight,attn_k.weight,attn_v.weight,attn_output.weight,"
        "attn_qkv.weight,attn_gate.weight,"
        "ssm_alpha.weight,ssm_beta.weight,ssm_out.weight,"
        "ffn_gate.weight,ffn_up.weight,ffn_down.weight";
    int   rank  = 16;
    float alpha = 32.0f;
    uint32_t seed = 1234;
    int   layer_first = 0;
    int   layer_last  = -1; // inclusive, -1 means the last one
};

static void print_usage(const char * argv0) {
    printf("usage: %s --model FNAME [options]\n\n", argv0);
    printf("  -m, --model FNAME     base model (GGUF)\n");
    printf("  -o, --outfile FNAME   output adapter (default: lora-init.gguf)\n");
    printf("  -r, --rank N          LoRA rank (default: 16)\n");
    printf("      --alpha F         LoRA alpha (default: 32)\n");
    printf("      --targets LIST    comma separated tensor name suffixes\n");
    printf("      --seed N          RNG seed (default: 1234)\n");
    printf("      --layers A:B      only blocks A..B, inclusive (default: all)\n");
    printf("                        the backward pass stops at the lowest trained block,\n");
    printf("                        so a higher A cuts activation memory\n");
}

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos < s.size()) {
        const size_t end = s.find(',', pos);
        const std::string tok = s.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        if (!tok.empty()) {
            out.push_back(tok);
        }
        if (end == std::string::npos) {
            break;
        }
        pos = end + 1;
    }
    return out;
}

static bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main(int argc, char ** argv) {
    init_params params;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s needs a value\n", arg.c_str());
                exit(1);
            }
            return argv[++i];
        };

        if (arg == "-m" || arg == "--model") {
            params.model = next();
        } else if (arg == "-o" || arg == "--outfile") {
            params.outfile = next();
        } else if (arg == "-r" || arg == "--rank") {
            params.rank = std::stoi(next());
        } else if (arg == "--alpha") {
            params.alpha = std::stof(next());
        } else if (arg == "--targets") {
            params.targets = next();
        } else if (arg == "--seed") {
            params.seed = (uint32_t) std::stoul(next());
        } else if (arg == "--layers") {
            const std::string val = next();
            const size_t sep = val.find(':');
            if (sep == std::string::npos) {
                fprintf(stderr, "error: --layers wants A:B\n");
                return 1;
            }
            params.layer_first = std::stoi(val.substr(0, sep));
            params.layer_last  = std::stoi(val.substr(sep + 1));
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (params.model.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    const std::vector<std::string> targets = split_csv(params.targets);

    ggml_context * ctx_base = nullptr;
    gguf_context * gguf_base = gguf_init_from_file(params.model.c_str(), { /*no_alloc =*/ true, /*ctx =*/ &ctx_base });
    if (!gguf_base) {
        fprintf(stderr, "error: cannot read %s\n", params.model.c_str());
        return 1;
    }

    std::string arch = "unknown";
    {
        const int64_t kid = gguf_find_key(gguf_base, "general.architecture");
        if (kid >= 0) {
            arch = gguf_get_val_str(gguf_base, kid);
        }
    }

    std::vector<ggml_tensor *> picked;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_base); t; t = ggml_get_next_tensor(ctx_base, t)) {
        if (ggml_n_dims(t) != 2) {
            continue;
        }
        const std::string name = ggml_get_name(t);

        // the layer range only filters per-block tensors, others (output, token_embd) pass through
        int layer = -1;
        if (sscanf(name.c_str(), "blk.%d.", &layer) == 1) {
            if (layer < params.layer_first) {
                continue;
            }
            if (params.layer_last >= 0 && layer > params.layer_last) {
                continue;
            }
        }

        for (const auto & suffix : targets) {
            if (ends_with(name, suffix)) {
                picked.push_back(t);
                break;
            }
        }
    }

    if (picked.empty()) {
        fprintf(stderr, "error: no tensor matched the target list\n");
        return 1;
    }

    // one context holding A and B for every target
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*2*picked.size() + 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_out = ggml_init(ip);

    gguf_context * gguf_out = gguf_init_empty();
    gguf_set_val_str(gguf_out, "general.type",         "adapter");
    gguf_set_val_str(gguf_out, "general.architecture", arch.c_str());
    gguf_set_val_str(gguf_out, "adapter.type",         "lora");
    gguf_set_val_f32(gguf_out, "adapter.lora.alpha",   params.alpha);

    std::mt19937 rng(params.seed);

    std::vector<std::vector<float>> data;
    data.reserve(2*picked.size());

    int64_t n_params = 0;

    for (ggml_tensor * t : picked) {
        const int64_t n_in  = t->ne[0];
        const int64_t n_out = t->ne[1];

        ggml_tensor * a = ggml_new_tensor_2d(ctx_out, GGML_TYPE_F32, n_in, params.rank);
        ggml_tensor * b = ggml_new_tensor_2d(ctx_out, GGML_TYPE_F32, params.rank, n_out);

        ggml_set_name(a, (std::string(ggml_get_name(t)) + ".lora_a").c_str());
        ggml_set_name(b, (std::string(ggml_get_name(t)) + ".lora_b").c_str());

        // kaiming-uniform for A, zeros for B
        const float bound = 1.0f/std::sqrt((float) n_in);
        std::uniform_real_distribution<float> dist(-bound, bound);

        data.emplace_back(n_in*params.rank);
        for (auto & v : data.back()) {
            v = dist(rng);
        }
        gguf_add_tensor(gguf_out, a);
        gguf_set_tensor_data(gguf_out, ggml_get_name(a), data.back().data());

        data.emplace_back(params.rank*n_out, 0.0f);
        gguf_add_tensor(gguf_out, b);
        gguf_set_tensor_data(gguf_out, ggml_get_name(b), data.back().data());

        n_params += (n_in + n_out)*params.rank;
    }

    if (!gguf_write_to_file(gguf_out, params.outfile.c_str(), false)) {
        fprintf(stderr, "error: cannot write %s\n", params.outfile.c_str());
        return 1;
    }

    printf("%s: arch %s, rank %d, alpha %.1f\n", __func__, arch.c_str(), params.rank, params.alpha);
    printf("%s: %zu target tensors, %" PRId64 " trainable parameters (%.1f MiB as f32)\n",
            __func__, picked.size(), n_params, n_params*sizeof(float)/1024.0/1024.0);
    printf("%s: wrote %s\n", __func__, params.outfile.c_str());

    gguf_free(gguf_out);
    gguf_free(gguf_base);
    ggml_free(ctx_out);
    ggml_free(ctx_base);

    return 0;
}
