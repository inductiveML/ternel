#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

struct model_run {
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> continuation;
    std::vector<float> logits;
    int32_t n_vocab = 0;
};

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "validate-llama-distribution: %s\n", message);
    std::exit(1);
}

void quiet_log(enum ggml_log_level, const char *, void *) {
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int32_t count = -llama_tokenize(
        vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, true, true);
    if (count <= 0) {
        fail("tokenization length query failed");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(count));
    const int32_t written = llama_tokenize(
        vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), count, true, true);
    if (written != count) {
        fail("tokenization failed");
    }
    return tokens;
}

llama_token argmax_token(const float * logits, int32_t n_vocab) {
    return static_cast<llama_token>(
        std::max_element(logits, logits + n_vocab) - logits);
}

model_run run_model(
        const std::string & path,
        const std::string & prompt,
        int steps,
        const std::vector<llama_token> * forced_continuation) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    model_params.use_mmap = false;
    model_params.check_tensors = true;

    llama_model * model = llama_model_load_from_file(path.c_str(), model_params);
    if (model == nullptr) {
        fail("model load failed");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    model_run result;
    result.n_vocab = llama_vocab_n_tokens(vocab);
    result.prompt_tokens = tokenize(vocab, prompt);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max<uint32_t>(128, result.prompt_tokens.size() + steps + 8);
    context_params.n_batch = std::max<uint32_t>(32, result.prompt_tokens.size());
    context_params.n_ubatch = context_params.n_batch;
    context_params.no_perf = false;

    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        llama_model_free(model);
        fail("context creation failed");
    }

    auto append_logits = [&]() {
        const float * current = llama_get_logits_ith(context, -1);
        if (current == nullptr) {
            fail("logits unavailable");
        }
        result.logits.insert(result.logits.end(), current, current + result.n_vocab);
        return argmax_token(current, result.n_vocab);
    };

    if (llama_decode(context, llama_batch_get_one(
            result.prompt_tokens.data(), static_cast<int32_t>(result.prompt_tokens.size()))) != 0) {
        fail("prompt decode failed");
    }
    llama_token next = append_logits();

    for (int step = 0; step < steps; ++step) {
        const llama_token token = forced_continuation == nullptr
            ? next
            : forced_continuation->at(static_cast<size_t>(step));
        result.continuation.push_back(token);
        if (llama_decode(context, llama_batch_get_one(&result.continuation.back(), 1)) != 0) {
            fail("generation decode failed");
        }
        next = append_logits();
    }

    llama_free(context);
    llama_model_free(model);
    return result;
}

void print_token_array(const std::vector<llama_token> & tokens) {
    std::putchar('[');
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) {
            std::putchar(',');
        }
        std::printf("%d", tokens[i]);
    }
    std::putchar(']');
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr,
            "usage: %s BASELINE.gguf TQ1.gguf [STEPS=16] [PROMPT]\n", argv[0]);
        return 2;
    }
    const int steps = argc >= 4 ? std::stoi(argv[3]) : 16;
    const std::string prompt = argc >= 5
        ? argv[4]
        : "The capital of France is";
    if (steps < 1 || steps > 128) {
        fail("steps must be in [1, 128]");
    }

    llama_log_set(quiet_log, nullptr);
    ggml_log_set(quiet_log, nullptr);
    ggml_backend_load_all();
    const model_run baseline = run_model(argv[1], prompt, steps, nullptr);
    const model_run tq1 = run_model(argv[2], prompt, steps, &baseline.continuation);

    if (baseline.prompt_tokens != tq1.prompt_tokens) {
        fail("tokenizers produced different prompt tokens");
    }
    if (baseline.n_vocab != tq1.n_vocab || baseline.logits.size() != tq1.logits.size()) {
        fail("logit shapes differ");
    }

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    double max_abs = 0.0;
    uint64_t nonfinite_baseline = 0;
    uint64_t nonfinite_tq1 = 0;
    uint64_t argmax_matches = 0;
    const size_t rows = static_cast<size_t>(steps + 1);

    for (size_t row = 0; row < rows; ++row) {
        const float * a = baseline.logits.data() + row * baseline.n_vocab;
        const float * b = tq1.logits.data() + row * tq1.n_vocab;
        argmax_matches += argmax_token(a, baseline.n_vocab) == argmax_token(b, tq1.n_vocab);
        for (int32_t col = 0; col < baseline.n_vocab; ++col) {
            if (!std::isfinite(a[col])) {
                ++nonfinite_baseline;
            }
            if (!std::isfinite(b[col])) {
                ++nonfinite_tq1;
            }
            const double da = a[col];
            const double db = b[col];
            const double delta = std::abs(da - db);
            max_abs = std::max(max_abs, delta);
            sum_abs += delta;
            sum_sq += delta * delta;
            dot += da * db;
            norm_a += da * da;
            norm_b += db * db;
        }
    }

    const double count = static_cast<double>(baseline.logits.size());
    const double cosine = dot / std::sqrt(norm_a * norm_b);
    std::printf("{\n");
    std::printf("  \"status\": \"%s\",\n",
        nonfinite_baseline == 0 && nonfinite_tq1 == 0 ? "PASS" : "FAIL_NONFINITE");
    std::printf("  \"prompt_tokens\": ");
    print_token_array(baseline.prompt_tokens);
    std::printf(",\n  \"forced_continuation\": ");
    print_token_array(baseline.continuation);
    std::printf(",\n");
    std::printf("  \"steps_compared\": %zu,\n", rows);
    std::printf("  \"vocab\": %d,\n", baseline.n_vocab);
    std::printf("  \"logits_compared\": %zu,\n", baseline.logits.size());
    std::printf("  \"max_abs_error\": %.17g,\n", max_abs);
    std::printf("  \"mean_abs_error\": %.17g,\n", sum_abs / count);
    std::printf("  \"rmse\": %.17g,\n", std::sqrt(sum_sq / count));
    std::printf("  \"cosine_similarity\": %.17g,\n", cosine);
    std::printf("  \"argmax_matches\": %llu,\n", static_cast<unsigned long long>(argmax_matches));
    std::printf("  \"argmax_total\": %zu,\n", rows);
    std::printf("  \"nonfinite_baseline\": %llu,\n",
        static_cast<unsigned long long>(nonfinite_baseline));
    std::printf("  \"nonfinite_tq1\": %llu\n",
        static_cast<unsigned long long>(nonfinite_tq1));
    std::printf("}\n");
    llama_backend_free();
    return nonfinite_baseline == 0 && nonfinite_tq1 == 0 ? 0 : 1;
}
