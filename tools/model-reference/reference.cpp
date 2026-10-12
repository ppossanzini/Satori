#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

static std::string hex(const std::string & bytes) {
    static const char * digits = "0123456789abcdef";
    std::string result;
    for (unsigned char c : bytes) { result += digits[c >> 4]; result += digits[c & 15]; }
    return result;
}

static void save_floats(const fs::path & path, const float * values, size_t count) {
    // Emit explicit little-endian IEEE-754 bytes regardless of host byte order.
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) throw std::runtime_error("Nonfinite reference value");
        uint32_t bits;
        std::memcpy(&bits, values + i, 4);
        for (int shift = 0; shift < 32; shift += 8) output.put(char((bits >> shift) & 255));
    }
}

struct capture {
    fs::path directory;
    std::string case_id;
    json tensors = json::array();
    bool enabled = true;
};

static bool callback(ggml_tensor * tensor, bool ask, void * state) {
    auto & data = *static_cast<capture *>(state);
    std::string name(tensor->name);
    bool selected = data.enabled && (name == "result_norm" || name.rfind("l_out-", 0) == 0 ||
        name == "attn_norm-0" || name == "Qcur-0" || name == "Kcur-0" || name == "Vcur-0" ||
        name == "attn_out-0" || name == "ffn_inp-0" || name == "ffn_norm-0" || name == "ffn_out-0");
    if (ask) return selected;
    if (!selected) return true;
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || ggml_nelements(tensor) < 288)
        throw std::runtime_error("Unexpected intermediate tensor layout: " + name);
    size_t offset = ggml_nbytes(tensor) - 288 * sizeof(float);
    std::vector<float> values(288);
    ggml_backend_tensor_get(tensor, values.data(), offset, values.size() * sizeof(float));
    std::string filename = data.case_id + "-" + name + ".f32le";
    save_floats(data.directory / filename, values.data(), values.size());
    json entry = {{"name", name}, {"file", filename}, {"elements", 288}, {"selection", "last token"}};
    auto found = std::find_if(data.tensors.begin(), data.tensors.end(), [&](const json & item) { return item["name"] == name; });
    if (found == data.tensors.end()) data.tensors.push_back(entry);
    else *found = entry;
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    std::vector<llama_token> tokens(text.size() * 4 + 32);
    int count = llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), int(tokens.size()), special, false);
    if (count < 0) throw std::runtime_error("Tokenizer buffer too small");
    tokens.resize(count);
    return tokens;
}

static std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::vector<char> bytes(tokens.size() * 128 + 128);
    int count = llama_detokenize(vocab, tokens.data(), int(tokens.size()), bytes.data(), int(bytes.size()), true, false);
    if (count < 0) throw std::runtime_error("Detokenizer buffer too small");
    return std::string(bytes.data(), count);
}

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: reference <model.gguf> <output-directory>");
        fs::path directory(argv[2]);
        fs::create_directories(directory);
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.use_extra_bufts = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_params), llama_model_free);
        if (!model) throw std::runtime_error("Model load failed");
        auto vocab = llama_model_get_vocab(model.get());
        std::vector<std::pair<std::string, std::string>> prompts = {
            {"empty", ""}, {"story", "Once upon a time"}, {"italian", "Ciao, perché il cielo è blu?"},
            {"cjk", "你好，世界"}, {"emoji", "Hello 🌍🙂"}, {"whitespace", "  a\t\nb  "},
            {"literal_special", "<s>hello</s>"}, {"nul", std::string("a\0b", 3)}
        };
        json result = {{"tokenizer_cases", json::array()}, {"forward_cases", json::array()},
            {"configuration", {{"n_ctx",128}, {"threads",1}, {"kv_type","F32"}, {"flash_attention",false},
                {"gpu_layers",0}, {"parse_special",false}, {"add_bos",true}, {"add_eos",false}}}};
        for (const auto & [id, text] : prompts) {
            for (bool add_special : {false, true}) {
                auto tokens = tokenize(vocab, text, add_special);
                json pieces = json::array();
                for (auto token : tokens) {
                    char buffer[256];
                    int length = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, false);
                    if (length < 0) throw std::runtime_error("Piece buffer too small");
                    pieces.push_back(hex(std::string(buffer, length)));
                }
                result["tokenizer_cases"].push_back({{"id",id}, {"text",text}, {"utf8_hex",hex(text)},
                    {"add_special",add_special}, {"tokens",tokens}, {"piece_hex",pieces},
                    {"decoded_utf8_hex",hex(detokenize(vocab,tokens))}});
            }
        }
        for (int index : {0, 1, 2}) {
            const auto & [id, text] = prompts[index];
            auto tokens = tokenize(vocab, text, true);
            capture data{directory, id};
            auto params = llama_context_default_params();
            params.n_ctx = 128; params.n_batch = 128; params.n_ubatch = 128; params.n_seq_max = 1;
            params.n_threads = 1; params.n_threads_batch = 1;
            params.type_k = GGML_TYPE_F32; params.type_v = GGML_TYPE_F32;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            params.offload_kqv = false; params.no_perf = true;
            params.cb_eval = callback; params.cb_eval_user_data = &data;
            std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), params), llama_free);
            if (!context) throw std::runtime_error("Context creation failed");
            if (llama_decode(context.get(), llama_batch_get_one(tokens.data(), int(tokens.size()))) != 0)
                throw std::runtime_error("Prompt forward failed");
            const int count = llama_vocab_n_tokens(vocab);
            float * logits = llama_get_logits_ith(context.get(), -1);
            std::string filename = id + "-logits.f32le";
            save_floats(directory / filename, logits, count);
            data.enabled = false;
            json generated = json::array();
            json margins = json::array();
            for (int step = 0; step < 12; ++step) {
                logits = llama_get_logits_ith(context.get(), -1);
                llama_token token = int(std::max_element(logits, logits + count) - logits);
                float runner_up = -INFINITY;
                for (int i = 0; i < count; ++i) if (i != token) runner_up = std::max(runner_up, logits[i]);
                generated.push_back(token); margins.push_back(logits[token] - runner_up);
                if (token == llama_vocab_eos(vocab)) break;
                if (llama_decode(context.get(), llama_batch_get_one(&token, 1)) != 0)
                    throw std::runtime_error("Incremental forward failed");
            }
            result["forward_cases"].push_back({{"id",id}, {"tokens",tokens}, {"logits_file",filename},
                {"logits_count",count}, {"intermediates",data.tensors}, {"greedy_tokens",generated}, {"greedy_top2_margins",margins}});
        }
        std::ofstream output(directory / "reference.json");
        output << std::setw(2) << result << '\n';
        model.reset();
        llama_backend_free();
    } catch (const std::exception & exception) { std::cerr << exception.what() << '\n'; return 1; }
}
