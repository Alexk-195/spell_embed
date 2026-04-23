// @file_purpose Compact Character-Level Word Embedding
#include "word_embed.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <cctype>
#include <fstream>
#include <cstdint>

using namespace word_embed;

namespace {

// =============================================================================
// UTF-8 Helper
// =============================================================================

std::vector<std::string> utf8_split(const std::string& str) {
    std::vector<std::string> result;
    for (size_t i = 0; i < str.size(); ) {
        size_t char_len = 1;
        unsigned char c = str[i];
        if ((c & 0x80) == 0) {
            char_len = 1;  // ASCII
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;  // 2-byte UTF-8
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;  // 3-byte UTF-8
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;  // 4-byte UTF-8
        }
        if (i + char_len <= str.size()) {
            result.push_back(str.substr(i, char_len));
        }
        i += char_len;
    }
    return result;
}

// =============================================================================
// CharEncoder - Maps UTF-8 characters to encoding vectors
// =============================================================================

class CharEncoder {
public:
    CharEncoder(const std::string& alphabet, 
                const std::unordered_map<std::string, std::vector<std::pair<std::string, double>>>& similarities) {
        
        // Split alphabet into UTF-8 characters
        chars_ = utf8_split(alphabet);
        K_ = chars_.size();
        
        // Build character to index map
        for (size_t i = 0; i < chars_.size(); ++i) {
            char_to_index_[chars_[i]] = i;
        }
        
        // Precompute encoding vectors for all characters
        for (const auto& ch : chars_) {
            encodings_[ch] = build_encoding(ch, similarities);
        }
    }
    
    const std::vector<float>& encode(const std::string& ch) const {
        auto it = encodings_.find(ch);
        if (it != encodings_.end()) {
            return it->second;
        }
        static const std::vector<float> zero_vec;
        return zero_vec;
    }
    
    std::vector<float> ngram_vector(const std::string& gram) const {
        std::vector<float> result;
        auto chars = utf8_split(gram);
        for (const auto& ch : chars) {
            const auto& enc = encode(ch);
            result.insert(result.end(), enc.begin(), enc.end());
        }
        return result;
    }
    
    size_t size() const { return K_; }
    
private:
    std::vector<float> build_encoding(const std::string& ch, 
                const std::unordered_map<std::string, std::vector<std::pair<std::string, double>>>& similarities) {
        std::vector<float> v(K_, 0.0f);
        
        // Set one-hot for character
        auto it = char_to_index_.find(ch);
        if (it != char_to_index_.end()) {
            v[it->second] = 1.0f;
        }
        
        // Add single-char similarities for soft encoding
        auto sim_it = similarities.find(ch);
        if (sim_it != similarities.end()) {
            for (const auto& [target, weight] : sim_it->second) {
                // Check if target is single UTF-8 character
                auto target_chars = utf8_split(target);
                if (target_chars.size() == 1) {
                    auto target_it = char_to_index_.find(target_chars[0]);
                    if (target_it != char_to_index_.end()) {
                        v[target_it->second] += static_cast<float>(weight);
                    }
                }
            }
            
            // Normalize to unit length
            float norm = 0.0f;
            for (float val : v) {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (float& val : v) {
                    val /= norm;
                }
            }
        }
        
        return v;
    }
    
    std::vector<std::string> chars_;
    size_t K_;
    std::unordered_map<std::string, size_t> char_to_index_;
    std::unordered_map<std::string, std::vector<float>> encodings_;
};

} // anonymous namespace

// =============================================================================
// Matrix utilities
// =============================================================================

class Matrix {
public:
    static std::vector<std::vector<float>> random_sparse_contractive(
            size_t rows, size_t cols, float alpha, float sparsity, std::mt19937& rng) {
        
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        
        // Initialize dense random matrix
        std::vector<std::vector<float>> A(rows, std::vector<float>(cols));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                A[i][j] = normal(rng);
            }
        }
        
        // Apply sparsity mask
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                if (uniform(rng) < sparsity) {
                    A[i][j] = 0.0f;
                }
            }
        }
        
        // Compute spectral norm using power iteration
        float spectral_norm = compute_spectral_norm(A);
        
        // Rescale to target spectral radius
        if (spectral_norm > 0) {
            float scale = alpha / spectral_norm;
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    A[i][j] *= scale;
                }
            }
        }
        
        return A;
    }
    
    static std::vector<std::vector<float>> random_projection(
            size_t rows, size_t cols, std::mt19937& rng) {
        std::normal_distribution<float> normal(0.0f, 1.0f);
        float scale = 1.0f / std::sqrt(static_cast<float>(cols));
        
        std::vector<std::vector<float>> B(rows, std::vector<float>(cols));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                B[i][j] = normal(rng) * scale;
            }
        }
        return B;
    }
    
    static std::vector<float> mat_vec_mul(
            const std::vector<std::vector<float>>& A,
            const std::vector<float>& v) {
        std::vector<float> result(A.size(), 0.0f);
        for (size_t i = 0; i < A.size(); ++i) {
            for (size_t j = 0; j < v.size(); ++j) {
                result[i] += A[i][j] * v[j];
            }
        }
        return result;
    }
    
private:
    static float compute_spectral_norm(const std::vector<std::vector<float>>& A, int iterations = 20) {
        if (A.empty() || A[0].empty()) return 0.0f;
        
        size_t n = A.size();
        size_t m = A[0].size();
        
        // Initialize random vector
        std::vector<float> v(m, 1.0f / std::sqrt(static_cast<float>(m)));
        
        // Power iteration
        for (int iter = 0; iter < iterations; ++iter) {
            // v = A^T * A * v
            std::vector<float> Av = mat_vec_mul(A, v);
            std::vector<float> AtAv(m, 0.0f);
            for (size_t j = 0; j < m; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    AtAv[j] += A[i][j] * Av[i];
                }
            }
            
            // Normalize
            float norm = 0.0f;
            for (float val : AtAv) {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (size_t j = 0; j < m; ++j) {
                    v[j] = AtAv[j] / norm;
                }
            }
        }
        
        // Compute ||A * v||
        std::vector<float> Av = mat_vec_mul(A, v);
        float result = 0.0f;
        for (float val : Av) {
            result += val * val;
        }
        return std::sqrt(result);
    }
};

// =============================================================================
// WordEmbedderImpl - Main implementation
// =============================================================================

class WordEmbedderImpl : public WordEmbedder {
public:
    // Constructor with random matrix generation
    WordEmbedderImpl(const WordEmbedConfig& config) 
        : config_(config), encoder_(config.alphabet, config.similarities) {
        
        std::mt19937 rng(config.seed);
        
        // Create sparse contractive matrix A
        A_ = Matrix::random_sparse_contractive(
            config.dimensions, config.dimensions, 
            config.alpha, config.sparsity, rng);
        
        // Create projection matrices B for each scale
        for (int scale : config.scales) {
            size_t input_dim = scale * encoder_.size();
            Bs_[scale] = Matrix::random_projection(config.dimensions, input_dim, rng);
        }
        
        // Precompute multi-char expansions
        for (const auto& [char_str, targets] : config.similarities) {
            for (const auto& [target, weight] : targets) {
                if (target.size() > 1) {
                    expansions_[char_str].push_back({target, weight});
                }
            }
        }
    }
    
    // Constructor with pre-loaded matrices
    WordEmbedderImpl(const WordEmbedConfig& config,
                     const std::vector<std::vector<float>>& A,
                     const std::unordered_map<int, std::vector<std::vector<float>>>& Bs)
        : config_(config), encoder_(config.alphabet, config.similarities), A_(A), Bs_(Bs) {
        
        // Precompute multi-char expansions
        for (const auto& [char_str, targets] : config.similarities) {
            for (const auto& [target, weight] : targets) {
                if (target.size() > 1) {
                    expansions_[char_str].push_back({target, weight});
                }
            }
        }
    }
    
    void embed(const std::string& word, std::vector<float>& embedding) const override {
        embedding = embed_raw(word);
        
        // Expansion blending for multi-char similarities
        if (!expansions_.empty()) {
            auto expanded_forms = expand_word(word);
            if (!expanded_forms.empty()) {
                double total_weight = 1.0;
                for (const auto& [exp_word, weight] : expanded_forms) {
                    auto v_exp = embed_raw(exp_word);
                    for (size_t i = 0; i < embedding.size(); ++i) {
                        embedding[i] += static_cast<float>(weight) * v_exp[i];
                    }
                    total_weight += weight;
                }
                float scale = 1.0f / static_cast<float>(total_weight);
                for (float& val : embedding) {
                    val *= scale;
                }
            }
        }
        
        // Optional normalization
        float norm = 0.0f;
        for (float val : embedding) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm > 1e-8f) {
            for (float& val : embedding) {
                val /= norm;
            }
        }
    }
    
private:
    struct NGram {
        int scale;
        std::string gram;
    };
    
    std::vector<NGram> collect_ngrams(const std::string& word) const {
        std::vector<NGram> ngrams;
        auto chars = utf8_split(word);
        
        for (size_t i = 0; i < chars.size(); ++i) {
            for (int scale : config_.scales) {
                if (i + scale <= chars.size()) {
                    // Concatenate UTF-8 characters to form ngram
                    std::string gram;
                    for (size_t j = i; j < i + scale; ++j) {
                        gram += chars[j];
                    }
                    ngrams.push_back({scale, gram});
                }
            }
        }
        return ngrams;
    }
    
    std::vector<std::pair<std::string, double>> expand_word(const std::string& word) const {
        std::vector<std::pair<std::string, double>> expansions;
        std::string lower_word = to_lower(word);
        
        // Split into UTF-8 characters
        auto chars = utf8_split(lower_word);
        
        // Check each character for expansions
        for (size_t i = 0; i < chars.size(); ++i) {
            auto it = expansions_.find(chars[i]);
            if (it != expansions_.end()) {
                for (const auto& [replacement, weight] : it->second) {
                    // Build expanded word by replacing this character
                    std::string expanded;
                    for (size_t j = 0; j < chars.size(); ++j) {
                        if (j == i) {
                            expanded += replacement;
                        } else {
                            expanded += chars[j];
                        }
                    }
                    expansions.push_back({expanded, weight});
                }
            }
        }
        return expansions;
    }
    
    std::vector<float> embed_raw(const std::string& word) const {
        // Add start boundary marker and convert to lowercase
        std::string processed = "^" + to_lower(word);
        
        // Collect n-grams
        auto ngrams = collect_ngrams(processed);
        size_t total_steps = ngrams.size();
        
        if (total_steps == 0) {
            return std::vector<float>(config_.dimensions, 0.0f);
        }
        
        // Initialize state vectors
        std::vector<float> v_seq(config_.dimensions, 0.0f);
        std::vector<float> v_bag(config_.dimensions, 0.0f);
        
        // Process each n-gram
        for (const auto& ng : ngrams) {
            auto x = encoder_.ngram_vector(ng.gram);
            auto projected = Matrix::mat_vec_mul(Bs_.at(ng.scale), x);
            
            // Sequential update: v = v + tanh(A @ v + projected)
            auto Av = Matrix::mat_vec_mul(A_, v_seq);
            for (size_t i = 0; i < config_.dimensions; ++i) {
                v_seq[i] += std::tanh(Av[i] + projected[i]);
            }
            
            // Bag accumulation
            for (size_t i = 0; i < config_.dimensions; ++i) {
                v_bag[i] += projected[i];
            }
        }
        
        // Apply length normalization with configurable strength
        float ln = static_cast<float>(config_.length_norm);
        float bag_norm = std::pow(static_cast<float>(total_steps), ln);
        float seq_norm = std::pow(static_cast<float>(total_steps), 0.5f * ln);
        
        std::vector<float> result(config_.dimensions);
        for (size_t i = 0; i < config_.dimensions; ++i) {
            result[i] = v_seq[i] / seq_norm + v_bag[i] / bag_norm;
        }
        
        return result;
    }
    
    static std::string to_lower(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(),
                      [](unsigned char c) { return std::tolower(c); });
        return result;
    }
    
    WordEmbedConfig config_;
    CharEncoder encoder_;
    std::vector<std::vector<float>> A_;
    std::unordered_map<int, std::vector<std::vector<float>>> Bs_;
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> expansions_;
};

// =============================================================================
// Factory methods
// =============================================================================

std::shared_ptr<WordEmbedder> WordEmbedder::create(const WordEmbedConfig& config) {
    return std::make_shared<WordEmbedderImpl>(config);
}

std::shared_ptr<WordEmbedder> WordEmbedder::create_from_file(const std::string& filepath, const WordEmbedConfig& config) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open matrix file: " + filepath);
    }
    
    // Read header
    uint32_t num_scales, d, K;
    file.read(reinterpret_cast<char*>(&num_scales), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&K), sizeof(uint32_t));
    
    // Read scales array
    std::vector<int> scales(num_scales);
    for (uint32_t i = 0; i < num_scales; ++i) {
        uint32_t scale;
        file.read(reinterpret_cast<char*>(&scale), sizeof(uint32_t));
        scales[i] = static_cast<int>(scale);
    }
    
    // Verify dimensions match config
    if (d != static_cast<uint32_t>(config.dimensions)) {
        throw std::runtime_error("Matrix file dimensions don't match config");
    }
    
    // Read A matrix
    std::vector<std::vector<float>> A(d, std::vector<float>(d));
    for (uint32_t i = 0; i < d; ++i) {
        for (uint32_t j = 0; j < d; ++j) {
            float val;
            file.read(reinterpret_cast<char*>(&val), sizeof(float));
            A[i][j] = val;
        }
    }
    
    // Read B matrices
    std::unordered_map<int, std::vector<std::vector<float>>> Bs;
    for (int scale : scales) {
        uint32_t rows = d;
        uint32_t cols = scale * K;
        std::vector<std::vector<float>> B(rows, std::vector<float>(cols));
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                float val;
                file.read(reinterpret_cast<char*>(&val), sizeof(float));
                B[i][j] = val;
            }
        }
        Bs[scale] = B;
    }
    
    file.close();
    
    return std::make_shared<WordEmbedderImpl>(config, A, Bs);
}

