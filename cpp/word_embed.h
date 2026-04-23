// @file_purpose Compact Character-Level Word Embedding
// derived from word_embed.py
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

// =============================================================================
// Compact Character-Level Word Embedding
//
// A training-free, deterministic algorithm that embeds words into fixed-size
// vectors where L2 distance correlates with edit distance.
// =============================================================================

namespace word_embed {

// Configuration for word embedding
struct WordEmbedConfig {
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz^";
    std::vector<int> scales = {2, 3, 4};
    int dimensions = 64;       // embedding dimensionality
    double alpha = 0.9;         // spectral radius target for A matrix
    double sparsity = 0.85;     // fraction of zeros in A matrix
    int seed = 42;            // random seed for reproducibility
    double length_norm = 1.0;   // length normalization strength [0, 1]
    
    // Character similarities: map from string to list of (target, weight) pairs
    // Use string keys to support multi-byte UTF-8 characters like "ö", "ä", "ü", "ß"
    // e.g., {"ö": {{"o", 0.7}, {"oe", 0.5}}}
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> similarities;
};

/**
 * Abstract base class for word embedding. The main method is `embed` which takes a word
 * and produces its embedding vector.
 * Factory methods `create` and `create_from_file` are provided for constructing instances.
 */
class WordEmbedder {
public:
    /// @brief Embed a word into a vector. The resulting embedding should be normalized to unit length.
    /// @param word 
    /// @param embedding 
    virtual void embed(const std::string& word, std::vector<float> & embedding) const = 0;
    
    /// @brief Create a WordEmbedder instance with the given configuration. This will generate random matrices internally.
    /// @param config Configuration object (alphabet, scales, etc.) that should match the matrices    /// @return 
    static std::shared_ptr<WordEmbedder> create(const WordEmbedConfig& config);
    
    /// @brief Create a WordEmbedder instance by loading matrices from a binary file. The file format should match the one produced by `save_matrices`.
    /// @param filepath File path to the binary file containing the matrices.
    /// @param config Configuration object (alphabet, scales, etc.) that should match the matrices
    static std::shared_ptr<WordEmbedder> create_from_file(const std::string& filepath, const WordEmbedConfig& config);
};

} // namespace word_embed

