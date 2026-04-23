#include "word_embed.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>

using namespace word_embed;

float compute_distance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dist = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        float diff = v1[i] - v2[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

int main() {
    // Configure with default parameters matching the Python version
    WordEmbedConfig config;
    config.dimensions = 256;  // Match Python default
    config.scales = {2, 3, 4};
    config.alpha = 0.9;
    config.sparsity = 0.85;
    config.seed = 42;
    config.alphabet = "abcdefghijklmnopqrstuvwxyz^";
    
    auto embedder = WordEmbedder::create(config);
    
    // Test various word pairs
    std::vector<std::pair<std::string, std::string>> test_pairs = {
        {"cat", "car"},
        {"cat", "bat"},
        {"orange", "oranges"},
        {"create", "created"},
        {"guarantee", "guarante"},
        {"receive", "recieve"},
        {"cat", "dog"},
        {"apple", "orange"}
    };
    
    std::cout << "Word Pair Distances (L2 on normalized embeddings):\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& [word1, word2] : test_pairs) {
        std::vector<float> emb1, emb2;
        embedder->embed(word1, emb1);
        embedder->embed(word2, emb2);
        
        float dist = compute_distance(emb1, emb2);
        
        std::cout << std::left << std::setw(12) << word1 
                  << " vs " << std::setw(12) << word2 
                  << " : " << std::fixed << std::setprecision(4) << dist 
                  << std::endl;
    }
    
    return 0;
}
