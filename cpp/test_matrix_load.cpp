#include "word_embed.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace word_embed;

float l2_distance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dist = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        float diff = v1[i] - v2[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

int main() {
    // Configuration matching Python test
    WordEmbedConfig config;
    config.alphabet = "abcdefghijklmnopqrstuvwxyzöäüß^";
    config.scales = {2, 3, 4};
    config.dimensions = 64;
    config.alpha = 0.9;
    config.sparsity = 0.85;
    config.seed = 42;
    config.length_norm = 1.0;
    
    // Character similarities for German (using string keys for UTF-8)
    config.similarities = {
        {"ö", {{"o", 0.4}, {"oe", 0.95}}},
        {"ä", {{"a", 0.4}, {"ae", 0.95}}},
        {"ü", {{"u", 0.4}, {"ue", 0.95}}},
        {"ß", {{"ss", 0.95}}}
    };
    
    std::cout << "Loading matrices from ../matrices.bin..." << std::endl;
    
    try {
        auto embedder = WordEmbedder::create_from_file("../matrices.bin", config);
        std::cout << "Matrices loaded successfully." << std::endl;
        
        // Test with the same words as Python
        std::vector<std::pair<std::string, std::string>> test_words = {
            {"cat", "car"},
            {"guarantee", "guarentee"},
            {"könig", "koenig"},
            {"straße", "strasse"}
        };
        
        std::cout << "\nC++ embeddings (with loaded matrices):" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        
        for (const auto& [w1, w2] : test_words) {
            std::vector<float> v1, v2;
            embedder->embed(w1, v1);
            embedder->embed(w2, v2);
            
            // Debug: print first few dimensions
            if (w1 == "cat") {
                std::cout << "\nDebug - cat embedding (first 5 dims): ";
                for (int i = 0; i < 5 && i < v1.size(); ++i) {
                    std::cout << v1[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Debug - car embedding (first 5 dims): ";
                for (int i = 0; i < 5 && i < v2.size(); ++i) {
                    std::cout << v2[i] << " ";
                }
                std::cout << std::endl;
            }
            
            float dist = l2_distance(v1, v2);
            
            std::cout << "  " << std::setw(12) << std::left << w1 
                      << " vs " << std::setw(12) << std::left << w2 
                      << ": " << dist << std::endl;
        }
        
        std::cout << "\n✓ C++ matrix loading test complete!" << std::endl;
        std::cout << "Compare distances with Python output above." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
