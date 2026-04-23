#include "word_embed.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstdint>

using namespace word_embed;

// Forward declare to access internals (hacky but for debugging)
extern void debug_print_loaded_matrices(const std::string& filepath, const WordEmbedConfig& config);

int main() {
    WordEmbedConfig config;
    config.alphabet = "abcdefghijklmnopqrstuvwxyzöäüß^";
    config.scales = {2, 3, 4};
    config.dimensions = 64;
    
    debug_print_loaded_matrices("../matrices.bin", config);
    
    return 0;
}

// Debug function to print loaded matrices
void debug_print_loaded_matrices(const std::string& filepath, const WordEmbedConfig& config) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }
    
    // Read header
    uint32_t num_scales, d, K;
    file.read(reinterpret_cast<char*>(&num_scales), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&K), sizeof(uint32_t));
    
    std::cout << "Header:" << std::endl;
    std::cout << "  num_scales: " << num_scales << std::endl;
    std::cout << "  d: " << d << std::endl;
    std::cout << "  K: " << K << std::endl;
    
    // Read scales
    std::vector<uint32_t> scales(num_scales);
    for (uint32_t i = 0; i < num_scales; ++i) {
        file.read(reinterpret_cast<char*>(&scales[i]), sizeof(uint32_t));
    }
    std::cout << "  scales: [";
    for (size_t i = 0; i < scales.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << scales[i];
    }
    std::cout << "]" << std::endl;
    
    // Read first 5 elements of A matrix
    std::cout << "\nA matrix first row (first 5 elements): [";
    std::cout << std::fixed << std::setprecision(8);
    for (int i = 0; i < 5; ++i) {
        float val;
        file.read(reinterpret_cast<char*>(&val), sizeof(float));
        if (i > 0) std::cout << ", ";
        std::cout << val;
    }
    std::cout << "]" << std::endl;
    
    file.close();
}
