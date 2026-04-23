#include "word_embed.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <numeric>

using namespace word_embed;

// Compute L2 distance between two vectors
float l2_distance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dist = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        float diff = v1[i] - v2[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

// Compute average distance for a list of word pairs
float avg_distance(const std::vector<std::tuple<std::string, std::string, std::string>>& pairs,
                   const std::map<std::string, std::vector<float>>& vectors) {
    float sum = 0.0f;
    for (const auto& [w1, w2, desc] : pairs) {
        sum += l2_distance(vectors.at(w1), vectors.at(w2));
    }
    return sum / pairs.size();
}

int main() {
    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    
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
    
    auto embedder = WordEmbedder::create_from_file("../bin/matrices.bin", config);
    
    // -------------------------------------------------------------------------
    // Test groups
    // -------------------------------------------------------------------------
    
    std::map<std::string, std::vector<std::tuple<std::string, std::string, std::string>>> groups;
    
    groups["Misspellings (expect: small distances)"] = {
        {"guarantee", "guarante", "1 deletion (middle)"},
        {"accept", "acept", "1 deletion (middle)"},
        {"library", "libary", "1 deletion (middle)"},
        {"calendar", "calender", "1 substitution (middle)"},
        {"guarantee", "guarentee", "1 substitution"},
        {"separate", "seperate", "1 substitution (common mistake)"},
        {"definite", "definate", "1 substitution (common mistake)"},
        {"receive", "recieve", "transposition (ie→ei)"},
        {"necessary", "neccessary", "1 insertion (double c)"},
        {"necessary", "necesary", "1 deletion"},
        {"accept", "except", "1 substitution at start"},
        {"knight", "night", "1 deletion at start"},
        {"their", "there", "1 substitution + deletion at end"},
        {"16-bit", "16 bit", "1 substitution with letters both not in alphabet(treated as same)"},
        {"16-bit", "16bit", "1 insertion of letter not in alphabet"}
    };
    
    groups["Morphological (expect: small distances)"] = {
        {"orange", "oranges", "plural suffix -s"},
        {"run", "running", "verb form -ning"},
        {"happy", "happily", "adverb form -ily"},
        {"create", "created", "past tense -d"},
        {"create", "creating", "gerund -ing"},
        {"quick", "quickly", "adverb -ly"}
    };
    
    groups["Position symmetry (expect: similar distances for all)"] = {
        {"cat", "bat", "edit at start (c→b)"},
        {"cat", "car", "edit at end (t→r)"},
        {"cat", "cot", "edit in middle (a→o)"},
        {"lamp", "ramp", "edit at start (l→r)"},
        {"lamp", "lump", "edit in middle (a→u)"},
        {"lamp", "lamb", "edit at end (p→b)"}
    };
    
    groups["Unrelated words (expect: large distances)"] = {
        {"guarantee", "banana", "long vs medium"},
        {"cat", "banana", "short vs medium"},
        {"apple", "orange", "same length"},
        {"cat", "dog", "short vs short"},
        {"elephant", "telephone", "shared letters, unrelated"},
        {"happy", "river", "same length, no overlap"}
    };
    
    groups["Anagram / reversal (expect: large distances)"] = {
        {"listen", "silent", "anagram"},
        {"star", "rats", "reversal"},
        {"dog", "god", "reversal"},
        {"space", "capes", "anagram"}
    };
    
    groups["German umlaut→base (expect: small distances after similarity)"] = {
        {"könig", "konig", "ö→o"},
        {"schön", "schon", "ö→o"},
        {"größe", "grosse", "ö→o + ß→ss (length change)"},
        {"über", "uber", "ü→u"},
        {"müller", "muller", "ü→u"},
        {"fähre", "fahre", "ä→a"},
        {"bär", "bar", "ä→a"},
        {"hände", "hande", "ä→a"}
    };
    
    groups["German umlaut→transcription (expect: small distances after similarity)"] = {
        {"könig", "koenig", "ö→oe"},
        {"schön", "schoen", "ö→oe"},
        {"über", "ueber", "ü→ue"},
        {"müller", "mueller", "ü→ue"},
        {"fähre", "faehre", "ä→ae"},
        {"bär", "baer", "ä→ae"},
        {"hände", "haende", "ä→ae"}
    };
    
    groups["German ß variants (expect: small distances after similarity)"] = {
        {"straße", "strasse", "ß→ss"},
        {"groß", "gross", "ß→ss"},
        {"fuß", "fuss", "ß→ss"},
        {"maß", "mass", "ß→ss"},
        {"heißen", "heissen", "ß→ss"},
        {"blumenstrauß", "blumenstrauss", "ß→ss"},
        {"blumenstraße", "blumenstrasse", "ß→ss"}
    };
    
    groups["German unrelated (expect: large distances)"] = {
        {"könig", "straße", "unrelated"},
        {"über", "hände", "unrelated"},
        {"schön", "müller", "unrelated"},
        {"bär", "fuß", "unrelated"},
        {"strauß", "straße", "unrelated"}
    };
    
    groups["Country-specific spelling (US vs UK/AU English) (expect: small distances)"] = {
        {"colour", "color", "-our vs -or"},
        {"favour", "favor", "-our vs -or"},
        {"honour", "honor", "-our vs -or"},
        {"behaviour", "behavior", "-our vs -or"},
        {"centre", "center", "-re vs -er"},
        {"theatre", "theater", "-re vs -er"},
        {"metre", "meter", "-re vs -er"},
        {"organise", "organize", "-ise vs -ize"},
        {"realise", "realize", "-ise vs -ize"},
        {"recognise", "recognize", "-ise vs -ize"},
        {"travelling", "traveling", "double-l inflection"},
        {"cancelling", "canceling", "double-l inflection"},
        {"labelled", "labeled", "double-l inflection"},
        {"catalogue", "catalog", "-ogue vs -og"},
        {"dialogue", "dialog", "-ogue vs -og"},
        {"defence", "defense", "-ence vs -ense"},
        {"offence", "offense", "-ence vs -ense"},
        {"licence", "license", "-ence vs -ense"},
        {"programme", "program", "-mme vs -m"},
        {"grey", "gray", "-ey vs -ay"},
        {"jewellery", "jewelry", "double-l vs single"},
        {"mould", "mold", "-ould vs -old"},
        {"tyre", "tire", "y vs i"},
        {"aluminium", "aluminum", "AU/UK vs US"}
    };
    
    // -------------------------------------------------------------------------
    // Collect all unique words and compute embeddings
    // -------------------------------------------------------------------------
    
    std::set<std::string> all_words_set;
    for (const auto& [group_name, pairs] : groups) {
        for (const auto& [w1, w2, desc] : pairs) {
            all_words_set.insert(w1);
            all_words_set.insert(w2);
        }
    }
    
    std::map<std::string, std::vector<float>> vectors;
    for (const auto& word : all_words_set) {
        std::vector<float> embedding;
        embedder->embed(word, embedding);
        vectors[word] = embedding;
    }
    
    // -------------------------------------------------------------------------
    // Print results by group
    // -------------------------------------------------------------------------
    
    std::cout << std::fixed << std::setprecision(4);
    
    for (const auto& [group_name, pairs] : groups) {
        std::cout << "--- " << group_name << " ---\n";
        for (const auto& [w1, w2, desc] : pairs) {
            float dist = l2_distance(vectors[w1], vectors[w2]);
            std::cout << "  " << std::setw(12) << std::left << w1 
                      << " vs " << std::setw(12) << std::left << w2 
                      << ": " << std::setw(8) << std::right << dist
                      << "   (" << desc << ")\n";
        }
        std::cout << "\n";
    }
    
    // -------------------------------------------------------------------------
    // Summary statistics
    // -------------------------------------------------------------------------
    
    std::vector<std::string> similar_group_names = {
        "Misspellings (expect: small distances)",
        "Morphological (expect: small distances)",
        "Country-specific spelling (US vs UK/AU English) (expect: small distances)"
    };
    
    std::vector<std::string> distant_group_names = {
        "Unrelated words (expect: large distances)",
        "Anagram / reversal (expect: large distances)"
    };
    
    std::vector<std::string> german_similar_group_names = {
        "German umlaut→base (expect: small distances after similarity)",
        "German umlaut→transcription (expect: small distances after similarity)",
        "German ß variants (expect: small distances after similarity)"
    };
    
    std::vector<std::string> german_distant_group_names = {
        "German unrelated (expect: large distances)"
    };
    
    std::vector<std::tuple<std::string, std::string, std::string>> similar_pairs;
    std::vector<std::tuple<std::string, std::string, std::string>> distant_pairs;
    std::vector<std::tuple<std::string, std::string, std::string>> german_similar_pairs;
    std::vector<std::tuple<std::string, std::string, std::string>> german_distant_pairs;
    
    for (const auto& name : similar_group_names) {
        const auto& pairs = groups[name];
        similar_pairs.insert(similar_pairs.end(), pairs.begin(), pairs.end());
    }
    
    for (const auto& name : distant_group_names) {
        const auto& pairs = groups[name];
        distant_pairs.insert(distant_pairs.end(), pairs.begin(), pairs.end());
    }
    
    for (const auto& name : german_similar_group_names) {
        const auto& pairs = groups[name];
        german_similar_pairs.insert(german_similar_pairs.end(), pairs.begin(), pairs.end());
    }
    
    for (const auto& name : german_distant_group_names) {
        const auto& pairs = groups[name];
        german_distant_pairs.insert(german_distant_pairs.end(), pairs.begin(), pairs.end());
    }
    
    float avg_sim = avg_distance(similar_pairs, vectors);
    float avg_far = avg_distance(distant_pairs, vectors);
    float ratio = avg_far / avg_sim;
    
    std::cout << "=================================================================\n";
    std::cout << "SUMMARY: Separation quality (higher = better discrimination)\n";
    std::cout << "=================================================================\n\n";
    
    std::cout << "  " << std::setw(35) << std::left << "Metric" 
              << std::setw(10) << std::right << "Ours\n";
    std::cout << "  " << std::string(35, '-') << " " << std::string(8, '-') << "\n";
    std::cout << "  " << std::setw(35) << std::left << "Avg similar distance" 
              << std::setw(10) << std::right << avg_sim << "\n";
    std::cout << "  " << std::setw(35) << std::left << "Avg unrelated distance" 
              << std::setw(10) << std::right << avg_far << "\n";
    std::cout << "  " << std::setw(35) << std::left << "Separation ratio (far/similar)" 
              << std::setw(8) << std::right << std::setprecision(2) << ratio << "x\n";
    std::cout << std::setprecision(4);
    std::cout << "\n";
    
    std::cout << "  " << std::setw(35) << std::left << "Embedding dimension" 
              << std::setw(10) << std::right << config.dimensions << "\n";
    std::cout << "  " << std::setw(35) << std::left << "Model size (approx)" 
              << std::setw(10) << std::right << "~100KB\n";
    std::cout << "  " << std::setw(35) << std::left << "Requires training data" 
              << std::setw(10) << std::right << "No\n";
    std::cout << "  " << std::setw(35) << std::left << "Handles any string" 
              << std::setw(10) << std::right << "Yes\n";
    std::cout << "\n";
    
    // -------------------------------------------------------------------------
    // German-specific summary
    // -------------------------------------------------------------------------
    
    float avg_gsim = avg_distance(german_similar_pairs, vectors);
    float avg_gfar = avg_distance(german_distant_pairs, vectors);
    float ratio_g = avg_gfar / avg_gsim;
    
    std::cout << "=================================================================\n";
    std::cout << "GERMAN: Separation quality (umlaut/ß variants vs unrelated)\n";
    std::cout << "=================================================================\n\n";
    std::cout << "  " << std::setw(40) << std::left << "Avg similar distance (ö→o,oe etc)" 
              << std::setw(10) << std::right << avg_gsim << "\n";
    std::cout << "  " << std::setw(40) << std::left << "Avg unrelated distance" 
              << std::setw(10) << std::right << avg_gfar << "\n";
    std::cout << "  " << std::setw(40) << std::left << "Separation ratio (far/similar)" 
              << std::setw(8) << std::right << std::setprecision(2) << ratio_g << "x\n";
    std::cout << "\n";
    
    // -------------------------------------------------------------------------
    // Length normalization comparison
    // -------------------------------------------------------------------------
    
    std::vector<std::tuple<std::string, std::string, std::string>> length_test_pairs = {
        {"cat", "cot", "3 chars"},
        {"lamp", "lump", "4 chars"},
        {"happy", "hippy", "5 chars"},
        {"bridge", "briage", "6 chars"},
        {"calendar", "calender", "8 chars"},
        {"separate", "seperate", "8 chars"},
        {"guarantee", "guarentee", "9 chars"},
        {"committee", "comittee", "9 chars"}
    };
    
    std::vector<double> ln_values = {0.0, 0.25, 0.5, 0.75, 1.0};
    
    std::cout << "==============================================================================\n";
    std::cout << "LENGTH NORMALIZATION: same edit type, varying word length\n";
    std::cout << "  length_norm=1.0 (default): long words → smaller distances\n";
    std::cout << "  length_norm=0.0:           distances independent of word length\n";
    std::cout << "==============================================================================\n\n";
    
    std::cout << "  " << std::setw(28) << std::left << "Pair";
    for (double ln : ln_values) {
        std::cout << "  ln=" << std::fixed << std::setprecision(2) << ln << "  ";
    }
    std::cout << "\n";
    
    std::cout << "  " << std::string(28, '-');
    for (size_t i = 0; i < ln_values.size(); ++i) {
        std::cout << "  " << std::string(8, '-');
    }
    std::cout << "\n";
    
    std::cout << std::setprecision(4);
    for (const auto& [w1, w2, desc] : length_test_pairs) {
        std::cout << "  " << std::setw(12) << std::right << w1 
                  << "/" << std::setw(14) << std::left << w2;
        
        for (double ln : ln_values) {
            WordEmbedConfig ln_config = config;
            ln_config.length_norm = ln;
            auto ln_embedder = WordEmbedder::create(ln_config);
            
            std::vector<float> v1, v2;
            ln_embedder->embed(w1, v1);
            ln_embedder->embed(w2, v2);
            float dist = l2_distance(v1, v2);
            
            std::cout << "  " << std::setw(8) << std::right << dist;
        }
        std::cout << "   (" << desc << ")\n";
    }
    std::cout << "\n";
    
    return 0;
}

