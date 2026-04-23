g++ -std=c++17 -O2 -o test_word_embed test_word_embed.cpp word_embed.cpp -lm
echo "test_word_embed built successfully"

g++ -std=c++17 -O2 -o test_matrix_load test_matrix_load.cpp word_embed.cpp -lm
echo "test_matrix_load built successfully"
