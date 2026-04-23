#include <iostream>
#include <string>
#include <vector>

// Helper to iterate through UTF-8 characters
std::vector<std::string> utf8_chars(const std::string& str) {
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
        result.push_back(str.substr(i, char_len));
        i += char_len;
    }
    return result;
}

int main() {
    std::string test = "könig";
    std::cout << "String: " << test << std::endl;
    std::cout << "Byte length: " << test.size() << std::endl;
    
    auto chars = utf8_chars(test);
    std::cout << "UTF-8 character count: " << chars.size() << std::endl;
    for (const auto& ch : chars) {
       std::cout << "  char: " << ch << " (bytes: " << ch.size() << ")" << std::endl;
    }
    
    return 0;
}
