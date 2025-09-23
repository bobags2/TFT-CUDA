#include <iostream>
#include <vector>

// Simple C++ function for testing compilation
extern "C" {
    void test_function() {
        std::cout << "Simple C++ compilation test successful" << std::endl;
    }
}

int main() {
    test_function();
    return 0;
}