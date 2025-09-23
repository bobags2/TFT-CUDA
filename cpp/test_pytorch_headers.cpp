#include <iostream>

// Test if we can include PyTorch headers without template errors
#define TORCH_EXTENSION_NAME test_pytorch
#define _GLIBCXX_USE_CXX11_ABI 1

// Disable all warnings that cause template instantiation to fail
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wplacement-new"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wstringop-overflow"

// Critical: Include standard new operators (no custom definitions needed)
#include <new>

// Include minimal PyTorch headers
#include <torch/torch.h>

#pragma GCC diagnostic pop

int main() {
    std::cout << "PyTorch header inclusion test successful" << std::endl;
    
    // Test basic tensor creation
    auto tensor = torch::zeros({2, 3});
    std::cout << "Tensor creation successful: " << tensor.sizes() << std::endl;
    
    return 0;
}