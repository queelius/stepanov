#include <iostream>
#include <cassert>
#include <vector>
#include "stepanov/allocators.hpp"

using namespace stepanov;

int main() {
    std::cout << "Testing bitmapped block allocator...\n";

    bitmapped_block_allocator<128, 64> bitmap;

    assert(bitmap.capacity() == 64);
    assert(bitmap.free_blocks() == 64);

    // Allocate some blocks
    std::vector<void*> blocks;
    for (int i = 0; i < 10; ++i) {
        void* p = bitmap.allocate(128);
        assert(p != nullptr);
        blocks.push_back(p);
    }

    std::cout << "  Allocated 10 blocks\n";
    std::cout << "  Free blocks: " << bitmap.free_blocks() << "\n";

    // Deallocate
    for (void* p : blocks) {
        bitmap.deallocate(p, 128);
    }

    std::cout << "  Free blocks after deallocation: " << bitmap.free_blocks() << "\n";
    assert(bitmap.free_blocks() == 64);

    std::cout << "PASSED\n";
    return 0;
}