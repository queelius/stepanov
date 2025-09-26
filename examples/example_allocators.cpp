// Example demonstrating compositional allocators in the Stepanov library
// Following generic programming principles for zero-overhead abstraction

#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <chrono>
#include <iomanip>
#include "stepanov/allocators.hpp"

using namespace stepanov;
using namespace std::chrono;

// Example 1: Simple composition for embedded systems
void example_embedded_allocator() {
    std::cout << "\n=== EMBEDDED SYSTEM ALLOCATOR ===\n";
    std::cout << "Fixed memory, no heap allocation allowed\n\n";

    // Composition: small objects from stack, larger from fixed pool
    using embedded_alloc = fallback_allocator<
        stack_allocator<4096>,                    // 4KB stack buffer for small allocs
        fixed_pool_allocator<256, 32>             // 32 blocks of 256 bytes for larger
    >;

    embedded_alloc allocator;

    // Small allocations use stack
    void* small1 = allocator.allocate(64);
    void* small2 = allocator.allocate(128);
    std::cout << "Allocated 64 + 128 bytes from stack\n";

    // Larger allocation uses pool
    void* large = allocator.allocate(256);
    std::cout << "Allocated 256 bytes from fixed pool\n";

    // Stack allocation fails when exhausted, falls back to pool
    void* test = allocator.allocate(4000);  // Too big for remaining stack
    if (test) {
        std::cout << "Large allocation fell back to pool\n";
    }

    allocator.deallocate(small1, 64);
    allocator.deallocate(small2, 128);
    allocator.deallocate(large, 256);
    if (test) allocator.deallocate(test, 4000);
}

// Example 2: High-performance game allocator
void example_game_allocator() {
    std::cout << "\n=== GAME ENGINE ALLOCATOR ===\n";
    std::cout << "Optimized for different object sizes with statistics\n\n";

    // Composition: segregate by size, track stats, add debug info
    using game_alloc = stats_allocator<
        segregator_allocator<
            64,                                    // Threshold
            fixed_pool_allocator<64, 1024>,       // Small: pool for particles
            segregator_allocator<
                1024,                              // Second threshold
                freelist_allocator<heap_allocator>, // Medium: recycled heap
                heap_allocator                     // Large: direct heap
            >
        >
    >;

    game_alloc allocator;

    // Simulate game allocations
    std::vector<std::pair<void*, std::size_t>> objects;

    // Small objects (particles, bullets)
    for (int i = 0; i < 100; ++i) {
        void* p = allocator.allocate(32);
        objects.push_back({p, 32});
    }
    std::cout << "Allocated 100 small objects (particles)\n";

    // Medium objects (enemies, items)
    for (int i = 0; i < 20; ++i) {
        void* p = allocator.allocate(512);
        objects.push_back({p, 512});
    }
    std::cout << "Allocated 20 medium objects (enemies)\n";

    // Large objects (level data)
    void* level = allocator.allocate(8192);
    std::cout << "Allocated large object (level data)\n";

    // Print statistics
    const auto& stats = allocator.stats();
    std::cout << "\nAllocation Statistics:\n";
    std::cout << "  Total allocations: " << stats.allocations << "\n";
    std::cout << "  Total bytes: " << stats.bytes_allocated << "\n";
    std::cout << "  Peak memory: " << stats.peak_bytes << " bytes\n";

    // Cleanup
    for (auto [p, size] : objects) {
        allocator.deallocate(p, size);
    }
    allocator.deallocate(level, 8192);
}

// Example 3: Server application with per-request allocation
void example_server_allocator() {
    std::cout << "\n=== SERVER REQUEST ALLOCATOR ===\n";
    std::cout << "Region-based allocation per request\n\n";

    // Simulate handling multiple requests
    for (int request = 1; request <= 3; ++request) {
        std::cout << "Processing request #" << request << "\n";

        // Each request gets its own region allocator
        region_allocator<heap_allocator> request_allocator(8192);

        // Allocate various objects for request processing
        void* buffer = request_allocator.allocate(1024);
        void* response = request_allocator.allocate(2048);
        void* temp = request_allocator.allocate(512);

        // Process request...
        std::cout << "  Allocated " << request_allocator.bytes_allocated()
                  << " bytes for request\n";

        // All memory automatically freed when allocator goes out of scope
    }
    std::cout << "All request memory automatically freed\n";
}

// Example 4: Complex real-world composition
void example_production_allocator() {
    std::cout << "\n=== PRODUCTION SYSTEM ALLOCATOR ===\n";
    std::cout << "Complex composition for real-world application\n\n";

    // Debug header for tracking
    struct alloc_info {
        std::size_t size;
        std::uint32_t magic = 0xDEADBEEF;
        std::uint64_t timestamp;
    };

    // Build a sophisticated allocator:
    // - Add debug headers
    // - Track statistics
    // - Use different strategies by size
    // - Thread-safe
    using production_alloc = synchronized_allocator<
        affixed_allocator<
            stats_allocator<
                segregator_allocator<
                    256,                               // Small threshold
                    bitmapped_block_allocator<256, 128>, // Bitmap for small
                    segregator_allocator<
                        4096,                          // Medium threshold
                        slab_allocator<char, 64>,     // Slab for medium
                        freelist_allocator<heap_allocator> // Freelist for large
                    >
                >
            >,
            alloc_info,                               // Prefix
            void                                       // No suffix
        >
    >;

    production_alloc allocator;

    // Make various allocations
    auto start = high_resolution_clock::now();

    void* p1 = allocator.allocate(100);
    void* p2 = allocator.allocate(500);
    void* p3 = allocator.allocate(5000);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    std::cout << "Allocated 3 objects of different sizes\n";
    std::cout << "Time taken: " << duration << " microseconds\n";

    // Access debug info
    auto* info = allocator.inner().prefix<alloc_info>(p1);
    std::cout << "\nDebug info for first allocation:\n";
    std::cout << "  Size: " << info->size << "\n";
    std::cout << "  Magic: 0x" << std::hex << info->magic << std::dec << "\n";

    // Get statistics
    const auto& stats = allocator.inner().inner().stats();
    std::cout << "\nAllocation statistics:\n";
    std::cout << "  Allocations: " << stats.allocations << "\n";
    std::cout << "  Current bytes: " << stats.current_bytes << "\n";

    // Cleanup
    allocator.deallocate(p1, 100);
    allocator.deallocate(p2, 500);
    allocator.deallocate(p3, 5000);
}

// Example 5: STL container integration
void example_stl_integration() {
    std::cout << "\n=== STL CONTAINER INTEGRATION ===\n";
    std::cout << "Using custom allocators with STL containers\n\n";

    // Create a high-performance allocator for containers
    using pool_type = fixed_pool_allocator<32, 1024>;
    using fallback_type = fallback_allocator<pool_type, heap_allocator>;
    using tracked_type = stats_allocator<fallback_type>;

    tracked_type allocator;

    // Adapt for STL use
    using vec_alloc = stl_allocator_adaptor<int, tracked_type>;
    std::vector<int, vec_alloc> vec{vec_alloc(&allocator)};

    // Add elements
    for (int i = 0; i < 100; ++i) {
        vec.push_back(i * i);
    }

    std::cout << "Vector with " << vec.size() << " elements\n";

    // Use with list
    using list_alloc = stl_allocator_adaptor<std::string, tracked_type>;
    std::list<std::string, list_alloc> list{list_alloc(&allocator)};

    list.push_back("Hello");
    list.push_back("World");
    list.push_back("From");
    list.push_back("Custom");
    list.push_back("Allocator");

    std::cout << "List with " << list.size() << " strings\n";

    // Print statistics
    const auto& stats = allocator.stats();
    std::cout << "\nTotal allocations: " << stats.allocations << "\n";
    std::cout << "Peak memory: " << stats.peak_bytes << " bytes\n";
    std::cout << "Current memory: " << stats.current_bytes << " bytes\n";
}

// Example 6: Performance comparison
void example_performance_comparison() {
    std::cout << "\n=== PERFORMANCE COMPARISON ===\n";
    std::cout << "Comparing allocator compositions\n\n";

    constexpr int iterations = 10000;
    constexpr int alloc_size = 256;

    // Test 1: Direct heap
    {
        heap_allocator alloc;
        auto start = high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            void* p = alloc.allocate(alloc_size);
            std::memset(p, 0, alloc_size);
            alloc.deallocate(p, alloc_size);
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        std::cout << "Heap allocator: " << duration << " μs\n";
    }

    // Test 2: Fixed pool
    {
        fixed_pool_allocator<256, 1> alloc;
        auto start = high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            void* p = alloc.allocate(alloc_size);
            if (p) {
                std::memset(p, 0, alloc_size);
                alloc.deallocate(p, alloc_size);
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        std::cout << "Fixed pool: " << duration << " μs\n";
    }

    // Test 3: Region allocator
    {
        region_allocator<> alloc(alloc_size * 100);
        auto start = high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            void* p = alloc.allocate(alloc_size);
            std::memset(p, 0, alloc_size);
            if (i % 100 == 99) {
                alloc.deallocate_all();
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        std::cout << "Region allocator: " << duration << " μs\n";
    }

    // Test 4: Composed allocator
    {
        using composed = fallback_allocator<
            fixed_pool_allocator<256, 1>,
            heap_allocator
        >;
        composed alloc;
        auto start = high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            void* p = alloc.allocate(alloc_size);
            std::memset(p, 0, alloc_size);
            alloc.deallocate(p, alloc_size);
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        std::cout << "Composed (pool+heap): " << duration << " μs\n";
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  COMPOSITIONAL ALLOCATORS SHOWCASE\n";
    std::cout << "  Stepanov Library Implementation\n";
    std::cout << "========================================\n";

    example_embedded_allocator();
    example_game_allocator();
    example_server_allocator();
    example_production_allocator();
    example_stl_integration();
    example_performance_comparison();

    std::cout << "\n========================================\n";
    std::cout << "All examples completed successfully!\n";
    std::cout << "========================================\n";

    return 0;
}