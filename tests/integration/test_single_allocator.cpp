#include <iostream>
#include <cassert>
#include <cstring>
#include "stepanov/allocators.hpp"

using namespace stepanov;

void test_heap() {
    std::cout << "Testing heap allocator...\n";
    heap_allocator alloc;
    void* p = alloc.allocate(100);
    assert(p != nullptr);
    std::memset(p, 42, 100);
    alloc.deallocate(p, 100);
    std::cout << "  PASSED\n";
}

void test_fixed_pool() {
    std::cout << "Testing fixed pool allocator...\n";
    fixed_pool_allocator<64, 10> pool;
    void* p1 = pool.allocate(64);
    void* p2 = pool.allocate(32);
    assert(p1 != nullptr);
    assert(p2 != nullptr);
    pool.deallocate(p1, 64);
    pool.deallocate(p2, 32);
    std::cout << "  PASSED\n";
}

void test_region() {
    std::cout << "Testing region allocator...\n";
    region_allocator<> region(1024);
    void* p1 = region.allocate(100);
    void* p2 = region.allocate(200);
    assert(p1 != nullptr);
    assert(p2 != nullptr);
    region.deallocate_all();
    std::cout << "  PASSED\n";
}

void test_freelist() {
    std::cout << "Testing freelist allocator...\n";
    freelist_allocator<heap_allocator> freelist;

    // Allocate and deallocate to build freelist
    void* p1 = freelist.allocate(100);
    void* p2 = freelist.allocate(200);
    assert(p1 != nullptr);
    assert(p2 != nullptr);

    freelist.deallocate(p1, 100);
    freelist.deallocate(p2, 200);

    // Should reuse from freelist
    void* p3 = freelist.allocate(100);
    void* p4 = freelist.allocate(200);
    assert(p3 != nullptr);
    assert(p4 != nullptr);

    freelist.deallocate(p3, 100);
    freelist.deallocate(p4, 200);

    std::cout << "  PASSED\n";
}

void test_fallback() {
    std::cout << "Testing fallback allocator...\n";

    using alloc_type = fallback_allocator<
        fixed_pool_allocator<64, 2>,
        heap_allocator
    >;

    alloc_type alloc;

    // First two should use pool
    void* p1 = alloc.allocate(64);
    void* p2 = alloc.allocate(64);
    assert(p1 != nullptr);
    assert(p2 != nullptr);

    // Third should fallback to heap
    void* p3 = alloc.allocate(64);
    assert(p3 != nullptr);

    alloc.deallocate(p1, 64);
    alloc.deallocate(p2, 64);
    alloc.deallocate(p3, 64);

    std::cout << "  PASSED\n";
}

void test_segregator() {
    std::cout << "Testing segregator allocator...\n";

    using alloc_type = segregator_allocator<
        256,
        fixed_pool_allocator<256, 10>,
        heap_allocator
    >;

    alloc_type alloc;

    // Small allocation should use pool
    void* small = alloc.allocate(100);
    assert(small != nullptr);

    // Large allocation should use heap
    void* large = alloc.allocate(500);
    assert(large != nullptr);

    alloc.deallocate(small, 100);
    alloc.deallocate(large, 500);

    std::cout << "  PASSED\n";
}

void test_stats() {
    std::cout << "Testing stats allocator...\n";

    stats_allocator<heap_allocator> alloc;

    void* p1 = alloc.allocate(100);
    void* p2 = alloc.allocate(200);

    assert(alloc.stats().allocations == 2);
    assert(alloc.stats().bytes_allocated == 300);

    alloc.deallocate(p1, 100);

    assert(alloc.stats().deallocations == 1);
    assert(alloc.stats().current_bytes == 200);

    alloc.deallocate(p2, 200);

    std::cout << "  PASSED\n";
}

int main() {
    test_heap();
    test_fixed_pool();
    test_region();
    test_freelist();
    test_fallback();
    test_segregator();
    test_stats();

    std::cout << "\nAll tests passed!\n";
    return 0;
}