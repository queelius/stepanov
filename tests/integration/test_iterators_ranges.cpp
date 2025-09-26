#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include "../include/stepanov/iterators.hpp"
#include "../include/stepanov/ranges.hpp"
#include "../include/stepanov/memory.hpp"
#include "../include/stepanov/parallel.hpp"

using namespace stepanov;
using stepanov::iterator_distance;
using stepanov::iterator_advance;
using stepanov::iterator_next;
using stepanov::iterator_prev;

// Test iterator utilities
void test_iterators() {
    std::cout << "Testing iterators...\n";

    std::vector<int> v = {1, 2, 3, 4, 5};

    // Test distance
    auto dist = iterator_distance(v.begin(), v.end());
    assert(dist == 5);

    // Test advance
    auto it = v.begin();
    iterator_advance(it, 3);
    assert(*it == 4);

    // Test next/prev
    auto next_it = iterator_next(v.begin(), 2);
    assert(*next_it == 3);

    auto prev_it = iterator_prev(v.end(), 2);
    assert(*prev_it == 4);

    // Test iterator_pair
    iterator_pair range(v.begin(), v.end());
    assert(!range.empty());
    assert(range.size() == 5);

    // Test counted_iterator
    counted_iterator counted(v.begin(), 3);
    assert(counted.count() == 3);
    assert(*counted == 1);
    ++counted;
    assert(counted.count() == 2);
    assert(*counted == 2);

    std::cout << "Iterator tests passed!\n";
}

// Test range adaptors
void test_ranges() {
    std::cout << "Testing ranges...\n";

    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Test filter view
    auto is_even = [](int x) { return x % 2 == 0; };
    filter_view filtered(v, is_even);

    std::vector<int> filtered_result;
    for (auto x : filtered) {
        filtered_result.push_back(x);
    }
    assert(filtered_result == std::vector<int>({2, 4, 6, 8, 10}));

    // Test transform view
    auto double_it = [](int x) { return x * 2; };
    transform_view transformed(v, double_it);

    std::vector<int> transformed_result;
    for (auto x : transformed) {
        transformed_result.push_back(x);
    }
    assert(transformed_result == std::vector<int>({2, 4, 6, 8, 10, 12, 14, 16, 18, 20}));

    // Test take view
    take_view taken(v, 3);
    std::vector<int> taken_result;
    for (auto x : taken) {
        taken_result.push_back(x);
    }
    assert(taken_result == std::vector<int>({1, 2, 3}));

    // Test drop view
    drop_view dropped(v, 7);
    std::vector<int> dropped_result;
    for (auto x : dropped) {
        dropped_result.push_back(x);
    }
    assert(dropped_result == std::vector<int>({8, 9, 10}));

    // Test enumerate view
    enumerate_view enumerated(v);
    std::vector<std::pair<std::ptrdiff_t, int>> enum_result;
    for (auto [idx, val] : enumerated) {
        enum_result.push_back({idx, val});
    }
    assert(enum_result.size() == 10);
    assert(enum_result[0] == std::make_pair(std::ptrdiff_t(0), 1));
    assert(enum_result[9] == std::make_pair(std::ptrdiff_t(9), 10));

    // Test zip view
    std::vector<char> chars = {'a', 'b', 'c', 'd', 'e'};
    zip_view zipped(v, chars);

    std::vector<std::pair<int, char>> zip_result;
    for (auto [num, ch] : zipped) {
        zip_result.push_back({num, ch});
    }
    assert(zip_result.size() == 5);
    assert(zip_result[0] == std::make_pair(1, 'a'));
    assert(zip_result[4] == std::make_pair(5, 'e'));

    // Test range composition using pipe operator
    auto pipeline = v | views::filter(is_even) | views::transform(double_it) | views::take(3);

    std::vector<int> pipeline_result;
    for (auto x : pipeline) {
        pipeline_result.push_back(x);
    }
    assert(pipeline_result == std::vector<int>({4, 8, 12}));

    std::cout << "Range tests passed!\n";
}

// Test memory allocators
void test_memory() {
    std::cout << "Testing memory allocators...\n";

    // Test stack allocator
    {
        stack_allocator<1024> stack_alloc;

        void* p1 = stack_alloc.allocate(100);
        assert(p1 != nullptr);
        auto used1 = stack_alloc.used();
        assert(used1 >= 100 && used1 <= 100 + alignof(std::max_align_t));

        void* p2 = stack_alloc.allocate(200);
        assert(p2 != nullptr);
        auto used2 = stack_alloc.used();
        assert(used2 >= 300 && used2 <= 300 + 2 * alignof(std::max_align_t));

        assert(stack_alloc.available() == 1024 - used2);
    }

    // Test arena allocator
    {
        arena_allocator<> arena(256);

        int* p1 = arena.construct<int>(42);
        assert(*p1 == 42);

        double* p2 = arena.construct<double>(3.14);
        assert(*p2 == 3.14);

        struct TestStruct {
            int x;
            double y;
            TestStruct(int x_, double y_) : x(x_), y(y_) {}
        };

        TestStruct* p3 = arena.construct<TestStruct>(10, 2.5);
        assert(p3->x == 10);
        assert(p3->y == 2.5);

        arena.destroy(p3);
    }

    // Test memory pool
    {
        fixed_memory_pool<int> pool;

        int* p1 = pool.allocate();
        *p1 = 100;

        int* p2 = pool.allocate();
        *p2 = 200;

        assert(pool.allocated_count() == 2);

        pool.deallocate(p1);
        assert(pool.deallocated_count() == 1);
        assert(pool.in_use_count() == 1);

        int* p3 = pool.allocate(); // Should reuse p1's memory
        *p3 = 300;

        pool.deallocate(p2);
        pool.deallocate(p3);
    }

    // Test aligned allocator
    {
        aligned_allocator<float, 32> aligned_alloc;

        float* p = aligned_alloc.allocate(16);
        assert(reinterpret_cast<std::uintptr_t>(p) % 32 == 0);

        for (int i = 0; i < 16; ++i) {
            p[i] = static_cast<float>(i);
        }

        aligned_alloc.deallocate(p, 16);
    }

    std::cout << "Memory tests passed!\n";
}

// Test parallel algorithms
void test_parallel() {
    std::cout << "Testing parallel algorithms...\n";

    // Test thread pool
    {
        thread_pool pool(4);

        std::vector<std::future<int>> futures;
        for (int i = 0; i < 10; ++i) {
            futures.push_back(pool.enqueue([i] { return i * i; }));
        }

        std::vector<int> results;
        for (auto& f : futures) {
            results.push_back(f.get());
        }

        assert(results == std::vector<int>({0, 1, 4, 9, 16, 25, 36, 49, 64, 81}));
    }

    // Test lock-free queue
    {
        lock_free_queue<int> queue;

        queue.push(1);
        queue.push(2);
        queue.push(3);

        int val;
        assert(queue.pop(val) && val == 1);
        assert(queue.pop(val) && val == 2);
        assert(queue.pop(val) && val == 3);
        assert(!queue.pop(val));
    }

    // Test lock-free stack
    {
        lock_free_stack<int> stack;

        stack.push(1);
        stack.push(2);
        stack.push(3);

        int val;
        assert(stack.pop(val) && val == 3);
        assert(stack.pop(val) && val == 2);
        assert(stack.pop(val) && val == 1);
        assert(!stack.pop(val));
    }

    // Test parallel reduce
    {
        std::vector<int> v(10000);
        std::iota(v.begin(), v.end(), 1);

        auto sum = parallel::reduce(v.begin(), v.end(), 0, std::plus<int>());
        assert(sum == 10000 * 10001 / 2);
    }

    // Test parallel scan
    {
        std::vector<int> v = {1, 2, 3, 4, 5};
        std::vector<int> result(5);

        parallel::inclusive_scan(v.begin(), v.end(), result.begin(), 0, std::plus<int>());
        assert(result == std::vector<int>({1, 3, 6, 10, 15}));
    }

    // Test parallel partition
    {
        std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        auto mid = parallel::partition(v.begin(), v.end(), [](int x) { return x <= 5; });

        assert(std::distance(v.begin(), mid) == 5);
        assert(std::all_of(v.begin(), mid, [](int x) { return x <= 5; }));
        assert(std::all_of(mid, v.end(), [](int x) { return x > 5; }));
    }

    // Test parallel sort
    {
        std::vector<int> v = {9, 3, 7, 1, 5, 10, 2, 8, 4, 6};
        parallel::merge_sort(v.begin(), v.end(), std::less<int>());
        assert(std::is_sorted(v.begin(), v.end()));
    }

    // Test SIMD operations
    {
        const size_t n = 16;
        alignas(32) float a[n], b[n], result[n];

        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i * 2);
        }

        parallel::simd::add_vectors_avx(a, b, result, n);

        for (size_t i = 0; i < n; ++i) {
            assert(result[i] == a[i] + b[i]);
        }

        float dot = parallel::simd::dot_product_avx(a, b, n);
        float expected = 0;
        for (size_t i = 0; i < n; ++i) {
            expected += a[i] * b[i];
        }
        assert(std::abs(dot - expected) < 0.001f);
    }

    // Test parallel for_each
    {
        std::vector<int> v(1000);
        parallel::for_each(v.begin(), v.end(), [](int& x) { x = 42; });
        assert(std::all_of(v.begin(), v.end(), [](int x) { return x == 42; }));
    }

    // Test execution policies
    {
        std::vector<int> v = {1, 2, 3, 4, 5};
        for_each(par, v.begin(), v.end(), [](int& x) { x *= 2; });
        assert(v == std::vector<int>({2, 4, 6, 8, 10}));
    }

    std::cout << "Parallel tests passed!\n";
}

// Demonstrate composability
void test_composability() {
    std::cout << "Testing composability...\n";

    // Combine ranges with parallel algorithms
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 1);

    // Create a pipeline: filter even numbers, double them, take first 10
    auto pipeline = data | views::filter([](int x) { return x % 2 == 0; })
                        | views::transform([](int x) { return x * 2; })
                        | views::take(10);

    // Process in parallel
    std::vector<int> result;
    result.reserve(10);
    for (auto x : pipeline) {
        result.push_back(x);
    }

    assert(result == std::vector<int>({4, 8, 12, 16, 20, 24, 28, 32, 36, 40}));

    // Use custom allocator with STL containers
    {
        // Test arena allocator with direct usage
        arena_allocator<> arena(1024);
        int* p = static_cast<int*>(arena.allocate(sizeof(int) * 3));
        p[0] = 1;
        p[1] = 2;
        p[2] = 3;
        assert(p[0] == 1 && p[1] == 2 && p[2] == 3);
    }

    std::cout << "Composability tests passed!\n";
}

int main() {
    try {
        test_iterators();
        test_ranges();
        test_memory();
        test_parallel();
        test_composability();

        std::cout << "\nAll tests passed successfully!\n";
        std::cout << "The library demonstrates:\n";
        std::cout << "  - Generic iterator concepts and utilities\n";
        std::cout << "  - Lazy range adaptors with composition\n";
        std::cout << "  - Custom memory allocators (stack, arena, pool)\n";
        std::cout << "  - Parallel algorithms and data structures\n";
        std::cout << "  - SIMD vectorization\n";
        std::cout << "  - Composable components following Stepanov's principles\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}