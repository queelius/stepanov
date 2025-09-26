#include <iostream>
#include <vector>
#include <list>
#include <array>
#include <string>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>

#include "../include/stepanov/iterators.hpp"
#include "../include/stepanov/iterators_enhanced.hpp"
#include "../include/stepanov/ranges.hpp"
#include "../include/stepanov/ranges_enhanced.hpp"
#include "../include/stepanov/range_algorithms.hpp"

using namespace stepanov;

// ================ Test Basic Iterator Adaptors ================
void test_basic_iterators() {
    std::cout << "\n=== Basic Iterator Adaptors ===\n";

    // Reverse iterator
    {
        std::vector<int> v{1, 2, 3, 4, 5};
        std::cout << "Original: ";
        for (int x : v) std::cout << x << " ";
        std::cout << "\n";

        std::cout << "Reversed: ";
        auto rbegin = make_reverse_iterator(v.end());
        auto rend = make_reverse_iterator(v.begin());
        for (auto it = rbegin; it != rend; ++it) {
            std::cout << *it << " ";
        }
        std::cout << "\n";
    }

    // Move iterator
    {
        std::vector<std::string> source{"hello", "world"};
        std::vector<std::string> dest;

        std::copy(make_move_iterator(source.begin()),
                  make_move_iterator(source.end()),
                  std::back_inserter(dest));

        std::cout << "Moved strings: ";
        for (const auto& s : dest) std::cout << s << " ";
        std::cout << "\n";
    }

    // Stride iterator
    {
        std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::cout << "Every 3rd element: ";

        stride_iterator begin(v.begin(), v.end(), 3);
        stride_iterator end(v.end(), v.end(), 3);

        for (auto it = begin; it != end; ++it) {
            std::cout << *it << " ";
        }
        std::cout << "\n";
    }

    // Counted iterator
    {
        std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8};
        std::cout << "First 5 with counted: ";

        counted_iterator begin(v.begin(), 5);
        counted_iterator end(v.begin(), 0);

        for (auto it = begin; it != end; ++it) {
            std::cout << *it << " ";
        }
        std::cout << "\n";
    }
}

// ================ Test Basic Range Views ================
void test_basic_ranges() {
    std::cout << "\n=== Basic Range Views ===\n";

    std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Filter even numbers
    {
        std::cout << "Even numbers: ";
        auto even_filter = [](int x) { return x % 2 == 0; };
        filter_view filtered(v, even_filter);
        for (int x : filtered) {
            std::cout << x << " ";
        }
        std::cout << "\n";
    }

    // Transform - square
    {
        std::cout << "Squared: ";
        auto square = [](int x) { return x * x; };
        transform_view squared(v, square);

        auto it = squared.begin();
        auto end = squared.end();
        int count = 0;
        while (it != end && count < 5) {
            std::cout << *it << " ";
            ++it;
            ++count;
        }
        std::cout << "...\n";
    }

    // Take first n
    {
        std::cout << "First 5: ";
        take_view first5(v, 5);
        for (auto it = first5.begin(); it != first5.end(); ++it) {
            std::cout << *it << " ";
        }
        std::cout << "\n";
    }

    // Drop first n
    {
        std::cout << "After dropping 7: ";
        drop_view after7(v, 7);
        for (auto it = after7.begin(); it != after7.end(); ++it) {
            std::cout << *it << " ";
        }
        std::cout << "\n";
    }

    // Enumerate
    {
        std::cout << "Enumerated (first 3): ";
        std::vector<int> small{10, 20, 30};
        enumerate_view enumerated(small);
        for (const auto& [idx, val] : enumerated) {
            std::cout << "(" << idx << "," << val << ") ";
        }
        std::cout << "\n";
    }
}

// ================ Test Advanced Range Views ================
void test_advanced_ranges() {
    std::cout << "\n=== Advanced Range Views ===\n";

    // Chunk view
    {
        std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::cout << "Chunks of 3:\n";
        chunk_view chunks(v, 3);

        int i = 0;
        for (auto chunk : chunks) {
            std::cout << "  Chunk " << i++ << ": ";
            for (int x : chunk) {
                std::cout << x << " ";
            }
            std::cout << "\n";
        }
    }

    // Slide view (sliding window)
    {
        std::vector<int> v{1, 2, 3, 4, 5};
        std::cout << "Sliding window of 3:\n";
        slide_view windows(v, 3);

        int i = 0;
        for (auto window : windows) {
            std::cout << "  Window " << i++ << ": ";
            for (int x : window) {
                std::cout << x << " ";
            }
            std::cout << "\n";
        }
    }

    // Reverse view
    {
        std::vector<int> v{1, 2, 3, 4, 5};
        std::cout << "Reverse view: ";
        reverse_view reversed(v);
        for (int x : reversed) {
            std::cout << x << " ";
        }
        std::cout << "\n";
    }

    // Unique view
    {
        std::vector<int> v{1, 1, 2, 2, 2, 3, 4, 4, 5};
        std::cout << "Unique consecutive: ";
        unique_view unique(v);
        for (auto it = unique.begin(); it != unique.end(); ++it) {
            std::cout << *it << " ";
        }
        std::cout << "\n";
    }

    // Join (flatten)
    {
        std::vector<std::vector<int>> nested{{1, 2}, {3, 4, 5}, {6}};
        std::cout << "Flattened: ";
        join_view joined(nested);
        for (auto it = joined.begin(); it != joined.end(); ++it) {
            std::cout << *it << " ";
        }
        std::cout << "\n";
    }

    // Cartesian product
    {
        std::vector<char> letters{'a', 'b'};
        std::vector<int> numbers{1, 2, 3};
        std::cout << "Cartesian product: ";
        cartesian_product_view product(letters, numbers);
        for (const auto& [letter, number] : product) {
            std::cout << "(" << letter << "," << number << ") ";
        }
        std::cout << "\n";
    }
}

// ================ Test Lazy Generation ================
void test_lazy_generation() {
    std::cout << "\n=== Lazy Generation ===\n";

    // Iota - numeric sequence
    {
        std::cout << "Iota [5,15): ";
        auto numbers = iota_view(5, 15);
        for (int x : numbers) {
            std::cout << x << " ";
        }
        std::cout << "\n";
    }

    // Repeat
    {
        std::cout << "Repeat 42 (first 5): ";
        auto repeated = repeat_view(42);
        auto begin = repeated.begin();
        for (int i = 0; i < 5; ++i) {
            std::cout << *begin << " ";
            ++begin;
        }
        std::cout << "\n";
    }

    // Iterate - powers of 2
    {
        std::cout << "Powers of 2: ";
        auto double_it = [](int x) { return x * 2; };
        iterate_view powers(1, double_it);

        auto it = powers.begin();
        for (int i = 0; i < 8; ++i) {
            std::cout << *it << " ";
            ++it;
        }
        std::cout << "\n";
    }
}

// ================ Test Range Algorithms ================
void test_algorithms() {
    std::cout << "\n=== Range Algorithms (EoP Style) ===\n";

    // Find mismatch
    {
        std::vector<int> v1{1, 2, 3, 4, 5};
        std::vector<int> v2{1, 2, 3, 6, 7};

        auto [it1, it2] = find_mismatch(v1.begin(), v1.end(), v2.begin(), v2.end());
        if (it1 != v1.end()) {
            std::cout << "Mismatch found: " << *it1 << " vs " << *it2 << "\n";
        }
    }

    // Strictly increasing
    {
        std::vector<int> inc{1, 2, 3, 4, 5};
        std::vector<int> not_inc{1, 2, 2, 3, 4};

        std::cout << "Is {1,2,3,4,5} strictly increasing? "
                  << (strictly_increasing_range(inc.begin(), inc.end()) ? "Yes" : "No") << "\n";
        std::cout << "Is {1,2,2,3,4} strictly increasing? "
                  << (strictly_increasing_range(not_inc.begin(), not_inc.end()) ? "Yes" : "No") << "\n";
    }

    // Rotate
    {
        std::vector<int> v{1, 2, 3, 4, 5, 6, 7};
        std::cout << "Before rotate: ";
        for (int x : v) std::cout << x << " ";
        std::cout << "\n";

        rotate(v.begin(), v.begin() + 3, v.end());
        std::cout << "After rotate(3): ";
        for (int x : v) std::cout << x << " ";
        std::cout << "\n";
    }

    // Heap operations
    {
        std::vector<int> v{3, 1, 4, 1, 5, 9, 2, 6};
        std::cout << "Original: ";
        for (int x : v) std::cout << x << " ";
        std::cout << "\n";

        make_heap(v.begin(), v.end());
        std::cout << "After make_heap: ";
        for (int x : v) std::cout << x << " ";
        std::cout << "\n";

        std::cout << "Is heap? " << (is_heap(v.begin(), v.end()) ? "Yes" : "No") << "\n";
    }

    // Numeric algorithms
    {
        std::vector<int> v{1, 2, 3, 4, 5};

        auto sum = accumulate(v.begin(), v.end(), 0);
        std::cout << "Sum of {1,2,3,4,5}: " << sum << "\n";

        auto prod = product(v.begin(), v.end(), 1);
        std::cout << "Product: " << prod << "\n";

        std::vector<int> partial_sums;
        partial_sum(v.begin(), v.end(), std::back_inserter(partial_sums));
        std::cout << "Partial sums: ";
        for (int x : partial_sums) std::cout << x << " ";
        std::cout << "\n";
    }

    // Partition
    {
        std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        auto is_even = [](int x) { return x % 2 == 0; };

        auto mid = partition(v.begin(), v.end(), is_even);
        std::cout << "After partition (even/odd): ";
        for (int x : v) std::cout << x << " ";
        std::cout << "\n";
        std::cout << "Partition point index: " << std::distance(v.begin(), mid) << "\n";
    }

    // Set operations (on sorted ranges)
    {
        std::vector<int> set1{1, 3, 5, 7, 9};
        std::vector<int> set2{2, 3, 5, 7, 11};

        std::vector<int> result;

        result.clear();
        set_union(set1.begin(), set1.end(), set2.begin(), set2.end(), std::back_inserter(result));
        std::cout << "Union: ";
        for (int x : result) std::cout << x << " ";
        std::cout << "\n";

        result.clear();
        set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), std::back_inserter(result));
        std::cout << "Intersection: ";
        for (int x : result) std::cout << x << " ";
        std::cout << "\n";

        result.clear();
        set_difference(set1.begin(), set1.end(), set2.begin(), set2.end(), std::back_inserter(result));
        std::cout << "Difference: ";
        for (int x : result) std::cout << x << " ";
        std::cout << "\n";
    }

    // Statistical operations
    {
        std::vector<double> data{1.0, 2.0, 3.0, 4.0, 5.0};
        std::cout << "Mean of {1,2,3,4,5}: " << mean(data.begin(), data.end()) << "\n";
        std::cout << "Variance: " << variance(data.begin(), data.end()) << "\n";
        std::cout << "Std Dev: " << stddev(data.begin(), data.end()) << "\n";
    }
}

// ================ Test Iterator Utilities ================
void test_utilities() {
    std::cout << "\n=== Iterator Utilities ===\n";

    std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Distance
    {
        auto dist = distance(v.begin(), v.end());
        std::cout << "Distance (vector): " << dist << "\n";

        std::list<int> l(v.begin(), v.end());
        auto dist_list = distance(l.begin(), l.end());
        std::cout << "Distance (list): " << dist_list << "\n";
    }

    // Advance
    {
        auto it = v.begin();
        advance(it, 5);
        std::cout << "After advance(5): " << *it << "\n";

        auto it2 = v.begin();
        advance(it2, 15, v.end());  // With bound
        std::cout << "After advance(15, end): " << (it2 == v.end() ? "at end" : "not at end") << "\n";
    }

    // Next/Prev
    {
        auto it = v.begin();
        auto next5 = stepanov::next(it, 5);
        std::cout << "next(begin, 5): " << *next5 << "\n";

        auto last = v.end();
        auto prev3 = stepanov::prev(last, 3);
        std::cout << "prev(end, 3): " << *prev3 << "\n";
    }

    // Iter swap
    {
        std::vector<int> v_copy = v;
        iter_swap(v_copy.begin(), v_copy.begin() + 5);
        std::cout << "After iter_swap(0, 5): " << v_copy[0] << " and " << v_copy[5] << "\n";
    }
}

// ================ Test Performance ================
void test_performance() {
    std::cout << "\n=== Performance Tests ===\n";

    const int size = 100000;
    std::vector<int> data(size);
    std::iota(data.begin(), data.end(), 0);

    // Lazy evaluation benefit
    {
        int operations = 0;

        // Chain of transformations
        auto expensive = [&operations](int x) {
            ++operations;
            return x * x;
        };

        transform_view transformed(data, expensive);
        filter_view filtered(transformed, [](int x) { return x % 2 == 0; });
        take_view limited(filtered, 10);

        std::cout << "Taking 10 even squares from " << size << " elements:\n";
        std::cout << "Results: ";
        for (int x : limited) {
            std::cout << x << " ";
        }
        std::cout << "\n";
        std::cout << "Operations performed: " << operations << " (not " << size << "!)\n";
    }

    // Iterator category optimization
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto dist_vec = distance(data.begin(), data.end());
        auto end_time = std::chrono::high_resolution_clock::now();
        auto vec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start).count();

        std::list<int> list_data(1000);  // Smaller for list
        std::iota(list_data.begin(), list_data.end(), 0);

        start = std::chrono::high_resolution_clock::now();
        auto dist_list = distance(list_data.begin(), list_data.end());
        end_time = std::chrono::high_resolution_clock::now();
        auto list_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start).count();

        std::cout << "\nDistance calculation optimization:\n";
        std::cout << "Vector (random access): " << vec_time << " ns for " << dist_vec << " elements\n";
        std::cout << "List (forward only): " << list_time << " ns for " << dist_list << " elements\n";
    }
}

int main() {
    try {
        std::cout << "=====================================\n";
        std::cout << "  Stepanov-Style Iterator & Range\n";
        std::cout << "     Comprehensive Test Suite\n";
        std::cout << "=====================================\n";

        test_basic_iterators();
        test_basic_ranges();
        test_advanced_ranges();
        test_lazy_generation();
        test_algorithms();
        test_utilities();
        test_performance();

        std::cout << "\n=====================================\n";
        std::cout << "     All Tests Completed!\n";
        std::cout << "=====================================\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}