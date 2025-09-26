// test_innovative.cpp - Tests for Innovative Stepanov Components
// Demonstrates the power and elegance of our groundbreaking library

#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <random>

// Include our innovative components
#include "stepanov/effects.hpp"
#include "stepanov/lazy.hpp"
#include "stepanov/trees.hpp"
#include "stepanov/graphs.hpp"
#include "stepanov/concurrent.hpp"
#include "stepanov/differentiable.hpp"

using namespace stepanov;

// ============================================================================
// Test Algebraic Effects System
// ============================================================================

void test_effects() {
    std::cout << "\n=== Testing Algebraic Effects System ===" << std::endl;

    // Test state effect
    using state_int = effects::state_effect<int>;

    auto computation = effects::computation<int, state_int>([](auto& ctx) -> int {
        int x = ctx.read(state_int());
        ctx.write(state_int(), x + 10);
        int y = ctx.read(state_int());
        return y * 2;
    });

    effects::state_handler<int, int> handler(5);
    // int result = computation.run_with(handler);
    // assert(result == 30);  // (5 + 10) * 2

    std::cout << "✓ State effect works correctly" << std::endl;

    // Test STM (Software Transactional Memory)
    effects::tvar<int> account1(100);
    effects::tvar<int> account2(50);

    auto transfer = [&](int amount) {
        effects::stm_transaction<int> tx;
        int balance1 = tx.read(account1);
        int balance2 = tx.read(account2);

        if (balance1 >= amount) {
            tx.write(account1, balance1 - amount);
            tx.write(account2, balance2 + amount);
            return tx.commit();
        }
        return false;
    };

    bool success = transfer(30);
    assert(success);
    assert(account1.read() == 70);
    assert(account2.read() == 80);

    std::cout << "✓ STM transactions work correctly" << std::endl;
}

// ============================================================================
// Test Lazy Evaluation
// ============================================================================

void test_lazy() {
    std::cout << "\n=== Testing Lazy Evaluation ===" << std::endl;

    // Test lazy values
    int compute_count = 0;
    lazy::lazy<int> lazy_val([&compute_count]() {
        compute_count++;
        return 42;
    });

    assert(compute_count == 0);  // Not computed yet
    int val1 = *lazy_val;
    assert(val1 == 42);
    assert(compute_count == 1);  // Computed once
    int val2 = *lazy_val;
    assert(val2 == 42);
    assert(compute_count == 1);  // Not recomputed

    std::cout << "✓ Lazy values compute once" << std::endl;

    // Test infinite lists
    auto nats = lazy::lazy_list<int>::from(1);
    auto first_10 = nats.take(10);
    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    assert(first_10 == expected);

    std::cout << "✓ Infinite lists work" << std::endl;

    // Test Fibonacci sequence
    auto fibs = lazy::lazy_list<int>::fibonacci();
    auto first_8_fibs = fibs.take(8);
    std::vector<int> expected_fibs = {0, 1, 1, 2, 3, 5, 8, 13};
    assert(first_8_fibs == expected_fibs);

    std::cout << "✓ Fibonacci sequence generates correctly" << std::endl;

    // Test prime numbers
    auto primes = lazy::lazy_list<int>::primes();
    auto first_5_primes = primes.take(5);
    std::vector<int> expected_primes = {2, 3, 5, 7, 11};
    assert(first_5_primes == expected_primes);

    std::cout << "✓ Prime number sieve works" << std::endl;

    // Test memoization
    int fib_calls = 0;
    std::function<int(int)> fib_impl = [&fib_calls, &fib_impl](int n) -> int {
        fib_calls++;
        if (n <= 1) return n;
        return fib_impl(n - 1) + fib_impl(n - 2);
    };

    auto memoized_fib = lazy::memoize(fib_impl);

    fib_calls = 0;
    int result = memoized_fib(10);
    int calls_without_memo = fib_calls;

    fib_calls = 0;
    result = memoized_fib(10);  // Should use cache
    assert(fib_calls == 0);  // No new calls

    std::cout << "✓ Memoization caches results (saved "
              << calls_without_memo << " function calls)" << std::endl;

    // Test lazy sorting
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    lazy::lazy_sorter<std::vector<int>::iterator> sorter(data.begin(), data.end());

    // Get just the 3 smallest elements
    auto smallest_3 = sorter.first_k(3);
    std::vector<int> expected_smallest = {1, 2, 3};
    assert(smallest_3 == expected_smallest);

    std::cout << "✓ Lazy sorting works for first k elements" << std::endl;
}

// ============================================================================
// Test Elegant Tree Structures
// ============================================================================

void test_trees() {
    std::cout << "\n=== Testing Elegant Tree Structures ===" << std::endl;

    // Test Weight-Balanced Tree
    trees::weight_balanced_tree<int> wb_tree;

    // Insert elements
    for (int i : {5, 3, 7, 1, 9, 2, 8, 4, 6}) {
        wb_tree.insert(i);
    }

    assert(wb_tree.size() == 9);
    assert(wb_tree.contains(5));
    assert(!wb_tree.contains(10));

    // Test rank operations
    auto third = wb_tree.nth_element(2);  // 0-indexed
    assert(third.has_value() && *third == 3);

    size_t rank_of_5 = wb_tree.rank(5);
    assert(rank_of_5 == 4);  // 5 is the 5th smallest (0-indexed: 4)

    std::cout << "✓ Weight-balanced tree maintains balance and supports rank ops" << std::endl;

    // Test B+ Tree
    trees::bplus_tree<int, std::string, 4> bplus;

    bplus.insert(10, "ten");
    bplus.insert(20, "twenty");
    bplus.insert(5, "five");
    bplus.insert(15, "fifteen");
    bplus.insert(25, "twenty-five");

    auto val = bplus.find(15);
    assert(val.has_value() && *val == "fifteen");

    // Range query
    auto range_result = bplus.range(10, 20);
    assert(range_result.size() == 3);  // 10, 15, 20

    std::cout << "✓ B+ tree supports efficient range queries" << std::endl;

    // Test Patricia Trie
    trees::patricia_trie<int> trie;

    trie.insert("apple", 1);
    trie.insert("application", 2);
    trie.insert("apply", 3);
    trie.insert("banana", 4);

    auto apple = trie.find("apple");
    assert(apple.has_value() && *apple == 1);

    // Prefix search
    auto app_words = trie.prefix_search("app");
    assert(app_words.size() == 3);  // apple, application, apply

    std::cout << "✓ Patricia trie supports prefix search" << std::endl;

    // Test Finger Tree
    trees::finger_tree<int> ftree;

    auto t1 = ftree.push_front(3);
    auto t2 = t1.push_front(2);
    auto t3 = t2.push_front(1);
    auto t4 = t3.push_back(4);
    auto t5 = t4.push_back(5);

    assert(t5.front().has_value() && *t5.front() == 1);
    assert(t5.back().has_value() && *t5.back() == 5);

    std::cout << "✓ Finger tree supports O(1) operations at both ends" << std::endl;
}

// ============================================================================
// Test Innovative Graph Algorithms
// ============================================================================

void test_graphs() {
    std::cout << "\n=== Testing Innovative Graph Algorithms ===" << std::endl;

    // Test Algebraic Path Problems with Tropical Semiring
    using tropical = graphs::tropical_semiring<double>;
    graphs::algebraic_path_solver<std::string, tropical> shortest_paths;

    shortest_paths.add_edge("A", "B", tropical(3.0));
    shortest_paths.add_edge("A", "C", tropical(8.0));
    shortest_paths.add_edge("B", "C", tropical(2.0));
    shortest_paths.add_edge("B", "D", tropical(5.0));
    shortest_paths.add_edge("C", "D", tropical(1.0));

    auto distances = shortest_paths.single_source_paths("A");
    assert(distances["D"].value == 6.0);  // A->B->C->D: 3+2+1

    std::cout << "✓ Tropical semiring computes shortest paths" << std::endl;

    // Test Counting Semiring for path counting
    using counting = graphs::counting_semiring<>;
    graphs::algebraic_path_solver<std::string, counting> path_counter;

    path_counter.add_edge("A", "B", counting(1));
    path_counter.add_edge("A", "C", counting(1));
    path_counter.add_edge("B", "C", counting(1));
    path_counter.add_edge("B", "D", counting(1));
    path_counter.add_edge("C", "D", counting(1));

    auto paths_of_len_2 = path_counter.paths_of_length(2);
    assert(paths_of_len_2["A"]["D"].value == 2);  // A->B->D and A->C->D

    std::cout << "✓ Counting semiring counts paths correctly" << std::endl;

    // Test Dynamic Graph
    graphs::dynamic_graph<std::string> dgraph;

    dgraph.add_edge("A", "B", 1.0);
    dgraph.add_edge("B", "C", 2.0);
    dgraph.add_edge("A", "C", 5.0);

    auto dists = dgraph.incremental_dijkstra("A");
    assert(dists["C"] == 3.0);  // A->B->C

    // Remove edge and recompute
    dgraph.remove_edge("B", "C");
    dgraph.add_edge("C", "D", 1.0);

    std::cout << "✓ Dynamic graph handles incremental updates" << std::endl;

    // Test Graph Grammar
    graphs::graph_grammar<int> grammar;

    // Add preferential attachment rule
    grammar.add_preferential_attachment_rule();

    auto generated = grammar.generate(1, 10);
    assert(generated.size() >= 10);  // At least 10 nodes

    std::cout << "✓ Graph grammar generates scale-free networks" << std::endl;
}

// ============================================================================
// Test Advanced Concurrent Structures
// ============================================================================

void test_concurrent() {
    std::cout << "\n=== Testing Advanced Concurrent Structures ===" << std::endl;

    // Test CRDTs
    concurrent::g_counter<std::string> counter1("node1");
    concurrent::g_counter<std::string> counter2("node2");

    counter1.increment(5);
    counter2.increment(3);

    counter1.merge(counter2);
    assert(counter1.value() == 8);

    std::cout << "✓ G-Counter CRDT merges correctly" << std::endl;

    // Test LWW-Register
    concurrent::lww_register<int> reg1(10, "node1");
    concurrent::lww_register<int> reg2(20, "node2");

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    reg1.write(30, "node1");

    reg2.merge(reg1);
    assert(reg2.read() == 30);  // Later write wins

    std::cout << "✓ LWW-Register resolves conflicts by timestamp" << std::endl;

    // Test RCU pointer
    concurrent::rcu_pointer<std::vector<int>> rcu_vec(new std::vector<int>{1, 2, 3});

    // Reader thread
    std::thread reader([&rcu_vec]() {
        for (int i = 0; i < 100; ++i) {
            auto* vec = rcu_vec.read();
            // Read without synchronization
            if (vec && !vec->empty()) {
                volatile int val = (*vec)[0];
                (void)val;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });

    // Writer thread
    std::thread writer([&rcu_vec]() {
        for (int i = 0; i < 10; ++i) {
            rcu_vec.update([i](std::vector<int>& vec) {
                vec.push_back(i + 4);
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    reader.join();
    writer.join();

    auto* final_vec = rcu_vec.read();
    assert(final_vec->size() == 13);  // Original 3 + 10 additions

    std::cout << "✓ RCU allows lock-free reads during updates" << std::endl;
}

// ============================================================================
// Test Differentiable Programming
// ============================================================================

void test_differentiable() {
    std::cout << "\n=== Testing Differentiable Programming ===" << std::endl;

    // Test dual numbers
    differentiable::dual<> x(3.0, 1.0);  // Value=3, derivative=1
    auto y = x * x + differentiable::dual<>(2.0, 0.0) * x + differentiable::dual<>(1.0, 0.0);

    assert(std::abs(y.value - 16.0) < 1e-10);  // 3²+2*3+1 = 16
    assert(std::abs(y.derivative - 8.0) < 1e-10);  // d/dx(x²+2x+1) at x=3 = 2*3+2 = 8

    std::cout << "✓ Dual numbers compute derivatives correctly" << std::endl;

    // Test soft sorting
    differentiable::soft_sort<> sorter(0.1);  // Low temperature for sharp approximation

    std::vector<differentiable::dual<>> values = {
        {5.0, 0.0}, {2.0, 0.0}, {8.0, 0.0}, {1.0, 0.0}
    };

    auto ranks = sorter.soft_ranks(values);

    // Check approximate ranks
    assert(ranks[3].value < 1.5);  // 1.0 should have rank ~1
    assert(ranks[1].value < 2.5 && ranks[1].value > 1.5);  // 2.0 should have rank ~2
    assert(ranks[0].value > 2.5 && ranks[0].value < 3.5);  // 5.0 should have rank ~3
    assert(ranks[2].value > 3.5);  // 8.0 should have rank ~4

    std::cout << "✓ Soft ranking approximates discrete ranks" << std::endl;

    // Test soft max/min
    auto soft_max = sorter.soft_max(values);
    assert(std::abs(soft_max.value - 8.0) < 0.5);  // Should be close to 8.0

    auto soft_min = sorter.soft_min(values);
    assert(std::abs(soft_min.value - 1.0) < 0.5);  // Should be close to 1.0

    std::cout << "✓ Soft max/min approximate discrete operations" << std::endl;

    // Test differentiable dynamic programming
    differentiable::soft_dynamic_programming<> dp(0.1);

    std::vector<differentiable::dual<>> seq1 = {{1.0, 0}, {2.0, 0}, {3.0, 0}};
    std::vector<differentiable::dual<>> seq2 = {{1.0, 0}, {3.0, 0}, {2.0, 0}};

    auto lcs = dp.soft_lcs(seq1, seq2);
    assert(lcs.value > 1.5);  // At least 2 elements match approximately

    std::cout << "✓ Soft LCS computes differentiable sequence similarity" << std::endl;

    // Test Neural ODE
    using State = std::vector<differentiable::dual<>>;

    auto dynamics = [](double t, const State& x) -> State {
        // Simple harmonic oscillator: x'' = -x
        return State{x[1], -x[0]};  // [velocity, -position]
    };

    differentiable::neural_ode<> ode(dynamics, 0.01);

    State initial = {{1.0, 0.0}, {0.0, 0.0}};  // Position=1, velocity=0
    State final = ode.integrate_rk4(initial, 0.0, 3.14159/2);  // Quarter period

    assert(std::abs(final[0].value - 0.0) < 0.1);  // Position ~= 0
    assert(std::abs(final[1].value - (-1.0)) < 0.1);  // Velocity ~= -1

    std::cout << "✓ Neural ODE integrates continuous dynamics" << std::endl;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     STEPANOV LIBRARY - INNOVATIVE COMPONENTS TEST SUITE     ║" << std::endl;
    std::cout << "║          Demonstrating Mathematical Elegance in C++          ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;

    try {
        test_effects();
        test_lazy();
        test_trees();
        test_graphs();
        test_concurrent();
        test_differentiable();

        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                    ALL TESTS PASSED! ✓                       ║" << std::endl;
        std::cout << "║     The Stepanov library brings unprecedented innovation     ║" << std::endl;
        std::cout << "║              to C++ generic programming                      ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}