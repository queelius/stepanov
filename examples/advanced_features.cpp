#include <iostream>
#include <vector>
#include <chrono>
#include "../include/stepanov/tropical.hpp"
#include "../include/stepanov/succinct.hpp"
#include "../include/stepanov/compression.hpp"
#include "../include/stepanov/cache_oblivious.hpp"
#include "../include/stepanov/verify.hpp"

using namespace stepanov;

// Example 1: Tropical Mathematics for Shortest Paths
void tropical_example() {
    std::cout << "\n=== Tropical Mathematics Example ===" << std::endl;

    // Graph represented as adjacency matrix (distances)
    std::vector<std::vector<double>> graph = {
        {0, 4, INFINITY, INFINITY, INFINITY},
        {INFINITY, 0, 1, 5, INFINITY},
        {INFINITY, INFINITY, 0, INFINITY, 2},
        {INFINITY, INFINITY, -3, 0, INFINITY},
        {INFINITY, INFINITY, INFINITY, 1, 0}
    };

    // Convert to tropical matrix (min-plus algebra)
    tropical::tropical_matrix<tropical::min_plus<double>> M(5, 5);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            M(i, j) = tropical::min_plus<double>(graph[i][j]);
        }
    }

    // Compute all shortest paths with Kleene star
    auto all_paths = M.kleene_star();

    std::cout << "Shortest paths from node 0:" << std::endl;
    for (size_t j = 0; j < 5; ++j) {
        double dist = all_paths(0, j).value;
        if (dist != INFINITY) {
            std::cout << "  To node " << j << ": " << dist << std::endl;
        }
    }

    // Scheduling example with max-plus algebra
    tropical::scheduler<double> sched;
    std::vector<std::vector<double>> precedence = {
        {INFINITY, 2, INFINITY},
        {INFINITY, INFINITY, 3},
        {4, INFINITY, INFINITY}
    };
    std::vector<double> durations = {1, 2, 1.5};

    auto schedule = sched.compute_schedule(precedence, durations);
    std::cout << "\nOptimal schedule cycle time: " << schedule.cycle_time << std::endl;
}

// Example 2: Succinct Data Structures
void succinct_example() {
    std::cout << "\n=== Succinct Data Structures Example ===" << std::endl;

    // Succinct bit vector with rank/select
    succinct::bit_vector bv = {1,0,1,1,0,1,0,0,1,1,0,1};
    bv.build_auxiliary();

    std::cout << "Bit vector: 101101001101" << std::endl;
    std::cout << "rank1(6) = " << bv.rank1(6) << " (number of 1s before position 6)" << std::endl;
    std::cout << "select1(3) = " << bv.select1(3) << " (position of 4th 1)" << std::endl;

    // Wavelet tree for sequence operations
    std::vector<char> text = {'a','b','r','a','c','a','d','a','b','r','a'};
    succinct::wavelet_tree<char> wt(text);

    std::cout << "\nWavelet tree on 'abracadabra':" << std::endl;
    std::cout << "rank(7, 'a') = " << wt.rank(7, 'a') << " (occurrences of 'a' before position 7)" << std::endl;
    std::cout << "3rd smallest in range [2,8]: " << wt.quantile(2, 8, 2) << std::endl;

    // FM-Index for compressed pattern matching
    std::string genome = "ATCGATCGTAGCTAGCTAGCATCG$";
    succinct::fm_index fm(genome);

    std::string pattern = "TAG";
    std::cout << "\nFM-Index search for '" << pattern << "': " << fm.count(pattern) << " occurrences" << std::endl;
}

// Example 3: Advanced Compression
void compression_example() {
    std::cout << "\n=== Compression Example ===" << std::endl;

    // rANS entropy coding
    std::vector<uint32_t> frequencies = {10, 30, 25, 35};  // Symbol frequencies
    compression::rans_encoder encoder;
    encoder.initialize(frequencies);

    // Encode sequence
    std::vector<size_t> symbols = {0, 1, 2, 3, 2, 1, 0, 3, 2, 1};
    for (size_t sym : symbols) {
        encoder.encode_symbol(sym);
    }
    auto compressed = encoder.finish();

    std::cout << "rANS compression:" << std::endl;
    std::cout << "  Original: " << symbols.size() << " symbols" << std::endl;
    std::cout << "  Compressed: " << compressed.size() << " bytes" << std::endl;

    // Burrows-Wheeler Transform
    compression::bwt_compressor bwt;
    std::vector<uint8_t> data = {'b','a','n','a','n','a'};
    auto bwt_result = bwt.forward_transform(data);

    std::cout << "\nBWT of 'banana':" << std::endl;
    std::cout << "  Primary index: " << bwt_result.primary_index << std::endl;
    std::cout << "  Transformed: ";
    for (uint8_t b : bwt_result.transformed) {
        std::cout << (char)b;
    }
    std::cout << std::endl;

    // PPM context modeling
    compression::ppm_model<3> ppm;
    std::string text = "abracadabra";

    double total_prob = 1.0;
    for (char c : text) {
        double prob = ppm.predict(c);
        total_prob *= prob;
        ppm.update(c);
    }

    std::cout << "\nPPM compression potential:" << std::endl;
    std::cout << "  Text: " << text << std::endl;
    std::cout << "  Bits needed: " << -std::log2(total_prob) << std::endl;
}

// Example 4: Cache-Oblivious Algorithms
void cache_oblivious_example() {
    std::cout << "\n=== Cache-Oblivious Algorithms Example ===" << std::endl;

    // Van Emde Boas tree
    cache_oblivious::veb_tree<int> veb(1000);
    std::vector<int> values = {42, 17, 93, 5, 68, 31};

    for (int v : values) {
        veb.insert(v);
    }

    std::cout << "vEB tree search for 31: " << (veb.search(31) ? "found" : "not found") << std::endl;

    auto range = veb.range_query(20, 70);
    std::cout << "Range query [20, 70]: ";
    for (int v : range) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    // Cache-oblivious matrix multiplication
    cache_oblivious::matrix<double> A(3, 3);
    cache_oblivious::matrix<double> B(3, 3);

    // Initialize matrices
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            A(i, j) = i + j;
            B(i, j) = i * j;
        }
    }

    auto C = A * B;
    std::cout << "\nCache-oblivious matrix multiply result C[1][1]: " << C(1, 1) << std::endl;

    // Cache-oblivious B-tree
    cache_oblivious::cache_oblivious_btree<int, std::string> btree;
    btree.insert(10, "ten");
    btree.insert(20, "twenty");
    btree.insert(5, "five");

    auto result = btree.search(20);
    std::cout << "B-tree search for key 20: " << (result ? *result : "not found") << std::endl;
}

// Example 5: Formal Verification
void verification_example() {
    std::cout << "\n=== Formal Verification Example ===" << std::endl;

    // Design by Contract
    auto divide_safe = [](double a, double b) -> double {
        REQUIRES(b != 0);  // Precondition
        double result = a / b;
        ENSURES(std::abs(result * b - a) < 1e-10);  // Postcondition
        return result;
    };

    try {
        std::cout << "Safe division 10/2 = " << divide_safe(10, 2) << std::endl;
    } catch (const verify::contract_violation& e) {
        std::cout << "Contract violation: " << e.what() << std::endl;
    }

    // Refinement types
    verify::positive_t<int> pos_num(42);
    std::cout << "Positive number: " << pos_num.get() << std::endl;

    verify::probability_t<double> prob(0.7);
    std::cout << "Probability: " << prob.get() << std::endl;

    // Property-based testing
    auto int_gen = std::make_unique<verify::int_generator<int>>(1, 100);
    auto prop = [](int a, int b) { return a + b == b + a; };  // Commutativity

    verify::property_test<int, int> test(
        "Addition commutativity",
        prop,
        std::make_unique<verify::int_generator<int>>(1, 100),
        std::make_unique<verify::int_generator<int>>(1, 100),
        50
    );

    auto result = test.run();
    std::cout << "\nProperty test '" << result.property_name << "': "
              << (result.passed ? "PASSED" : "FAILED")
              << " (" << result.num_tests << " tests)" << std::endl;

    // Bounded model checking example
    struct Counter { int value; };

    verify::bounded_model_checker<Counter> checker(
        Counter{0},
        [](const Counter& s) -> std::vector<std::pair<Counter, std::string>> {
            return {
                {Counter{s.value + 1}, "increment"},
                {Counter{s.value - 1}, "decrement"}
            };
        },
        5  // Max depth
    );

    checker.add_invariant([](const Counter& s) { return s.value >= -10 && s.value <= 10; });

    auto violations = checker.check();
    std::cout << "Model checking found " << violations.size() << " violation(s)" << std::endl;
}

int main() {
    std::cout << "=== Stepanov Library Advanced Features Demo ===" << std::endl;

    tropical_example();
    succinct_example();
    compression_example();
    cache_oblivious_example();
    verification_example();

    std::cout << "\n=== All examples completed successfully ===" << std::endl;
    return 0;
}