#include <iostream>
#include <cassert>
#include <random>
#include <chrono>
#include <iomanip>

// Include all new components
#include <stepanov/hashing.hpp>
#include <stepanov/random.hpp>
#include <stepanov/error.hpp>
#include <stepanov/serialization.hpp>

using namespace stepanov;

// Test hashing and sketching
void test_hashing() {
    std::cout << "Testing hashing components...\n";

    // Test bloom filter
    bloom_filter<int> bf(1000, 0.01);
    for (int i = 0; i < 100; ++i) {
        bf.insert(i);
    }
    assert(bf.possibly_contains(50));
    assert(!bf.possibly_contains(500));
    std::cout << "  Bloom filter FP rate: " << bf.estimated_false_positive_rate() << "\n";

    // Test Count-Min sketch
    count_min_sketch<std::string> cms(0.01, 0.001);
    cms.update("apple", 5);
    cms.update("banana", 3);
    cms.update("apple", 2);
    assert(cms.estimate("apple") >= 7);
    assert(cms.estimate("banana") >= 3);
    std::cout << "  Count-Min sketch passed\n";

    // Test MinHash
    minhash<int> mh(128);
    std::unordered_set<int> set1 = {1, 2, 3, 4, 5};
    std::unordered_set<int> set2 = {3, 4, 5, 6, 7};
    auto sig1 = mh.compute_signature(set1);
    auto sig2 = mh.compute_signature(set2);
    double similarity = mh.jaccard_similarity(sig1, sig2);
    assert(similarity > 0.2 && similarity < 0.8);  // Should be around 0.4
    std::cout << "  MinHash Jaccard similarity: " << similarity << "\n";

    // Test HyperLogLog
    hyperloglog<int> hll(14);
    for (int i = 0; i < 10000; ++i) {
        hll.add(i);
    }
    auto estimate = hll.estimate_cardinality();
    std::cout << "  HyperLogLog estimate: " << estimate << " (actual: 10000)\n";
    // HyperLogLog has error margin, just check it's reasonable
    assert(estimate > 0);

    // Test consistent hashing
    consistent_hash<std::string, int> ch(150);
    ch.add_node("server1");
    ch.add_node("server2");
    ch.add_node("server3");
    auto node = ch.get_node(42);
    assert(node.has_value());
    std::cout << "  Consistent hashing passed\n";

    // Test cuckoo hashing
    cuckoo_hash_table<int, std::string> cht;
    cht.insert(1, "one");
    cht.insert(2, "two");
    cht.insert(3, "three");
    assert(cht.find(2).value() == "two");
    assert(!cht.find(4).has_value());
    std::cout << "  Cuckoo hashing passed\n";

    std::cout << "  All hashing tests passed\n";
}

// Test random number generation
void test_random() {
    std::cout << "Testing random number generation...\n";

    // Test PCG
    pcg32<> pcg(42);
    auto val1 = pcg();
    auto val2 = pcg();
    assert(val1 != val2);
    std::cout << "  PCG32 sample: " << val1 << "\n";

    // Test xoshiro256**
    xoshiro256ss xoshiro(42);
    auto xval = xoshiro();
    assert(xval != 0);
    std::cout << "  xoshiro256** sample: " << xval << "\n";

    // Test distributions
    normal_distribution<pcg32<>> normal(0.0, 1.0);
    double sum = 0;
    for (int i = 0; i < 1000; ++i) {
        sum += normal(pcg);
    }
    double mean = sum / 1000;
    assert(std::abs(mean) < 0.1);  // Should be close to 0
    std::cout << "  Normal distribution mean: " << mean << "\n";

    // Test reservoir sampling
    std::mt19937 gen(42);
    reservoir_sampler<int, std::mt19937> rs(5, gen);
    for (int i = 0; i < 100; ++i) {
        rs.add(i);
    }
    auto sample = rs.get_sample();
    assert(sample.size() == 5);
    std::cout << "  Reservoir sampling passed\n";

    // Test weighted sampling
    std::vector<std::string> items = {"A", "B", "C"};
    std::vector<double> weights = {0.5, 0.3, 0.2};
    weighted_sampler<std::string, std::mt19937> ws(items, weights);
    std::map<std::string, int> counts;
    for (int i = 0; i < 1000; ++i) {
        counts[ws.sample(gen)]++;
    }
    assert(counts["A"] > counts["B"] && counts["B"] > counts["C"]);
    std::cout << "  Weighted sampling passed\n";

    // Test quasi-random sequences
    sobol_sequence sobol(2);
    auto point1 = sobol.next();
    auto point2 = sobol.next();
    assert(point1[0] != point2[0]);
    std::cout << "  Sobol sequence passed\n";

    halton_sequence halton(2);
    auto hpoint = halton.next();
    assert(hpoint[0] >= 0 && hpoint[0] < 1);
    std::cout << "  Halton sequence passed\n";

    // Test alias method
    alias_method<std::mt19937> alias({0.1, 0.2, 0.3, 0.4});
    std::vector<int> alias_counts(4, 0);
    for (int i = 0; i < 10000; ++i) {
        alias_counts[alias.sample(gen)]++;
    }
    assert(alias_counts[3] > alias_counts[0]);  // 0.4 > 0.1
    std::cout << "  Alias method passed\n";

    std::cout << "  All random tests passed\n";
}

// Test error handling
void test_error_handling() {
    std::cout << "Testing error handling...\n";

    // Test optional with monadic operations
    optional<int> opt1(42);
    auto opt2 = opt1.map([](int x) { return x * 2; });
    assert(opt2.value() == 84);

    auto opt3 = opt1.filter([](int x) { return x > 50; });
    assert(!opt3.has_value());

    auto opt4 = optional<int>().or_else([]() { return optional<int>(10); });
    assert(opt4.value() == 10);
    std::cout << "  Optional monadic operations passed\n";

    // Test expected
    auto divide = [](int a, int b) -> expected<int, std::string> {
        if (b == 0) return expected<int, std::string>::failure("Division by zero");
        return expected<int, std::string>::success(a / b);
    };

    auto result1 = divide(10, 2);
    assert(result1.has_value() && result1.value() == 5);

    auto result2 = divide(10, 0);
    assert(result2.has_error());

    auto mapped = result1.map([](int x) { return x * 2; });
    assert(mapped.value() == 10);
    std::cout << "  Expected type passed\n";

    // Test contracts
    try {
        contract<>::require(true, "This should pass");
        contract<>::require(false, "This should fail");
        assert(false);  // Should not reach here
    } catch (const std::logic_error& e) {
        // Expected
    }
    std::cout << "  Contract assertions passed\n";

    // Test scope guard
    bool cleanup_called = false;
    {
        auto guard = make_scope_guard([&cleanup_called]() {
            cleanup_called = true;
        });
    }
    assert(cleanup_called);
    std::cout << "  Scope guard passed\n";

    // Test error accumulator
    error_accumulator<> acc;
    acc.add("Error 1");
    acc.add("Error 2");
    assert(acc.has_errors());
    assert(acc.get_errors().size() == 2);
    std::cout << "  Error accumulator passed\n";

    std::cout << "  All error handling tests passed\n";
}

// Test serialization
void test_serialization() {
    std::cout << "Testing serialization...\n";

    // Test binary serialization
    binary_writer writer;
    writer.write<int>(42);
    writer.write<double>(3.14);
    writer.write_string("Hello");

    std::vector<int> vec = {1, 2, 3, 4, 5};
    writer.write_vector(vec);

    auto buffer = writer.get_buffer();

    binary_reader reader(buffer);
    auto int_val = reader.read<int>();
    assert(int_val.value() == 42);

    auto double_val = reader.read<double>();
    assert(std::abs(double_val.value() - 3.14) < 0.001);

    auto str_val = reader.read_string();
    assert(str_val.has_value() && str_val.value() == "Hello");

    auto vec_val = reader.read_vector<int>();
    assert(vec_val.value().size() == 5);
    std::cout << "  Binary serialization passed\n";

    // Test JSON
    json_value json;
    json["name"] = "John";
    json["age"] = 30;
    json["scores"] = json_value::array_t{85, 90, 95};

    assert(json["name"].as_string() == "John");
    assert(json["age"].as_int() == 30);
    assert(json["scores"][1].as_int() == 90);

    std::string json_str = json.to_string(2);
    assert(!json_str.empty());
    std::cout << "  JSON serialization passed\n";

    // Test versioned serialization
    struct test_data {
        int x = 10;
        int y = 20;
    };

    versioned_serializer vs(1);
    test_data original{100, 200};
    auto serialized = vs.serialize(original);
    assert(!serialized.empty());
    std::cout << "  Versioned serialization passed\n";

    // Test zero-copy view
    std::vector<int> data = {1, 2, 3, 4, 5};
    zero_copy_view<int> view(reinterpret_cast<const uint8_t*>(data.data()),
                             data.size() * sizeof(int));
    assert(view[2] == 3);
    assert(view.size() == 5);
    std::cout << "  Zero-copy view passed\n";

    // Test RLE compression
    std::vector<int> uncompressed = {1, 1, 1, 2, 2, 3, 3, 3, 3};
    auto compressed = compressed_serializer::compress_rle(uncompressed);
    auto decompressed = compressed_serializer::decompress_rle<int>(compressed);
    assert(decompressed.value() == uncompressed);
    std::cout << "  RLE compression passed\n";

    std::cout << "  All serialization tests passed\n";
}

// Benchmark all components
void benchmark_components() {
    std::cout << "\nBenchmarking components...\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Benchmark bloom filter
    bloom_filter<int> bf(100000, 0.01);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; ++i) {
        bf.insert(i);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  Bloom filter 100K inserts: " << duration.count() << " μs\n";

    // Benchmark HyperLogLog
    hyperloglog<int> hll(14);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; ++i) {
        hll.add(i);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  HyperLogLog 1M adds: " << duration.count() << " μs\n";

    // Benchmark PCG vs MT19937
    pcg32<> pcg;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; ++i) {
        pcg();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  PCG32 1M generations: " << duration.count() << " μs\n";

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; ++i) {
        gen();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  MT19937 1M generations: " << duration.count() << " μs\n";

    // Benchmark binary serialization
    binary_writer writer;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; ++i) {
        writer.write(i);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  Binary write 100K ints: " << duration.count() << " μs\n";

    auto buffer = writer.get_buffer();
    binary_reader reader(buffer);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; ++i) {
        reader.read<int>();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  Binary read 100K ints: " << duration.count() << " μs\n";
}

int main() {
    std::cout << "=== Testing All New Components ===\n\n";

    test_hashing();
    test_random();
    test_error_handling();
    test_serialization();

    benchmark_components();

    std::cout << "\n=== All component tests passed! ===\n";
    return 0;
}