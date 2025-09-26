#include <stepanov/compression/benchmark.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
#include <cstdio>

using namespace stepanov::compression::benchmark;

// Test compressor that achieves known compression ratio
class test_compressor : public compressor_interface {
private:
    double target_ratio_;

public:
    explicit test_compressor(double ratio) : target_ratio_(ratio) {}

    std::vector<uint8_t> compress(const std::vector<uint8_t>& input) override {
        size_t compressed_size = static_cast<size_t>(input.size() / target_ratio_);
        return std::vector<uint8_t>(compressed_size, 0xAB);
    }

    std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed) override {
        size_t original_size = static_cast<size_t>(compressed.size() * target_ratio_);
        return std::vector<uint8_t>(original_size, 0xCD);
    }

    std::string name() const override {
        return "Test_" + std::to_string(static_cast<int>(target_ratio_)) + "x";
    }

    std::string description() const override {
        return "Test compressor with fixed " + std::to_string(target_ratio_) + "x ratio";
    }
};

// Non-virtual test compressor for concept testing
class simple_test_compressor {
private:
    double target_ratio_;

public:
    explicit simple_test_compressor(double ratio = 2.0) : target_ratio_(ratio) {}

    std::vector<uint8_t> compress(const std::vector<uint8_t>& input) const {
        size_t compressed_size = static_cast<size_t>(input.size() / target_ratio_);
        return std::vector<uint8_t>(compressed_size, 0xAB);
    }

    std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed) const {
        size_t original_size = static_cast<size_t>(compressed.size() * target_ratio_);
        return std::vector<uint8_t>(original_size, 0xCD);
    }

    std::string name() const {
        return "SimpleTest";
    }

    std::string description() const {
        return "Simple test compressor";
    }
};

void test_corpus_creation() {
    std::cout << "Testing corpus creation..." << std::endl;

    test_corpus corpus("Test", "test");
    corpus.add_text_sample("Hello, World!");
    corpus.add_text_sample("Testing 123");

    std::vector<uint8_t> binary = {0x00, 0xFF, 0xAA, 0x55};
    corpus.add_sample(binary);

    assert(corpus.samples().size() == 3);
    assert(corpus.characteristics().name == "Test");
    assert(corpus.characteristics().type == "test");
    assert(corpus.characteristics().file_count == 3);

    std::cout << "  ✓ Corpus creation works" << std::endl;
}

void test_standard_corpora() {
    std::cout << "Testing standard corpora..." << std::endl;

    auto canterbury = canterbury_corpus();
    assert(!canterbury.samples().empty());
    assert(canterbury.characteristics().entropy > 0);

    auto text = text_corpus();
    assert(!text.samples().empty());

    auto synthetic = synthetic_corpus();
    assert(!synthetic.samples().empty());

    std::cout << "  ✓ Standard corpora load correctly" << std::endl;
}

void test_compression_algorithms() {
    std::cout << "Testing compression algorithms..." << std::endl;

    // Test RLE
    simple_rle_compressor rle;
    std::vector<uint8_t> rle_input = {1, 1, 1, 2, 2, 3};
    auto rle_compressed = rle.compress(rle_input);
    auto rle_decompressed = rle.decompress(rle_compressed);
    assert(rle_decompressed == rle_input);
    std::cout << "  ✓ RLE compression works" << std::endl;

    // Test Delta
    simple_delta_compressor delta;
    std::vector<uint8_t> delta_input = {10, 11, 12, 13, 14};
    auto delta_compressed = delta.compress(delta_input);
    auto delta_decompressed = delta.decompress(delta_compressed);
    assert(delta_decompressed == delta_input);
    std::cout << "  ✓ Delta compression works" << std::endl;
}

void test_benchmark_runner() {
    std::cout << "Testing benchmark runner..." << std::endl;

    benchmark_runner runner;

    // Add test algorithms
    runner.add_algorithm(std::make_unique<test_compressor>(2.0));
    runner.add_algorithm(std::make_unique<test_compressor>(3.0));

    // Add simple corpus
    test_corpus corpus("Small", "test");
    corpus.add_text_sample("AAABBBCCC");
    runner.add_corpus(corpus);

    runner.set_runs_per_test(2);
    runner.set_verbose(false);

    // Run benchmarks
    runner.run();

    std::cout << "  ✓ Benchmark runner executes" << std::endl;
}

void test_metrics_calculation() {
    std::cout << "Testing metrics calculation..." << std::endl;

    // Create corpus with known characteristics
    test_corpus corpus("Metrics", "test");

    // Highly compressible data
    std::vector<uint8_t> compressible(1000, 0xAA);
    corpus.add_sample(compressible);

    auto characteristics = corpus.characteristics();
    assert(characteristics.total_size == 1000);

    // Entropy should be low for repeated data
    assert(characteristics.entropy < 1.0);

    std::cout << "  ✓ Metrics calculation works" << std::endl;
    std::cout << "    Entropy of repeated data: " << characteristics.entropy << " bits/byte" << std::endl;
}

void test_benchmark_suite_api() {
    std::cout << "Testing benchmark suite API..." << std::endl;

    benchmark_suite suite;

    // Test fluent interface
    suite.add_algorithm(simple_test_compressor(2.0))
         .add_algorithm(simple_test_compressor(4.0))
         .add_corpus(synthetic_corpus())
         .verbose(false)
         .runs_per_test(1);

    auto results = suite.run();

    // Test results interface
    std::cout << "\n--- Test Output ---\n";
    results.print_summary();
    results.recommend_best();
    std::cout << "--- End Test Output ---\n\n";

    std::cout << "  ✓ Benchmark suite API works" << std::endl;
}

void test_compression_concepts() {
    std::cout << "Testing compression concepts..." << std::endl;

    // Verify concept checking
    static_assert(CompressorConcept<simple_rle_compressor>);
    static_assert(CompressorConcept<simple_delta_compressor>);
    static_assert(CompressorConcept<simple_test_compressor>);

    std::cout << "  ✓ Compression concepts satisfied" << std::endl;
}

void test_csv_export() {
    std::cout << "Testing CSV export..." << std::endl;

    benchmark_suite suite;
    suite.add_algorithm(simple_test_compressor(2.0))
         .add_corpus(text_corpus())
         .verbose(false)
         .runs_per_test(1);

    auto results = suite.run();
    results.export_csv("test_output.csv");

    // Check if file was created
    std::ifstream file("test_output.csv");
    assert(file.is_open());

    std::string header;
    std::getline(file, header);
    assert(header.find("Corpus") != std::string::npos);
    assert(header.find("Algorithm") != std::string::npos);
    assert(header.find("Compression Ratio") != std::string::npos);

    file.close();
    std::remove("test_output.csv");

    std::cout << "  ✓ CSV export works" << std::endl;
}

void run_performance_comparison() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              PERFORMANCE COMPARISON DEMONSTRATION                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    benchmark_suite suite;

    // Add real algorithms
    suite.add_algorithm(simple_rle_compressor())
         .add_algorithm(simple_delta_compressor());

    // Add different data types
    suite.add_corpus(text_corpus())
         .add_corpus(synthetic_corpus());

    suite.verbose(false)
         .runs_per_test(5);

    auto results = suite.run();
    results.print_summary();
    results.recommend_best();
}

int main() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║            COMPRESSION BENCHMARK TEST SUITE                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    try {
        test_corpus_creation();
        test_standard_corpora();
        test_compression_algorithms();
        test_benchmark_runner();
        test_metrics_calculation();
        test_benchmark_suite_api();
        test_compression_concepts();
        test_csv_export();

        std::cout << "\n✅ All tests passed!\n";

        // Run demonstration
        run_performance_comparison();

        std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                     TEST SUITE COMPLETE                            ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

        std::cout << "The benchmark suite demonstrates:\n";
        std::cout << "  • Clean, composable API design\n";
        std::cout << "  • Statistical rigor in measurements\n";
        std::cout << "  • Beautiful visualization of results\n";
        std::cout << "  • Deep insights into compression behavior\n";
        std::cout << "  • Connection between compression and intelligence\n\n";

        std::cout << "\"To measure is to understand.\"\n";
        std::cout << "                    - Lord Kelvin\n\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}