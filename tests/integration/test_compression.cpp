#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <stepanov/compression.hpp>

using namespace stepanov::compression;

// Test data generators
std::vector<byte> generate_text_data() {
    std::string text = R"(
        The Stepanov compression library embodies the principles of generic programming.
        It provides simple, elegant, and composable compression algorithms.
        Each algorithm does one thing well, following the Unix philosophy.
        The library emphasizes clarity over cleverness, speed over maximum compression.
        This is a test of text compression, with repeated words and patterns.
        Compression, compression, compression - repetition helps compression algorithms.
    )";
    return std::vector<byte>(text.begin(), text.end());
}

std::vector<byte> generate_repetitive_data(size_t size = 10000) {
    std::vector<byte> data(size);
    for (size_t i = 0; i < size; ++i) {
        // Create runs of same values
        data[i] = (i / 100) % 256;
    }
    return data;
}

std::vector<byte> generate_sparse_data(size_t size = 10000) {
    std::vector<byte> data(size, 0);
    // Sparse data with occasional non-zero values
    for (size_t i = 0; i < size; i += 37) {
        data[i] = i % 256;
    }
    return data;
}

std::vector<byte> generate_random_data(size_t size = 10000) {
    std::vector<byte> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = (i * 2654435761U) % 256;  // Simple hash for pseudo-random
    }
    return data;
}

// Test a single compressor
template<typename Compressor>
void test_compressor(const std::string& name, Compressor& comp,
                     const std::vector<byte>& data) {
    std::cout << "\n=== Testing " << name << " ===" << std::endl;

    // Compress
    auto compressed = comp.compress(data);

    // Decompress
    auto decompressed = comp.decompress(compressed);

    // Check correctness
    bool correct = (decompressed.size() == data.size()) &&
                   std::equal(data.begin(), data.end(), decompressed.begin());

    // Display results
    std::cout << "Original size:    " << std::setw(8) << data.size() << " bytes" << std::endl;
    std::cout << "Compressed size:  " << std::setw(8) << compressed.data.size() << " bytes" << std::endl;
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2)
              << compressed.compression_ratio() << std::endl;
    std::cout << "Savings:          " << std::fixed << std::setprecision(1)
              << (1.0 - compressed.compression_ratio()) * 100 << "%" << std::endl;
    std::cout << "Correctness:      " << (correct ? "PASSED" : "FAILED") << std::endl;

    if (!correct && decompressed.size() > 0) {
        // Find first mismatch for debugging
        auto mismatch = std::mismatch(data.begin(), data.end(), decompressed.begin());
        if (mismatch.first != data.end()) {
            size_t pos = std::distance(data.begin(), mismatch.first);
            std::cout << "First mismatch at position " << pos << std::endl;
        }
    }
}

// Test transform
void test_transforms() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "      TESTING TRANSFORMS" << std::endl;
    std::cout << "========================================" << std::endl;

    // Test run-length encoding
    {
        run_length_transform rle;
        std::vector<byte> data = {1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5};

        auto encoded = rle.forward(data);
        auto decoded = rle.inverse(encoded);

        std::cout << "\nRun-Length Transform:" << std::endl;
        std::cout << "Original size: " << data.size() << std::endl;
        std::cout << "Encoded size:  " << encoded.size() << std::endl;
        std::cout << "Round-trip:    " << (data == decoded ? "PASSED" : "FAILED") << std::endl;
    }

    // Test move-to-front
    {
        move_to_front_transform mtf;
        std::vector<byte> data = {1, 1, 2, 1, 2, 2, 3, 2, 3, 3, 3};

        auto encoded = mtf.forward(data);
        auto decoded = mtf.inverse(encoded);

        std::cout << "\nMove-to-Front Transform:" << std::endl;
        std::cout << "Original: ";
        for (auto b : data) std::cout << (int)b << " ";
        std::cout << "\nEncoded:  ";
        for (auto b : encoded) std::cout << (int)b << " ";
        std::cout << "\nRound-trip: " << (data == decoded ? "PASSED" : "FAILED") << std::endl;
    }
}

// Test pipeline composition
void test_pipeline() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "      TESTING PIPELINE COMPOSITION" << std::endl;
    std::cout << "========================================" << std::endl;

    auto text_data = generate_text_data();

    // Create a pipeline: MTF -> RLE -> Fast LZ
    auto pipeline = make_pipeline(
        move_to_front_transform(),
        run_length_transform(),
        fast_lz_compressor()
    );

    auto compressed = pipeline.compress(text_data);

    std::cout << "\nPipeline: MTF -> RLE -> Fast LZ" << std::endl;
    std::cout << "Original size:    " << text_data.size() << " bytes" << std::endl;
    std::cout << "Compressed size:  " << compressed.data.size() << " bytes" << std::endl;
    std::cout << "Compression ratio: " << compressed.compression_ratio() << std::endl;
}

// Test compression utilities
void test_utilities() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "      TESTING UTILITIES" << std::endl;
    std::cout << "========================================" << std::endl;

    // Test data analysis
    auto text = generate_text_data();
    auto repetitive = generate_repetitive_data(1000);
    auto sparse = generate_sparse_data(1000);
    auto random = generate_random_data(1000);

    std::cout << "\nData Analysis:" << std::endl;
    auto print_type = [](data_type t) {
        switch(t) {
            case data_type::text: return "TEXT";
            case data_type::repetitive: return "REPETITIVE";
            case data_type::sparse: return "SPARSE";
            case data_type::binary: return "BINARY";
            default: return "UNKNOWN";
        }
    };

    std::cout << "Text data:       " << print_type(analyze_data(text)) << std::endl;
    std::cout << "Repetitive data: " << print_type(analyze_data(repetitive)) << std::endl;
    std::cout << "Sparse data:     " << print_type(analyze_data(sparse)) << std::endl;
    std::cout << "Random data:     " << print_type(analyze_data(random)) << std::endl;

    // Test compression detection
    fast_lz_compressor compressor;
    auto compressed = compressor.compress(text);

    std::cout << "\nCompression Detection:" << std::endl;
    std::cout << "Original text:     " << (is_compressed(text) ? "COMPRESSED" : "NOT COMPRESSED") << std::endl;
    std::cout << "Compressed data:   " << (is_compressed(compressed.data) ? "COMPRESSED" : "NOT COMPRESSED") << std::endl;
}

// Test arithmetic coding with different models
void test_arithmetic_coding() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "      TESTING ARITHMETIC CODING" << std::endl;
    std::cout << "========================================" << std::endl;

    auto text = generate_text_data();

    // Test with uniform model
    {
        arithmetic_coder<uniform_model> coder;
        auto compressed = coder.compress(text);

        std::cout << "\nArithmetic Coding with Uniform Model:" << std::endl;
        std::cout << "Original size:    " << text.size() << " bytes" << std::endl;
        std::cout << "Compressed size:  " << compressed.data.size() << " bytes" << std::endl;
        std::cout << "Compression ratio: " << compressed.compression_ratio() << std::endl;
    }

    // Test with adaptive model
    {
        arithmetic_coder<adaptive_model> coder;
        auto compressed = coder.compress(text);

        std::cout << "\nArithmetic Coding with Adaptive Model:" << std::endl;
        std::cout << "Original size:    " << text.size() << " bytes" << std::endl;
        std::cout << "Compressed size:  " << compressed.data.size() << " bytes" << std::endl;
        std::cout << "Compression ratio: " << compressed.compression_ratio() << std::endl;
    }
}

// Main test suite
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   STEPANOV COMPRESSION LIBRARY TEST" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nPhilosophy: Simple, Fast, Composable" << std::endl;
    std::cout << "80% of compression needs with 20% of complexity" << std::endl;

    // Test transforms
    test_transforms();

    // Generate test data
    auto text_data = generate_text_data();
    auto repetitive_data = generate_repetitive_data();
    auto sparse_data = generate_sparse_data();
    auto random_data = generate_random_data();

    // Test each compressor
    std::cout << "\n========================================" << std::endl;
    std::cout << "      TESTING INDIVIDUAL COMPRESSORS" << std::endl;
    std::cout << "========================================" << std::endl;

    // LZ77 - Good general purpose
    {
        lz77_compressor lz77;
        std::cout << "\n--- LZ77 on Different Data Types ---" << std::endl;
        test_compressor("LZ77 (Text)", lz77, text_data);
        test_compressor("LZ77 (Repetitive)", lz77, repetitive_data);
        test_compressor("LZ77 (Sparse)", lz77, sparse_data);
        test_compressor("LZ77 (Random)", lz77, random_data);
    }

    // Fast LZ - Speed focused
    {
        fast_lz_compressor fast_lz;
        std::cout << "\n--- Fast LZ (Speed > Ratio) ---" << std::endl;
        test_compressor("Fast LZ (Text)", fast_lz, text_data);
        test_compressor("Fast LZ (Repetitive)", fast_lz, repetitive_data);
    }

    // Test pipeline composition
    test_pipeline();

    // Test utilities
    test_utilities();

    // Test arithmetic coding
    test_arithmetic_coding();

    // Performance comparison
    std::cout << "\n========================================" << std::endl;
    std::cout << "      PERFORMANCE COMPARISON" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nCompressing 100KB of text data:" << std::endl;
    auto large_text = generate_text_data();
    for (int i = 0; i < 200; ++i) {
        auto more = generate_text_data();
        large_text.insert(large_text.end(), more.begin(), more.end());
    }

    {
        lz77_compressor lz77;
        auto result = benchmark(lz77, large_text);
        std::cout << "\nLZ77:" << std::endl;
        std::cout << "  Compression ratio: " << result.compression_ratio << std::endl;
        std::cout << "  Compress time:    " << result.compress_time_ms << " ms" << std::endl;
        std::cout << "  Decompress time:  " << result.decompress_time_ms << " ms" << std::endl;
        std::cout << "  Round-trip OK:    " << (result.successful ? "YES" : "NO") << std::endl;
    }

    {
        fast_lz_compressor fast_lz;
        auto result = benchmark(fast_lz, large_text);
        std::cout << "\nFast LZ:" << std::endl;
        std::cout << "  Compression ratio: " << result.compression_ratio << std::endl;
        std::cout << "  Compress time:    " << result.compress_time_ms << " ms" << std::endl;
        std::cout << "  Decompress time:  " << result.decompress_time_ms << " ms" << std::endl;
        std::cout << "  Round-trip OK:    " << (result.successful ? "YES" : "NO") << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "          TEST SUITE COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nThe library demonstrates that elegant," << std::endl;
    std::cout << "simple code can achieve practical compression" << std::endl;
    std::cout << "without over-engineering or complexity." << std::endl;

    return 0;
}