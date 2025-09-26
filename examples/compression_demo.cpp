#include <iostream>
#include <string>
#include <vector>
#include "include/stepanov/compression.hpp"

using namespace stepanov::compression;

// Demonstrate the elegant, composable API
int main() {
    // Example 1: Simple compression with automatic algorithm selection
    std::string message = "The quick brown fox jumps over the lazy dog. "
                          "The quick brown fox jumps over the lazy dog.";

    std::vector<byte> data(message.begin(), message.end());

    std::cout << "Original: " << message << "\n";
    std::cout << "Size: " << data.size() << " bytes\n\n";

    // Fast compression - prioritizes speed
    {
        fast_lz_compressor compressor;
        auto compressed = compressor.compress(data);
        auto decompressed = compressor.decompress(compressed);

        std::cout << "Fast LZ Compression:\n";
        std::cout << "  Compressed size: " << compressed.data.size() << " bytes\n";
        std::cout << "  Ratio: " << compressed.compression_ratio() << "\n";
        std::cout << "  Savings: " << (1.0 - compressed.compression_ratio()) * 100 << "%\n\n";
    }

    // Example 2: Composable pipeline for better compression
    {
        auto pipeline = make_pipeline(
            move_to_front_transform(),  // Improve locality
            run_length_transform(),      // Handle repetition
            fast_lz_compressor()         // Final compression
        );

        auto compressed = pipeline.compress(data);

        std::cout << "Pipeline (MTF->RLE->LZ):\n";
        std::cout << "  Compressed size: " << compressed.data.size() << " bytes\n";
        std::cout << "  Ratio: " << compressed.compression_ratio() << "\n\n";
    }

    // Example 3: Generic arithmetic coding with custom model
    {
        // Use adaptive model that learns data statistics
        arithmetic_coder<adaptive_model> coder;
        auto compressed = coder.compress(data);

        std::cout << "Arithmetic Coding (Adaptive):\n";
        std::cout << "  Compressed size: " << compressed.data.size() << " bytes\n";
        std::cout << "  Ratio: " << compressed.compression_ratio() << "\n\n";
    }

    // Example 4: Smart compression based on data analysis
    {
        auto type = analyze_data(data);
        std::cout << "Data Analysis:\n";
        std::cout << "  Detected type: ";
        switch(type) {
            case data_type::text: std::cout << "TEXT\n"; break;
            case data_type::repetitive: std::cout << "REPETITIVE\n"; break;
            case data_type::sparse: std::cout << "SPARSE\n"; break;
            case data_type::binary: std::cout << "BINARY\n"; break;
            default: std::cout << "UNKNOWN\n";
        }

        // Let the library suggest the best compressor
        std::cout << "  Recommended: ";
        if (type == data_type::text) {
            std::cout << "Huffman or LZ77\n";
        } else if (type == data_type::repetitive || type == data_type::sparse) {
            std::cout << "Run-length encoding or Fast LZ\n";
        } else {
            std::cout << "General purpose LZ77\n";
        }
    }

    // Example 5: Check if data is already compressed
    {
        fast_lz_compressor comp;
        auto compressed = comp.compress(data);

        std::cout << "\nCompression Detection:\n";
        std::cout << "  Original: " << (is_compressed(data) ? "COMPRESSED" : "UNCOMPRESSED") << "\n";
        std::cout << "  After compression: " << (is_compressed(compressed.data) ? "COMPRESSED" : "UNCOMPRESSED") << "\n";
    }

    std::cout << "\n=== Design Philosophy ===\n";
    std::cout << "- Simple: Each algorithm does one thing well\n";
    std::cout << "- Fast: Speed prioritized over maximum compression\n";
    std::cout << "- Composable: Build custom pipelines easily\n";
    std::cout << "- Generic: Works with any probability model\n";
    std::cout << "- Practical: 80% of needs with 20% of complexity\n";

    return 0;
}