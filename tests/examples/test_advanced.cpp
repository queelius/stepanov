#include <iostream>
#include <vector>
#include <iomanip>
#include <stepanov/tropical.hpp>
#include <stepanov/succinct.hpp>
#include <stepanov/compression.hpp>
#include <stepanov/verify.hpp>

using namespace stepanov;

int main() {
    std::cout << "=== Testing Stepanov Advanced Features ===" << std::endl;

    // 1. Tropical Mathematics
    std::cout << "\n1. TROPICAL MATHEMATICS" << std::endl;
    std::cout << "------------------------" << std::endl;

    // Example: Find shortest paths in a graph
    tropical::tropical_matrix<tropical::min_plus<double>> dist(4, 4);

    // Graph edges (adjacency matrix)
    dist(0, 1) = tropical::min_plus<double>(3);
    dist(0, 2) = tropical::min_plus<double>(8);
    dist(1, 2) = tropical::min_plus<double>(2);
    dist(1, 3) = tropical::min_plus<double>(5);
    dist(2, 3) = tropical::min_plus<double>(1);

    // Compute shortest paths with matrix powers
    auto dist2 = dist * dist;  // 2-hop paths
    auto dist3 = dist2 * dist;  // 3-hop paths

    std::cout << "Direct path 0->3: " << dist(0, 3).value << std::endl;
    std::cout << "2-hop path 0->3: " << dist2(0, 3).value << std::endl;
    std::cout << "3-hop path 0->3: " << dist3(0, 3).value << std::endl;

    // 2. Succinct Data Structures
    std::cout << "\n2. SUCCINCT DATA STRUCTURES" << std::endl;
    std::cout << "----------------------------" << std::endl;

    // Create a bit vector with rank/select support
    succinct::bit_vector bv = {1,1,0,1,0,0,1,1,1,0,1,0};
    std::cout << "Bit vector: 110100111010" << std::endl;
    std::cout << "Size: " << bv.size() << " bits" << std::endl;
    std::cout << "Number of 1s (rank1): " << bv.rank1(bv.size()) << std::endl;
    std::cout << "Position of 3rd 1 (select1): " << bv.select1(2) << std::endl;

    // FM-Index for text compression and search
    std::string text = "mississippi$";
    succinct::fm_index fm(text);
    std::cout << "\nText: " << text << std::endl;
    std::cout << "Pattern 'iss' occurs: " << fm.count("iss") << " times" << std::endl;
    std::cout << "Pattern 'sip' occurs: " << fm.count("sip") << " times" << std::endl;

    // 3. Advanced Compression
    std::cout << "\n3. ADVANCED COMPRESSION" << std::endl;
    std::cout << "------------------------" << std::endl;

    // Using LZ77 compression
    compression::lz77_compressor lz77;
    std::string text_to_compress = "the quick brown fox jumps over the lazy dog. the quick brown fox jumps again.";
    std::vector<uint8_t> input(text_to_compress.begin(), text_to_compress.end());

    auto compressed_lz77 = lz77.compress(input);
    auto decompressed_lz77 = lz77.decompress(compressed_lz77);

    std::cout << "Original size: " << input.size() << " bytes" << std::endl;
    std::cout << "Compressed size: " << compressed_lz77.data.size() << " bytes" << std::endl;
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2)
              << (double)input.size() / compressed_lz77.data.size() << ":1" << std::endl;
    std::cout << "Decompression successful: " << (input == decompressed_lz77 ? "Yes" : "No") << std::endl;

    // 4. Formal Verification
    std::cout << "\n4. FORMAL VERIFICATION" << std::endl;
    std::cout << "-----------------------" << std::endl;

    // Refinement types
    try {
        verify::positive_t<int> positive_num(42);
        std::cout << "Created positive number: " << positive_num.get() << std::endl;

        verify::non_negative_t<double> non_neg(0.0);
        std::cout << "Created non-negative number: " << non_neg.get() << std::endl;

        // This would throw if we tried negative
        // verify::positive_t<int> bad(-5);  // Contract violation!

    } catch (const verify::contract_violation& e) {
        std::cout << "Contract violated: " << e.what() << std::endl;
    }

    // Property-based testing
    auto test_commutativity = []() {
        auto gen1 = std::make_unique<verify::int_generator<int>>(1, 100);
        auto gen2 = std::make_unique<verify::int_generator<int>>(1, 100);

        auto property = [](int a, int b) -> bool {
            return (a + b) == (b + a);  // Addition is commutative
        };

        verify::property_test<int, int> test(
            "Addition commutativity",
            property,
            std::move(gen1),
            std::move(gen2),
            100
        );

        return test.run();
    };

    auto result = test_commutativity();
    std::cout << "Property test '" << result.property_name << "': "
              << (result.passed ? "PASS" : "FAIL")
              << " (" << result.num_tests << " tests in "
              << result.duration.count() << "ms)" << std::endl;

    std::cout << "\n=== All Tests Completed Successfully ===" << std::endl;

    return 0;
}