// Compression Algorithms Showcase
// Demonstrates the compositional compression framework

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <stepanov/compression/lz77.hpp>
#include <stepanov/compression/ans.hpp>
#include <stepanov/compression/grammar.hpp>
#include <stepanov/compression/ml.hpp>
#include <stepanov/compression/compositional.hpp>

using namespace stepanov::compression;
using namespace std;
using namespace chrono;

// Helper to display compression stats
void show_compression_stats(const string& name, size_t original, size_t compressed) {
    double ratio = 100.0 * (1.0 - double(compressed) / original);
    cout << "  " << left << setw(20) << name << ": ";
    cout << original << " → " << compressed << " bytes ";
    cout << "(" << fixed << setprecision(1) << ratio << "% reduction)\n";
}

int main() {
    cout << "=== Compression Algorithms Showcase ===\n\n";

    // Test data with different characteristics
    string repetitive = "abcabcabcabcabcabcabcabcabcabc";
    string english = "The quick brown fox jumps over the lazy dog. The dog was really lazy.";
    string dna = "ATCGATCGATCGATCGCGCGTATATATGCGCATCGATCG";
    string code = "for(int i=0; i<n; ++i) { sum += array[i]; array[i] *= 2; }";

    // 1. LZ77 - Dictionary-based compression
    cout << "1. LZ77 COMPRESSION\n";
    cout << "-------------------\n";
    cout << "Exploits repetition through back-references:\n\n";

    lz77_compressor lz77;

    auto compressed_rep = lz77.compress(repetitive);
    show_compression_stats("Repetitive text", repetitive.size(), compressed_rep.size());

    auto compressed_eng = lz77.compress(english);
    show_compression_stats("English text", english.size(), compressed_eng.size());

    auto compressed_dna = lz77.compress(dna);
    show_compression_stats("DNA sequence", dna.size(), compressed_dna.size());

    // Verify decompression
    auto decompressed = lz77.decompress(compressed_rep);
    cout << "\n  Decompression test: " << (decompressed == repetitive ? "✓ PASS" : "✗ FAIL") << "\n\n";

    // 2. ANS - Asymmetric Numeral Systems
    cout << "2. ANS COMPRESSION\n";
    cout << "------------------\n";
    cout << "Near-optimal entropy coding:\n\n";

    ans_compressor ans;

    // ANS adapts to symbol frequencies
    string skewed = "aaaaaabbbcc";  // Skewed distribution
    auto compressed_skewed = ans.compress(skewed);
    show_compression_stats("Skewed distribution", skewed.size(), compressed_skewed.size());

    string uniform = "abcdefghijk";  // Uniform distribution
    auto compressed_uniform = ans.compress(uniform);
    show_compression_stats("Uniform distribution", uniform.size(), compressed_uniform.size());

    cout << "\n  ANS achieves near-entropy compression!\n\n";

    // 3. Grammar-based compression
    cout << "3. GRAMMAR-BASED COMPRESSION\n";
    cout << "----------------------------\n";
    cout << "Represents text as context-free grammar:\n\n";

    grammar_compressor grammar;

    string structured = "if (x > 0) { return x; } if (y > 0) { return y; }";
    auto compressed_struct = grammar.compress(structured);

    cout << "  Original: \"" << structured << "\"\n";
    cout << "  Grammar rules discovered:\n";
    for (const auto& rule : compressed_struct.rules) {
        cout << "    " << rule.lhs << " → " << rule.rhs << "\n";
    }
    show_compression_stats("Structured text", structured.size(), compressed_struct.size());
    cout << "\n";

    // 4. ML-based compression
    cout << "4. ML-BASED COMPRESSION\n";
    cout << "-----------------------\n";
    cout << "Uses machine learning for prediction:\n\n";

    ml_compressor ml;

    // ML compression learns patterns
    string pattern = "0101010101010101010101010101";
    auto compressed_pattern = ml.compress(pattern);
    show_compression_stats("Pattern sequence", pattern.size(), compressed_pattern.size());

    string natural = "The compression algorithm learns from the data";
    auto compressed_natural = ml.compress(natural);
    show_compression_stats("Natural language", natural.size(), compressed_natural.size());

    cout << "\n  ML compression adapts to data characteristics!\n\n";

    // 5. Compositional Framework
    cout << "5. COMPOSITIONAL FRAMEWORK\n";
    cout << "--------------------------\n";
    cout << "Combine algorithms for best results:\n\n";

    // Create hybrid compressor: LZ77 → ANS
    compositional_compressor hybrid;
    hybrid.add_stage(make_unique<lz77_compressor>());
    hybrid.add_stage(make_unique<ans_compressor>());

    cout << "Hybrid (LZ77 → ANS) compression:\n";

    auto hybrid_compressed = hybrid.compress(english);
    show_compression_stats("English text", english.size(), hybrid_compressed.size());

    // Compare with individual algorithms
    cout << "\nComparison with individual algorithms:\n";
    show_compression_stats("LZ77 alone", english.size(), compressed_eng.size());
    show_compression_stats("ANS alone", english.size(), ans.compress(english).size());
    show_compression_stats("Hybrid", english.size(), hybrid_compressed.size());

    cout << "\n  Hybrid often outperforms individual algorithms!\n\n";

    // 6. Performance comparison
    cout << "6. PERFORMANCE COMPARISON\n";
    cout << "------------------------\n";

    // Generate larger test data
    string large_data;
    for (int i = 0; i < 1000; ++i) {
        large_data += english;
    }

    cout << "Compressing " << large_data.size() / 1024 << " KB of text:\n\n";

    // Time each algorithm
    auto benchmark = [](const string& name, auto& compressor, const string& data) {
        auto start = high_resolution_clock::now();
        auto compressed = compressor.compress(data);
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(end - start).count() / 1000.0;

        cout << "  " << left << setw(15) << name << ": ";
        cout << right << setw(8) << fixed << setprecision(2) << duration << " ms";
        cout << "  (ratio: " << setprecision(1);
        cout << 100.0 * compressed.size() / data.size() << "%)\n";

        return compressed;
    };

    benchmark("LZ77", lz77, large_data);
    benchmark("ANS", ans, large_data);
    benchmark("Grammar", grammar, large_data);
    benchmark("ML", ml, large_data);
    benchmark("Hybrid", hybrid, large_data);

    cout << "\n";

    // 7. Adaptive compression
    cout << "7. ADAPTIVE COMPRESSION\n";
    cout << "-----------------------\n";

    cout << "Framework automatically selects best algorithm:\n\n";

    adaptive_compressor adaptive;
    adaptive.add_algorithm("lz77", make_unique<lz77_compressor>());
    adaptive.add_algorithm("ans", make_unique<ans_compressor>());
    adaptive.add_algorithm("grammar", make_unique<grammar_compressor>());

    cout << "Testing on different data types:\n";

    auto test_adaptive = [&](const string& name, const string& data) {
        auto result = adaptive.compress_auto(data);
        cout << "  " << left << setw(20) << name << ": ";
        cout << "selected " << result.algorithm_used << "\n";
    };

    test_adaptive("Repetitive", repetitive);
    test_adaptive("English", english);
    test_adaptive("DNA", dna);
    test_adaptive("Code", code);

    cout << "\nKey Insight: The compositional framework enables\n";
    cout << "mixing and matching algorithms to achieve\n";
    cout << "compression ratios beyond any single method!\n";

    return 0;
}