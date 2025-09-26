/**
 * Universal Learning Through Compression
 * =======================================
 *
 * This example demonstrates that compression IS learning.
 * We build a universal classifier that can learn any pattern
 * without explicit features, parameters, or training algorithms.
 *
 * The key insight: The ability to compress data is mathematically
 * equivalent to understanding its patterns.
 */

#include <stepanov/compression.hpp>
#include <stepanov/algorithms.hpp>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>

namespace stepanov::examples {

/**
 * Universal Classifier based on Kolmogorov Complexity
 *
 * This classifier works on the principle that similar data compresses
 * well together. No neural networks, no parameters, just pure
 * information theory.
 */
class universal_classifier {
private:
    struct class_model {
        std::string label;
        std::string accumulated_data;
        stepanov::compressor<stepanov::lz77> comp;
    };

    std::vector<class_model> models;

public:
    /**
     * Train by simply concatenating examples.
     * The "model" is the data itself.
     */
    void train(const std::string& data, const std::string& label) {
        auto it = std::find_if(models.begin(), models.end(),
            [&](const auto& m) { return m.label == label; });

        if (it != models.end()) {
            it->accumulated_data += data;
        } else {
            models.push_back({label, data, {}});
        }
    }

    /**
     * Classify by finding which class compresses best with the input.
     * This implements the Normalized Compression Distance (NCD).
     */
    std::string classify(const std::string& input) {
        if (models.empty()) return "";

        std::string best_label;
        double best_score = std::numeric_limits<double>::max();

        for (const auto& model : models) {
            // NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
            auto cx = model.comp.compress(input).size();
            auto cy = model.comp.compress(model.accumulated_data).size();
            auto cxy = model.comp.compress(input + model.accumulated_data).size();

            double ncd = static_cast<double>(cxy - std::min(cx, cy))
                       / std::max(cx, cy);

            if (ncd < best_score) {
                best_score = ncd;
                best_label = model.label;
            }
        }

        return best_label;
    }

    /**
     * Predict the next element in a sequence by finding the continuation
     * that compresses best with the history.
     */
    template<typename T>
    T predict_next(const std::vector<T>& sequence,
                   const std::vector<T>& candidates) {
        if (candidates.empty()) return T{};

        std::string seq_str = serialize(sequence);
        T best_candidate = candidates[0];
        double best_ratio = 0;

        for (const auto& candidate : candidates) {
            std::string with_candidate = seq_str + serialize({candidate});
            auto compressed = stepanov::compress(with_candidate);
            double ratio = stepanov::compression_ratio(with_candidate, compressed);

            if (ratio > best_ratio) {
                best_ratio = ratio;
                best_candidate = candidate;
            }
        }

        return best_candidate;
    }

private:
    template<typename T>
    std::string serialize(const std::vector<T>& v) {
        std::string result;
        for (const auto& item : v) {
            result += std::to_string(item) + ",";
        }
        return result;
    }
};

/**
 * Demonstration: Language Detection Without Linguistic Knowledge
 */
void demo_language_detection() {
    std::cout << "=== Language Detection via Compression ===\n\n";

    universal_classifier classifier;

    // Train with example texts
    classifier.train(
        "The quick brown fox jumps over the lazy dog. "
        "This is an English sentence with common words.",
        "English"
    );

    classifier.train(
        "Le renard brun rapide saute par-dessus le chien paresseux. "
        "Ceci est une phrase française avec des mots communs.",
        "French"
    );

    classifier.train(
        "Der schnelle braune Fuchs springt über den faulen Hund. "
        "Dies ist ein deutscher Satz mit häufigen Wörtern.",
        "German"
    );

    classifier.train(
        "El rápido zorro marrón salta sobre el perro perezoso. "
        "Esta es una oración en español con palabras comunes.",
        "Spanish"
    );

    // Test classification
    std::vector<std::pair<std::string, std::string>> tests = {
        {"Hello world, how are you today?", "English"},
        {"Bonjour le monde, comment allez-vous?", "French"},
        {"Hallo Welt, wie geht es dir?", "German"},
        {"Hola mundo, ¿cómo estás?", "Spanish"},
        {"The weather is nice.", "English"},
        {"Das Wetter ist schön.", "German"}
    };

    for (const auto& [text, expected] : tests) {
        auto predicted = classifier.classify(text);
        std::cout << "Text: \"" << text << "\"\n";
        std::cout << "Predicted: " << predicted
                  << " | Expected: " << expected
                  << " | " << (predicted == expected ? "✓" : "✗") << "\n\n";
    }
}

/**
 * Demonstration: Pattern Learning in Sequences
 */
void demo_sequence_learning() {
    std::cout << "=== Sequence Pattern Learning ===\n\n";

    // Learn arithmetic sequences
    {
        std::vector<int> sequence = {2, 4, 6, 8, 10, 12};
        universal_classifier predictor;

        // Convert to string for compression
        std::string seq_str;
        for (int n : sequence) seq_str += std::to_string(n) + " ";

        // Try different continuations
        std::vector<std::pair<int, double>> candidates;
        for (int next = 13; next <= 16; ++next) {
            std::string test = seq_str + std::to_string(next);
            auto compressed = stepanov::compress(test);
            double ratio = stepanov::compression_ratio(test, compressed);
            candidates.push_back({next, ratio});
        }

        // Best continuation has highest compression
        auto best = std::max_element(candidates.begin(), candidates.end(),
            [](auto& a, auto& b) { return a.second < b.second; });

        std::cout << "Sequence: ";
        for (int n : sequence) std::cout << n << " ";
        std::cout << "?\n";
        std::cout << "Predicted next: " << best->first
                  << " (compression ratio: " << best->second << ")\n\n";
    }

    // Learn Fibonacci sequence
    {
        std::vector<int> fibonacci = {1, 1, 2, 3, 5, 8, 13, 21};
        std::string fib_str;
        for (int n : fibonacci) fib_str += std::to_string(n) + " ";

        std::vector<std::pair<int, double>> candidates;
        for (int next = 30; next <= 40; ++next) {
            std::string test = fib_str + std::to_string(next);
            auto compressed = stepanov::compress(test);
            double ratio = stepanov::compression_ratio(test, compressed);
            candidates.push_back({next, ratio});
        }

        auto best = std::max_element(candidates.begin(), candidates.end(),
            [](auto& a, auto& b) { return a.second < b.second; });

        std::cout << "Fibonacci: ";
        for (int n : fibonacci) std::cout << n << " ";
        std::cout << "?\n";
        std::cout << "Predicted next: " << best->first << "\n";
        std::cout << "Actual next: 34\n\n";
    }
}

/**
 * Demonstration: Anomaly Detection Through Incompressibility
 */
void demo_anomaly_detection() {
    std::cout << "=== Anomaly Detection via Compression ===\n\n";

    // Normal pattern: Regular heartbeat
    std::string normal_pattern = "lub-dub lub-dub lub-dub lub-dub lub-dub ";

    // Test various patterns
    std::vector<std::pair<std::string, std::string>> patterns = {
        {"lub-dub lub-dub lub-dub lub-dub", "Normal"},
        {"lub-dub lub-dub skip lub-dub", "Anomaly (skip)"},
        {"lub-dub-dub lub-dub lub-dub", "Anomaly (extra beat)"},
        {"lub-dub lub-dub lub-dub", "Normal (shorter)"},
        {"random noise xyz abc 123", "Anomaly (noise)"}
    };

    for (const auto& [pattern, label] : patterns) {
        // Measure how well it compresses with normal
        auto combined = normal_pattern + pattern;
        auto compressed = stepanov::compress(combined);
        double ratio = stepanov::compression_ratio(combined, compressed);

        // High compression = similar to normal
        // Low compression = anomaly
        bool is_normal = ratio > 0.5;

        std::cout << "Pattern: \"" << pattern << "\"\n";
        std::cout << "Compression ratio: " << ratio << "\n";
        std::cout << "Detection: " << (is_normal ? "Normal" : "Anomaly")
                  << " | Expected: " << label << "\n\n";
    }
}

/**
 * Demonstration: Clustering Without Distance Metrics
 */
void demo_compression_clustering() {
    std::cout << "=== Clustering via Compression ===\n\n";

    // Data points (disguised as strings for compression)
    std::vector<std::pair<std::string, std::string>> data = {
        // Cluster 1: Programming languages
        {"python java cpp csharp", "Programming"},
        {"javascript typescript rust go", "Programming"},
        {"haskell ocaml lisp scheme", "Programming"},

        // Cluster 2: Animals
        {"dog cat mouse rabbit", "Animals"},
        {"lion tiger bear wolf", "Animals"},
        {"eagle hawk sparrow crow", "Animals"},

        // Cluster 3: Colors
        {"red blue green yellow", "Colors"},
        {"purple orange pink brown", "Colors"},
        {"black white gray silver", "Colors"}
    };

    // Compute compression distance matrix
    std::cout << "Compression Distance Matrix:\n";
    std::cout << "      ";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << "D" << i << "    ";
    }
    std::cout << "\n";

    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << "D" << i << "   ";
        for (size_t j = 0; j < data.size(); ++j) {
            if (i == j) {
                std::cout << "0.00  ";
            } else {
                // Normalized Compression Distance
                auto xi = stepanov::compress(data[i].first);
                auto xj = stepanov::compress(data[j].first);
                auto xij = stepanov::compress(data[i].first + data[j].first);

                double ncd = static_cast<double>(xij.size() - std::min(xi.size(), xj.size()))
                           / std::max(xi.size(), xj.size());

                std::cout << std::fixed << std::setprecision(2) << ncd << "  ";
            }
        }
        std::cout << " | " << data[i].second << "\n";
    }

    std::cout << "\nNote: Lower values indicate similarity. "
              << "Same-category items cluster together!\n\n";
}

/**
 * The Philosophical Demonstration:
 * Compression = Understanding = Intelligence
 */
void demo_philosophical() {
    std::cout << "=== The Deep Connection ===\n\n";

    std::cout << "Consider these equivalent statements:\n\n";

    std::cout << "1. 'I understand this data'\n";
    std::cout << "   = 'I can find patterns in it'\n";
    std::cout << "   = 'I can compress it efficiently'\n\n";

    std::cout << "2. 'This AI learned from the data'\n";
    std::cout << "   = 'It can predict the next element'\n";
    std::cout << "   = 'It found a shorter description'\n\n";

    std::cout << "3. 'These items are similar'\n";
    std::cout << "   = 'They share common patterns'\n";
    std::cout << "   = 'They compress well together'\n\n";

    // Demonstrate with a simple proof
    std::string random = "xk3j9dk2ld9sk2ld9ak2ld";  // Random
    std::string pattern = "abcabcabcabcabcabcabc";  // Pattern

    auto random_compressed = stepanov::compress(random);
    auto pattern_compressed = stepanov::compress(pattern);

    std::cout << "Random string:  \"" << random << "\"\n";
    std::cout << "Compressed size: " << random_compressed.size()
              << " (ratio: " << stepanov::compression_ratio(random, random_compressed)
              << ")\n\n";

    std::cout << "Pattern string: \"" << pattern << "\"\n";
    std::cout << "Compressed size: " << pattern_compressed.size()
              << " (ratio: " << stepanov::compression_ratio(pattern, pattern_compressed)
              << ")\n\n";

    std::cout << "Conclusion: Understanding IS compression.\n";
    std::cout << "           Intelligence IS pattern finding.\n";
    std::cout << "           Learning IS model compression.\n\n";

    std::cout << "This is not metaphor. This is mathematics.\n";
    std::cout << "And Stepanov makes it practical.\n";
}

} // namespace stepanov::examples

int main() {
    using namespace stepanov::examples;

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Universal Learning Through Compression                 ║\n";
    std::cout << "║     Demonstrating: Compression = Intelligence              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    demo_language_detection();
    demo_sequence_learning();
    demo_anomaly_detection();
    demo_compression_clustering();
    demo_philosophical();

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  'The best model of the data is the shortest program      ║\n";
    std::cout << "║   that generates it.' - Ray Solomonoff                    ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║  Stepanov: Where compression theory meets practical code.  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    return 0;
}