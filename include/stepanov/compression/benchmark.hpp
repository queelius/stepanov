#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "../concepts.hpp"

namespace stepanov::compression::benchmark {

// ==========================================
// Core Metric Types
// ==========================================

struct compression_metrics {
    double compression_ratio;      // original_size / compressed_size
    double compression_speed_mbps;  // MB/s for compression
    double decompression_speed_mbps; // MB/s for decompression
    size_t peak_memory_bytes;      // Peak memory usage
    size_t avg_memory_bytes;       // Average memory usage
    double cpu_utilization;        // 0.0 to 1.0 per core

    // Advanced metrics
    double kolmogorov_approximation; // K(x) ≈ |C(x)|
    double entropy_estimate;         // H(X) in bits per byte
    double incompressibility_score;  // 0 = highly compressible, 1 = incompressible
    double stability_variance;       // Variance across similar inputs

    // Statistical measures
    size_t sample_count;
    double confidence_interval;
    double std_deviation;
};

struct corpus_characteristics {
    std::string name;
    std::string type;  // text, binary, structured, media, synthetic
    size_t total_size;
    size_t file_count;
    double entropy;
    double redundancy;
    std::map<std::string, double> feature_scores;
};

// ==========================================
// Compression Algorithm Interface
// ==========================================

template<typename T>
concept CompressorConcept = requires(T t, const std::vector<uint8_t>& input) {
    { t.compress(input) } -> std::convertible_to<std::vector<uint8_t>>;
    { t.decompress(input) } -> std::convertible_to<std::vector<uint8_t>>;
    { t.name() } -> std::convertible_to<std::string>;
    { t.description() } -> std::convertible_to<std::string>;
};

class compressor_interface {
public:
    virtual ~compressor_interface() = default;
    virtual std::vector<uint8_t> compress(const std::vector<uint8_t>& input) = 0;
    virtual std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed) = 0;
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
    virtual size_t memory_usage() const { return 0; }
};

// ==========================================
// Test Corpus Management
// ==========================================

class test_corpus {
private:
    std::string name_;
    std::string type_;
    std::vector<std::vector<uint8_t>> samples_;
    corpus_characteristics characteristics_;

public:
    test_corpus(const std::string& name, const std::string& type)
        : name_(name), type_(type) {}

    void add_sample(const std::vector<uint8_t>& data) {
        samples_.push_back(data);
        update_characteristics();
    }

    void add_text_sample(const std::string& text) {
        std::vector<uint8_t> data(text.begin(), text.end());
        add_sample(data);
    }

    const std::vector<std::vector<uint8_t>>& samples() const {
        return samples_;
    }

    corpus_characteristics characteristics() const {
        return characteristics_;
    }

private:
    void update_characteristics() {
        characteristics_.name = name_;
        characteristics_.type = type_;
        characteristics_.file_count = samples_.size();

        size_t total = 0;
        for (const auto& sample : samples_) {
            total += sample.size();
        }
        characteristics_.total_size = total;

        // Calculate entropy
        if (!samples_.empty()) {
            std::vector<size_t> freq(256, 0);
            size_t total_bytes = 0;

            for (const auto& sample : samples_) {
                for (uint8_t byte : sample) {
                    freq[byte]++;
                    total_bytes++;
                }
            }

            double entropy = 0.0;
            for (size_t count : freq) {
                if (count > 0) {
                    double p = static_cast<double>(count) / total_bytes;
                    entropy -= p * std::log2(p);
                }
            }
            characteristics_.entropy = entropy;
            characteristics_.redundancy = 8.0 - entropy;  // Max 8 bits per byte
        }
    }
};

// ==========================================
// Standard Test Corpora
// ==========================================

inline test_corpus canterbury_corpus() {
    test_corpus corpus("Canterbury", "mixed");

    // Simplified samples representing Canterbury corpus characteristics
    corpus.add_text_sample("To be or not to be, that is the question.");
    corpus.add_text_sample("int main() { printf(\"Hello, World!\"); return 0; }");
    corpus.add_text_sample("<html><body><h1>Title</h1><p>Content</p></body></html>");

    // Add binary-like data
    std::vector<uint8_t> binary_sample;
    for (int i = 0; i < 256; ++i) {
        binary_sample.push_back(static_cast<uint8_t>(i));
        binary_sample.push_back(static_cast<uint8_t>(i ^ 0xFF));
    }
    corpus.add_sample(binary_sample);

    return corpus;
}

inline test_corpus text_corpus() {
    test_corpus corpus("Text", "text");

    corpus.add_text_sample(R"(
        In the beginning was the Word, and the Word was with God,
        and the Word was God. All things were made through him,
        and without him was not any thing made that was made.
    )");

    corpus.add_text_sample(R"(
        We hold these truths to be self-evident, that all men are created equal,
        that they are endowed by their Creator with certain unalienable Rights,
        that among these are Life, Liberty and the pursuit of Happiness.
    )");

    corpus.add_text_sample(R"(
        #include <iostream>
        template<typename T>
        T fibonacci(T n) {
            if (n <= 1) return n;
            return fibonacci(n-1) + fibonacci(n-2);
        }
    )");

    return corpus;
}

inline test_corpus synthetic_corpus() {
    test_corpus corpus("Synthetic", "synthetic");

    // Highly compressible: repeated pattern
    std::vector<uint8_t> repeated(10000, 0xAA);
    corpus.add_sample(repeated);

    // Incompressible: random data
    std::vector<uint8_t> random_data;
    uint32_t seed = 12345;
    for (size_t i = 0; i < 10000; ++i) {
        seed = seed * 1103515245 + 12345;
        random_data.push_back(static_cast<uint8_t>(seed >> 16));
    }
    corpus.add_sample(random_data);

    // Semi-compressible: pattern with noise
    std::vector<uint8_t> noisy_pattern;
    for (size_t i = 0; i < 10000; ++i) {
        if (i % 10 < 7) {
            noisy_pattern.push_back(static_cast<uint8_t>('A' + (i % 26)));
        } else {
            seed = seed * 1103515245 + 12345;
            noisy_pattern.push_back(static_cast<uint8_t>(seed >> 16));
        }
    }
    corpus.add_sample(noisy_pattern);

    return corpus;
}

// ==========================================
// Benchmark Runner
// ==========================================

class benchmark_runner {
private:
    std::vector<std::unique_ptr<compressor_interface>> algorithms_;
    std::vector<test_corpus> corpora_;
    std::map<std::string, std::map<std::string, compression_metrics>> results_;
    size_t runs_per_test_ = 5;
    bool verbose_ = false;

public:
    void add_algorithm(std::unique_ptr<compressor_interface> algo) {
        algorithms_.push_back(std::move(algo));
    }

    template<CompressorConcept T>
    void add_algorithm(T&& compressor) {
        class wrapper : public compressor_interface {
            T comp_;
        public:
            explicit wrapper(T c) : comp_(std::move(c)) {}
            std::vector<uint8_t> compress(const std::vector<uint8_t>& input) override {
                return comp_.compress(input);
            }
            std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed) override {
                return comp_.decompress(compressed);
            }
            std::string name() const override { return comp_.name(); }
            std::string description() const override { return comp_.description(); }
        };

        algorithms_.push_back(std::make_unique<wrapper>(std::forward<T>(compressor)));
    }

    void add_corpus(const test_corpus& corpus) {
        corpora_.push_back(corpus);
    }

    void set_runs_per_test(size_t runs) {
        runs_per_test_ = runs;
    }

    void set_verbose(bool v) {
        verbose_ = v;
    }

    void run() {
        results_.clear();

        for (const auto& corpus : corpora_) {
            if (verbose_) {
                std::cout << "\n=== Testing corpus: " << corpus.characteristics().name
                         << " ===\n";
            }

            for (const auto& algo : algorithms_) {
                if (verbose_) {
                    std::cout << "  Testing " << algo->name() << "...\n";
                }

                auto metrics = benchmark_algorithm(*algo, corpus);
                results_[corpus.characteristics().name][algo->name()] = metrics;
            }
        }
    }

    void print_summary() const {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              COMPRESSION BENCHMARK RESULTS                         ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";

        for (const auto& [corpus_name, algo_results] : results_) {
            std::cout << "\n┌─── Corpus: " << corpus_name << " ───┐\n";

            // Print header
            std::cout << std::left << std::setw(20) << "Algorithm"
                     << std::right << std::setw(12) << "Ratio"
                     << std::setw(15) << "Comp MB/s"
                     << std::setw(15) << "Decomp MB/s"
                     << std::setw(15) << "Memory KB"
                     << std::setw(10) << "Entropy"
                     << "\n";
            std::cout << std::string(87, '-') << "\n";

            // Sort by compression ratio
            std::vector<std::pair<std::string, compression_metrics>> sorted_results(
                algo_results.begin(), algo_results.end()
            );
            std::sort(sorted_results.begin(), sorted_results.end(),
                [](const auto& a, const auto& b) {
                    return a.second.compression_ratio > b.second.compression_ratio;
                });

            for (const auto& [algo_name, metrics] : sorted_results) {
                std::cout << std::left << std::setw(20) << algo_name
                         << std::right << std::fixed << std::setprecision(2)
                         << std::setw(12) << metrics.compression_ratio << "x"
                         << std::setw(15) << metrics.compression_speed_mbps
                         << std::setw(15) << metrics.decompression_speed_mbps
                         << std::setw(15) << (metrics.peak_memory_bytes / 1024)
                         << std::setw(10) << metrics.entropy_estimate
                         << "\n";
            }
        }

        print_visualization();
    }

    void print_visualization() const {
        std::cout << "\n┌─── Compression-Speed Tradeoff ───┐\n";
        std::cout << "│ Speed                             │\n";
        std::cout << "│ (MB/s)                           │\n";
        std::cout << "│ 1000 ┤                           │\n";

        // ASCII art scatter plot
        const int width = 35;
        const int height = 10;
        std::vector<std::string> plot(height, std::string(width, ' '));

        double max_ratio = 0, max_speed = 0;
        for (const auto& [corpus, algos] : results_) {
            for (const auto& [name, metrics] : algos) {
                max_ratio = std::max(max_ratio, metrics.compression_ratio);
                max_speed = std::max(max_speed, metrics.compression_speed_mbps);
            }
        }

        // Place points
        char marker = 'A';
        std::map<char, std::string> legend;

        for (const auto& [corpus, algos] : results_) {
            for (const auto& [name, metrics] : algos) {
                int x = static_cast<int>((metrics.compression_ratio / max_ratio) * (width - 1));
                int y = height - 1 - static_cast<int>((metrics.compression_speed_mbps / max_speed) * (height - 1));

                if (x >= 0 && x < width && y >= 0 && y < height) {
                    plot[y][x] = marker;
                    legend[marker] = name.substr(0, 10);
                    marker++;
                    if (marker > 'Z') marker = 'a';
                }
            }
        }

        // Print plot
        for (int i = 0; i < height; ++i) {
            std::cout << "│ ";
            if (i == 0) std::cout << " 100 ┤";
            else if (i == height - 1) std::cout << "   0 ┤";
            else std::cout << "     │";
            std::cout << plot[i] << "│\n";
        }

        std::cout << "│     └────┴────┴────┴────┴────┤\n";
        std::cout << "│     0   2   4   6   8   10     │\n";
        std::cout << "│         Compression Ratio       │\n";
        std::cout << "└─────────────────────────────────┘\n";

        // Print legend
        std::cout << "\nLegend: ";
        for (const auto& [mark, name] : legend) {
            std::cout << mark << "=" << name << "  ";
        }
        std::cout << "\n";
    }

    void recommend_best() const {
        std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    RECOMMENDATIONS                                 ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";

        for (const auto& [corpus_name, algo_results] : results_) {
            std::cout << "\n▶ For " << corpus_name << " data:\n";

            // Find best for different criteria
            auto best_ratio = std::max_element(algo_results.begin(), algo_results.end(),
                [](const auto& a, const auto& b) {
                    return a.second.compression_ratio < b.second.compression_ratio;
                });

            auto best_speed = std::max_element(algo_results.begin(), algo_results.end(),
                [](const auto& a, const auto& b) {
                    return a.second.compression_speed_mbps < b.second.compression_speed_mbps;
                });

            auto best_balanced = std::max_element(algo_results.begin(), algo_results.end(),
                [](const auto& a, const auto& b) {
                    double score_a = a.second.compression_ratio * std::sqrt(a.second.compression_speed_mbps);
                    double score_b = b.second.compression_ratio * std::sqrt(b.second.compression_speed_mbps);
                    return score_a < score_b;
                });

            std::cout << "  • Best compression: " << best_ratio->first
                     << " (" << std::fixed << std::setprecision(2)
                     << best_ratio->second.compression_ratio << "x)\n";

            std::cout << "  • Fastest: " << best_speed->first
                     << " (" << best_speed->second.compression_speed_mbps << " MB/s)\n";

            std::cout << "  • Best balanced: " << best_balanced->first << "\n";

            // Analyze data characteristics
            analyze_corpus_fit(corpus_name, algo_results);
        }
    }

    void export_csv(const std::string& filename) const {
        std::ofstream out(filename);

        // Header
        out << "Corpus,Algorithm,Compression Ratio,Compression Speed (MB/s),"
            << "Decompression Speed (MB/s),Peak Memory (KB),Entropy,Kolmogorov Approx\n";

        for (const auto& [corpus_name, algo_results] : results_) {
            for (const auto& [algo_name, metrics] : algo_results) {
                out << corpus_name << ","
                    << algo_name << ","
                    << metrics.compression_ratio << ","
                    << metrics.compression_speed_mbps << ","
                    << metrics.decompression_speed_mbps << ","
                    << (metrics.peak_memory_bytes / 1024) << ","
                    << metrics.entropy_estimate << ","
                    << metrics.kolmogorov_approximation << "\n";
            }
        }
    }

private:
    compression_metrics benchmark_algorithm(compressor_interface& algo,
                                           const test_corpus& corpus) {
        compression_metrics metrics{};
        std::vector<double> ratios, comp_speeds, decomp_speeds;

        for (const auto& sample : corpus.samples()) {
            for (size_t run = 0; run < runs_per_test_; ++run) {
                // Measure compression
                auto comp_start = std::chrono::high_resolution_clock::now();
                auto compressed = algo.compress(sample);
                auto comp_end = std::chrono::high_resolution_clock::now();

                // Measure decompression
                auto decomp_start = std::chrono::high_resolution_clock::now();
                auto decompressed = algo.decompress(compressed);
                auto decomp_end = std::chrono::high_resolution_clock::now();

                // Calculate metrics
                double ratio = static_cast<double>(sample.size()) / compressed.size();
                ratios.push_back(ratio);

                auto comp_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    comp_end - comp_start).count();
                double comp_speed = (sample.size() / 1024.0 / 1024.0) /
                                  (comp_duration / 1000000.0);
                comp_speeds.push_back(comp_speed);

                auto decomp_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    decomp_end - decomp_start).count();
                double decomp_speed = (sample.size() / 1024.0 / 1024.0) /
                                    (decomp_duration / 1000000.0);
                decomp_speeds.push_back(decomp_speed);

                // Verify correctness
                if (decompressed != sample) {
                    std::cerr << "Warning: Decompression mismatch for "
                             << algo.name() << "\n";
                }
            }
        }

        // Calculate averages and statistics
        metrics.compression_ratio = calculate_mean(ratios);
        metrics.compression_speed_mbps = calculate_mean(comp_speeds);
        metrics.decompression_speed_mbps = calculate_mean(decomp_speeds);
        metrics.std_deviation = calculate_std_dev(ratios);
        metrics.sample_count = ratios.size();
        metrics.confidence_interval = 1.96 * metrics.std_deviation /
                                     std::sqrt(metrics.sample_count);

        // Estimate advanced metrics
        metrics.kolmogorov_approximation = estimate_kolmogorov(corpus, metrics.compression_ratio);
        metrics.entropy_estimate = corpus.characteristics().entropy;
        metrics.incompressibility_score = calculate_incompressibility(metrics.compression_ratio);
        metrics.stability_variance = calculate_variance(ratios);

        // Memory usage (simplified estimate)
        metrics.peak_memory_bytes = algo.memory_usage();
        if (metrics.peak_memory_bytes == 0) {
            // Estimate based on algorithm characteristics
            metrics.peak_memory_bytes = estimate_memory_usage(algo.name(), corpus);
        }
        metrics.avg_memory_bytes = metrics.peak_memory_bytes * 0.7;

        return metrics;
    }

    double calculate_mean(const std::vector<double>& values) const {
        if (values.empty()) return 0.0;
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    }

    double calculate_std_dev(const std::vector<double>& values) const {
        if (values.size() < 2) return 0.0;
        double mean = calculate_mean(values);
        double sq_sum = 0.0;
        for (double v : values) {
            sq_sum += (v - mean) * (v - mean);
        }
        return std::sqrt(sq_sum / (values.size() - 1));
    }

    double calculate_variance(const std::vector<double>& values) const {
        double std_dev = calculate_std_dev(values);
        return std_dev * std_dev;
    }

    double estimate_kolmogorov(const test_corpus& corpus, double compression_ratio) const {
        // K(x) ≈ |C(x)| where C is the best compressor
        // Normalized by original size for comparison
        return 1.0 / compression_ratio;
    }

    double calculate_incompressibility(double compression_ratio) const {
        // Score from 0 (highly compressible) to 1 (incompressible)
        // Ratio near 1.0 means incompressible
        return 1.0 / compression_ratio;
    }

    size_t estimate_memory_usage(const std::string& algo_name,
                                 const test_corpus& corpus) const {
        // Heuristic estimates based on algorithm type
        size_t avg_sample_size = corpus.characteristics().total_size /
                                corpus.characteristics().file_count;

        if (algo_name.find("lz") != std::string::npos) {
            return avg_sample_size * 3;  // Dictionary + buffers
        } else if (algo_name.find("arithmetic") != std::string::npos) {
            return avg_sample_size * 2;  // Model + buffer
        } else if (algo_name.find("huffman") != std::string::npos) {
            return avg_sample_size + 65536;  // Tree + buffer
        }
        return avg_sample_size * 2;  // Default estimate
    }

    void analyze_corpus_fit(const std::string& corpus_name,
                           const std::map<std::string, compression_metrics>& algo_results) const {
        std::cout << "\n  Analysis:\n";

        // Find corpus characteristics
        for (const auto& corpus : corpora_) {
            if (corpus.characteristics().name == corpus_name) {
                double entropy = corpus.characteristics().entropy;

                if (entropy < 3.0) {
                    std::cout << "    → Highly redundant data (entropy="
                             << std::fixed << std::setprecision(2) << entropy
                             << " bits/byte)\n";
                    std::cout << "    → Dictionary-based methods excel here\n";
                } else if (entropy < 6.0) {
                    std::cout << "    → Moderate redundancy (entropy="
                             << std::fixed << std::setprecision(2) << entropy
                             << " bits/byte)\n";
                    std::cout << "    → Context-modeling algorithms recommended\n";
                } else {
                    std::cout << "    → Low redundancy (entropy="
                             << std::fixed << std::setprecision(2) << entropy
                             << " bits/byte)\n";
                    std::cout << "    → Consider specialized or no compression\n";
                }

                // Check if compression is worthwhile
                auto best = std::max_element(algo_results.begin(), algo_results.end(),
                    [](const auto& a, const auto& b) {
                        return a.second.compression_ratio < b.second.compression_ratio;
                    });

                if (best->second.compression_ratio < 1.1) {
                    std::cout << "    ⚠ Compression not recommended (max ratio only "
                             << best->second.compression_ratio << "x)\n";
                }

                break;
            }
        }
    }
};

// ==========================================
// Sample Compression Algorithms
// ==========================================

class simple_rle_compressor {
public:
    std::vector<uint8_t> compress(const std::vector<uint8_t>& input) {
        std::vector<uint8_t> output;
        if (input.empty()) return output;

        uint8_t current = input[0];
        uint8_t count = 1;

        for (size_t i = 1; i < input.size(); ++i) {
            if (input[i] == current && count < 255) {
                count++;
            } else {
                output.push_back(count);
                output.push_back(current);
                current = input[i];
                count = 1;
            }
        }

        output.push_back(count);
        output.push_back(current);

        return output;
    }

    std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed) {
        std::vector<uint8_t> output;

        for (size_t i = 0; i < compressed.size(); i += 2) {
            uint8_t count = compressed[i];
            uint8_t value = compressed[i + 1];

            for (uint8_t j = 0; j < count; ++j) {
                output.push_back(value);
            }
        }

        return output;
    }

    std::string name() const { return "RLE"; }
    std::string description() const { return "Simple Run-Length Encoding"; }
};

class simple_delta_compressor {
public:
    std::vector<uint8_t> compress(const std::vector<uint8_t>& input) {
        if (input.empty()) return {};

        std::vector<uint8_t> output;
        output.push_back(input[0]);  // Store first value

        for (size_t i = 1; i < input.size(); ++i) {
            int delta = static_cast<int>(input[i]) - static_cast<int>(input[i-1]);
            output.push_back(static_cast<uint8_t>(delta + 128));  // Offset to handle negative
        }

        return output;
    }

    std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed) {
        if (compressed.empty()) return {};

        std::vector<uint8_t> output;
        output.push_back(compressed[0]);

        for (size_t i = 1; i < compressed.size(); ++i) {
            int delta = static_cast<int>(compressed[i]) - 128;
            int value = static_cast<int>(output.back()) + delta;
            output.push_back(static_cast<uint8_t>(value & 0xFF));
        }

        return output;
    }

    std::string name() const { return "Delta"; }
    std::string description() const { return "Delta Encoding"; }
};

// ==========================================
// Benchmark Suite Builder
// ==========================================

class benchmark_suite {
private:
    benchmark_runner runner_;

public:
    benchmark_suite() {
        runner_.set_verbose(false);
        runner_.set_runs_per_test(3);
    }

    template<CompressorConcept T>
    benchmark_suite& add_algorithm(T&& compressor) {
        runner_.add_algorithm(std::forward<T>(compressor));
        return *this;
    }

    benchmark_suite& add_corpus(const test_corpus& corpus) {
        runner_.add_corpus(corpus);
        return *this;
    }

    benchmark_suite& verbose(bool v = true) {
        runner_.set_verbose(v);
        return *this;
    }

    benchmark_suite& runs_per_test(size_t n) {
        runner_.set_runs_per_test(n);
        return *this;
    }

    class results {
        const benchmark_runner* runner_;
    public:
        explicit results(const benchmark_runner* r) : runner_(r) {}

        void print_summary() const {
            runner_->print_summary();
        }

        void recommend_best() const {
            runner_->recommend_best();
        }

        void export_csv(const std::string& filename) const {
            runner_->export_csv(filename);
        }
    };

    results run() {
        runner_.run();
        return results(&runner_);
    }
};

} // namespace stepanov::compression::benchmark