#ifndef STEPANOV_COMPRESSION_KOLMOGOROV_HPP
#define STEPANOV_COMPRESSION_KOLMOGOROV_HPP

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <limits>
#include <span>
#include <concepts>

namespace stepanov::compression::kolmogorov {

// ============================================================================
// Kolmogorov Complexity Approximations
// ============================================================================
// These algorithms approximate the uncomputable Kolmogorov complexity
// through practical compression methods, providing fundamental measures
// of information content and similarity.

// Lempel-Ziv Complexity: Measures the randomness/complexity of a sequence
// by counting the minimum number of distinct patterns needed to generate it
template<typename Iterator>
requires std::forward_iterator<Iterator>
class lempel_ziv_complexity {
    using value_type = typename std::iterator_traits<Iterator>::value_type;

public:
    // Compute the LZ complexity of a sequence
    // This counts the minimum number of "innovations" needed to generate the sequence
    static size_t compute(Iterator first, Iterator last) {
        if (first == last) return 0;

        size_t complexity = 0;
        auto current = first;
        auto lookahead = std::next(first);

        while (lookahead != last) {
            // Check if current substring starting at lookahead exists in history
            auto history_end = lookahead;
            bool found_in_history = false;

            while (lookahead != last) {
                // Search for [history_end, lookahead+1) in [first, history_end)
                auto search_result = std::search(first, history_end,
                                                history_end, std::next(lookahead));

                if (search_result != history_end) {
                    // Pattern found in history, extend
                    ++lookahead;
                    found_in_history = true;
                } else {
                    // New pattern discovered
                    if (found_in_history) {
                        // We had a match but can't extend further
                        ++lookahead;
                    }
                    break;
                }
            }

            ++complexity;
            current = lookahead;
            if (lookahead != last) ++lookahead;
        }

        return complexity;
    }

    // Normalized LZ complexity (between 0 and 1)
    // Divides by theoretical maximum for random sequence
    static double normalized_complexity(Iterator first, Iterator last) {
        size_t n = std::distance(first, last);
        if (n == 0) return 0.0;

        size_t lz = compute(first, last);

        // For binary sequences, max complexity ≈ n / log₂(n)
        // For alphabet of size k, max ≈ n / log_k(n)
        std::unordered_map<value_type, bool> alphabet;
        for (auto it = first; it != last; ++it) {
            alphabet[*it] = true;
        }
        size_t k = alphabet.size();

        double max_complexity = n / std::log2(static_cast<double>(n)) * std::log2(static_cast<double>(k));
        return lz / max_complexity;
    }
};

// Normalized Compression Distance (NCD)
// Universal similarity metric based on compression
// NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
// where C(x) is the compressed size of x
template<typename Compressor>
class normalized_compression_distance {
    Compressor compressor;

public:
    // Compute NCD between two sequences
    template<typename Range1, typename Range2>
    double compute(const Range1& x, const Range2& y) {
        // Get compressed sizes
        auto cx = compress_size(x);
        auto cy = compress_size(y);

        // Concatenate x and y
        std::vector<typename Range1::value_type> xy;
        xy.reserve(x.size() + y.size());
        xy.insert(xy.end(), x.begin(), x.end());
        xy.insert(xy.end(), y.begin(), y.end());

        auto cxy = compress_size(xy);

        // NCD formula
        double min_c = std::min(cx, cy);
        double max_c = std::max(cx, cy);

        if (max_c == 0) return 0.0;

        double ncd = (cxy - min_c) / max_c;

        // Clamp to [0, 1] due to compressor imperfections
        return std::clamp(ncd, 0.0, 1.0);
    }

    // Information Distance: ID(x,y) = max(C(x|y), C(y|x))
    // Approximated as: max(C(xy) - C(y), C(yx) - C(x))
    template<typename Range1, typename Range2>
    double information_distance(const Range1& x, const Range2& y) {
        auto cx = compress_size(x);
        auto cy = compress_size(y);

        // xy concatenation
        std::vector<typename Range1::value_type> xy;
        xy.reserve(x.size() + y.size());
        xy.insert(xy.end(), x.begin(), x.end());
        xy.insert(xy.end(), y.begin(), y.end());
        auto cxy = compress_size(xy);

        // yx concatenation
        std::vector<typename Range2::value_type> yx;
        yx.reserve(x.size() + y.size());
        yx.insert(yx.end(), y.begin(), y.end());
        yx.insert(yx.end(), x.begin(), x.end());
        auto cyx = compress_size(yx);

        return std::max(cxy - cy, cyx - cx);
    }

private:
    template<typename Range>
    size_t compress_size(const Range& data) {
        auto compressed = compressor.compress(data);
        return compressed.size();
    }
};

// Minimum Description Length (MDL) Principle
// Model selection via compression - the best model is the one that
// provides the shortest description of the data
template<typename T>
class mdl_model_selector {
public:
    struct model {
        virtual ~model() = default;

        // Description length of the model itself
        virtual double model_complexity() const = 0;

        // Description length of data given the model
        virtual double data_complexity_given_model(const std::vector<T>& data) const = 0;

        // Total description length
        double total_description_length(const std::vector<T>& data) const {
            return model_complexity() + data_complexity_given_model(data);
        }
    };

    // Select best model according to MDL principle
    static std::shared_ptr<model> select_model(
        const std::vector<T>& data,
        const std::vector<std::shared_ptr<model>>& candidates
    ) {
        if (candidates.empty()) return nullptr;

        auto best = candidates[0];
        double best_mdl = best->total_description_length(data);

        for (size_t i = 1; i < candidates.size(); ++i) {
            double mdl = candidates[i]->total_description_length(data);
            if (mdl < best_mdl) {
                best_mdl = mdl;
                best = candidates[i];
            }
        }

        return best;
    }

    // Two-Part MDL: Explicitly separate model and data encoding
    class two_part_mdl {
        double log2_model_count;  // log₂ of number of possible models

    public:
        two_part_mdl(size_t num_models) : log2_model_count(std::log2(num_models)) {}

        double compute(std::shared_ptr<model> m, const std::vector<T>& data) {
            return log2_model_count + m->model_complexity() +
                   m->data_complexity_given_model(data);
        }
    };

    // Normalized Maximum Likelihood (NML) - optimal universal model
    class nml_model : public model {
        size_t parameter_count;
        double sample_size;

    public:
        nml_model(size_t params, size_t n)
            : parameter_count(params), sample_size(n) {}

        double model_complexity() const override {
            // Fisher information approximation
            return 0.5 * parameter_count * std::log2(sample_size / (2.0 * M_PI));
        }

        double data_complexity_given_model(const std::vector<T>& data) const override {
            // Compute negative log-likelihood
            // This is problem-specific and would need specialization
            return data.size() * std::log2(256.0);  // Placeholder
        }
    };
};

// Algorithmic Mutual Information
// I(x:y) = K(x) + K(y) - K(x,y)
// Measures the amount of information shared between two objects
template<typename Compressor>
class algorithmic_mutual_information {
    Compressor compressor;

public:
    template<typename Range1, typename Range2>
    double compute(const Range1& x, const Range2& y) {
        auto kx = compress_size(x);
        auto ky = compress_size(y);

        // Concatenate for joint complexity
        std::vector<typename Range1::value_type> xy;
        xy.reserve(x.size() + y.size());
        xy.insert(xy.end(), x.begin(), x.end());
        xy.insert(xy.end(), y.begin(), y.end());

        auto kxy = compress_size(xy);

        // I(x:y) = K(x) + K(y) - K(x,y)
        return static_cast<double>(kx + ky - kxy);
    }

    // Normalized mutual information: I(x:y) / max(K(x), K(y))
    template<typename Range1, typename Range2>
    double normalized_mutual_information(const Range1& x, const Range2& y) {
        double mi = compute(x, y);
        auto kx = compress_size(x);
        auto ky = compress_size(y);

        double max_k = std::max(kx, ky);
        if (max_k == 0) return 0.0;

        return mi / max_k;
    }

private:
    template<typename Range>
    size_t compress_size(const Range& data) {
        auto compressed = compressor.compress(data);
        return compressed.size();
    }
};

// Bennett's Logical Depth - Computational complexity measure
// Measures the number of computation steps in the shortest program
template<typename Iterator>
class logical_depth {
    using value_type = typename std::iterator_traits<Iterator>::value_type;

public:
    // Approximate logical depth using compression with time bounds
    static double compute(Iterator first, Iterator last, size_t max_steps = 1000000) {
        size_t n = std::distance(first, last);
        if (n == 0) return 0.0;

        // Use iterative compression to approximate depth
        // Each iteration represents a "computation step"
        std::vector<value_type> current(first, last);
        std::vector<value_type> compressed;

        size_t depth = 0;
        size_t prev_size = current.size();

        for (size_t step = 0; step < max_steps; ++step) {
            compressed = compress_iteration(current);

            if (compressed.size() >= prev_size) {
                // No more compression possible
                break;
            }

            depth += prev_size - compressed.size();
            prev_size = compressed.size();
            current = std::move(compressed);

            if (current.size() <= 1) break;
        }

        return static_cast<double>(depth) / n;
    }

private:
    static std::vector<value_type> compress_iteration(const std::vector<value_type>& data) {
        // Simple RLE as approximation
        if (data.empty()) return {};

        std::vector<value_type> result;
        value_type current = data[0];
        size_t count = 1;

        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i] == current && count < 255) {
                ++count;
            } else {
                result.push_back(current);
                if (count > 1) {
                    result.push_back(static_cast<value_type>(count));
                }
                current = data[i];
                count = 1;
            }
        }

        result.push_back(current);
        if (count > 1) {
            result.push_back(static_cast<value_type>(count));
        }

        return result;
    }
};

// Sophistication / Effective Complexity
// Measures the length of the shortest "meaningful" description
template<typename T>
class sophistication_measure {
public:
    // Compute sophistication using two-part code optimization
    static double compute(const std::vector<T>& data, size_t model_bits = 16) {
        // Find the model that minimizes: |model| + |data given model|
        // while keeping |model| ≤ model_bits

        double min_total = std::numeric_limits<double>::max();
        double best_model_size = 0;

        // Try different model complexities
        for (size_t m = 1; m <= model_bits; ++m) {
            double model_size = m;
            double data_given_model = estimate_conditional_complexity(data, m);
            double total = model_size + data_given_model;

            if (total < min_total) {
                min_total = total;
                best_model_size = model_size;
            }
        }

        return best_model_size;
    }

private:
    static double estimate_conditional_complexity(const std::vector<T>& data, size_t model_complexity) {
        // Estimate complexity of data given a model of specified complexity
        // More complex models can capture more patterns

        // Simple approximation: complex models reduce data entropy
        double base_entropy = data.size() * 8.0;  // Assume 8 bits per element
        double reduction = std::min(model_complexity * 10.0, base_entropy * 0.9);

        return base_entropy - reduction;
    }
};

} // namespace stepanov::compression::kolmogorov

#endif // STEPANOV_COMPRESSION_KOLMOGOROV_HPP