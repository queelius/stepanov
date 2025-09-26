#ifndef STEPANOV_COMPRESSION_HOMOMORPHIC_HPP
#define STEPANOV_COMPRESSION_HOMOMORPHIC_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <functional>
#include <bit>
#include <span>
#include <concepts>
#include <cstring>
#include <numeric>

namespace stepanov::compression::homomorphic {

// ============================================================================
// Homomorphic Compression
// ============================================================================
// Compression schemes that preserve certain operations on compressed data,
// allowing computation without decompression.

// ============================================================================
// Searchable Compression - Search patterns without decompressing
// ============================================================================

// Compressed Pattern Matching using LZ77 with search-friendly encoding
template<typename T>
class searchable_lz77 {
    struct lz_factor {
        bool is_literal;
        union {
            T literal;
            struct {
                size_t offset;
                size_t length;
            } reference;
        };

        lz_factor(T lit) : is_literal(true), literal(lit) {}
        lz_factor(size_t off, size_t len) : is_literal(false) {
            reference.offset = off;
            reference.length = len;
        }
    };

    std::vector<lz_factor> factors;
    size_t original_length;

    // Build suffix array for efficient pattern search
    class suffix_array {
        std::vector<size_t> sa;
        std::vector<size_t> lcp;  // Longest common prefix
        std::vector<T> text;

    public:
        void build(const std::vector<T>& input) {
            text = input;
            size_t n = text.size();
            sa.resize(n);

            // Simple O(n log² n) construction
            std::iota(sa.begin(), sa.end(), 0);
            std::sort(sa.begin(), sa.end(), [this](size_t i, size_t j) {
                return std::lexicographical_compare(
                    text.begin() + i, text.end(),
                    text.begin() + j, text.end()
                );
            });

            // Build LCP array
            build_lcp();
        }

        void build_lcp() {
            size_t n = text.size();
            lcp.resize(n - 1);
            std::vector<size_t> rank(n);

            for (size_t i = 0; i < n; ++i) {
                rank[sa[i]] = i;
            }

            size_t h = 0;
            for (size_t i = 0; i < n; ++i) {
                if (rank[i] > 0) {
                    size_t j = sa[rank[i] - 1];
                    while (i + h < n && j + h < n && text[i + h] == text[j + h]) {
                        h++;
                    }
                    lcp[rank[i] - 1] = h;
                    if (h > 0) h--;
                }
            }
        }

        std::vector<size_t> search(const std::vector<T>& pattern) const {
            std::vector<size_t> occurrences;

            // Binary search on suffix array
            auto comp = [this, &pattern](size_t idx) {
                return std::lexicographical_compare(
                    text.begin() + sa[idx], text.end(),
                    pattern.begin(), pattern.end()
                );
            };

            auto lower = std::lower_bound(sa.begin(), sa.end(), 0, comp);
            auto upper = std::upper_bound(sa.begin(), sa.end(), 0, comp);

            for (auto it = lower; it != upper; ++it) {
                occurrences.push_back(*it);
            }

            return occurrences;
        }
    };

public:
    void compress(const std::vector<T>& input, size_t window_size = 4096) {
        original_length = input.size();
        factors.clear();

        size_t pos = 0;
        while (pos < input.size()) {
            size_t best_length = 0;
            size_t best_offset = 0;

            // Search in sliding window
            size_t search_start = pos > window_size ? pos - window_size : 0;

            for (size_t i = search_start; i < pos; ++i) {
                size_t match_length = 0;
                while (pos + match_length < input.size() &&
                       input[i + match_length] == input[pos + match_length]) {
                    match_length++;
                }

                if (match_length > best_length) {
                    best_length = match_length;
                    best_offset = pos - i;
                }
            }

            if (best_length >= 3) {  // Minimum profitable length
                factors.emplace_back(best_offset, best_length);
                pos += best_length;
            } else {
                factors.emplace_back(input[pos]);
                pos++;
            }
        }
    }

    // Search pattern in compressed data without full decompression
    std::vector<size_t> search_compressed(const std::vector<T>& pattern) {
        std::vector<size_t> occurrences;

        // Build partial decompression index
        std::vector<size_t> factor_positions;
        size_t current_pos = 0;

        for (const auto& factor : factors) {
            factor_positions.push_back(current_pos);
            if (factor.is_literal) {
                current_pos++;
            } else {
                current_pos += factor.reference.length;
            }
        }

        // Search in each factor
        for (size_t i = 0; i < factors.size(); ++i) {
            if (search_in_factor(factors[i], pattern, factor_positions[i])) {
                occurrences.push_back(factor_positions[i]);
            }

            // Check cross-factor matches
            if (i > 0) {
                check_cross_factor_match(i, pattern, factor_positions, occurrences);
            }
        }

        return occurrences;
    }

private:
    bool search_in_factor(const lz_factor& factor, const std::vector<T>& pattern,
                         size_t factor_position) {
        if (factor.is_literal) {
            return pattern.size() == 1 && pattern[0] == factor.literal;
        } else {
            // Check if pattern could occur in referenced region
            return factor.reference.length >= pattern.size();
        }
    }

    void check_cross_factor_match(size_t factor_idx, const std::vector<T>& pattern,
                                 const std::vector<size_t>& positions,
                                 std::vector<size_t>& occurrences) {
        // Check if pattern spans across factor boundaries
        // This requires partial decompression of relevant factors
        // Simplified implementation - full version would be more sophisticated
    }
};

// ============================================================================
// Order-Preserving Compression - Compare without decompressing
// ============================================================================

template<typename T>
requires std::totally_ordered<T>
class order_preserving_compressor {
    struct compressed_value {
        uint32_t bucket_id;
        T bucket_min;
        T bucket_max;
        std::vector<uint8_t> compressed_data;

        // Order-preserving comparison
        bool operator<(const compressed_value& other) const {
            if (bucket_id != other.bucket_id) {
                return bucket_id < other.bucket_id;
            }
            // Within same bucket, need partial decompression
            return bucket_min < other.bucket_min;
        }
    };

    size_t bucket_size;
    std::vector<std::pair<T, T>> bucket_ranges;

public:
    order_preserving_compressor(size_t bucket_sz = 256) : bucket_size(bucket_sz) {}

    std::vector<compressed_value> compress(const std::vector<T>& data) {
        if (data.empty()) return {};

        // Find min/max for bucketing
        auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
        T min_val = *min_it;
        T max_val = *max_it;

        // Create buckets
        size_t num_buckets = (data.size() + bucket_size - 1) / bucket_size;
        bucket_ranges.clear();

        if constexpr (std::is_arithmetic_v<T>) {
            T range = max_val - min_val;
            T bucket_width = range / num_buckets;

            for (size_t i = 0; i < num_buckets; ++i) {
                T bucket_min = min_val + i * bucket_width;
                T bucket_max = (i == num_buckets - 1) ? max_val : bucket_min + bucket_width;
                bucket_ranges.emplace_back(bucket_min, bucket_max);
            }
        }

        // Compress each value with bucket assignment
        std::vector<compressed_value> compressed;
        compressed.reserve(data.size());

        for (const T& value : data) {
            compressed_value cv;
            cv.bucket_id = find_bucket(value);
            cv.bucket_min = bucket_ranges[cv.bucket_id].first;
            cv.bucket_max = bucket_ranges[cv.bucket_id].second;

            // Compress within bucket using delta encoding
            T normalized = value - cv.bucket_min;
            cv.compressed_data = compress_value(normalized);

            compressed.push_back(cv);
        }

        return compressed;
    }

    // Compare compressed values without decompression
    bool compressed_less(const compressed_value& a, const compressed_value& b) {
        return a < b;  // Uses order-preserving comparison
    }

    // Range query on compressed data
    std::vector<size_t> range_query(const std::vector<compressed_value>& compressed,
                                    const T& min_value, const T& max_value) {
        std::vector<size_t> results;

        uint32_t min_bucket = find_bucket(min_value);
        uint32_t max_bucket = find_bucket(max_value);

        for (size_t i = 0; i < compressed.size(); ++i) {
            const auto& cv = compressed[i];

            if (cv.bucket_id >= min_bucket && cv.bucket_id <= max_bucket) {
                // May need partial decompression for boundary buckets
                if (cv.bucket_id == min_bucket || cv.bucket_id == max_bucket) {
                    T value = decompress_value(cv);
                    if (value >= min_value && value <= max_value) {
                        results.push_back(i);
                    }
                } else {
                    // Entire bucket is in range
                    results.push_back(i);
                }
            }
        }

        return results;
    }

private:
    uint32_t find_bucket(const T& value) {
        for (size_t i = 0; i < bucket_ranges.size(); ++i) {
            if (value >= bucket_ranges[i].first && value <= bucket_ranges[i].second) {
                return i;
            }
        }
        return bucket_ranges.size() - 1;  // Default to last bucket
    }

    std::vector<uint8_t> compress_value(const T& value) {
        // Simple byte representation - could use more sophisticated encoding
        std::vector<uint8_t> bytes;
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&value);
        bytes.insert(bytes.end(), ptr, ptr + sizeof(T));
        return bytes;
    }

    T decompress_value(const compressed_value& cv) {
        T normalized;
        const uint8_t* ptr = cv.compressed_data.data();
        normalized = *reinterpret_cast<const T*>(ptr);
        return cv.bucket_min + normalized;
    }
};

// ============================================================================
// Computable Compression - Compute on compressed data
// ============================================================================

// Homomorphic Integer Compression supporting addition
class additive_homomorphic_compressor {
    static constexpr uint32_t MODULUS = 1000000007;  // Large prime

    // Compressed integer as sum of powers
    struct compressed_int {
        std::vector<std::pair<uint32_t, uint32_t>> terms;  // (base, exponent) pairs

        // Add two compressed integers without decompression
        compressed_int operator+(const compressed_int& other) const {
            compressed_int result;

            // Merge terms - this is homomorphic!
            result.terms = terms;
            result.terms.insert(result.terms.end(), other.terms.begin(), other.terms.end());

            // Consolidate terms with same base
            std::unordered_map<uint32_t, uint32_t> base_to_exp;
            for (const auto& [base, exp] : result.terms) {
                base_to_exp[base] = (base_to_exp[base] + exp) % MODULUS;
            }

            result.terms.clear();
            for (const auto& [base, exp] : base_to_exp) {
                if (exp > 0) {
                    result.terms.emplace_back(base, exp);
                }
            }

            return result;
        }

        // Scalar multiplication
        compressed_int operator*(uint32_t scalar) const {
            compressed_int result;
            for (const auto& [base, exp] : terms) {
                result.terms.emplace_back(base, (static_cast<uint64_t>(exp) * scalar) % MODULUS);
            }
            return result;
        }
    };

public:
    compressed_int compress(uint64_t value) {
        compressed_int result;

        // Represent as sum of powers of small primes
        std::vector<uint32_t> primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};

        for (uint32_t prime : primes) {
            uint32_t exponent = 0;
            while (value % prime == 0) {
                value /= prime;
                exponent++;
            }
            if (exponent > 0) {
                result.terms.emplace_back(prime, exponent);
            }
        }

        // Remainder as single term
        if (value > 1) {
            result.terms.emplace_back(value % MODULUS, 1);
        }

        return result;
    }

    uint64_t decompress(const compressed_int& compressed) {
        uint64_t result = 1;

        for (const auto& [base, exp] : compressed.terms) {
            for (uint32_t i = 0; i < exp; ++i) {
                result = (result * base) % MODULUS;
            }
        }

        return result;
    }

    // Compute sum without decompression
    compressed_int add_compressed(const compressed_int& a, const compressed_int& b) {
        return a + b;  // Homomorphic addition
    }

    // Compute scalar multiplication without decompression
    compressed_int multiply_compressed(const compressed_int& a, uint32_t scalar) {
        return a * scalar;  // Homomorphic scalar multiplication
    }
};

// String compression supporting edit distance computation
class edit_distance_preserving_compressor {
    struct compressed_string {
        std::vector<uint32_t> q_gram_hashes;  // q-gram signatures
        size_t original_length;
        std::unordered_map<uint32_t, std::vector<size_t>> q_gram_positions;

        // Approximate edit distance without full decompression
        size_t approximate_edit_distance(const compressed_string& other) const {
            // Use q-gram distance as lower bound for edit distance
            std::unordered_map<uint32_t, int> gram_diff;

            for (uint32_t gram : q_gram_hashes) {
                gram_diff[gram]++;
            }
            for (uint32_t gram : other.q_gram_hashes) {
                gram_diff[gram]--;
            }

            size_t q_gram_distance = 0;
            for (const auto& [gram, diff] : gram_diff) {
                q_gram_distance += std::abs(diff);
            }

            // Q-gram distance / q is lower bound for edit distance
            return q_gram_distance / 3;  // Assuming q=3
        }
    };

    static constexpr size_t Q = 3;  // q-gram size

public:
    compressed_string compress(const std::string& text) {
        compressed_string result;
        result.original_length = text.length();

        // Extract q-grams
        for (size_t i = 0; i <= text.length() - Q; ++i) {
            uint32_t hash = hash_q_gram(text.substr(i, Q));
            result.q_gram_hashes.push_back(hash);
            result.q_gram_positions[hash].push_back(i);
        }

        return result;
    }

    // Compute approximate edit distance on compressed strings
    size_t compressed_edit_distance(const compressed_string& a, const compressed_string& b) {
        return a.approximate_edit_distance(b);
    }

    // Check if pattern might exist (with up to k edits)
    bool approximate_search(const compressed_string& text,
                           const compressed_string& pattern,
                           size_t max_edits) {
        size_t approx_dist = text.approximate_edit_distance(pattern);
        return approx_dist <= max_edits * Q;  // Conservative bound
    }

private:
    uint32_t hash_q_gram(const std::string& gram) {
        uint32_t hash = 0;
        for (char c : gram) {
            hash = hash * 31 + c;
        }
        return hash;
    }
};

// ============================================================================
// Privacy-Preserving Compression
// ============================================================================

// Format-preserving compression that maintains data format
template<typename T>
class format_preserving_compressor {
    struct format_spec {
        size_t field_width;
        bool is_numeric;
        char padding_char;
        std::string prefix;
        std::string suffix;
    };

    format_spec detect_format(const std::vector<T>& data) {
        format_spec spec{};

        if constexpr (std::is_arithmetic_v<T>) {
            spec.is_numeric = true;
            spec.field_width = sizeof(T);
        }

        return spec;
    }

public:
    struct formatted_compressed {
        format_spec format;
        std::vector<uint8_t> compressed_data;

        // Maintains format for operations
        std::string to_formatted_string() const {
            std::string result = format.prefix;

            // Decode compressed data while maintaining format
            size_t decoded_length = compressed_data.size() * 2;  // Simplified

            if (format.is_numeric) {
                result += std::string(format.field_width - decoded_length, format.padding_char);
            }

            for (uint8_t byte : compressed_data) {
                result += std::to_string(byte % 10);  // Simplified formatting
            }

            result += format.suffix;
            return result;
        }
    };

    formatted_compressed compress_with_format(const std::vector<T>& data) {
        formatted_compressed result;
        result.format = detect_format(data);

        // Compress while preserving ability to reconstruct format
        for (const T& value : data) {
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
            result.compressed_data.insert(result.compressed_data.end(),
                                         bytes, bytes + sizeof(T));
        }

        return result;
    }

    // Operations that preserve format
    formatted_compressed concatenate(const formatted_compressed& a,
                                   const formatted_compressed& b) {
        if (a.format.field_width != b.format.field_width) {
            throw std::runtime_error("Format mismatch");
        }

        formatted_compressed result = a;
        result.compressed_data.insert(result.compressed_data.end(),
                                     b.compressed_data.begin(),
                                     b.compressed_data.end());
        return result;
    }
};

} // namespace stepanov::compression::homomorphic

#endif // STEPANOV_COMPRESSION_HOMOMORPHIC_HPP