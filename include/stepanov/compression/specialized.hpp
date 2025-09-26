#ifndef STEPANOV_COMPRESSION_SPECIALIZED_HPP
#define STEPANOV_COMPRESSION_SPECIALIZED_HPP

#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <bit>
#include <span>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <queue>
#include <concepts>

namespace stepanov::compression::specialized {

// ============================================================================
// Specialized Domain Compressors
// ============================================================================
// Highly optimized compression algorithms for specific data types and domains,
// exploiting domain-specific patterns and structures.

// ============================================================================
// Time Series Compression
// ============================================================================

// Gorilla Compression - Facebook's algorithm for floating-point time series
class gorilla_compressor {
    struct bit_writer {
        std::vector<uint64_t> buffer;
        size_t bit_pos = 0;

        void write_bits(uint64_t value, size_t num_bits) {
            if (bit_pos % 64 + num_bits > 64) {
                buffer.push_back(0);
            }

            size_t word_idx = bit_pos / 64;
            size_t bit_offset = bit_pos % 64;

            buffer[word_idx] |= (value << bit_offset);
            if (bit_offset + num_bits > 64) {
                buffer[word_idx + 1] |= (value >> (64 - bit_offset));
            }

            bit_pos += num_bits;
        }

        void write_bit(bool bit) {
            write_bits(bit ? 1 : 0, 1);
        }
    };

    bit_writer writer;
    uint64_t prev_timestamp = 0;
    uint64_t prev_delta = 0;
    uint64_t prev_value_bits = 0;

public:
    // Compress timestamp (using delta-of-delta encoding)
    void compress_timestamp(uint64_t timestamp) {
        if (prev_timestamp == 0) {
            // First timestamp - write full value
            writer.write_bits(timestamp, 64);
            prev_timestamp = timestamp;
            return;
        }

        uint64_t delta = timestamp - prev_timestamp;

        if (prev_delta == 0) {
            // Second timestamp - write delta
            writer.write_bits(delta, 27);  // Gorilla uses 27 bits for first delta
            prev_delta = delta;
            prev_timestamp = timestamp;
            return;
        }

        // Delta-of-delta encoding
        int64_t delta_of_delta = static_cast<int64_t>(delta) - static_cast<int64_t>(prev_delta);

        if (delta_of_delta == 0) {
            writer.write_bit(0);  // Same delta
        } else if (delta_of_delta >= -63 && delta_of_delta <= 64) {
            writer.write_bits(0b10, 2);  // 7-bit delta-of-delta
            writer.write_bits(delta_of_delta + 63, 7);
        } else if (delta_of_delta >= -255 && delta_of_delta <= 256) {
            writer.write_bits(0b110, 3);  // 9-bit delta-of-delta
            writer.write_bits(delta_of_delta + 255, 9);
        } else if (delta_of_delta >= -2047 && delta_of_delta <= 2048) {
            writer.write_bits(0b1110, 4);  // 12-bit delta-of-delta
            writer.write_bits(delta_of_delta + 2047, 12);
        } else {
            writer.write_bits(0b1111, 4);  // Full 32-bit delta-of-delta
            writer.write_bits(delta_of_delta, 32);
        }

        prev_delta = delta;
        prev_timestamp = timestamp;
    }

    // Compress floating-point value (using XOR with previous value)
    void compress_value(double value) {
        uint64_t value_bits = std::bit_cast<uint64_t>(value);

        if (prev_value_bits == 0) {
            // First value - write full
            writer.write_bits(value_bits, 64);
            prev_value_bits = value_bits;
            return;
        }

        uint64_t xor_value = value_bits ^ prev_value_bits;

        if (xor_value == 0) {
            writer.write_bit(0);  // Same value
        } else {
            writer.write_bit(1);

            // Find leading and trailing zeros in XOR
            size_t leading_zeros = std::countl_zero(xor_value);
            size_t trailing_zeros = std::countr_zero(xor_value);

            // Check if we can use previous block info
            static size_t prev_leading = 0, prev_trailing = 0;

            if (leading_zeros >= prev_leading && trailing_zeros >= prev_trailing) {
                writer.write_bit(0);  // Use previous block
            } else {
                writer.write_bit(1);  // New block info

                // Write leading zeros (6 bits)
                writer.write_bits(leading_zeros, 6);

                // Write meaningful bits length (6 bits)
                size_t meaningful_bits = 64 - leading_zeros - trailing_zeros;
                writer.write_bits(meaningful_bits - 1, 6);

                prev_leading = leading_zeros;
                prev_trailing = trailing_zeros;
            }

            // Write meaningful bits
            size_t meaningful_bits = 64 - leading_zeros - trailing_zeros;
            writer.write_bits(xor_value >> trailing_zeros, meaningful_bits);
        }

        prev_value_bits = value_bits;
    }

    std::vector<uint64_t> finish() {
        // Pad to word boundary
        if (writer.bit_pos % 64 != 0) {
            writer.buffer.push_back(0);
        }
        return std::move(writer.buffer);
    }
};

// Sprintz Compression - For integer sensor data with predictable patterns
template<typename T>
requires std::integral<T>
class sprintz_compressor {
    std::vector<T> buffer;
    size_t buffer_size;

    // Fire-and-forget predictor
    T predict(const std::vector<T>& history, size_t pos) {
        if (pos < 2) return pos > 0 ? history[pos - 1] : 0;

        // Linear prediction: x[i] = 2*x[i-1] - x[i-2]
        return 2 * history[pos - 1] - history[pos - 2];
    }

    // Bit-packing for small values
    std::vector<uint8_t> bit_pack(const std::vector<T>& values) {
        if (values.empty()) return {};

        // Find maximum bits needed
        T max_val = *std::max_element(values.begin(), values.end(),
            [](T a, T b) { return std::abs(a) < std::abs(b); });

        // Convert to unsigned for bit_width
        using unsigned_t = std::make_unsigned_t<T>;
        unsigned_t unsigned_max = static_cast<unsigned_t>(std::abs(max_val));
        size_t bits_needed = max_val == 0 ? 1 : std::bit_width(unsigned_max) + 1;

        std::vector<uint8_t> packed;
        packed.push_back(bits_needed);  // Header

        size_t bit_pos = 0;
        uint64_t accumulator = 0;

        for (T val : values) {
            accumulator |= (static_cast<uint64_t>(val) & ((1ULL << bits_needed) - 1)) << bit_pos;
            bit_pos += bits_needed;

            while (bit_pos >= 8) {
                packed.push_back(accumulator & 0xFF);
                accumulator >>= 8;
                bit_pos -= 8;
            }
        }

        if (bit_pos > 0) {
            packed.push_back(accumulator & 0xFF);
        }

        return packed;
    }

public:
    sprintz_compressor(size_t buf_size = 8) : buffer_size(buf_size) {}

    std::vector<uint8_t> compress(const std::vector<T>& data) {
        std::vector<uint8_t> compressed;

        for (size_t i = 0; i < data.size(); i += buffer_size) {
            size_t block_size = std::min(buffer_size, data.size() - i);
            std::vector<T> residuals;

            for (size_t j = 0; j < block_size; ++j) {
                T predicted = predict(data, i + j);
                T residual = data[i + j] - predicted;
                residuals.push_back(residual);
            }

            // Bit-pack residuals
            auto packed = bit_pack(residuals);
            compressed.push_back(packed.size());  // Block header
            compressed.insert(compressed.end(), packed.begin(), packed.end());
        }

        return compressed;
    }
};

// ============================================================================
// Columnar Compression for Databases
// ============================================================================

template<typename T>
class columnar_compressor {
public:
    enum encoding_type {
        RLE,           // Run-length encoding
        DICTIONARY,    // Dictionary encoding
        BIT_PACKED,    // Bit-packing for integers
        DELTA,         // Delta encoding
        FRAME_OF_REF   // Frame of reference encoding
    };

    struct compressed_column {
        encoding_type encoding;
        std::vector<uint8_t> data;
        std::unordered_map<T, uint32_t> dictionary;  // For dictionary encoding
        T reference_value;  // For frame of reference
    };

    // Analyze column and choose best encoding
    compressed_column compress(const std::vector<T>& column) {
        // Calculate statistics
        auto stats = analyze_column(column);

        // Choose best encoding based on statistics
        if (stats.run_length_efficiency > 0.3) {
            return compress_rle(column);
        } else if (stats.cardinality < column.size() / 10) {
            return compress_dictionary(column);
        } else if constexpr (std::is_integral_v<T>) {
            if (stats.is_sorted || stats.is_monotonic) {
                return compress_delta(column);
            } else if (stats.value_range < (1ULL << 16)) {
                return compress_frame_of_reference(column);
            }
        }

        return compress_bit_packed(column);
    }

private:
    struct column_stats {
        size_t cardinality;
        double run_length_efficiency;
        bool is_sorted;
        bool is_monotonic;
        T min_value;
        T max_value;
        uint64_t value_range;
    };

    column_stats analyze_column(const std::vector<T>& column) {
        column_stats stats{};
        if (column.empty()) return stats;

        std::unordered_map<T, size_t> value_counts;
        size_t run_count = 1;
        T prev = column[0];
        bool sorted = true;
        bool monotonic = true;

        stats.min_value = column[0];
        stats.max_value = column[0];

        for (size_t i = 0; i < column.size(); ++i) {
            value_counts[column[i]]++;

            if (i > 0) {
                if (column[i] == prev) {
                    run_count++;
                }
                if (column[i] < prev) {
                    sorted = false;
                    monotonic = false;
                } else if (column[i] == prev) {
                    monotonic = false;
                }
            }

            stats.min_value = std::min(stats.min_value, column[i]);
            stats.max_value = std::max(stats.max_value, column[i]);
            prev = column[i];
        }

        stats.cardinality = value_counts.size();
        stats.run_length_efficiency = static_cast<double>(run_count) / column.size();
        stats.is_sorted = sorted;
        stats.is_monotonic = monotonic;

        if constexpr (std::is_integral_v<T>) {
            stats.value_range = static_cast<uint64_t>(stats.max_value - stats.min_value);
        }

        return stats;
    }

    compressed_column compress_rle(const std::vector<T>& column) {
        compressed_column result{RLE, {}, {}, T{}};

        if (column.empty()) return result;

        T current = column[0];
        uint32_t count = 1;

        for (size_t i = 1; i < column.size(); ++i) {
            if (column[i] == current && count < UINT32_MAX) {
                count++;
            } else {
                // Write run
                append_bytes(result.data, current);
                append_bytes(result.data, count);

                current = column[i];
                count = 1;
            }
        }

        // Write final run
        append_bytes(result.data, current);
        append_bytes(result.data, count);

        return result;
    }

    compressed_column compress_dictionary(const std::vector<T>& column) {
        compressed_column result{DICTIONARY, {}, {}, T{}};

        // Build dictionary
        std::unordered_map<T, uint32_t> dict;
        uint32_t next_id = 0;

        for (const T& val : column) {
            if (dict.find(val) == dict.end()) {
                dict[val] = next_id++;
            }
        }

        result.dictionary = dict;

        // Encode column using dictionary
        size_t bits_needed = std::bit_width(next_id - 1);

        for (const T& val : column) {
            uint32_t id = dict[val];
            append_bits(result.data, id, bits_needed);
        }

        return result;
    }

    compressed_column compress_delta(const std::vector<T>& column) {
        compressed_column result{DELTA, {}, {}, T{}};

        if (column.empty()) return result;

        result.reference_value = column[0];
        append_bytes(result.data, result.reference_value);

        T prev = column[0];
        for (size_t i = 1; i < column.size(); ++i) {
            T delta = column[i] - prev;
            append_varint(result.data, delta);
            prev = column[i];
        }

        return result;
    }

    compressed_column compress_frame_of_reference(const std::vector<T>& column) {
        compressed_column result{FRAME_OF_REF, {}, {}, T{}};

        if (column.empty()) return result;

        // Find minimum value as reference
        result.reference_value = *std::min_element(column.begin(), column.end());
        append_bytes(result.data, result.reference_value);

        // Store offsets from reference
        std::vector<T> offsets;
        for (const T& val : column) {
            offsets.push_back(val - result.reference_value);
        }

        // Bit-pack offsets
        T max_offset = *std::max_element(offsets.begin(), offsets.end());
        size_t bits_needed = std::bit_width(max_offset);

        result.data.push_back(bits_needed);
        for (const T& offset : offsets) {
            append_bits(result.data, offset, bits_needed);
        }

        return result;
    }

    compressed_column compress_bit_packed(const std::vector<T>& column) {
        compressed_column result{BIT_PACKED, {}, {}, T{}};

        if constexpr (std::is_integral_v<T>) {
            T max_val = *std::max_element(column.begin(), column.end());
            size_t bits_needed = std::bit_width(max_val);

            result.data.push_back(bits_needed);
            for (const T& val : column) {
                append_bits(result.data, val, bits_needed);
            }
        } else {
            // Fallback to byte storage
            for (const T& val : column) {
                append_bytes(result.data, val);
            }
        }

        return result;
    }

    template<typename U>
    void append_bytes(std::vector<uint8_t>& vec, const U& value) {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        vec.insert(vec.end(), bytes, bytes + sizeof(U));
    }

    void append_bits(std::vector<uint8_t>& vec, uint64_t value, size_t bits) {
        static uint64_t accumulator = 0;
        static size_t bit_pos = 0;

        accumulator |= value << bit_pos;
        bit_pos += bits;

        while (bit_pos >= 8) {
            vec.push_back(accumulator & 0xFF);
            accumulator >>= 8;
            bit_pos -= 8;
        }
    }

    void append_varint(std::vector<uint8_t>& vec, int64_t value) {
        // ZigZag encoding for signed integers
        uint64_t encoded = (value << 1) ^ (value >> 63);

        while (encoded >= 0x80) {
            vec.push_back((encoded & 0x7F) | 0x80);
            encoded >>= 7;
        }
        vec.push_back(encoded & 0x7F);
    }
};

// ============================================================================
// Graph Compression
// ============================================================================

// WebGraph-style compression for large graphs
class webgraph_compressor {
    struct compressed_adjacency_list {
        uint32_t reference_node;  // Copy list from this node (0 if none)
        std::vector<int32_t> residuals;  // Differences from reference
        uint8_t encoding_flags;  // Encoding methods used
    };

    // Locality: nearby nodes often have similar adjacency lists
    size_t find_best_reference(const std::vector<std::vector<uint32_t>>& graph,
                               size_t node, size_t window) {
        size_t best_ref = 0;
        size_t best_similarity = 0;

        size_t start = node > window ? node - window : 0;
        for (size_t ref = start; ref < node && ref < graph.size(); ++ref) {
            size_t similarity = compute_similarity(graph[node], graph[ref]);
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_ref = ref;
            }
        }

        return best_similarity > graph[node].size() / 2 ? best_ref : 0;
    }

    size_t compute_similarity(const std::vector<uint32_t>& list1,
                              const std::vector<uint32_t>& list2) {
        size_t common = 0;
        size_t i = 0, j = 0;

        while (i < list1.size() && j < list2.size()) {
            if (list1[i] == list2[j]) {
                common++;
                i++;
                j++;
            } else if (list1[i] < list2[j]) {
                i++;
            } else {
                j++;
            }
        }

        return common;
    }

public:
    std::vector<compressed_adjacency_list> compress(
        const std::vector<std::vector<uint32_t>>& graph) {

        std::vector<compressed_adjacency_list> compressed;
        compressed.reserve(graph.size());

        for (size_t node = 0; node < graph.size(); ++node) {
            compressed_adjacency_list adj_list;

            // Find reference node for copy-list compression
            adj_list.reference_node = find_best_reference(graph, node, 32);

            if (adj_list.reference_node > 0) {
                // Compute residuals from reference
                const auto& ref_list = graph[adj_list.reference_node];
                const auto& curr_list = graph[node];

                // Differential encoding
                size_t ref_idx = 0, curr_idx = 0;
                while (curr_idx < curr_list.size()) {
                    if (ref_idx < ref_list.size() && ref_list[ref_idx] == curr_list[curr_idx]) {
                        adj_list.residuals.push_back(0);  // Same as reference
                        ref_idx++;
                        curr_idx++;
                    } else if (ref_idx < ref_list.size() && ref_list[ref_idx] < curr_list[curr_idx]) {
                        adj_list.residuals.push_back(-static_cast<int32_t>(ref_list[ref_idx]));
                        ref_idx++;
                    } else {
                        adj_list.residuals.push_back(curr_list[curr_idx]);
                        curr_idx++;
                    }
                }
            } else {
                // Gap encoding for adjacency list
                if (!graph[node].empty()) {
                    adj_list.residuals.push_back(graph[node][0]);
                    for (size_t i = 1; i < graph[node].size(); ++i) {
                        adj_list.residuals.push_back(graph[node][i] - graph[node][i-1]);
                    }
                }
            }

            compressed.push_back(adj_list);
        }

        return compressed;
    }
};

// K²-tree for adjacency matrix compression
class k2tree_compressor {
    struct k2tree_node {
        std::array<bool, 4> bits;  // 2x2 subdivision
        std::array<std::unique_ptr<k2tree_node>, 4> children;

        bool is_leaf() const {
            return !children[0] && !children[1] && !children[2] && !children[3];
        }
    };

    std::unique_ptr<k2tree_node> root;
    size_t matrix_size;

    std::unique_ptr<k2tree_node> build_tree(
        const std::vector<std::vector<bool>>& matrix,
        size_t row_start, size_t row_end,
        size_t col_start, size_t col_end) {

        auto node = std::make_unique<k2tree_node>();

        if (row_end - row_start == 1 && col_end - col_start == 1) {
            // Base case: single cell
            node->bits[0] = matrix[row_start][col_start];
            return node;
        }

        size_t row_mid = (row_start + row_end) / 2;
        size_t col_mid = (col_start + col_end) / 2;

        // Check each quadrant
        bool q0 = has_ones(matrix, row_start, row_mid, col_start, col_mid);
        bool q1 = has_ones(matrix, row_start, row_mid, col_mid, col_end);
        bool q2 = has_ones(matrix, row_mid, row_end, col_start, col_mid);
        bool q3 = has_ones(matrix, row_mid, row_end, col_mid, col_end);

        node->bits[0] = q0;
        node->bits[1] = q1;
        node->bits[2] = q2;
        node->bits[3] = q3;

        // Recursively build children for non-empty quadrants
        if (q0) node->children[0] = build_tree(matrix, row_start, row_mid, col_start, col_mid);
        if (q1) node->children[1] = build_tree(matrix, row_start, row_mid, col_mid, col_end);
        if (q2) node->children[2] = build_tree(matrix, row_mid, row_end, col_start, col_mid);
        if (q3) node->children[3] = build_tree(matrix, row_mid, row_end, col_mid, col_end);

        return node;
    }

    bool has_ones(const std::vector<std::vector<bool>>& matrix,
                  size_t row_start, size_t row_end,
                  size_t col_start, size_t col_end) {
        for (size_t i = row_start; i < row_end; ++i) {
            for (size_t j = col_start; j < col_end; ++j) {
                if (matrix[i][j]) return true;
            }
        }
        return false;
    }

public:
    void compress(const std::vector<std::vector<bool>>& adjacency_matrix) {
        matrix_size = adjacency_matrix.size();

        // Pad to power of 2 if necessary
        size_t padded_size = 1;
        while (padded_size < matrix_size) padded_size *= 2;

        std::vector<std::vector<bool>> padded(padded_size, std::vector<bool>(padded_size, false));
        for (size_t i = 0; i < matrix_size; ++i) {
            for (size_t j = 0; j < matrix_size; ++j) {
                padded[i][j] = adjacency_matrix[i][j];
            }
        }

        root = build_tree(padded, 0, padded_size, 0, padded_size);
    }

    // Check if edge (i,j) exists - O(log n) time
    bool has_edge(size_t i, size_t j) const {
        if (i >= matrix_size || j >= matrix_size) return false;

        return query(root.get(), i, j, 0, matrix_size);
    }

private:
    bool query(const k2tree_node* node, size_t i, size_t j,
               size_t offset, size_t length) const {
        if (!node) return false;

        if (length == 1) {
            return node->bits[0];
        }

        size_t half = length / 2;
        size_t quadrant = 0;

        if (i >= offset + half) {
            quadrant += 2;
            i -= half;
        }
        if (j >= offset + half) {
            quadrant += 1;
            j -= half;
        }

        if (!node->bits[quadrant]) return false;

        return query(node->children[quadrant].get(), i, j, offset, half);
    }
};

} // namespace stepanov::compression::specialized

#endif // STEPANOV_COMPRESSION_SPECIALIZED_HPP