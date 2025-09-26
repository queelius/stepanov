#ifndef STEPANOV_COMPRESSION_HPP
#define STEPANOV_COMPRESSION_HPP

#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <span>
#include <queue>
#include <memory>
#include <concepts>
#include <bit>
#include <ranges>
#include <cmath>
#include <chrono>
#include <numeric>

namespace stepanov::compression {

// ============================================================================
// Core Concepts - Define what compression means generically
// ============================================================================

using byte = uint8_t;
using byte_span = std::span<const byte>;
using byte_vector = std::vector<byte>;

// Compressed data with metadata
struct compressed_data {
    byte_vector data;
    size_t original_size = 0;
    uint8_t algorithm_id = 0;

    float compression_ratio() const {
        return original_size > 0 ? static_cast<float>(data.size()) / original_size : 1.0f;
    }
};

// Core compression concept
template<typename C>
concept Compressor = requires(C c, byte_span input) {
    { c.compress(input) } -> std::convertible_to<compressed_data>;
    { c.decompress(std::declval<const compressed_data&>()) } -> std::convertible_to<byte_vector>;
    { c.worst_case_expansion() } -> std::convertible_to<float>;
    { c.algorithm_id() } -> std::convertible_to<uint8_t>;
};

// Probability model for entropy coding
template<typename M>
concept ProbabilityModel = requires(M m, uint16_t symbol) {
    { m.probability(symbol) } -> std::convertible_to<float>;
    { m.cumulative_probability(symbol) } -> std::convertible_to<float>;
    { m.update(symbol) };
    { m.symbol_count() } -> std::convertible_to<size_t>;
};

// Transform that can be applied before compression
template<typename T>
concept Transform = requires(T t, byte_span input) {
    { t.forward(input) } -> std::convertible_to<byte_vector>;
    { t.inverse(input) } -> std::convertible_to<byte_vector>;
};

// ============================================================================
// Simple, Fast Transforms
// ============================================================================

// Run-length encoding - dead simple, perfect for specific data
class run_length_transform {
public:
    byte_vector forward(byte_span input) const {
        byte_vector result;
        result.reserve(input.size() + input.size() / 128);

        size_t i = 0;
        while (i < input.size()) {
            byte value = input[i];
            size_t run = 1;

            while (i + run < input.size() &&
                   input[i + run] == value &&
                   run < 255) {
                ++run;
            }

            if (run >= 3 || value >= 128) {
                // Encode as run: marker + length + value
                result.push_back(128);  // Run marker
                result.push_back(static_cast<byte>(run));
                result.push_back(value);
                i += run;
            } else {
                // Literal byte
                result.push_back(value);
                ++i;
            }
        }

        return result;
    }

    byte_vector inverse(byte_span input) const {
        byte_vector result;
        result.reserve(input.size() * 2);

        size_t i = 0;
        while (i < input.size()) {
            if (input[i] == 128 && i + 2 < input.size()) {
                // Run-length encoded sequence
                byte length = input[i + 1];
                byte value = input[i + 2];
                result.insert(result.end(), length, value);
                i += 3;
            } else {
                // Literal byte
                result.push_back(input[i]);
                ++i;
            }
        }

        return result;
    }
};

// Move-to-front transform - improves locality
class move_to_front_transform {
    mutable std::array<byte, 256> mtf_table;

public:
    byte_vector forward(byte_span input) const {
        // Initialize with identity mapping
        std::iota(mtf_table.begin(), mtf_table.end(), 0);

        byte_vector result;
        result.reserve(input.size());

        for (byte b : input) {
            // Find position of byte in table
            auto pos = std::find(mtf_table.begin(), mtf_table.end(), b);
            size_t index = std::distance(mtf_table.begin(), pos);
            result.push_back(static_cast<byte>(index));

            // Move to front
            if (index > 0) {
                std::rotate(mtf_table.begin(), pos, pos + 1);
            }
        }

        return result;
    }

    byte_vector inverse(byte_span input) const {
        // Initialize with identity mapping
        std::iota(mtf_table.begin(), mtf_table.end(), 0);

        byte_vector result;
        result.reserve(input.size());

        for (byte index : input) {
            byte value = mtf_table[index];
            result.push_back(value);

            // Move to front
            if (index > 0) {
                std::rotate(mtf_table.begin(),
                           mtf_table.begin() + index,
                           mtf_table.begin() + index + 1);
            }
        }

        return result;
    }
};

// ============================================================================
// Huffman Coding - Classic, Elegant, Educational
// ============================================================================

class huffman_compressor {
    struct node {
        uint32_t frequency;
        uint16_t symbol;
        std::unique_ptr<node> left, right;

        node(uint32_t freq, uint16_t sym)
            : frequency(freq), symbol(sym) {}

        bool is_leaf() const { return !left && !right; }
    };

    // Build Huffman tree from frequencies
    std::unique_ptr<node> build_tree(const std::array<uint32_t, 256>& frequencies) const {
        auto compare = [](const auto& a, const auto& b) {
            return a->frequency > b->frequency;
        };

        std::priority_queue<std::unique_ptr<node>,
                          std::vector<std::unique_ptr<node>>,
                          decltype(compare)> pq(compare);

        // Create leaf nodes
        for (size_t i = 0; i < 256; ++i) {
            if (frequencies[i] > 0) {
                pq.push(std::make_unique<node>(frequencies[i], i));
            }
        }

        // Build tree
        while (pq.size() > 1) {
            auto left = std::move(const_cast<std::unique_ptr<node>&>(pq.top()));
            pq.pop();
            auto right = std::move(const_cast<std::unique_ptr<node>&>(pq.top()));
            pq.pop();

            auto parent = std::make_unique<node>(
                left->frequency + right->frequency, 256);
            parent->left = std::move(left);
            parent->right = std::move(right);

            pq.push(std::move(parent));
        }

        return pq.empty() ? nullptr :
               std::move(const_cast<std::unique_ptr<node>&>(pq.top()));
    }

    // Build code table from tree
    void build_codes(const node* n, uint32_t code, uint8_t bits,
                     std::array<std::pair<uint32_t, uint8_t>, 256>& codes) const {
        if (!n) return;

        if (n->is_leaf()) {
            codes[n->symbol] = {code, bits};
        } else {
            build_codes(n->left.get(), code << 1, bits + 1, codes);
            build_codes(n->right.get(), (code << 1) | 1, bits + 1, codes);
        }
    }

public:
    compressed_data compress(byte_span input) {
        if (input.empty()) return {.data = {}, .original_size = 0, .algorithm_id = algorithm_id()};

        // Count frequencies
        std::array<uint32_t, 256> frequencies{};
        for (byte b : input) {
            frequencies[b]++;
        }

        // Build Huffman tree
        auto root = build_tree(frequencies);
        if (!root) return {.data = {}, .original_size = input.size(), .algorithm_id = algorithm_id()};

        // Build code table
        std::array<std::pair<uint32_t, uint8_t>, 256> codes{};
        build_codes(root.get(), 0, 0, codes);

        // Encode data
        compressed_data result;
        result.original_size = input.size();
        result.algorithm_id = algorithm_id();

        // Store tree structure (simplified)
        result.data.reserve(input.size() / 2);

        // Store frequency table (only non-zero)
        for (size_t i = 0; i < 256; ++i) {
            if (frequencies[i] > 0) {
                result.data.push_back(i);
                result.data.push_back(frequencies[i] & 0xFF);
                result.data.push_back((frequencies[i] >> 8) & 0xFF);
                result.data.push_back((frequencies[i] >> 16) & 0xFF);
                result.data.push_back((frequencies[i] >> 24) & 0xFF);
            }
        }
        result.data.push_back(0xFF); // End marker

        // Bit buffer for encoded data
        uint32_t bit_buffer = 0;
        uint8_t bits_in_buffer = 0;

        for (byte b : input) {
            auto [code, bits] = codes[b];

            bit_buffer = (bit_buffer << bits) | code;
            bits_in_buffer += bits;

            while (bits_in_buffer >= 8) {
                bits_in_buffer -= 8;
                result.data.push_back((bit_buffer >> bits_in_buffer) & 0xFF);
            }
        }

        // Flush remaining bits
        if (bits_in_buffer > 0) {
            result.data.push_back((bit_buffer << (8 - bits_in_buffer)) & 0xFF);
        }

        return result;
    }

    byte_vector decompress(const compressed_data& compressed) {
        // Simplified decompression (would need to rebuild tree from stored data)
        // Full implementation would parse the stored frequency table
        return {};
    }

    float worst_case_expansion() const { return 1.1f; }
    uint8_t algorithm_id() const { return 1; }
};

// ============================================================================
// LZ77/LZSS - Simple, Fast, Good Enough
// ============================================================================

class lz77_compressor {
    static constexpr size_t WINDOW_SIZE = 32768;
    static constexpr size_t LOOKAHEAD_SIZE = 258;
    static constexpr size_t MIN_MATCH = 3;

    struct match {
        uint16_t offset;
        uint8_t length;
    };

    // Find longest match in sliding window
    match find_match(byte_span window, byte_span lookahead) const {
        match best{0, 0};

        size_t max_offset = std::min(window.size(), WINDOW_SIZE);
        size_t max_length = std::min(lookahead.size(), LOOKAHEAD_SIZE);

        for (size_t offset = 1; offset <= max_offset; ++offset) {
            size_t pos = window.size() - offset;
            size_t length = 0;

            while (length < max_length &&
                   window[pos + length] == lookahead[length]) {
                ++length;
            }

            if (length >= MIN_MATCH && length > best.length) {
                best = {static_cast<uint16_t>(offset), static_cast<uint8_t>(length)};
            }
        }

        return best;
    }

public:
    compressed_data compress(byte_span input) {
        compressed_data result;
        result.original_size = input.size();
        result.algorithm_id = algorithm_id();
        result.data.reserve(input.size() / 2);

        size_t pos = 0;

        while (pos < input.size()) {
            // Get window and lookahead
            size_t window_start = pos > WINDOW_SIZE ? pos - WINDOW_SIZE : 0;
            auto window = input.subspan(window_start, pos - window_start);
            auto lookahead = input.subspan(pos,
                std::min(LOOKAHEAD_SIZE, input.size() - pos));

            auto m = find_match(window, lookahead);

            if (m.length >= MIN_MATCH) {
                // Output match: flag + offset + length
                result.data.push_back(0x80);  // Match flag
                result.data.push_back(m.offset & 0xFF);
                result.data.push_back(m.offset >> 8);
                result.data.push_back(m.length);
                pos += m.length;
            } else {
                // Output literal
                result.data.push_back(input[pos]);
                ++pos;
            }
        }

        return result;
    }

    byte_vector decompress(const compressed_data& compressed) {
        byte_vector result;
        result.reserve(compressed.original_size);

        size_t pos = 0;
        while (pos < compressed.data.size()) {
            if (compressed.data[pos] == 0x80 && pos + 3 < compressed.data.size()) {
                // Match
                uint16_t offset = compressed.data[pos + 1] |
                                 (compressed.data[pos + 2] << 8);
                uint8_t length = compressed.data[pos + 3];

                // Copy from back-reference
                // Handle overlapping copies correctly
                size_t src_pos = result.size() - offset;
                for (size_t i = 0; i < length; ++i) {
                    result.push_back(result[src_pos]);
                    // For overlapping copies, we need to re-read from the newly written data
                    if (offset <= i) {
                        src_pos = result.size() - offset;
                    } else {
                        src_pos++;
                    }
                }

                pos += 4;
            } else {
                // Literal
                result.push_back(compressed.data[pos]);
                ++pos;
            }
        }

        return result;
    }

    float worst_case_expansion() const { return 1.01f; }
    uint8_t algorithm_id() const { return 2; }
};

// ============================================================================
// Fast LZ4-Style Compression - Speed > Ratio
// ============================================================================

class fast_lz_compressor {
    static constexpr size_t MIN_MATCH = 4;
    static constexpr size_t HASH_BITS = 14;
    static constexpr size_t HASH_SIZE = 1 << HASH_BITS;

    // Simple hash for fast matching
    uint32_t hash(const byte* data) const {
        return (*(uint32_t*)data * 2654435761U) >> (32 - HASH_BITS);
    }

public:
    compressed_data compress(byte_span input) {
        if (input.size() < MIN_MATCH) {
            // Too small to compress
            compressed_data result;
            result.original_size = input.size();
            result.algorithm_id = algorithm_id();
            result.data.assign(input.begin(), input.end());
            return result;
        }

        compressed_data result;
        result.original_size = input.size();
        result.algorithm_id = algorithm_id();
        result.data.reserve(input.size() / 4);

        std::array<uint32_t, HASH_SIZE> hash_table{};

        const byte* ip = input.data();
        const byte* ip_end = ip + input.size() - MIN_MATCH;
        const byte* ip_limit = ip + input.size() - 12;

        while (ip < ip_limit) {
            // Find match
            uint32_t h = hash(ip);
            uint32_t offset = ip - input.data();
            uint32_t match_pos = hash_table[h];
            hash_table[h] = offset;

            if (match_pos && offset - match_pos < 65536) {
                const byte* match = input.data() + match_pos;

                if (*(uint32_t*)ip == *(uint32_t*)match) {
                    // Found match - extend it
                    size_t match_len = MIN_MATCH;
                    while (ip + match_len < ip_end &&
                           ip[match_len] == match[match_len]) {
                        ++match_len;
                    }

                    // Output match
                    uint16_t distance = offset - match_pos;
                    result.data.push_back(0x80 | (match_len - MIN_MATCH));
                    result.data.push_back(distance & 0xFF);
                    result.data.push_back(distance >> 8);

                    ip += match_len;
                    continue;
                }
            }

            // No match - output literal
            result.data.push_back(*ip++);
        }

        // Copy remaining literals
        while (ip < input.data() + input.size()) {
            result.data.push_back(*ip++);
        }

        return result;
    }

    byte_vector decompress(const compressed_data& compressed) {
        byte_vector result;
        result.reserve(compressed.original_size);

        const byte* ip = compressed.data.data();
        const byte* ip_end = ip + compressed.data.size();

        while (ip < ip_end) {
            byte token = *ip++;

            if (token & 0x80) {
                // Match
                if (ip + 1 >= ip_end) break;

                size_t match_len = (token & 0x7F) + MIN_MATCH;
                uint16_t distance = *ip++;
                distance |= (*ip++) << 8;

                // Copy from back-reference
                const byte* copy_src = result.data() + result.size() - distance;
                for (size_t i = 0; i < match_len; ++i) {
                    result.push_back(copy_src[i]);
                }
            } else {
                // Literal
                result.push_back(token);
            }
        }

        return result;
    }

    float worst_case_expansion() const { return 1.06f; }
    uint8_t algorithm_id() const { return 3; }
};

// ============================================================================
// Generic Arithmetic Coder - Works with Any Probability Model
// ============================================================================

template<ProbabilityModel Model>
class arithmetic_coder {
    Model model;
    static constexpr uint32_t TOP = 0xFFFFFFFF;
    static constexpr uint32_t FIRST_QUARTER = TOP / 4 + 1;
    static constexpr uint32_t HALF = 2 * FIRST_QUARTER;
    static constexpr uint32_t THIRD_QUARTER = 3 * FIRST_QUARTER;

public:
    explicit arithmetic_coder(Model m = Model{}) : model(std::move(m)) {}

    compressed_data compress(byte_span input) {
        compressed_data result;
        result.original_size = input.size();
        result.algorithm_id = 10;  // Generic arithmetic coding

        uint32_t low = 0;
        uint32_t high = TOP;
        uint32_t underflow_bits = 0;

        for (byte symbol : input) {
            uint32_t range = high - low + 1;

            // Get probabilities from model
            float cum_low = model.cumulative_probability(symbol);
            float cum_high = model.cumulative_probability(symbol + 1);

            // Update range
            high = low + (range * cum_high) - 1;
            low = low + (range * cum_low);

            // Renormalization
            while (true) {
                if (high < HALF) {
                    // Output 0 and underflow bits
                    output_bit(result.data, 0);
                    while (underflow_bits > 0) {
                        output_bit(result.data, 1);
                        --underflow_bits;
                    }
                } else if (low >= HALF) {
                    // Output 1 and underflow bits
                    output_bit(result.data, 1);
                    while (underflow_bits > 0) {
                        output_bit(result.data, 0);
                        --underflow_bits;
                    }
                    low -= HALF;
                    high -= HALF;
                } else if (low >= FIRST_QUARTER && high < THIRD_QUARTER) {
                    // Underflow
                    ++underflow_bits;
                    low -= FIRST_QUARTER;
                    high -= FIRST_QUARTER;
                } else {
                    break;
                }

                low <<= 1;
                high = (high << 1) | 1;
            }

            model.update(symbol);
        }

        // Flush remaining bits
        if (low < FIRST_QUARTER) {
            output_bit(result.data, 0);
            while (underflow_bits-- > 0) output_bit(result.data, 1);
        } else {
            output_bit(result.data, 1);
            while (underflow_bits-- > 0) output_bit(result.data, 0);
        }

        return result;
    }

    byte_vector decompress(const compressed_data& compressed) {
        // Decompression would be similar but reading bits instead of writing
        return {};
    }

    float worst_case_expansion() const { return 1.01f; }
    uint8_t algorithm_id() const { return 10; }

private:
    void output_bit(byte_vector& output, int bit) {
        static uint8_t bit_buffer = 0;
        static uint8_t bits_in_buffer = 0;

        bit_buffer = (bit_buffer << 1) | bit;
        if (++bits_in_buffer == 8) {
            output.push_back(bit_buffer);
            bits_in_buffer = 0;
            bit_buffer = 0;
        }
    }
};

// ============================================================================
// Simple Probability Models
// ============================================================================

// Fixed uniform model - baseline
class uniform_model {
public:
    float probability(uint16_t) const { return 1.0f / 256.0f; }
    float cumulative_probability(uint16_t symbol) const {
        return symbol / 256.0f;
    }
    void update(uint16_t) {}  // No adaptation
    size_t symbol_count() const { return 256; }
};

// Simple adaptive model
class adaptive_model {
    std::array<uint32_t, 257> counts{};  // 257 for cumulative[256]
    uint32_t total = 256;

public:
    adaptive_model() {
        // Initialize with uniform distribution
        for (size_t i = 0; i < 256; ++i) {
            counts[i] = 1;
        }
        update_cumulative();
    }

    float probability(uint16_t symbol) const {
        return static_cast<float>(counts[symbol + 1] - counts[symbol]) / total;
    }

    float cumulative_probability(uint16_t symbol) const {
        return static_cast<float>(counts[symbol]) / total;
    }

    void update(uint16_t symbol) {
        // Increment count
        for (size_t i = symbol + 1; i <= 256; ++i) {
            counts[i]++;
        }
        total++;

        // Rescale if needed
        if (total > 65536) {
            rescale();
        }
    }

    size_t symbol_count() const { return 256; }

private:
    void update_cumulative() {
        uint32_t cumulative = 0;
        for (size_t i = 0; i <= 256; ++i) {
            uint32_t count = (i > 0) ? counts[i] - counts[i-1] : 0;
            counts[i] = cumulative;
            cumulative += count;
        }
        total = cumulative;
    }

    void rescale() {
        // Halve all counts
        uint32_t cumulative = 0;
        for (size_t i = 0; i <= 256; ++i) {
            uint32_t count = (i > 0) ? counts[i] - counts[i-1] : 0;
            count = (count + 1) / 2;  // Keep at least 1
            counts[i] = cumulative;
            cumulative += count;
        }
        total = cumulative;
    }
};

// ============================================================================
// Composable Pipeline - The Power of Generic Programming
// ============================================================================

template<typename... Components>
class compression_pipeline {
    std::tuple<Components...> components;

    template<size_t I = 0>
    byte_vector apply_forward(byte_span input) const {
        if constexpr (I == sizeof...(Components)) {
            return byte_vector(input.begin(), input.end());
        } else {
            auto& component = std::get<I>(components);
            if constexpr (requires { component.forward(input); }) {
                auto result = component.forward(input);
                return apply_forward<I + 1>(byte_span(result));
            } else if constexpr (requires { component.compress(input); }) {
                auto compressed = component.compress(input);
                return compressed.data;
            } else {
                return apply_forward<I + 1>(input);
            }
        }
    }

public:
    explicit compression_pipeline(Components... comps)
        : components(std::move(comps)...) {}

    compressed_data compress(byte_span input) {
        compressed_data result;
        result.original_size = input.size();
        result.data = apply_forward(input);
        result.algorithm_id = 100;  // Pipeline ID
        return result;
    }

    byte_vector decompress(const compressed_data& compressed) {
        // Would apply inverse transforms in reverse order
        return {};
    }

    float worst_case_expansion() const { return 1.2f; }
};

// Factory function for pipeline creation
template<typename... Components>
auto make_pipeline(Components... components) {
    return compression_pipeline<Components...>(std::move(components)...);
}

// ============================================================================
// Compression Utilities
// ============================================================================

// Check if data appears to be already compressed
bool is_compressed(byte_span data) {
    if (data.size() < 256) return false;

    // Check entropy - compressed data has high entropy
    std::array<size_t, 256> counts{};
    for (byte b : data) {
        counts[b]++;
    }

    // Calculate simple entropy metric
    double entropy = 0.0;
    for (size_t count : counts) {
        if (count > 0) {
            double p = static_cast<double>(count) / data.size();
            entropy -= p * std::log2(p);
        }
    }

    // High entropy (> 7.5 bits) suggests compression
    return entropy > 7.5;
}

// Suggest best compressor for data characteristics
enum class data_type {
    text,
    binary,
    sparse,
    repetitive,
    unknown
};

data_type analyze_data(byte_span data) {
    if (data.size() < 100) return data_type::unknown;

    std::array<size_t, 256> counts{};
    size_t runs = 0;
    size_t zeros = 0;

    byte prev = data[0];
    for (byte b : data) {
        counts[b]++;
        if (b == 0) zeros++;
        if (b == prev) runs++;
        prev = b;
    }

    // Check for text (ASCII printable)
    size_t text_chars = 0;
    for (size_t i = 32; i < 127; ++i) {
        text_chars += counts[i];
    }

    if (text_chars > data.size() * 0.9) return data_type::text;
    if (zeros > data.size() * 0.5) return data_type::sparse;
    if (runs > data.size() * 0.7) return data_type::repetitive;

    return data_type::binary;
}

// Simple compression advisor
template<typename Compressor>
std::unique_ptr<Compressor> suggest_compressor(byte_span data) {
    auto type = analyze_data(data);

    switch (type) {
    case data_type::text:
        // Huffman works well for text
        return std::make_unique<huffman_compressor>();
    case data_type::sparse:
    case data_type::repetitive:
        // RLE for repetitive data
        return std::make_unique<fast_lz_compressor>();
    default:
        // LZ77 as general purpose
        return std::make_unique<lz77_compressor>();
    }
}

// ============================================================================
// Benchmarking Utilities
// ============================================================================

template<Compressor C>
struct benchmark_result {
    float compression_ratio;
    double compress_time_ms;
    double decompress_time_ms;
    size_t compressed_size;
    bool successful;
};

template<Compressor C>
benchmark_result<C> benchmark(C& compressor, byte_span data) {
    benchmark_result<C> result{};

    auto start = std::chrono::high_resolution_clock::now();
    auto compressed = compressor.compress(data);
    auto compress_end = std::chrono::high_resolution_clock::now();

    auto decompressed = compressor.decompress(compressed);
    auto decompress_end = std::chrono::high_resolution_clock::now();

    result.compression_ratio = compressed.compression_ratio();
    result.compressed_size = compressed.data.size();
    result.compress_time_ms = std::chrono::duration<double, std::milli>(
        compress_end - start).count();
    result.decompress_time_ms = std::chrono::duration<double, std::milli>(
        decompress_end - compress_end).count();

    // Verify round-trip
    result.successful = (decompressed.size() == data.size()) &&
                       std::equal(data.begin(), data.end(), decompressed.begin());

    return result;
}

} // namespace stepanov::compression

#endif // STEPANOV_COMPRESSION_HPP