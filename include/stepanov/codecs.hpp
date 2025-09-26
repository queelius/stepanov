#pragma once

/**
 * stepanov::codecs - Bit-level encoding/decoding with algebraic composition
 *
 * Integrates concepts from PPC (Packed Codecs) library following Stepanov principles:
 * - Generic programming with zero-cost abstractions
 * - Algebraic composition of codecs
 * - Type erasure for runtime polymorphism when needed
 * - Efficient bit-level operations
 */

#include <cstdint>
#include <vector>
#include <concepts>
#include <bit>
#include <span>
#include <array>
#include <limits>
#include <string>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "type_erasure.hpp"
#include "concepts.hpp"

namespace stepanov::codecs {

// =============================================================================
// Bit-level I/O abstractions
// =============================================================================

class bit_writer {
private:
    std::vector<std::uint8_t>& buffer_;
    std::uint8_t current_byte_ = 0;
    int bit_position_ = 0;

public:
    explicit bit_writer(std::vector<std::uint8_t>& buffer)
        : buffer_(buffer) {}

    void write_bit(bool bit) {
        if (bit) {
            current_byte_ |= (1u << bit_position_);
        }
        if (++bit_position_ == 8) {
            flush();
        }
    }

    void write_bits(std::uint64_t value, int count) {
        for (int i = 0; i < count; ++i) {
            write_bit((value >> i) & 1u);
        }
    }

    void flush() {
        if (bit_position_ > 0) {
            buffer_.push_back(current_byte_);
            current_byte_ = 0;
            bit_position_ = 0;
        }
    }

    void align_byte() { flush(); }

    std::size_t position() const {
        return buffer_.size() * 8 + bit_position_;
    }

    ~bit_writer() { flush(); }
};

class bit_reader {
private:
    std::span<const std::uint8_t> data_;
    std::size_t byte_position_ = 0;
    std::uint8_t current_byte_ = 0;
    int bit_position_ = 8;  // Start at 8 to force initial read

public:
    explicit bit_reader(std::span<const std::uint8_t> data)
        : data_(data) {}

    bool read_bit() {
        if (bit_position_ == 8) {
            if (byte_position_ >= data_.size()) {
                throw std::runtime_error("bit_reader: end of stream");
            }
            current_byte_ = data_[byte_position_++];
            bit_position_ = 0;
        }
        return (current_byte_ >> bit_position_++) & 1u;
    }

    std::uint64_t read_bits(int count) {
        std::uint64_t value = 0;
        for (int i = 0; i < count; ++i) {
            value |= static_cast<std::uint64_t>(read_bit()) << i;
        }
        return value;
    }

    void align_byte() {
        if (bit_position_ != 0) {
            bit_position_ = 8;
        }
    }

    std::size_t position() const {
        return (byte_position_ - (bit_position_ == 8 ? 0 : 1)) * 8 + bit_position_;
    }

    bool eof() const {
        return byte_position_ >= data_.size() && bit_position_ == 8;
    }
};

// =============================================================================
// Codec concepts
// =============================================================================

template<typename C, typename T>
concept encoder = requires(const C& codec, const T& value, bit_writer& writer) {
    { codec.encode(value, writer) } -> std::same_as<void>;
};

template<typename C, typename T>
concept decoder = requires(const C& codec, bit_reader& reader) {
    { codec.decode(reader) } -> std::same_as<T>;
};

template<typename C, typename T>
concept codec = encoder<C, T> && decoder<C, T>;

// =============================================================================
// Fundamental codecs
// =============================================================================

/**
 * Unary codec - simplest variable-length encoding
 * Encodes n as n zeros followed by a one
 */
struct unary_codec {
    template<std::unsigned_integral T>
    void encode(T value, bit_writer& writer) const {
        for (T i = 0; i < value; ++i) {
            writer.write_bit(false);
        }
        writer.write_bit(true);
    }

    template<std::unsigned_integral T>
    T decode(bit_reader& reader) const {
        T count = 0;
        while (!reader.read_bit()) {
            ++count;
        }
        return count;
    }

    std::string name() const { return "unary"; }
};

/**
 * Elias gamma codec - universal code for positive integers
 */
struct elias_gamma_codec {
    template<std::unsigned_integral T>
    void encode(T value, bit_writer& writer) const {
        if (value == 0) {
            throw std::invalid_argument("elias_gamma: cannot encode zero");
        }

        // Find the position of the most significant bit
        int len = std::bit_width(value);

        // Write len-1 zeros
        for (int i = 0; i < len - 1; ++i) {
            writer.write_bit(false);
        }

        // Write the value in binary (MSB to LSB)
        for (int i = len - 1; i >= 0; --i) {
            writer.write_bit((value >> i) & 1u);
        }
    }

    template<std::unsigned_integral T>
    T decode(bit_reader& reader) const {
        // Count leading zeros
        int zeros = 0;
        while (!reader.read_bit()) {
            ++zeros;
        }

        // Read zeros more bits
        T value = 1;
        for (int i = 0; i < zeros; ++i) {
            value = (value << 1) | reader.read_bit();
        }

        return value;
    }

    std::string name() const { return "elias_gamma"; }
};

/**
 * Elias delta codec - more efficient for larger numbers
 */
struct elias_delta_codec {
    template<std::unsigned_integral T>
    void encode(T value, bit_writer& writer) const {
        if (value == 0) {
            throw std::invalid_argument("elias_delta: cannot encode zero");
        }

        int len = std::bit_width(value);

        // Encode len using gamma
        elias_gamma_codec gamma;
        gamma.encode(static_cast<T>(len), writer);

        // Write remaining bits of value (skip MSB which is always 1)
        for (int i = len - 2; i >= 0; --i) {
            writer.write_bit((value >> i) & 1u);
        }
    }

    template<std::unsigned_integral T>
    T decode(bit_reader& reader) const {
        elias_gamma_codec gamma;
        auto len = gamma.decode<std::size_t>(reader);

        // Read remaining bits
        T value = 1;
        for (std::size_t i = 0; i < len - 1; ++i) {
            value = (value << 1) | reader.read_bit();
        }

        return value;
    }

    std::string name() const { return "elias_delta"; }
};

/**
 * Fixed-width codec - encodes values using a fixed number of bits
 */
template<int BitWidth>
struct fixed_codec {
    static_assert(BitWidth > 0 && BitWidth <= 64);

    template<std::unsigned_integral T>
    void encode(T value, bit_writer& writer) const {
        if (value >= (1ull << BitWidth)) {
            throw std::overflow_error("fixed_codec: value too large");
        }
        writer.write_bits(value, BitWidth);
    }

    template<std::unsigned_integral T>
    T decode(bit_reader& reader) const {
        return static_cast<T>(reader.read_bits(BitWidth));
    }

    std::string name() const {
        return "fixed_" + std::to_string(BitWidth);
    }
};

/**
 * Fibonacci codec - based on Fibonacci sequence
 * More complex but has interesting mathematical properties
 */
struct fibonacci_codec {
private:
    static constexpr std::size_t max_fibs = 92;

    template<typename T>
    static constexpr std::array<T, max_fibs> compute_fibonacci() {
        std::array<T, max_fibs> fibs{};
        fibs[0] = 1;
        fibs[1] = 2;
        for (std::size_t i = 2; i < max_fibs; ++i) {
            if (fibs[i-1] > std::numeric_limits<T>::max() / 2) break;
            fibs[i] = fibs[i-1] + fibs[i-2];
        }
        return fibs;
    }

public:
    template<std::unsigned_integral T>
    void encode(T value, bit_writer& writer) const {
        if (value == 0) {
            writer.write_bit(true);
            writer.write_bit(true);
            return;
        }

        auto fibs = compute_fibonacci<T>();
        ++value; // Map to positive integers

        // Find Fibonacci representation
        std::array<bool, max_fibs> bits{};
        int highest = 0;

        for (int i = max_fibs - 1; i >= 0; --i) {
            if (fibs[i] <= value && fibs[i] > 0) {
                bits[i] = true;
                value -= fibs[i];
                highest = std::max(highest, i);
            }
        }

        // Write the representation (LSB to MSB)
        for (int i = 0; i <= highest; ++i) {
            writer.write_bit(bits[i]);
        }
        writer.write_bit(true); // Terminator
    }

    template<std::unsigned_integral T>
    T decode(bit_reader& reader) const {
        auto fibs = compute_fibonacci<T>();
        T value = 0;
        std::size_t i = 0;
        bool prev_bit = false;

        while (true) {
            bool bit = reader.read_bit();
            if (bit && prev_bit) {
                // Two consecutive 1s mark the end
                break;
            }
            if (bit && i < max_fibs) {
                value += fibs[i];
            }
            prev_bit = bit;
            ++i;
        }

        return value - 1; // Map back from positive integers
    }

    std::string name() const { return "fibonacci"; }
};

// =============================================================================
// Codec composition
// =============================================================================

/**
 * Sequential codec - encodes multiple values in sequence
 */
template<typename... Codecs>
struct sequential_codec {
    std::tuple<Codecs...> codecs;

    sequential_codec(Codecs... cs) : codecs(cs...) {}

    template<typename... Values>
    void encode(const std::tuple<Values...>& values, bit_writer& writer) const {
        encode_impl(values, writer, std::index_sequence_for<Codecs...>{});
    }

    template<typename... Values>
    std::tuple<Values...> decode(bit_reader& reader) const {
        return decode_impl<Values...>(reader, std::index_sequence_for<Codecs...>{});
    }

private:
    template<typename... Values, std::size_t... Is>
    void encode_impl(const std::tuple<Values...>& values, bit_writer& writer,
                    std::index_sequence<Is...>) const {
        ((std::get<Is>(codecs).encode(std::get<Is>(values), writer)), ...);
    }

    template<typename... Values, std::size_t... Is>
    std::tuple<Values...> decode_impl(bit_reader& reader,
                                      std::index_sequence<Is...>) const {
        return std::make_tuple(
            std::get<Is>(codecs).template decode<
                std::tuple_element_t<Is, std::tuple<Values...>>>(reader)...
        );
    }
};

/**
 * Delta codec - encodes differences between consecutive values
 */
template<typename BaseCodec>
struct delta_codec {
    BaseCodec base_codec;

    delta_codec(BaseCodec codec = {}) : base_codec(std::move(codec)) {}

    template<std::unsigned_integral T>
    void encode(std::span<const T> values, bit_writer& writer) const {
        if (values.empty()) return;

        // Encode first value directly
        base_codec.encode(values[0], writer);

        // Encode differences
        T prev = values[0];
        for (std::size_t i = 1; i < values.size(); ++i) {
            T diff = values[i] - prev;
            base_codec.encode(diff, writer);
            prev = values[i];
        }
    }

    template<std::unsigned_integral T>
    std::vector<T> decode(bit_reader& reader, std::size_t count) const {
        if (count == 0) return {};

        std::vector<T> values;
        values.reserve(count);

        // Decode first value
        T value = base_codec.template decode<T>(reader);
        values.push_back(value);

        // Decode differences and reconstruct
        for (std::size_t i = 1; i < count; ++i) {
            T diff = base_codec.template decode<T>(reader);
            value += diff;
            values.push_back(value);
        }

        return values;
    }

    std::string name() const {
        return "delta<" + base_codec.name() + ">";
    }
};

/**
 * Run-length codec - encodes runs of identical values
 */
template<typename ValueCodec, typename CountCodec>
struct run_length_codec {
    ValueCodec value_codec;
    CountCodec count_codec;

    run_length_codec(ValueCodec v = {}, CountCodec c = {})
        : value_codec(std::move(v)), count_codec(std::move(c)) {}

    template<typename T>
    void encode(std::span<const T> values, bit_writer& writer) const {
        if (values.empty()) return;

        std::size_t i = 0;
        while (i < values.size()) {
            T value = values[i];
            std::size_t count = 1;

            // Count consecutive identical values
            while (i + count < values.size() && values[i + count] == value) {
                ++count;
            }

            // Encode value and count
            value_codec.encode(value, writer);
            count_codec.encode(count, writer);

            i += count;
        }
    }

    template<typename T>
    std::vector<T> decode(bit_reader& reader, std::size_t total_count) const {
        std::vector<T> values;
        values.reserve(total_count);

        while (values.size() < total_count) {
            T value = value_codec.template decode<T>(reader);
            auto count = count_codec.template decode<std::size_t>(reader);

            for (std::size_t i = 0; i < count; ++i) {
                values.push_back(value);
            }
        }

        return values;
    }

    std::string name() const {
        return "run_length<" + value_codec.name() + "," + count_codec.name() + ">";
    }
};

// =============================================================================
// Adaptive codec selection
// =============================================================================

/**
 * Adaptive codec - selects best codec based on data characteristics
 */
template<typename T>
class adaptive_codec {
private:
    struct statistics {
        T min_value = std::numeric_limits<T>::max();
        T max_value = std::numeric_limits<T>::min();
        double mean = 0.0;
        double variance = 0.0;
        std::size_t run_count = 0;
        std::size_t unique_count = 0;
    };

    statistics analyze(std::span<const T> values) const {
        statistics stats;
        if (values.empty()) return stats;

        // Basic statistics
        double sum = 0.0;
        std::unordered_map<T, std::size_t> freq;

        for (const auto& v : values) {
            stats.min_value = std::min(stats.min_value, v);
            stats.max_value = std::max(stats.max_value, v);
            sum += static_cast<double>(v);
            ++freq[v];
        }

        stats.mean = sum / values.size();
        stats.unique_count = freq.size();

        // Variance
        double var_sum = 0.0;
        for (const auto& v : values) {
            double diff = static_cast<double>(v) - stats.mean;
            var_sum += diff * diff;
        }
        stats.variance = var_sum / values.size();

        // Run count
        for (std::size_t i = 1; i < values.size(); ++i) {
            if (values[i] != values[i-1]) {
                ++stats.run_count;
            }
        }

        return stats;
    }

    enum class codec_type : std::uint8_t {
        unary,
        elias_gamma,
        elias_delta,
        fixed_8,
        fixed_16,
        fixed_32,
        fibonacci
    };

    codec_type select_codec(const statistics& stats, std::size_t count) const {
        // Simple heuristic selection
        T range = stats.max_value - stats.min_value;

        if (stats.max_value < 8) {
            return codec_type::unary;
        } else if (stats.max_value < 256) {
            return codec_type::fixed_8;
        } else if (stats.max_value < 65536) {
            if (stats.variance > 1000) {
                return codec_type::elias_gamma;
            } else {
                return codec_type::fixed_16;
            }
        } else if (stats.unique_count < count / 4) {
            return codec_type::elias_delta;
        } else if (range < 4294967296ull) {
            return codec_type::fixed_32;
        } else {
            return codec_type::fibonacci;
        }
    }

public:
    void encode(std::span<const T> values, bit_writer& writer) const {
        auto stats = analyze(values);
        auto selected = select_codec(stats, values.size());

        // Write codec type
        writer.write_bits(static_cast<std::uint8_t>(selected), 3);

        // Encode with selected codec
        switch (selected) {
            case codec_type::unary:
                for (const auto& v : values) {
                    unary_codec{}.encode(v, writer);
                }
                break;
            case codec_type::elias_gamma:
                for (const auto& v : values) {
                    elias_gamma_codec{}.encode(v, writer);
                }
                break;
            case codec_type::elias_delta:
                for (const auto& v : values) {
                    elias_delta_codec{}.encode(v, writer);
                }
                break;
            case codec_type::fixed_8:
                for (const auto& v : values) {
                    fixed_codec<8>{}.encode(v, writer);
                }
                break;
            case codec_type::fixed_16:
                for (const auto& v : values) {
                    fixed_codec<16>{}.encode(v, writer);
                }
                break;
            case codec_type::fixed_32:
                for (const auto& v : values) {
                    fixed_codec<32>{}.encode(v, writer);
                }
                break;
            case codec_type::fibonacci:
                for (const auto& v : values) {
                    fibonacci_codec{}.encode(v, writer);
                }
                break;
        }
    }

    std::vector<T> decode(bit_reader& reader, std::size_t count) const {
        auto selected = static_cast<codec_type>(reader.read_bits(3));
        std::vector<T> values;
        values.reserve(count);

        switch (selected) {
            case codec_type::unary:
                for (std::size_t i = 0; i < count; ++i) {
                    values.push_back(unary_codec{}.decode<T>(reader));
                }
                break;
            case codec_type::elias_gamma:
                for (std::size_t i = 0; i < count; ++i) {
                    values.push_back(elias_gamma_codec{}.decode<T>(reader));
                }
                break;
            case codec_type::elias_delta:
                for (std::size_t i = 0; i < count; ++i) {
                    values.push_back(elias_delta_codec{}.decode<T>(reader));
                }
                break;
            case codec_type::fixed_8:
                for (std::size_t i = 0; i < count; ++i) {
                    values.push_back(fixed_codec<8>{}.decode<T>(reader));
                }
                break;
            case codec_type::fixed_16:
                for (std::size_t i = 0; i < count; ++i) {
                    values.push_back(fixed_codec<16>{}.decode<T>(reader));
                }
                break;
            case codec_type::fixed_32:
                for (std::size_t i = 0; i < count; ++i) {
                    values.push_back(fixed_codec<32>{}.decode<T>(reader));
                }
                break;
            case codec_type::fibonacci:
                for (std::size_t i = 0; i < count; ++i) {
                    values.push_back(fibonacci_codec{}.decode<T>(reader));
                }
                break;
        }

        return values;
    }

    std::string name() const { return "adaptive"; }
};

// =============================================================================
// Helper functions for convenient usage
// =============================================================================

template<typename Codec, typename T>
std::vector<std::uint8_t> encode(const Codec& codec, const T& value) {
    std::vector<std::uint8_t> buffer;
    bit_writer writer(buffer);
    codec.encode(value, writer);
    return buffer;
}

template<typename Codec, typename T>
T decode(const Codec& codec, std::span<const std::uint8_t> data) {
    bit_reader reader(data);
    return codec.template decode<T>(reader);
}

template<typename Codec, typename T>
std::size_t encoded_size(const Codec& codec, const T& value) {
    std::vector<std::uint8_t> buffer;
    bit_writer writer(buffer);
    codec.encode(value, writer);
    writer.flush();
    return buffer.size();
}

// =============================================================================
// Type-erased codec wrapper (uses any_codec from type_erasure.hpp)
// =============================================================================

template<typename T>
using any_integer_codec = any_codec<T, std::vector<std::uint8_t>>;

// Factory functions for creating type-erased codecs
template<typename T>
any_integer_codec<T> make_unary_codec() {
    return any_integer_codec<T>(
        [](const T& value) -> std::vector<std::uint8_t> {
            return encode(unary_codec{}, value);
        }
    );
}

template<typename T>
any_integer_codec<T> make_elias_gamma_codec() {
    return any_integer_codec<T>(
        [](const T& value) -> std::vector<std::uint8_t> {
            return encode(elias_gamma_codec{}, value);
        }
    );
}

template<typename T>
any_integer_codec<T> make_adaptive_codec() {
    return any_integer_codec<T>(
        [](const T& value) -> std::vector<std::uint8_t> {
            std::vector<T> single = {value};
            adaptive_codec<T> codec;
            std::vector<std::uint8_t> buffer;
            bit_writer writer(buffer);
            codec.encode(single, writer);
            return buffer;
        }
    );
}

} // namespace stepanov::codecs