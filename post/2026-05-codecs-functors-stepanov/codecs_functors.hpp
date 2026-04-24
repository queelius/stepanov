// codecs_functors.hpp
// Pedagogical implementation for the post "Bits Follow Types"
// (Stepanov series, slot 20). For the production version, see PFC:
// https://github.com/queelius/pfc

#pragma once

#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace codecs_functors {

// ---- Bit I/O ----------------------------------------------------------------

class BitWriter {
    std::span<std::uint8_t> buf_;
    std::size_t byte_idx_ = 0;
    std::uint8_t byte_ = 0;
    std::uint8_t bit_pos_ = 0;
public:
    explicit BitWriter(std::span<std::uint8_t> buf) noexcept : buf_(buf) {}

    void write(bool bit) noexcept {
        byte_ |= (bit ? std::uint8_t{1} : std::uint8_t{0}) << bit_pos_;
        if (++bit_pos_ == 8) {
            buf_[byte_idx_++] = byte_;
            byte_ = 0;
            bit_pos_ = 0;
        }
    }

    void align() noexcept {
        if (bit_pos_ > 0) {
            buf_[byte_idx_++] = byte_;
            byte_ = 0;
            bit_pos_ = 0;
        }
    }

    [[nodiscard]] std::size_t bytes_written() const noexcept {
        return byte_idx_ + (bit_pos_ > 0 ? 1 : 0);
    }
};

class BitReader {
    std::span<const std::uint8_t> buf_;
    std::size_t byte_idx_ = 0;
    std::uint8_t bit_pos_ = 0;
public:
    explicit BitReader(std::span<const std::uint8_t> buf) noexcept : buf_(buf) {}

    bool read() noexcept {
        bool bit = ((buf_[byte_idx_] >> bit_pos_) & 1) != 0;
        if (++bit_pos_ == 8) {
            ++byte_idx_;
            bit_pos_ = 0;
        }
        return bit;
    }

    // True while at least one BYTE remains in the buffer. Callers query this
    // only at item boundaries, never mid-item, so padding bits from the
    // writer's align() are never exposed as data.
    [[nodiscard]] bool peek() const noexcept {
        return byte_idx_ < buf_.size();
    }
};

// ---- Concepts ---------------------------------------------------------------

template<typename T>
concept BitSink = requires(T& s, bool bit) {
    { s.write(bit) } -> std::same_as<void>;
};

template<typename T>
concept BitSource = requires(T& s) {
    { s.read() } -> std::same_as<bool>;
    { s.peek() } -> std::convertible_to<bool>;
};

template<typename C>
concept Codec = requires {
    typename C::value_type;
};
// Note: a fuller Codec concept would require encode and decode functions
// against any BitSink and BitSource. The minimal form above is sufficient
// for the pedagogical examples in this post; the prose discusses the
// fuller requirements.

// ---- Leaf codec: Elias gamma -----------------------------------------------
// Encodes positive integers (>= 1).
// Codeword for n: (bit_width(n) - 1) zero bits, then a one bit,
// then the bits of n minus the leading 1, MSB first.
// Length: 2 * floor(log2(n)) + 1 bits.

struct Gamma {
    using value_type = std::uint64_t;

    template<BitSink S>
    static void encode(value_type n, S& sink) {
        assert(n >= 1 && "Gamma is undefined for n = 0");
        std::size_t bits = std::bit_width(n);
        for (std::size_t i = 0; i < bits - 1; ++i) sink.write(false);
        sink.write(true);
        for (std::size_t i = bits - 1; i > 0; --i) {
            sink.write(((n >> (i - 1)) & 1) != 0);
        }
    }

    template<BitSource S>
    static value_type decode(S& source) {
        std::size_t bits = 1;
        while (!source.read()) ++bits;
        value_type result = 1;
        for (std::size_t i = 1; i < bits; ++i) {
            result = (result << 1) | (source.read() ? value_type{1} : value_type{0});
        }
        return result;
    }
};

// ---- Combinator: Opt -- coproduct with unit (1 + T) -------------------------
// Wire format: 1 tag bit (1 = present, 0 = absent), then C's encoding if present.

template<typename C>
struct Opt {
    using value_type = std::optional<typename C::value_type>;

    template<BitSink S>
    static void encode(const value_type& v, S& sink) {
        if (v) {
            sink.write(true);
            C::encode(*v, sink);
        } else {
            sink.write(false);
        }
    }

    template<BitSource S>
    static value_type decode(S& source) {
        if (source.read()) {
            return C::decode(source);
        }
        return std::nullopt;
    }
};

// ---- Combinator: Either -- binary coproduct (A + B) -------------------------
// Wire format: 1 tag bit (0 = left/A, 1 = right/B), then the chosen branch.

template<typename A, typename B>
struct Either {
    using value_type = std::variant<typename A::value_type, typename B::value_type>;

    template<BitSink S>
    static void encode(const value_type& v, S& sink) {
        if (v.index() == 0) {
            sink.write(false);
            A::encode(std::get<0>(v), sink);
        } else {
            sink.write(true);
            B::encode(std::get<1>(v), sink);
        }
    }

    template<BitSource S>
    static value_type decode(S& source) {
        if (source.read()) {
            return value_type{std::in_place_index<1>, B::decode(source)};
        }
        return value_type{std::in_place_index<0>, A::decode(source)};
    }
};

// ---- Combinator: Either3 -- ternary coproduct (A + B + C) -------------------
// Wire format: 2 tag bits encode the branch index (LSB first):
//   00 -> A, 01 -> B, 10 -> C. Then the chosen branch.

template<typename A, typename B, typename C>
struct Either3 {
    using value_type = std::variant<typename A::value_type,
                                    typename B::value_type,
                                    typename C::value_type>;

    template<BitSink S>
    static void encode(const value_type& v, S& sink) {
        std::size_t idx = v.index();
        sink.write((idx & 0x1) != 0);
        sink.write((idx & 0x2) != 0);
        if (idx == 0) A::encode(std::get<0>(v), sink);
        else if (idx == 1) B::encode(std::get<1>(v), sink);
        else C::encode(std::get<2>(v), sink);
    }

    template<BitSource S>
    static value_type decode(S& source) {
        std::size_t idx = (source.read() ? 1u : 0u);
        idx |= (source.read() ? 2u : 0u);
        if (idx == 0) return value_type{std::in_place_index<0>, A::decode(source)};
        if (idx == 1) return value_type{std::in_place_index<1>, B::decode(source)};
        return value_type{std::in_place_index<2>, C::decode(source)};
    }
};

}  // namespace codecs_functors
