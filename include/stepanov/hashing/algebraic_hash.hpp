#pragma once

#include "../concepts.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <type_traits>

namespace stepanov {
namespace hashing {

// Hash value type with algebraic operations
template<std::size_t Bits>
class hash_value {
public:
    static constexpr std::size_t bits = Bits;
    static constexpr std::size_t bytes = Bits / 8;
    using storage_type = std::array<std::uint8_t, bytes>;

private:
    storage_type data_;

public:
    // Constructors
    hash_value() : data_{} {}

    explicit hash_value(const storage_type& data) : data_(data) {}

    hash_value(std::uint64_t val) : data_{} {
        static_assert(Bits >= 64, "Hash value too small for uint64_t");
        std::memcpy(data_.data(), &val, sizeof(val));
    }

    // Algebraic operations (forms a group under XOR)
    hash_value operator^(const hash_value& other) const {
        hash_value result;
        for (std::size_t i = 0; i < bytes; ++i) {
            result.data_[i] = data_[i] ^ other.data_[i];
        }
        return result;
    }

    hash_value& operator^=(const hash_value& other) {
        for (std::size_t i = 0; i < bytes; ++i) {
            data_[i] ^= other.data_[i];
        }
        return *this;
    }

    // Bitwise complement
    hash_value operator~() const {
        hash_value result;
        for (std::size_t i = 0; i < bytes; ++i) {
            result.data_[i] = ~data_[i];
        }
        return result;
    }

    // Rotation operations
    hash_value rotate_left(std::size_t n) const {
        n %= bits;
        if (n == 0) return *this;

        hash_value result;
        std::size_t byte_shift = n / 8;
        std::size_t bit_shift = n % 8;

        for (std::size_t i = 0; i < bytes; ++i) {
            std::size_t src = (i + byte_shift) % bytes;
            result.data_[i] = (data_[src] << bit_shift);
            if (bit_shift > 0) {
                std::size_t prev_src = (src + bytes - 1) % bytes;
                result.data_[i] |= (data_[prev_src] >> (8 - bit_shift));
            }
        }
        return result;
    }

    hash_value rotate_right(std::size_t n) const {
        return rotate_left(bits - (n % bits));
    }

    // Comparison
    bool operator==(const hash_value& other) const {
        return data_ == other.data_;
    }

    bool operator!=(const hash_value& other) const {
        return data_ != other.data_;
    }

    bool operator<(const hash_value& other) const {
        return data_ < other.data_;
    }

    // Conversion
    std::string to_hex() const {
        static constexpr char hex_chars[] = "0123456789abcdef";
        std::string result;
        result.reserve(bytes * 2);

        for (std::uint8_t byte : data_) {
            result.push_back(hex_chars[byte >> 4]);
            result.push_back(hex_chars[byte & 0x0F]);
        }
        return result;
    }

    // Access
    const storage_type& data() const { return data_; }
    storage_type& data() { return data_; }

    // Convert to smaller hash value
    template<std::size_t NewBits>
        requires (NewBits <= Bits && NewBits % 8 == 0)
    hash_value<NewBits> truncate() const {
        hash_value<NewBits> result;
        std::memcpy(result.data().data(), data_.data(), NewBits / 8);
        return result;
    }
};

// Common hash value sizes
using hash32 = hash_value<32>;
using hash64 = hash_value<64>;
using hash128 = hash_value<128>;
using hash256 = hash_value<256>;

// Polynomial rolling hash
template<typename T = std::uint64_t>
class polynomial_hash {
public:
    using value_type = T;

private:
    T hash_;
    T base_;
    T mod_;
    std::size_t length_;
    T base_power_;  // base^length mod mod

    static T mod_mul(T a, T b, T mod) {
        // Prevent overflow in multiplication
        if constexpr (sizeof(T) <= 4) {
            return static_cast<T>((static_cast<std::uint64_t>(a) * b) % mod);
        } else {
            // Use 128-bit arithmetic if available
            using U = std::conditional_t<sizeof(T) == 8, unsigned __int128, T>;
            return static_cast<T>((static_cast<U>(a) * b) % mod);
        }
    }

    static T mod_pow(T base, std::size_t exp, T mod) {
        T result = 1;
        T b = base % mod;
        while (exp > 0) {
            if (exp & 1) {
                result = mod_mul(result, b, mod);
            }
            b = mod_mul(b, b, mod);
            exp >>= 1;
        }
        return result;
    }

public:
    // Large prime for modulus (Mersenne prime 2^61 - 1 for 64-bit)
    static constexpr T default_mod() {
        if constexpr (sizeof(T) == 8) {
            return (T(1) << 61) - 1;
        } else {
            return (T(1) << 31) - 1;  // Mersenne prime 2^31 - 1 for 32-bit
        }
    }

    polynomial_hash(T base = 31, T mod = default_mod())
        : hash_(0), base_(base), mod_(mod), length_(0), base_power_(1) {}

    // Initialize with a sequence
    template<typename InputIt>
    polynomial_hash(InputIt first, InputIt last, T base = 31, T mod = default_mod())
        : polynomial_hash(base, mod) {
        for (auto it = first; it != last; ++it) {
            push_back(*it);
        }
    }

    // Rolling hash operations
    void push_back(T value) {
        hash_ = mod_mul(hash_, base_, mod_);
        hash_ = (hash_ + (value % mod_)) % mod_;
        base_power_ = mod_mul(base_power_, base_, mod_);
        ++length_;
    }

    void push_front(T value) {
        T val_contribution = mod_mul(value % mod_, base_power_, mod_);
        hash_ = (hash_ + val_contribution) % mod_;
        base_power_ = mod_mul(base_power_, base_, mod_);
        ++length_;
    }

    void pop_front(T value) {
        if (length_ == 0) return;

        --length_;
        base_power_ = mod_pow(base_, length_, mod_);

        T val_contribution = mod_mul(value % mod_, base_power_, mod_);
        hash_ = (hash_ + mod_ - val_contribution) % mod_;
    }

    void slide(T old_value, T new_value) {
        // Remove old value from front
        T old_contribution = mod_mul(old_value % mod_, base_power_, mod_);
        hash_ = (hash_ + mod_ - old_contribution) % mod_;

        // Shift everything
        hash_ = mod_mul(hash_, base_, mod_);

        // Add new value at back
        hash_ = (hash_ + (new_value % mod_)) % mod_;
    }

    T value() const { return hash_; }
    std::size_t length() const { return length_; }

    void clear() {
        hash_ = 0;
        length_ = 0;
        base_power_ = 1;
    }

    bool operator==(const polynomial_hash& other) const {
        return hash_ == other.hash_ && length_ == other.length_;
    }
};

// Universal polynomial hash family
template<typename T = std::uint64_t>
class universal_polynomial_hash {
public:
    using value_type = T;

private:
    std::vector<T> coefficients_;
    T mod_;

    static T mod_mul(T a, T b, T mod) {
        if constexpr (sizeof(T) <= 4) {
            return static_cast<T>((static_cast<std::uint64_t>(a) * b) % mod);
        } else {
            using U = std::conditional_t<sizeof(T) == 8, unsigned __int128, T>;
            return static_cast<T>((static_cast<U>(a) * b) % mod);
        }
    }

public:
    universal_polynomial_hash(std::size_t degree, T mod = polynomial_hash<T>::default_mod())
        : coefficients_(degree + 1), mod_(mod) {
        randomize();
    }

    void randomize() {
        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        std::uniform_int_distribution<T> dist(0, mod_ - 1);

        for (auto& coeff : coefficients_) {
            coeff = dist(gen);
        }
    }

    template<typename InputIt>
    T operator()(InputIt first, InputIt last) const {
        T result = 0;
        T x_power = 1;
        std::size_t i = 0;

        for (auto it = first; it != last && i < coefficients_.size(); ++it, ++i) {
            T value = static_cast<T>(*it) % mod_;
            result = (result + mod_mul(coefficients_[i], mod_mul(value, x_power, mod_), mod_)) % mod_;
            x_power = mod_mul(x_power, value, mod_);
        }

        return result;
    }

    T operator()(std::span<const T> data) const {
        return operator()(data.begin(), data.end());
    }
};

// Tabulation hashing with XOR universality
template<typename T, std::size_t Tables = sizeof(T)>
class xor_tabulation_hash {
public:
    using value_type = T;
    using output_type = std::uint64_t;

private:
    std::array<std::array<output_type, 256>, Tables> tables_;

    void initialize() {
        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        std::uniform_int_distribution<output_type> dist;

        for (auto& table : tables_) {
            for (auto& entry : table) {
                entry = dist(gen);
            }
        }
    }

public:
    xor_tabulation_hash() { initialize(); }

    void randomize() { initialize(); }

    output_type operator()(T value) const {
        output_type result = 0;
        const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&value);

        for (std::size_t i = 0; i < Tables && i < sizeof(T); ++i) {
            result ^= tables_[i][bytes[i]];
        }

        return result;
    }

    // Combine two hash values (XOR is universal)
    static output_type combine(output_type h1, output_type h2) {
        return h1 ^ h2;
    }
};

// Multiply-shift with optimal constants
template<typename T>
    requires std::unsigned_integral<T>
class golden_ratio_hash {
public:
    using value_type = T;
    using output_type = std::size_t;

private:
    static constexpr T golden_ratio_constant() {
        if constexpr (sizeof(T) == 8) {
            return 0x9e3779b97f4a7c15ULL;  // 2^64 / phi
        } else if constexpr (sizeof(T) == 4) {
            return 0x9e3779b9U;  // 2^32 / phi
        } else {
            return T(0.6180339887 * std::numeric_limits<T>::max());
        }
    }

    T multiplier_;
    std::size_t shift_;

public:
    golden_ratio_hash(std::size_t output_bits = sizeof(std::size_t) * 8)
        : multiplier_(golden_ratio_constant()),
          shift_(sizeof(T) * 8 - output_bits) {}

    output_type operator()(T value) const {
        return (multiplier_ * value) >> shift_;
    }

    void set_multiplier(T mult) { multiplier_ = mult | 1; }  // Ensure odd

    static golden_ratio_hash random(std::size_t output_bits = sizeof(std::size_t) * 8) {
        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        std::uniform_int_distribution<T> dist;

        golden_ratio_hash h(output_bits);
        h.set_multiplier(dist(gen));
        return h;
    }
};

// FNV-1a hash (Fowler-Noll-Vo)
template<typename T = std::uint64_t>
class fnv1a_hash {
public:
    using value_type = T;

private:
    T hash_;

    static constexpr T fnv_prime() {
        if constexpr (sizeof(T) == 8) {
            return 0x100000001b3ULL;
        } else {
            return 0x01000193U;
        }
    }

    static constexpr T fnv_offset_basis() {
        if constexpr (sizeof(T) == 8) {
            return 0xcbf29ce484222325ULL;
        } else {
            return 0x811c9dc5U;
        }
    }

public:
    fnv1a_hash() : hash_(fnv_offset_basis()) {}

    void update(std::uint8_t byte) {
        hash_ ^= byte;
        hash_ *= fnv_prime();
    }

    template<typename InputIt>
    void update(InputIt first, InputIt last) {
        for (auto it = first; it != last; ++it) {
            update(static_cast<std::uint8_t>(*it));
        }
    }

    void update(std::span<const std::uint8_t> data) {
        update(data.begin(), data.end());
    }

    T value() const { return hash_; }

    void reset() { hash_ = fnv_offset_basis(); }

    template<typename InputIt>
    static T hash(InputIt first, InputIt last) {
        fnv1a_hash h;
        h.update(first, last);
        return h.value();
    }
};

// Hash combiner using mixing functions
class hash_combiner {
public:
    template<typename T>
    static T combine(T h1, T h2) {
        // Based on boost::hash_combine
        h1 ^= h2 + T(0x9e3779b9) + (h1 << 6) + (h1 >> 2);
        return h1;
    }

    template<typename T, typename... Args>
    static T combine_multiple(T first, Args... args) {
        if constexpr (sizeof...(args) == 0) {
            return first;
        } else {
            return combine(first, combine_multiple(args...));
        }
    }

    // Symmetric combination (order-independent)
    template<typename T>
    static T combine_symmetric(T h1, T h2) {
        return h1 ^ h2;
    }

    // Ordered combination for sequences
    template<typename InputIt, typename T = std::uint64_t>
    static T combine_sequence(InputIt first, InputIt last) {
        T result = 0;
        for (auto it = first; it != last; ++it) {
            result = combine(result, static_cast<T>(*it));
        }
        return result;
    }
};

} // namespace hashing
} // namespace stepanov