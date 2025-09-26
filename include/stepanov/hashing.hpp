#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <concepts>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstring>
#include <deque>
#include "concepts.hpp"

namespace stepanov {

// Universal hash family concepts
template<typename H>
concept hash_function = requires(H h, typename H::input_type x) {
    typename H::input_type;
    typename H::output_type;
    { h(x) } -> std::convertible_to<typename H::output_type>;
};

template<typename H>
concept universal_hash_family = hash_function<H> && requires(H h) {
    { H::random() } -> std::same_as<H>;
};

// Basic universal hash functions

// Multiply-shift hashing
template<typename T>
    requires std::unsigned_integral<T>
class multiply_shift_hash {
public:
    using input_type = T;
    using output_type = std::size_t;

private:
    T a;
    std::size_t m;  // Number of bits in output

public:
    multiply_shift_hash(T a_val, std::size_t bits)
        : a(a_val | 1), m(bits) {}  // Ensure a is odd

    std::size_t operator()(T x) const {
        return (a * x) >> (sizeof(T) * 8 - m);
    }

    static multiply_shift_hash random(std::size_t bits = 32) {
        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        std::uniform_int_distribution<T> dist;
        return multiply_shift_hash(dist(gen), bits);
    }
};

// Tabulation hashing
template<typename T, std::size_t Chunks = sizeof(T)>
class tabulation_hash {
public:
    using input_type = T;
    using output_type = std::size_t;

private:
    std::array<std::array<std::size_t, 256>, Chunks> tables;

public:
    tabulation_hash() {
        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        std::uniform_int_distribution<std::size_t> dist;

        for (auto& table : tables) {
            for (auto& entry : table) {
                entry = dist(gen);
            }
        }
    }

    std::size_t operator()(T x) const {
        std::size_t hash = 0;
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&x);

        for (std::size_t i = 0; i < Chunks && i < sizeof(T); ++i) {
            hash ^= tables[i][bytes[i]];
        }

        return hash;
    }

    static tabulation_hash random() {
        return tabulation_hash();
    }
};

// Polynomial hashing for sequences
template<typename T>
class polynomial_hash {
public:
    using input_type = std::vector<T>;
    using output_type = std::size_t;

private:
    static constexpr std::size_t prime = 1000000007;
    std::size_t a;

public:
    explicit polynomial_hash(std::size_t a_val = 31) : a(a_val) {}

    std::size_t operator()(const std::vector<T>& seq) const {
        std::size_t hash = 0;
        std::size_t pow = 1;

        for (const auto& elem : seq) {
            hash = (hash + (std::hash<T>{}(elem) % prime) * pow) % prime;
            pow = (pow * a) % prime;
        }

        return hash;
    }

    static polynomial_hash random() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<std::size_t> dist(2, prime - 1);
        return polynomial_hash(dist(gen));
    }
};

// Bloom filter with optimal parameters
template<typename T, typename Hash = std::hash<T>>
class bloom_filter {
private:
    std::vector<bool> bits;
    std::size_t num_hash_functions;
    std::size_t num_bits;
    std::size_t num_elements;
    std::vector<std::size_t> seeds;

    std::size_t hash_with_seed(const T& item, std::size_t seed) const {
        Hash h;
        return (h(item) ^ (seed * 0x9e3779b9 + (seed << 6) + (seed >> 2))) % num_bits;
    }

public:
    // Constructor with optimal parameters
    bloom_filter(std::size_t expected_elements, double false_positive_rate = 0.01)
        : num_elements(0) {
        // Calculate optimal number of bits
        num_bits = static_cast<std::size_t>(
            -expected_elements * std::log(false_positive_rate) / (std::log(2) * std::log(2))
        );

        // Ensure reasonable size
        if (num_bits > 1000000000) {  // Cap at 1 billion bits
            num_bits = 1000000000;
        }
        if (num_bits < 64) {
            num_bits = 64;
        }

        // Calculate optimal number of hash functions
        num_hash_functions = static_cast<std::size_t>(
            (num_bits / expected_elements) * std::log(2)
        );

        if (num_hash_functions < 1) num_hash_functions = 1;
        if (num_hash_functions > 10) num_hash_functions = 10;

        bits.resize(num_bits, false);

        // Generate seeds for hash functions
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<std::size_t> dist;

        seeds.reserve(num_hash_functions);
        for (std::size_t i = 0; i < num_hash_functions; ++i) {
            seeds.push_back(dist(gen));
        }
    }

    void insert(const T& item) {
        for (std::size_t i = 0; i < num_hash_functions; ++i) {
            bits[hash_with_seed(item, seeds[i])] = true;
        }
        ++num_elements;
    }

    bool possibly_contains(const T& item) const {
        for (std::size_t i = 0; i < num_hash_functions; ++i) {
            if (!bits[hash_with_seed(item, seeds[i])]) {
                return false;
            }
        }
        return true;
    }

    double estimated_false_positive_rate() const {
        std::size_t set_bits = std::count(bits.begin(), bits.end(), true);
        double ratio = static_cast<double>(set_bits) / num_bits;
        return std::pow(ratio, num_hash_functions);
    }

    void clear() {
        std::fill(bits.begin(), bits.end(), false);
        num_elements = 0;
    }

    std::size_t size() const { return num_elements; }
    std::size_t bit_count() const { return num_bits; }
    std::size_t hash_count() const { return num_hash_functions; }
};

// Count-Min sketch for frequency estimation
template<typename T, typename Hash = std::hash<T>>
class count_min_sketch {
private:
    std::vector<std::vector<std::size_t>> counters;
    std::size_t width;
    std::size_t depth;
    std::vector<std::size_t> seeds;

    std::size_t hash_with_seed(const T& item, std::size_t seed) const {
        Hash h;
        return (h(item) ^ (seed * 0x9e3779b9 + (seed << 6) + (seed >> 2))) % width;
    }

public:
    count_min_sketch(std::size_t w, std::size_t d)
        : width(w), depth(d), counters(d, std::vector<std::size_t>(w, 0)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<std::size_t> dist;

        seeds.reserve(depth);
        for (std::size_t i = 0; i < depth; ++i) {
            seeds.push_back(dist(gen));
        }
    }

    // Constructor with error bounds
    count_min_sketch(double epsilon, double delta)
        : count_min_sketch(
            static_cast<std::size_t>(std::ceil(std::exp(1) / epsilon)),
            static_cast<std::size_t>(std::ceil(std::log(1 / delta)))
        ) {}

    void update(const T& item, std::size_t count = 1) {
        for (std::size_t i = 0; i < depth; ++i) {
            counters[i][hash_with_seed(item, seeds[i])] += count;
        }
    }

    std::size_t estimate(const T& item) const {
        std::size_t min_count = std::numeric_limits<std::size_t>::max();

        for (std::size_t i = 0; i < depth; ++i) {
            min_count = std::min(min_count, counters[i][hash_with_seed(item, seeds[i])]);
        }

        return min_count;
    }

    void clear() {
        for (auto& row : counters) {
            std::fill(row.begin(), row.end(), 0);
        }
    }
};

// MinHash for similarity estimation
template<typename T>
class minhash {
private:
    std::size_t num_permutations;
    std::vector<std::size_t> a_values;
    std::vector<std::size_t> b_values;
    static constexpr std::size_t prime = 4294967291;  // Large prime < 2^32

    std::size_t hash_function(const T& item, std::size_t a, std::size_t b) const {
        std::hash<T> h;
        std::size_t hash_val = h(item);
        return (a * hash_val + b) % prime;
    }

public:
    using signature_type = std::vector<std::size_t>;

    explicit minhash(std::size_t k = 128) : num_permutations(k) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<std::size_t> dist(1, prime - 1);

        a_values.reserve(k);
        b_values.reserve(k);

        for (std::size_t i = 0; i < k; ++i) {
            a_values.push_back(dist(gen));
            b_values.push_back(dist(gen));
        }
    }

    signature_type compute_signature(const std::unordered_set<T>& set) const {
        signature_type signature(num_permutations, std::numeric_limits<std::size_t>::max());

        for (const auto& item : set) {
            for (std::size_t i = 0; i < num_permutations; ++i) {
                std::size_t hash_val = hash_function(item, a_values[i], b_values[i]);
                signature[i] = std::min(signature[i], hash_val);
            }
        }

        return signature;
    }

    double jaccard_similarity(const signature_type& sig1, const signature_type& sig2) const {
        std::size_t matches = 0;

        for (std::size_t i = 0; i < num_permutations; ++i) {
            if (sig1[i] == sig2[i]) {
                ++matches;
            }
        }

        return static_cast<double>(matches) / num_permutations;
    }
};

// HyperLogLog for cardinality estimation
template<typename T, typename Hash = std::hash<T>>
class hyperloglog {
private:
    std::vector<uint8_t> registers;
    std::size_t m;  // Number of registers (must be power of 2)
    std::size_t b;  // Number of bits for register index

    std::size_t leading_zeros(std::size_t hash) const {
        if (hash == 0) return 64 - b;

        std::size_t zeros = 0;
        std::size_t mask = 1ULL << (63);

        while ((hash & mask) == 0 && zeros < 64) {
            ++zeros;
            mask >>= 1;
        }

        return zeros;
    }

    double alpha() const {
        if (m >= 128) return 0.7213 / (1 + 1.079 / m);
        if (m >= 64) return 0.709;
        if (m >= 32) return 0.697;
        if (m >= 16) return 0.673;
        return 0.5;
    }

public:
    explicit hyperloglog(std::size_t precision = 14) : b(precision) {
        if (precision < 4 || precision > 16) {
            throw std::invalid_argument("Precision must be between 4 and 16");
        }
        m = 1ULL << precision;
        registers.resize(m, 0);
    }

    void add(const T& item) {
        Hash h;
        std::size_t hash = h(item);

        std::size_t j = hash & ((1ULL << b) - 1);  // First b bits (mask properly)
        std::size_t w = hash >> b;                 // Remaining bits

        if (j >= registers.size()) j = registers.size() - 1;  // Safety check

        registers[j] = std::max(registers[j], static_cast<uint8_t>(leading_zeros(w) + 1));
    }

    std::size_t estimate_cardinality() const {
        double raw_estimate = 0;
        std::size_t zeros = 0;

        for (const auto& reg : registers) {
            raw_estimate += std::pow(2, -reg);
            if (reg == 0) ++zeros;
        }

        raw_estimate = alpha() * m * m / raw_estimate;

        // Apply bias correction
        if (raw_estimate <= 2.5 * m && zeros != 0) {
            // Small range correction
            return m * std::log(static_cast<double>(m) / zeros);
        } else if (raw_estimate <= (1.0 / 30.0) * (1ULL << 32)) {
            // No correction
            return static_cast<std::size_t>(raw_estimate);
        } else {
            // Large range correction
            return static_cast<std::size_t>(-std::pow(2, 32) * std::log(1 - raw_estimate / std::pow(2, 32)));
        }
    }

    void merge(const hyperloglog& other) {
        if (m != other.m) {
            throw std::invalid_argument("Cannot merge HyperLogLogs with different precisions");
        }

        for (std::size_t i = 0; i < m; ++i) {
            registers[i] = std::max(registers[i], other.registers[i]);
        }
    }

    void clear() {
        std::fill(registers.begin(), registers.end(), 0);
    }
};

// Consistent hashing for distributed systems
template<typename Node, typename Key, typename Hash = std::hash<Key>>
class consistent_hash {
private:
    std::map<std::size_t, Node> ring;
    std::size_t virtual_nodes;
    Hash hasher;

    std::size_t hash_node(const Node& node, std::size_t replica) const {
        std::string node_str = std::to_string(std::hash<Node>{}(node)) + "_" + std::to_string(replica);
        return std::hash<std::string>{}(node_str);
    }

public:
    explicit consistent_hash(std::size_t vnodes = 150)
        : virtual_nodes(vnodes) {}

    void add_node(const Node& node) {
        for (std::size_t i = 0; i < virtual_nodes; ++i) {
            std::size_t hash = hash_node(node, i);
            ring[hash] = node;
        }
    }

    void remove_node(const Node& node) {
        for (std::size_t i = 0; i < virtual_nodes; ++i) {
            std::size_t hash = hash_node(node, i);
            ring.erase(hash);
        }
    }

    std::optional<Node> get_node(const Key& key) const {
        if (ring.empty()) return std::nullopt;

        std::size_t hash = hasher(key);
        auto it = ring.lower_bound(hash);

        if (it == ring.end()) {
            it = ring.begin();
        }

        return it->second;
    }

    std::vector<Node> get_nodes(const Key& key, std::size_t n) const {
        std::vector<Node> nodes;
        if (ring.empty()) return nodes;

        std::size_t hash = hasher(key);
        auto it = ring.lower_bound(hash);

        std::unordered_set<Node> seen;

        while (nodes.size() < n && seen.size() < ring.size() / virtual_nodes) {
            if (it == ring.end()) {
                it = ring.begin();
            }

            if (seen.insert(it->second).second) {
                nodes.push_back(it->second);
            }

            ++it;
        }

        return nodes;
    }
};

// Cuckoo hashing
template<typename Key, typename Value, typename Hash1 = std::hash<Key>>
class cuckoo_hash_table {
private:
    struct bucket {
        bool occupied = false;
        Key key;
        Value value;
    };

    std::vector<bucket> table1;
    std::vector<bucket> table2;
    std::size_t size;
    std::size_t num_elements;
    Hash1 hash1;

    // Second hash function
    struct hash2 {
        std::size_t operator()(const Key& k) const {
            std::size_t h = std::hash<Key>{}(k);
            return h ^ (h >> 16) ^ (h << 16);
        }
    } hash2;

    static constexpr std::size_t max_iterations = 500;
    static constexpr double max_load_factor = 0.5;

    std::size_t index1(const Key& key) const {
        return hash1(key) % size;
    }

    std::size_t index2(const Key& key) const {
        return hash2(key) % size;
    }

    void rehash() {
        std::size_t new_size = size * 2;
        std::vector<bucket> new_table1(new_size);
        std::vector<bucket> new_table2(new_size);

        auto old_table1 = std::move(table1);
        auto old_table2 = std::move(table2);

        table1 = std::move(new_table1);
        table2 = std::move(new_table2);
        std::size_t old_size = size;
        size = new_size;
        num_elements = 0;

        for (const auto& b : old_table1) {
            if (b.occupied) {
                insert(b.key, b.value);
            }
        }

        for (const auto& b : old_table2) {
            if (b.occupied) {
                insert(b.key, b.value);
            }
        }
    }

public:
    explicit cuckoo_hash_table(std::size_t initial_size = 16)
        : size(initial_size), num_elements(0),
          table1(initial_size), table2(initial_size) {}

    bool insert(const Key& key, const Value& value) {
        if (static_cast<double>(num_elements) / (2 * size) > max_load_factor) {
            rehash();
        }

        Key current_key = key;
        Value current_value = value;

        for (std::size_t i = 0; i < max_iterations; ++i) {
            // Try table 1
            std::size_t idx1 = index1(current_key);
            if (!table1[idx1].occupied) {
                table1[idx1] = {true, current_key, current_value};
                ++num_elements;
                return true;
            }

            // Swap with table 1
            std::swap(current_key, table1[idx1].key);
            std::swap(current_value, table1[idx1].value);

            // Try table 2
            std::size_t idx2 = index2(current_key);
            if (!table2[idx2].occupied) {
                table2[idx2] = {true, current_key, current_value};
                ++num_elements;
                return true;
            }

            // Swap with table 2
            std::swap(current_key, table2[idx2].key);
            std::swap(current_value, table2[idx2].value);
        }

        // Rehash if we couldn't insert
        rehash();
        return insert(current_key, current_value);
    }

    std::optional<Value> find(const Key& key) const {
        std::size_t idx1 = index1(key);
        if (table1[idx1].occupied && table1[idx1].key == key) {
            return table1[idx1].value;
        }

        std::size_t idx2 = index2(key);
        if (table2[idx2].occupied && table2[idx2].key == key) {
            return table2[idx2].value;
        }

        return std::nullopt;
    }

    bool erase(const Key& key) {
        std::size_t idx1 = index1(key);
        if (table1[idx1].occupied && table1[idx1].key == key) {
            table1[idx1].occupied = false;
            --num_elements;
            return true;
        }

        std::size_t idx2 = index2(key);
        if (table2[idx2].occupied && table2[idx2].key == key) {
            table2[idx2].occupied = false;
            --num_elements;
            return true;
        }

        return false;
    }

    std::size_t count() const { return num_elements; }
    std::size_t capacity() const { return 2 * size; }
    double load_factor() const {
        return static_cast<double>(num_elements) / (2 * size);
    }
};

// Rolling hash (Rabin fingerprint)
template<typename T>
class rolling_hash {
private:
    std::size_t hash_value;
    std::deque<T> window;
    std::size_t window_size;
    static constexpr std::size_t base = 256;
    static constexpr std::size_t prime = 1000000007;
    std::size_t base_power;

    std::size_t mod_pow(std::size_t base, std::size_t exp, std::size_t mod) const {
        std::size_t result = 1;
        base %= mod;

        while (exp > 0) {
            if (exp & 1) result = (result * base) % mod;
            base = (base * base) % mod;
            exp >>= 1;
        }

        return result;
    }

public:
    explicit rolling_hash(std::size_t size)
        : hash_value(0), window_size(size) {
        base_power = mod_pow(base, window_size - 1, prime);
    }

    void push(const T& item) {
        std::size_t item_hash = std::hash<T>{}(item) % prime;

        if (window.size() == window_size) {
            // Remove oldest element
            std::size_t old_hash = std::hash<T>{}(window.front()) % prime;
            hash_value = (hash_value + prime - (old_hash * base_power) % prime) % prime;
            window.pop_front();
        }

        // Add new element
        hash_value = (hash_value * base + item_hash) % prime;
        window.push_back(item);
    }

    std::size_t get_hash() const { return hash_value; }

    bool is_full() const { return window.size() == window_size; }

    void clear() {
        hash_value = 0;
        window.clear();
    }
};

// Perfect hashing (minimal perfect hash function) - simplified version
template<typename Key>
class perfect_hash {
private:
    std::vector<Key> keys;
    std::vector<std::size_t> g_values;
    std::size_t n;

    std::size_t hash1(const Key& key) const {
        return std::hash<Key>{}(key) % n;
    }

    std::size_t hash2(const Key& key, std::size_t g) const {
        return (std::hash<Key>{}(key) ^ g) % n;
    }

public:
    void build(const std::vector<Key>& input_keys) {
        keys = input_keys;
        n = keys.size();
        g_values.resize(n, 0);

        // Simplified: In a real implementation, this would use
        // more sophisticated techniques like CHD or BBHash
        for (std::size_t i = 0; i < n; ++i) {
            g_values[i] = i;
        }
    }

    std::optional<std::size_t> perfect_hash_value(const Key& key) const {
        std::size_t h1 = hash1(key);
        if (h1 >= n) return std::nullopt;

        std::size_t result = hash2(key, g_values[h1]);

        // Verify this is actually the key
        if (result < keys.size() && keys[result] == key) {
            return result;
        }

        return std::nullopt;
    }
};

} // namespace stepanov