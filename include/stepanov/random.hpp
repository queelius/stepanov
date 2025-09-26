#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
#include <optional>
#include <functional>
#include <unordered_map>
#include "concepts.hpp"

namespace stepanov {

// Random number generator concepts
template<typename G>
concept random_generator = requires(G g) {
    typename G::result_type;
    { G::min() } -> std::convertible_to<typename G::result_type>;
    { G::max() } -> std::convertible_to<typename G::result_type>;
    { g() } -> std::convertible_to<typename G::result_type>;
};

template<typename D>
concept distribution = requires(D d, std::mt19937 g) {
    typename D::result_type;
    { d(g) } -> std::convertible_to<typename D::result_type>;
    { d.min() } -> std::convertible_to<typename D::result_type>;
    { d.max() } -> std::convertible_to<typename D::result_type>;
};

// PCG (Permuted Congruential Generator)
template<typename T = uint64_t>
class pcg32 {
public:
    using result_type = uint32_t;

private:
    uint64_t state;
    uint64_t inc;

    uint32_t rotr32(uint32_t x, unsigned r) const {
        return x >> r | x << (32 - r);
    }

public:
    pcg32(uint64_t seed = 0x853c49e6748fea9bULL, uint64_t seq = 0xa02bdbf7bb3c0a7ULL)
        : state(0), inc((seq << 1) | 1) {
        (*this)();
        state += seed;
        (*this)();
    }

    result_type operator()() {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = ((oldstate >> 18) ^ oldstate) >> 27;
        uint32_t rot = oldstate >> 59;
        return rotr32(xorshifted, rot);
    }

    static constexpr result_type min() { return std::numeric_limits<result_type>::min(); }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    void seed(uint64_t s) {
        state = 0;
        (*this)();
        state += s;
        (*this)();
    }

    void discard(unsigned long long n) {
        for (unsigned long long i = 0; i < n; ++i) {
            (*this)();
        }
    }
};

// xoshiro256** - fast, high-quality PRNG
class xoshiro256ss {
public:
    using result_type = uint64_t;

private:
    std::array<uint64_t, 4> s;

    static uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    explicit xoshiro256ss(uint64_t seed = 0) {
        std::mt19937_64 seeder(seed);
        for (auto& state : s) {
            state = seeder();
        }
    }

    result_type operator()() {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rotl(s[3], 45);

        return result;
    }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    void jump() {
        // Jump ahead 2^128 calls
        static const uint64_t JUMP[] = {
            0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
            0xa9582618e03fc9aa, 0x39abdc4529b1661c
        };

        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;

        for (auto jump : JUMP) {
            for (int b = 0; b < 64; ++b) {
                if (jump & (1ULL << b)) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                (*this)();
            }
        }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }
};

// Distribution transformations

// Box-Muller transform for normal distribution
template<typename RNG>
    requires random_generator<RNG>
class normal_distribution {
private:
    double mean;
    double stddev;
    mutable bool has_spare;
    mutable double spare;

public:
    using result_type = double;

    normal_distribution(double m = 0.0, double s = 1.0)
        : mean(m), stddev(s), has_spare(false) {}

    template<typename Generator>
    result_type operator()(Generator& g) const {
        if (has_spare) {
            has_spare = false;
            return spare * stddev + mean;
        }

        has_spare = true;

        static constexpr double two_pi = 2.0 * M_PI;

        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        double u = uniform(g);
        double v = uniform(g);

        double mag = stddev * std::sqrt(-2.0 * std::log(u));

        spare = mag * std::cos(two_pi * v);
        return mag * std::sin(two_pi * v) + mean;
    }

    double min() const { return -std::numeric_limits<double>::infinity(); }
    double max() const { return std::numeric_limits<double>::infinity(); }
};

// Exponential distribution using inverse CDF
template<typename RNG>
    requires random_generator<RNG>
class exponential_distribution {
private:
    double lambda;

public:
    using result_type = double;

    explicit exponential_distribution(double l = 1.0) : lambda(l) {}

    template<typename Generator>
    result_type operator()(Generator& g) const {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        return -std::log(1.0 - uniform(g)) / lambda;
    }

    double min() const { return 0.0; }
    double max() const { return std::numeric_limits<double>::infinity(); }
};

// Poisson distribution using Knuth's algorithm
template<typename RNG>
    requires random_generator<RNG>
class poisson_distribution {
private:
    double lambda;
    double exp_lambda;

public:
    using result_type = unsigned int;

    explicit poisson_distribution(double l = 1.0)
        : lambda(l), exp_lambda(std::exp(-l)) {}

    template<typename Generator>
    result_type operator()(Generator& g) const {
        if (lambda > 30.0) {
            // For large lambda, use transformed rejection method
            return poisson_ptrs(g);
        }

        // Knuth's algorithm for small lambda
        unsigned int k = 0;
        double p = 1.0;

        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        do {
            ++k;
            p *= uniform(g);
        } while (p > exp_lambda);

        return k - 1;
    }

    template<typename Generator>
    result_type poisson_ptrs(Generator& g) const {
        // Transformed rejection method for large lambda
        double smu = std::sqrt(lambda);
        double d = 6.0 * lambda * lambda;
        double big_l = lambda - 1.1484;

        normal_distribution<Generator> normal(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        while (true) {
            double em = -1;
            double diff = 0;

            double g_val = normal(g);
            double em_cand = smu * g_val + lambda;

            if (em_cand >= 0.0) {
                em = std::floor(em_cand);
                diff = em - lambda;

                if (std::abs(diff) <= d * std::abs(g_val)) {
                    return static_cast<result_type>(em);
                }

                if (diff < 0) {
                    if (diff * diff < 2 * lambda * std::log(uniform(g))) {
                        return static_cast<result_type>(em);
                    }
                } else {
                    double log_fact = em * std::log(lambda / em) - 1.1393 + em - lambda;
                    if (log_fact > std::log(uniform(g))) {
                        return static_cast<result_type>(em);
                    }
                }
            }
        }
    }

    result_type min() const { return 0; }
    result_type max() const { return std::numeric_limits<result_type>::max(); }
};

// Geometric distribution
template<typename RNG>
    requires random_generator<RNG>
class geometric_distribution {
private:
    double p;

public:
    using result_type = unsigned int;

    explicit geometric_distribution(double prob = 0.5) : p(prob) {}

    template<typename Generator>
    result_type operator()(Generator& g) const {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        return static_cast<result_type>(std::floor(std::log(uniform(g)) / std::log(1.0 - p)));
    }

    result_type min() const { return 0; }
    result_type max() const { return std::numeric_limits<result_type>::max(); }
};

// Custom distribution via inverse CDF
template<typename T>
class custom_distribution {
private:
    std::vector<T> values;
    std::vector<double> cdf;

public:
    using result_type = T;

    custom_distribution(const std::vector<T>& vals, const std::vector<double>& probs) {
        if (vals.size() != probs.size()) {
            throw std::invalid_argument("Values and probabilities must have same size");
        }

        values = vals;
        cdf.resize(probs.size());

        std::partial_sum(probs.begin(), probs.end(), cdf.begin());

        // Normalize CDF
        double total = cdf.back();
        for (auto& c : cdf) {
            c /= total;
        }
    }

    template<typename Generator>
    result_type operator()(Generator& g) const {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        double u = uniform(g);

        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        std::size_t index = std::distance(cdf.begin(), it);

        return values[std::min(index, values.size() - 1)];
    }

    T min() const { return *std::min_element(values.begin(), values.end()); }
    T max() const { return *std::max_element(values.begin(), values.end()); }
};

// Quasi-random sequences

// Sobol sequence generator
class sobol_sequence {
private:
    std::size_t dimension;
    std::size_t count;
    std::vector<std::vector<uint32_t>> direction_numbers;
    std::vector<uint32_t> x;

    void initialize_direction_numbers() {
        // Simplified: use basic direction numbers
        // In practice, would load from precomputed tables
        direction_numbers.resize(dimension);

        for (std::size_t d = 0; d < dimension; ++d) {
            direction_numbers[d].resize(32);
            uint32_t v = 1;

            for (std::size_t i = 0; i < 32; ++i) {
                direction_numbers[d][i] = v;
                v <<= 1;
            }
        }
    }

public:
    explicit sobol_sequence(std::size_t dim = 2)
        : dimension(dim), count(0), x(dim, 0) {
        initialize_direction_numbers();
    }

    std::vector<double> next() {
        if (count == 0) {
            count = 1;
            return std::vector<double>(dimension, 0.0);
        }

        // Find rightmost zero bit
        uint32_t c = count - 1;
        uint32_t j = 0;

        while ((c & 1) == 1) {
            c >>= 1;
            ++j;
        }

        // Update state
        for (std::size_t d = 0; d < dimension; ++d) {
            x[d] ^= direction_numbers[d][j];
        }

        // Convert to [0,1)
        std::vector<double> point(dimension);
        for (std::size_t d = 0; d < dimension; ++d) {
            point[d] = x[d] / static_cast<double>(1ULL << 32);
        }

        ++count;
        return point;
    }

    void reset() {
        count = 0;
        std::fill(x.begin(), x.end(), 0);
    }
};

// Halton sequence generator
class halton_sequence {
private:
    std::size_t dimension;
    std::size_t index;
    std::vector<std::size_t> primes;

    static std::size_t nth_prime(std::size_t n) {
        static const std::vector<std::size_t> small_primes = {
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
            53, 59, 61, 67, 71, 73, 79, 83, 89, 97
        };

        if (n < small_primes.size()) {
            return small_primes[n];
        }

        // For larger dimensions, compute primes
        std::size_t candidate = small_primes.back() + 2;
        std::size_t found = small_primes.size();

        while (found <= n) {
            bool is_prime = true;

            for (std::size_t p : small_primes) {
                if (p * p > candidate) break;
                if (candidate % p == 0) {
                    is_prime = false;
                    break;
                }
            }

            if (is_prime) {
                ++found;
                if (found == n + 1) return candidate;
            }

            candidate += 2;
        }

        return candidate;
    }

    double halton_element(std::size_t i, std::size_t base) const {
        double f = 1.0;
        double r = 0.0;

        while (i > 0) {
            f /= base;
            r += f * (i % base);
            i /= base;
        }

        return r;
    }

public:
    explicit halton_sequence(std::size_t dim = 2)
        : dimension(dim), index(0) {
        primes.reserve(dim);
        for (std::size_t d = 0; d < dim; ++d) {
            primes.push_back(nth_prime(d));
        }
    }

    std::vector<double> next() {
        std::vector<double> point(dimension);

        for (std::size_t d = 0; d < dimension; ++d) {
            point[d] = halton_element(index, primes[d]);
        }

        ++index;
        return point;
    }

    void reset() { index = 0; }
};

// Random sampling algorithms

// Reservoir sampling
template<typename T, typename RNG>
    requires random_generator<RNG>
class reservoir_sampler {
private:
    std::vector<T> reservoir;
    std::size_t k;
    std::size_t count;
    RNG& generator;

public:
    reservoir_sampler(std::size_t sample_size, RNG& gen)
        : k(sample_size), count(0), generator(gen) {
        reservoir.reserve(k);
    }

    void add(const T& item) {
        if (count < k) {
            reservoir.push_back(item);
        } else {
            std::uniform_int_distribution<std::size_t> dist(0, count);
            std::size_t j = dist(generator);

            if (j < k) {
                reservoir[j] = item;
            }
        }

        ++count;
    }

    const std::vector<T>& get_sample() const { return reservoir; }

    void reset() {
        reservoir.clear();
        count = 0;
    }
};

// Weighted sampling with replacement
template<typename T, typename RNG>
    requires random_generator<RNG>
class weighted_sampler {
private:
    std::discrete_distribution<std::size_t> dist;
    std::vector<T> items;

public:
    weighted_sampler(const std::vector<T>& elements, const std::vector<double>& weights)
        : items(elements), dist(weights.begin(), weights.end()) {
        if (elements.size() != weights.size()) {
            throw std::invalid_argument("Items and weights must have same size");
        }
    }

    T sample(RNG& generator) {
        return items[dist(generator)];
    }

    std::vector<T> sample_n(std::size_t n, RNG& generator) {
        std::vector<T> samples;
        samples.reserve(n);

        for (std::size_t i = 0; i < n; ++i) {
            samples.push_back(sample(generator));
        }

        return samples;
    }
};

// Alias method for O(1) discrete sampling
template<typename RNG>
    requires random_generator<RNG>
class alias_method {
private:
    struct alias_table_entry {
        std::size_t alias;
        double probability;
    };

    std::vector<alias_table_entry> table;
    std::uniform_real_distribution<double> uniform_real;
    std::uniform_int_distribution<std::size_t> uniform_int;

public:
    explicit alias_method(const std::vector<double>& probabilities)
        : uniform_real(0.0, 1.0), uniform_int(0, probabilities.size() - 1) {
        std::size_t n = probabilities.size();
        table.resize(n);

        // Normalize probabilities
        double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
        std::vector<double> prob(n);

        for (std::size_t i = 0; i < n; ++i) {
            prob[i] = probabilities[i] * n / sum;
        }

        // Separate into small and large
        std::vector<std::size_t> small, large;

        for (std::size_t i = 0; i < n; ++i) {
            if (prob[i] < 1.0) {
                small.push_back(i);
            } else {
                large.push_back(i);
            }
        }

        // Build alias table
        while (!small.empty() && !large.empty()) {
            std::size_t s = small.back();
            small.pop_back();

            std::size_t l = large.back();

            table[s].probability = prob[s];
            table[s].alias = l;

            prob[l] = prob[l] + prob[s] - 1.0;

            if (prob[l] < 1.0) {
                small.push_back(l);
                large.pop_back();
            }
        }

        // Handle remaining entries
        while (!large.empty()) {
            std::size_t l = large.back();
            large.pop_back();
            table[l].probability = 1.0;
            table[l].alias = l;
        }

        while (!small.empty()) {
            std::size_t s = small.back();
            small.pop_back();
            table[s].probability = 1.0;
            table[s].alias = s;
        }
    }

    std::size_t sample(RNG& generator) {
        std::size_t i = uniform_int(generator);
        double u = uniform_real(generator);

        if (u < table[i].probability) {
            return i;
        } else {
            return table[i].alias;
        }
    }
};

// Stratified sampling
template<typename T, typename RNG>
    requires random_generator<RNG>
std::vector<T> stratified_sample(const std::vector<T>& population,
                                 std::size_t n_samples,
                                 RNG& generator) {
    std::size_t pop_size = population.size();

    if (n_samples >= pop_size) {
        return population;
    }

    std::vector<T> samples;
    samples.reserve(n_samples);

    double interval = static_cast<double>(pop_size) / n_samples;
    std::uniform_real_distribution<double> uniform(0.0, interval);

    for (std::size_t i = 0; i < n_samples; ++i) {
        double offset = i * interval + uniform(generator);
        std::size_t index = std::min(static_cast<std::size_t>(offset), pop_size - 1);
        samples.push_back(population[index]);
    }

    return samples;
}

} // namespace stepanov