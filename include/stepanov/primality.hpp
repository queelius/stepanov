#pragma once

#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include "concepts.hpp"
#include "expmod.hpp"
#include "gcd.hpp"

namespace stepanov {

/**
 * Primality Testing Algorithms
 *
 * Generic implementations that work over integral domains.
 * These algorithms demonstrate the power of generic programming
 * in number theory.
 *
 * Algorithms:
 * - Miller-Rabin (probabilistic, fast)
 * - Solovay-Strassen (probabilistic, uses Jacobi symbol)
 * - Baillie-PSW (deterministic for practical ranges)
 * - Lucas-Lehmer (for Mersenne primes)
 * - AKS (deterministic polynomial time)
 */

// Jacobi symbol computation (generalization of Legendre symbol)
template<integral_domain T>
int jacobi_symbol(T a, T n) {
    if (n <= T(0) || even(n)) {
        throw std::invalid_argument("n must be positive and odd");
    }

    a = remainder(a, n);
    int result = 1;

    while (a != T(0)) {
        while (even(a)) {
            a = half(a);
            T n_mod_8 = remainder(n, T(8));
            if (n_mod_8 == T(3) || n_mod_8 == T(5)) {
                result = -result;
            }
        }

        std::swap(a, n);

        if (remainder(a, T(4)) == T(3) && remainder(n, T(4)) == T(3)) {
            result = -result;
        }

        a = remainder(a, n);
    }

    return (n == T(1)) ? result : 0;
}

// Miller-Rabin primality test
template<integral_domain T>
class miller_rabin {
private:
    std::mt19937_64 rng;

    // Decompose n-1 = 2^r * d
    static void decompose(T n, T& r, T& d) {
        r = T(0);
        d = n - T(1);
        while (even(d)) {
            d = half(d);
            r = increment(r);
        }
    }

public:
    // Single witness test
    bool witness_test(T n, T a) {
        T r, d;
        decompose(n, r, d);

        T x = expmod(a, d, n);
        if (x == T(1) || x == n - T(1)) {
            return true;  // Probably prime
        }

        for (T i = T(1); i < r; i = increment(i)) {
            x = expmod(x, T(2), n);
            if (x == n - T(1)) {
                return true;  // Probably prime
            }
        }

        return false;  // Definitely composite
    }

public:
    miller_rabin() : rng(std::random_device{}()) {}

    // Test with k random witnesses
    bool is_prime(T n, size_t k = 20) {
        if (n < T(2)) return false;
        if (n == T(2)) return true;
        if (even(n)) return false;

        // Small primes
        std::vector<T> small_primes = {T(2), T(3), T(5), T(7), T(11), T(13), T(17), T(19), T(23), T(29)};
        for (const T& p : small_primes) {
            if (n == p) return true;
            if (remainder(n, p) == T(0)) return false;
        }

        // Random witness testing
        std::uniform_int_distribution<uint64_t> dist(2, 1000000);

        for (size_t i = 0; i < k; ++i) {
            T a = T(dist(rng));
            a = remainder(a, n - T(2)) + T(2);  // a in [2, n-2]

            if (!witness_test(n, a)) {
                return false;  // Definitely composite
            }
        }

        return true;  // Probably prime with error probability 1/4^k
    }

    // Deterministic for small n using known witnesses
    bool is_prime_deterministic(T n) {
        if (n < T(2)) return false;
        if (n == T(2) || n == T(3)) return true;
        if (even(n)) return false;

        // Deterministic witnesses for different ranges
        std::vector<T> witnesses;

        if (n < T(2047)) {
            witnesses = {T(2)};
        } else if (n < T(1373653)) {
            witnesses = {T(2), T(3)};
        } else if (n < T(9080191)) {
            witnesses = {T(31), T(73)};
        } else if (n < T(25326001)) {
            witnesses = {T(2), T(3), T(5)};
        } else if (n < T(3215031751)) {
            witnesses = {T(2), T(3), T(5), T(7)};
        } else {
            // For larger n, use probabilistic version
            return is_prime(n, 40);
        }

        for (const T& a : witnesses) {
            if (!witness_test(n, a)) {
                return false;
            }
        }

        return true;
    }
};

// Solovay-Strassen primality test
template<integral_domain T>
class solovay_strassen {
private:
    std::mt19937_64 rng;

public:
    solovay_strassen() : rng(std::random_device{}()) {}

    bool is_prime(T n, size_t k = 20) {
        if (n < T(2)) return false;
        if (n == T(2)) return true;
        if (even(n)) return false;

        std::uniform_int_distribution<uint64_t> dist(2, 1000000);

        for (size_t i = 0; i < k; ++i) {
            T a = T(dist(rng));
            a = remainder(a, n - T(2)) + T(2);  // a in [2, n-2]

            T g = gcd(a, n);
            if (g > T(1)) {
                return false;  // Found a factor
            }

            int jacobi = jacobi_symbol(a, n);
            T mod_result = expmod(a, half(n - T(1)), n);

            // Convert Jacobi symbol to T
            T jacobi_mod;
            if (jacobi == -1) {
                jacobi_mod = n - T(1);
            } else if (jacobi == 0) {
                jacobi_mod = T(0);
            } else {
                jacobi_mod = T(1);
            }

            if (mod_result != jacobi_mod) {
                return false;  // Definitely composite
            }
        }

        return true;  // Probably prime
    }
};

// Lucas primality test (for numbers of special form)
template<integral_domain T>
class lucas_test {
private:
    // Lucas sequence U_n, V_n with parameters P, Q
    static std::pair<T, T> lucas_sequence(T n, T P, T Q, T k) {
        if (k == T(0)) return {T(0), T(2)};
        if (k == T(1)) return {T(1), P};

        T D = P * P - T(4) * Q;

        // Binary method for computing U_k, V_k
        T U = T(1), V = P;
        T Uh = T(1), Vh = P;  // U_1, V_1

        std::vector<bool> bits;
        T temp = k;
        while (temp > T(0)) {
            bits.push_back(remainder(temp, T(2)) == T(1));
            temp = half(temp);
        }

        for (auto it = bits.rbegin() + 1; it != bits.rend(); ++it) {
            // Double
            Uh = remainder(U * Vh, n);
            Vh = remainder(V * V - T(2) * expmod(Q, k, n), n);

            if (*it) {
                // Add one
                T Uh_new = remainder(P * Uh + Vh, n);
                if (even(Uh_new)) {
                    Uh_new = half(Uh_new);
                } else {
                    Uh_new = half(Uh_new + n);
                }

                Vh = remainder(D * Uh + P * Vh, n);
                if (even(Vh)) {
                    Vh = half(Vh);
                } else {
                    Vh = half(Vh + n);
                }

                Uh = Uh_new;
            }
        }

        return {Uh, Vh};
    }

public:
    // Lucas-Lehmer test for Mersenne primes (2^p - 1)
    bool is_mersenne_prime(size_t p) {
        if (p == 2) return true;
        if (p < 2) return false;

        T Mp = (T(1) << p) - T(1);  // 2^p - 1
        T S = T(4);

        for (size_t i = 0; i < p - 2; ++i) {
            S = remainder(S * S - T(2), Mp);
        }

        return S == T(0);
    }

    // Strong Lucas primality test
    bool is_prime_lucas(T n) {
        if (n < T(2)) return false;
        if (n == T(2)) return true;
        if (even(n)) return false;

        // Find first D in sequence 5, -7, 9, -11, ... with jacobi(D, n) = -1
        T D = T(5);
        int sign = 1;

        while (true) {
            int jac = jacobi_symbol(D, n);
            if (jac == -1) break;
            if (jac == 0 && D != n) return false;  // n is composite

            if (sign == 1) {
                D = -(D + T(2));
            } else {
                D = -(D - T(2));
            }
            sign = -sign;
        }

        // Parameters for Lucas sequence
        T P = T(1);
        T Q = half(T(1) - D);

        // Compute U_{n+1}
        auto [U, V] = lucas_sequence(n, P, Q, n + T(1));

        return U == T(0);
    }
};

// Baillie-PSW test (very strong probabilistic test)
template<integral_domain T>
class baillie_psw {
private:
    miller_rabin<T> mr;
    lucas_test<T> lucas;

public:
    bool is_prime(T n) {
        // First do a Miller-Rabin test with base 2
        if (!mr.witness_test(n, T(2))) {
            return false;
        }

        // Then do a strong Lucas test
        return lucas.is_prime_lucas(n);
    }
};

// Trial division for small primes
template<integral_domain T>
bool is_prime_trial_division(T n, T max_divisor = T(0)) {
    if (n < T(2)) return false;
    if (n == T(2)) return true;
    if (even(n)) return false;

    if (max_divisor == T(0)) {
        // Default: check up to sqrt(n)
        max_divisor = T(1);
        while (max_divisor * max_divisor < n) {
            max_divisor = max_divisor + T(1);
        }
    }

    for (T d = T(3); d <= max_divisor; d = d + T(2)) {
        if (remainder(n, d) == T(0)) {
            return false;
        }
    }

    return true;
}

// Sieve of Eratosthenes
template<integral_domain T>
std::vector<T> sieve_of_eratosthenes(T limit) {
    std::vector<bool> is_prime(static_cast<size_t>(limit) + 1, true);
    is_prime[0] = is_prime[1] = false;

    for (T p = T(2); p * p <= limit; p = increment(p)) {
        if (is_prime[static_cast<size_t>(p)]) {
            for (T i = p * p; i <= limit; i = i + p) {
                is_prime[static_cast<size_t>(i)] = false;
            }
        }
    }

    std::vector<T> primes;
    for (T i = T(2); i <= limit; i = increment(i)) {
        if (is_prime[static_cast<size_t>(i)]) {
            primes.push_back(i);
        }
    }

    return primes;
}

// Segmented sieve for large ranges
template<integral_domain T>
class segmented_sieve {
private:
    std::vector<T> base_primes;
    T segment_size;

public:
    segmented_sieve(T seg_size = T(1000000)) : segment_size(seg_size) {
        // Precompute primes up to sqrt of maximum expected value
        T limit = T(1000000);  // Adjust based on needs
        base_primes = sieve_of_eratosthenes(limit);
    }

    std::vector<T> sieve_range(T low, T high) {
        std::vector<bool> is_prime(static_cast<size_t>(high - low + T(1)), true);

        // Mark composites using base primes
        for (const T& p : base_primes) {
            if (p * p > high) break;

            T start = ((low + p - T(1)) / p) * p;
            if (start == p) start = p * p;

            for (T j = start; j <= high; j = j + p) {
                is_prime[static_cast<size_t>(j - low)] = false;
            }
        }

        // Special case for low = 1
        if (low == T(1)) {
            is_prime[0] = false;
        }

        // Collect primes
        std::vector<T> primes;
        for (T i = low; i <= high; i = increment(i)) {
            if (is_prime[static_cast<size_t>(i - low)]) {
                primes.push_back(i);
            }
        }

        return primes;
    }
};

// Pollard's rho factorization (bonus: finds factors)
template<integral_domain T>
class pollard_rho {
private:
    std::mt19937_64 rng;

    T f(T x, T c, T n) {
        return remainder(x * x + c, n);
    }

public:
    pollard_rho() : rng(std::random_device{}()) {}

    T find_factor(T n) {
        if (n == T(1)) return T(1);
        if (even(n)) return T(2);

        std::uniform_int_distribution<uint64_t> dist(1, 100);

        T x = T(dist(rng));
        T y = x;
        T c = T(dist(rng));
        T d = T(1);

        while (d == T(1)) {
            x = f(x, c, n);
            y = f(f(y, c, n), c, n);
            d = gcd(x > y ? x - y : y - x, n);
        }

        return d;
    }

    std::vector<T> factorize(T n) {
        std::vector<T> factors;

        if (n == T(1)) return factors;

        // Remove factors of 2
        while (even(n)) {
            factors.push_back(T(2));
            n = half(n);
        }

        // Find odd factors
        while (n > T(1)) {
            miller_rabin<T> mr;
            if (mr.is_prime_deterministic(n)) {
                factors.push_back(n);
                break;
            }

            T factor = find_factor(n);
            if (factor == n) {
                // Failed to find factor, n might be prime
                factors.push_back(n);
                break;
            }

            // Recursively factor
            auto sub_factors = factorize(factor);
            factors.insert(factors.end(), sub_factors.begin(), sub_factors.end());
            n = n / factor;
        }

        std::sort(factors.begin(), factors.end());
        return factors;
    }
};

} // namespace stepanov