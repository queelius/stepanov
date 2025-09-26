#pragma once

#include <vector>
#include <algorithm>
#include <functional>
#include "concepts.hpp"
#include "math.hpp"

namespace stepanov {

/**
 * Combinatorial Algorithms
 *
 * Generic implementations that work over any ring, not just integers.
 * This enables modular arithmetic, polynomial rings, matrices, etc.
 *
 * Key algorithms:
 * - Binomial coefficients with Pascal's triangle
 * - Stirling numbers (both kinds)
 * - Integer partitions
 * - Catalan numbers
 * - Bell numbers
 * - Derangements
 */

// Generic binomial coefficient computation
template<ring T>
class binomial_coefficients {
private:
    std::vector<std::vector<T>> pascal;
    size_t max_n;

public:
    explicit binomial_coefficients(size_t n) : max_n(n) {
        pascal.resize(n + 1);
        for (size_t i = 0; i <= n; ++i) {
            pascal[i].resize(i + 1);
            pascal[i][0] = pascal[i][i] = T(1);
            for (size_t j = 1; j < i; ++j) {
                pascal[i][j] = pascal[i-1][j-1] + pascal[i-1][j];
            }
        }
    }

    T get(size_t n, size_t k) const {
        if (n > max_n || k > n) return T(0);
        return pascal[n][k];
    }

    // Generate entire row
    std::vector<T> row(size_t n) const {
        if (n > max_n) return {};
        return pascal[n];
    }

    // Compute using multiplicative formula (for large n, small k)
    static T compute(size_t n, size_t k) {
        if (k > n) return T(0);
        if (k == 0 || k == n) return T(1);
        if (k > n - k) k = n - k;  // Symmetry

        T result = T(1);
        for (size_t i = 0; i < k; ++i) {
            result = result * T(n - i);
            result = result / T(i + 1);  // Requires field
        }
        return result;
    }
};

// Stirling numbers of the first kind (unsigned)
// S1(n,k) = number of permutations of n elements with k cycles
template<ring T>
class stirling_first {
private:
    std::vector<std::vector<T>> s1;

public:
    stirling_first(size_t max_n) {
        s1.resize(max_n + 1);
        for (size_t n = 0; n <= max_n; ++n) {
            s1[n].resize(n + 1, T(0));
        }

        s1[0][0] = T(1);
        for (size_t n = 1; n <= max_n; ++n) {
            for (size_t k = 1; k <= n; ++k) {
                s1[n][k] = s1[n-1][k-1] + T(n-1) * s1[n-1][k];
            }
        }
    }

    T get(size_t n, size_t k) const {
        if (n >= s1.size() || k > n) return T(0);
        return s1[n][k];
    }

    // Signed version
    T get_signed(size_t n, size_t k) const {
        T val = get(n, k);
        return ((n - k) % 2 == 0) ? val : -val;
    }
};

// Stirling numbers of the second kind
// S2(n,k) = number of ways to partition n elements into k non-empty sets
template<ring T>
class stirling_second {
private:
    std::vector<std::vector<T>> s2;

public:
    stirling_second(size_t max_n) {
        s2.resize(max_n + 1);
        for (size_t n = 0; n <= max_n; ++n) {
            s2[n].resize(n + 1, T(0));
        }

        s2[0][0] = T(1);
        for (size_t n = 1; n <= max_n; ++n) {
            s2[n][0] = T(0);
            for (size_t k = 1; k <= n; ++k) {
                s2[n][k] = T(k) * s2[n-1][k] + s2[n-1][k-1];
            }
        }
    }

    T get(size_t n, size_t k) const {
        if (n >= s2.size() || k > n) return T(0);
        return s2[n][k];
    }

    // Explicit formula using inclusion-exclusion
    static T compute(size_t n, size_t k) {
        T result = T(0);
        T sign = T(1);

        for (size_t j = 0; j <= k; ++j) {
            T binom = binomial_coefficients<T>::compute(k, j);
            T term = binom * power(T(k - j), T(n));
            result = result + sign * term;
            sign = -sign;
        }

        // Divide by k!
        for (size_t i = 1; i <= k; ++i) {
            result = result / T(i);
        }

        return result;
    }
};

// Integer partitions
// p(n,k) = number of ways to partition n into at most k parts
template<ring T>
class integer_partitions {
private:
    std::vector<std::vector<T>> partitions;

public:
    integer_partitions(size_t max_n) {
        partitions.resize(max_n + 1);
        for (size_t n = 0; n <= max_n; ++n) {
            partitions[n].resize(n + 1, T(0));
        }

        // Base cases
        for (size_t k = 0; k <= max_n; ++k) {
            partitions[0][k] = T(1);
        }

        // Recurrence: p(n,k) = p(n,k-1) + p(n-k,k)
        for (size_t n = 1; n <= max_n; ++n) {
            for (size_t k = 1; k <= n; ++k) {
                partitions[n][k] = partitions[n][k-1];
                if (n >= k) {
                    partitions[n][k] = partitions[n][k] + partitions[n-k][k];
                }
            }
        }
    }

    T get(size_t n, size_t k) const {
        if (n >= partitions.size() || k > n) return T(0);
        return partitions[n][k];
    }

    T total(size_t n) const {
        if (n >= partitions.size()) return T(0);
        return partitions[n][n];
    }

    // Generate all partitions of n
    std::vector<std::vector<size_t>> generate_all(size_t n) const {
        std::vector<std::vector<size_t>> result;
        std::vector<size_t> current;
        generate_helper(n, n, current, result);
        return result;
    }

private:
    void generate_helper(size_t n, size_t max_part,
                        std::vector<size_t>& current,
                        std::vector<std::vector<size_t>>& result) const {
        if (n == 0) {
            result.push_back(current);
            return;
        }

        for (size_t i = std::min(n, max_part); i >= 1; --i) {
            current.push_back(i);
            generate_helper(n - i, i, current, result);
            current.pop_back();
        }
    }
};

// Catalan numbers
// C_n = number of binary trees with n internal nodes
template<ring T>
class catalan_numbers {
private:
    std::vector<T> catalan;

public:
    explicit catalan_numbers(size_t max_n) {
        catalan.resize(max_n + 1);
        catalan[0] = T(1);

        // Recurrence: C_n = sum(C_i * C_{n-1-i}) for i = 0 to n-1
        for (size_t n = 1; n <= max_n; ++n) {
            catalan[n] = T(0);
            for (size_t i = 0; i < n; ++i) {
                catalan[n] = catalan[n] + catalan[i] * catalan[n-1-i];
            }
        }
    }

    T get(size_t n) const {
        if (n >= catalan.size()) return T(0);
        return catalan[n];
    }

    // Direct formula: C_n = (2n choose n) / (n+1)
    static T compute(size_t n) {
        return binomial_coefficients<T>::compute(2*n, n) / T(n + 1);
    }
};

// Bell numbers
// B_n = number of partitions of a set with n elements
template<ring T>
class bell_numbers {
private:
    std::vector<T> bell;
    stirling_second<T> s2;

public:
    explicit bell_numbers(size_t max_n) : s2(max_n) {
        bell.resize(max_n + 1);

        for (size_t n = 0; n <= max_n; ++n) {
            bell[n] = T(0);
            for (size_t k = 0; k <= n; ++k) {
                bell[n] = bell[n] + s2.get(n, k);
            }
        }
    }

    T get(size_t n) const {
        if (n >= bell.size()) return T(0);
        return bell[n];
    }

    // Generate using Bell triangle (Aitken's array)
    static std::vector<T> generate_triangle(size_t max_n) {
        std::vector<std::vector<T>> triangle(max_n + 1);
        triangle[0] = {T(1)};

        for (size_t n = 1; n <= max_n; ++n) {
            triangle[n].resize(n + 1);
            triangle[n][0] = triangle[n-1][n-1];
            for (size_t k = 1; k <= n; ++k) {
                triangle[n][k] = triangle[n][k-1] + triangle[n-1][k-1];
            }
        }

        std::vector<T> result;
        for (const auto& row : triangle) {
            if (!row.empty()) {
                result.push_back(row[0]);
            }
        }
        return result;
    }
};

// Derangements
// D_n = number of permutations of n elements with no fixed points
template<ring T>
class derangements {
private:
    std::vector<T> derang;

public:
    explicit derangements(size_t max_n) {
        derang.resize(max_n + 1);
        if (max_n >= 0) derang[0] = T(1);
        if (max_n >= 1) derang[1] = T(0);

        // Recurrence: D_n = (n-1) * (D_{n-1} + D_{n-2})
        for (size_t n = 2; n <= max_n; ++n) {
            derang[n] = T(n - 1) * (derang[n-1] + derang[n-2]);
        }
    }

    T get(size_t n) const {
        if (n >= derang.size()) return T(0);
        return derang[n];
    }

    // Inclusion-exclusion formula
    static T compute(size_t n) {
        T result = T(0);
        T factorial = T(1);
        T sign = T(1);

        for (size_t k = 0; k <= n; ++k) {
            if (k > 0) factorial = factorial * T(k);
            result = result + sign / factorial;
            sign = -sign;
        }

        // Multiply by n!
        for (size_t i = 1; i <= n; ++i) {
            result = result * T(i);
        }

        return result;
    }
};

// Fibonacci numbers (generalized to any ring)
template<ring T>
class fibonacci {
private:
    std::vector<T> fib;

public:
    fibonacci(size_t max_n, T f0 = T(0), T f1 = T(1)) {
        fib.resize(max_n + 1);
        if (max_n >= 0) fib[0] = f0;
        if (max_n >= 1) fib[1] = f1;

        for (size_t n = 2; n <= max_n; ++n) {
            fib[n] = fib[n-1] + fib[n-2];
        }
    }

    T get(size_t n) const {
        if (n >= fib.size()) return T(0);
        return fib[n];
    }

    // Matrix exponentiation method for large n
    static T compute_large(size_t n) {
        if (n == 0) return T(0);
        if (n == 1) return T(1);

        // Use 2x2 matrix [[1,1],[1,0]]^n
        T a = T(1), b = T(1), c = T(1), d = T(0);
        T ta, tb, tc, td;

        size_t m = n - 1;
        while (m > 0) {
            if (m % 2 == 1) {
                ta = a * T(1) + b * T(1);
                tb = a * T(1) + b * T(0);
                tc = c * T(1) + d * T(1);
                td = c * T(1) + d * T(0);
                a = ta; b = tb; c = tc; d = td;
            }

            ta = a * a + b * c;
            tb = b * (a + d);
            tc = c * (a + d);
            td = b * c + d * d;
            a = ta; b = tb; c = tc; d = td;

            m /= 2;
        }

        return a;
    }
};

// Multinomial coefficients
template<ring T>
T multinomial_coefficient(size_t n, const std::vector<size_t>& k) {
    T result = T(1);
    size_t sum = 0;

    // Compute n! / (k1! * k2! * ... * km!)
    for (size_t i = 0; i < k.size(); ++i) {
        for (size_t j = 0; j < k[i]; ++j) {
            result = result * T(sum + j + 1);
            result = result / T(j + 1);
        }
        sum += k[i];
    }

    return result;
}

// Eulerian numbers
// A(n,k) = number of permutations of n elements with k ascents
template<ring T>
class eulerian_numbers {
private:
    std::vector<std::vector<T>> euler;

public:
    eulerian_numbers(size_t max_n) {
        euler.resize(max_n + 1);
        for (size_t n = 0; n <= max_n; ++n) {
            euler[n].resize(n + 1, T(0));
        }

        euler[0][0] = T(1);
        for (size_t n = 1; n <= max_n; ++n) {
            for (size_t k = 0; k < n; ++k) {
                euler[n][k] = T(k + 1) * euler[n-1][k] + T(n - k) * euler[n-1][k-1];
            }
        }
    }

    T get(size_t n, size_t k) const {
        if (n >= euler.size() || k >= n) return T(0);
        return euler[n][k];
    }
};

} // namespace stepanov