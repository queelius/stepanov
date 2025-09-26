#pragma once

#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>
#include "concepts.hpp"
#include "math.hpp"
#include "expmod.hpp"

namespace stepanov {

/**
 * Fast Fourier Transform for Arbitrary Rings
 *
 * Generalizes FFT beyond complex numbers to:
 * - Number Theoretic Transform (NTT) for modular arithmetic
 * - Polynomial multiplication over any ring with primitive roots
 * - Convolution in various algebraic structures
 *
 * Key insight: FFT works in any ring with a principal nth root of unity.
 * This includes finite fields Z/pZ where p = kn + 1 (primitive root exists).
 */

// Check if value is power of 2
constexpr bool is_power_of_2(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Complex FFT - the classical version
template<typename T>
requires std::floating_point<T>
class complex_fft {
private:
    using complex_t = std::complex<T>;
    static constexpr T PI = T(3.14159265358979323846);

    // Bit reversal permutation
    static void bit_reverse(std::vector<complex_t>& a) {
        size_t n = a.size();
        for (size_t i = 1, j = 0; i < n; ++i) {
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                std::swap(a[i], a[j]);
            }
        }
    }

public:
    // Forward FFT
    static std::vector<complex_t> fft(std::vector<complex_t> a, bool inverse = false) {
        size_t n = a.size();
        if (!is_power_of_2(n)) {
            throw std::invalid_argument("FFT size must be power of 2");
        }

        bit_reverse(a);

        for (size_t len = 2; len <= n; len <<= 1) {
            T angle = 2 * PI / len * (inverse ? -1 : 1);
            complex_t wlen(std::cos(angle), std::sin(angle));

            for (size_t i = 0; i < n; i += len) {
                complex_t w(1);
                for (size_t j = 0; j < len / 2; ++j) {
                    complex_t u = a[i + j];
                    complex_t v = a[i + j + len/2] * w;
                    a[i + j] = u + v;
                    a[i + j + len/2] = u - v;
                    w *= wlen;
                }
            }
        }

        if (inverse) {
            for (auto& x : a) {
                x /= T(n);
            }
        }

        return a;
    }

    // Polynomial multiplication using FFT
    static std::vector<T> multiply_polynomials(const std::vector<T>& p,
                                               const std::vector<T>& q) {
        size_t result_size = p.size() + q.size() - 1;
        size_t n = 1;
        while (n < result_size) n <<= 1;

        std::vector<complex_t> a(n), b(n);
        for (size_t i = 0; i < p.size(); ++i) a[i] = p[i];
        for (size_t i = 0; i < q.size(); ++i) b[i] = q[i];

        a = fft(a);
        b = fft(b);

        for (size_t i = 0; i < n; ++i) {
            a[i] *= b[i];
        }

        a = fft(a, true);

        std::vector<T> result(result_size);
        for (size_t i = 0; i < result_size; ++i) {
            result[i] = std::round(a[i].real());
        }

        return result;
    }
};

// Number Theoretic Transform (NTT) - FFT in modular arithmetic
template<typename T>
requires integral_domain<T>
class ntt {
private:
    T mod;
    T primitive_root;

    // Find primitive root modulo p
    static T find_primitive_root(T p) {
        // For common NTT primes
        if (p == T(998244353)) return T(3);
        if (p == T(1004535809)) return T(3);
        if (p == T(469762049)) return T(3);

        // General case - brute force for small primes
        std::vector<T> factors;
        T phi = p - T(1);
        T temp = phi;

        // Find prime factors of phi(p) = p-1
        for (T i = T(2); i * i <= temp; i = i + T(1)) {
            if (remainder(temp, i) == T(0)) {
                factors.push_back(i);
                while (remainder(temp, i) == T(0)) {
                    temp = quotient(temp, i);
                }
            }
        }
        if (temp > T(1)) {
            factors.push_back(temp);
        }

        // Test candidates
        for (T g = T(2); g < p; g = g + T(1)) {
            bool is_primitive = true;
            for (const T& factor : factors) {
                if (expmod(g, quotient(phi, factor), p) == T(1)) {
                    is_primitive = false;
                    break;
                }
            }
            if (is_primitive) {
                return g;
            }
        }

        return T(0);  // Not found
    }

    void bit_reverse(std::vector<T>& a) {
        size_t n = a.size();
        for (size_t i = 1, j = 0; i < n; ++i) {
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                std::swap(a[i], a[j]);
            }
        }
    }

public:
    ntt(T modulus) : mod(modulus), primitive_root(find_primitive_root(modulus)) {
        if (primitive_root == T(0)) {
            throw std::invalid_argument("No primitive root found for given modulus");
        }
    }

    // Forward NTT
    std::vector<T> transform(std::vector<T> a, bool inverse = false) {
        size_t n = a.size();
        if (!is_power_of_2(n)) {
            throw std::invalid_argument("NTT size must be power of 2");
        }

        // Check if n-th root of unity exists
        if (remainder(mod - T(1), T(n)) != T(0)) {
            throw std::invalid_argument("n-th root of unity doesn't exist for this modulus");
        }

        bit_reverse(a);

        // nth root of unity
        T wn = expmod(primitive_root, quotient(mod - T(1), T(n)), mod);
        if (inverse) {
            wn = expmod(wn, mod - T(2), mod);  // Multiplicative inverse
        }

        for (size_t len = 2; len <= n; len <<= 1) {
            T wlen = expmod(wn, T(n / len), mod);

            for (size_t i = 0; i < n; i += len) {
                T w = T(1);
                for (size_t j = 0; j < len / 2; ++j) {
                    T u = a[i + j];
                    T v = remainder(a[i + j + len/2] * w, mod);
                    a[i + j] = remainder(u + v, mod);
                    a[i + j + len/2] = remainder(u - v + mod, mod);
                    w = remainder(w * wlen, mod);
                }
            }
        }

        if (inverse) {
            T n_inv = expmod(T(n), mod - T(2), mod);  // Multiplicative inverse of n
            for (auto& x : a) {
                x = remainder(x * n_inv, mod);
            }
        }

        return a;
    }

    // Polynomial multiplication using NTT
    std::vector<T> multiply_polynomials(const std::vector<T>& p,
                                        const std::vector<T>& q) {
        size_t result_size = p.size() + q.size() - 1;
        size_t n = 1;
        while (n < result_size) n <<= 1;

        std::vector<T> a(n, T(0)), b(n, T(0));
        for (size_t i = 0; i < p.size(); ++i) a[i] = remainder(p[i], mod);
        for (size_t i = 0; i < q.size(); ++i) b[i] = remainder(q[i], mod);

        a = transform(a);
        b = transform(b);

        for (size_t i = 0; i < n; ++i) {
            a[i] = remainder(a[i] * b[i], mod);
        }

        a = transform(a, true);

        a.resize(result_size);
        return a;
    }
};

// Generic DFT for any ring with nth roots of unity
template<ring T>
class generic_dft {
private:
    std::vector<T> roots_of_unity;
    std::vector<T> inverse_roots;
    size_t n;

public:
    // Constructor takes the principal nth root of unity
    generic_dft(size_t size, T omega) : n(size) {
        roots_of_unity.resize(n);
        inverse_roots.resize(n);

        roots_of_unity[0] = T(1);
        for (size_t i = 1; i < n; ++i) {
            roots_of_unity[i] = roots_of_unity[i-1] * omega;
        }

        // Compute inverse roots (omega^(-i))
        T omega_inv = inverse(omega);  // Requires field
        inverse_roots[0] = T(1);
        for (size_t i = 1; i < n; ++i) {
            inverse_roots[i] = inverse_roots[i-1] * omega_inv;
        }
    }

    // Forward DFT using definition (O(n²) but works for any size)
    std::vector<T> dft(const std::vector<T>& a) {
        if (a.size() != n) {
            throw std::invalid_argument("Input size must match DFT size");
        }

        std::vector<T> result(n, T(0));

        for (size_t k = 0; k < n; ++k) {
            for (size_t j = 0; j < n; ++j) {
                size_t exp = (j * k) % n;
                result[k] = result[k] + a[j] * roots_of_unity[exp];
            }
        }

        return result;
    }

    // Inverse DFT
    std::vector<T> idft(const std::vector<T>& a) {
        if (a.size() != n) {
            throw std::invalid_argument("Input size must match DFT size");
        }

        std::vector<T> result(n, T(0));
        T n_inv = inverse(T(n));  // Requires field

        for (size_t k = 0; k < n; ++k) {
            for (size_t j = 0; j < n; ++j) {
                size_t exp = (j * k) % n;
                result[k] = result[k] + a[j] * inverse_roots[exp];
            }
            result[k] = result[k] * n_inv;
        }

        return result;
    }

private:
    // Helper to compute multiplicative inverse (for fields)
    T inverse(const T& a) {
        // This would need to be specialized for each field type
        return T(1) / a;  // Works for fields that support division
    }
};

// Convolution using FFT/NTT
template<typename T>
std::vector<T> convolution(const std::vector<T>& a, const std::vector<T>& b) {
    if constexpr (std::is_floating_point_v<T>) {
        return complex_fft<T>::multiply_polynomials(a, b);
    } else {
        // Use NTT with a suitable prime
        ntt<T> transform(T(998244353));
        return transform.multiply_polynomials(a, b);
    }
}

// Circular convolution
template<ring T>
std::vector<T> circular_convolution(const std::vector<T>& a,
                                    const std::vector<T>& b) {
    size_t n = a.size();
    if (b.size() != n) {
        throw std::invalid_argument("Vectors must have same size for circular convolution");
    }

    std::vector<T> result(n, T(0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i] = result[i] + a[j] * b[(i - j + n) % n];
        }
    }

    return result;
}

// 2D FFT for image processing and 2D convolutions
template<typename T>
requires std::floating_point<T>
class fft_2d {
private:
    using complex_t = std::complex<T>;

public:
    static std::vector<std::vector<complex_t>>
    transform_2d(std::vector<std::vector<complex_t>> matrix, bool inverse = false) {
        size_t rows = matrix.size();
        size_t cols = matrix[0].size();

        // Transform rows
        for (size_t i = 0; i < rows; ++i) {
            matrix[i] = complex_fft<T>::fft(matrix[i], inverse);
        }

        // Transform columns
        for (size_t j = 0; j < cols; ++j) {
            std::vector<complex_t> column(rows);
            for (size_t i = 0; i < rows; ++i) {
                column[i] = matrix[i][j];
            }

            column = complex_fft<T>::fft(column, inverse);

            for (size_t i = 0; i < rows; ++i) {
                matrix[i][j] = column[i];
            }
        }

        return matrix;
    }
};

// Bluestein's algorithm for arbitrary-size FFT
template<typename T>
requires std::floating_point<T>
class bluestein_fft {
private:
    using complex_t = std::complex<T>;
    static constexpr T PI = T(3.14159265358979323846);

public:
    static std::vector<complex_t> transform(const std::vector<complex_t>& a) {
        size_t n = a.size();

        // Find next power of 2 >= 2n-1
        size_t m = 1;
        while (m < 2 * n - 1) m <<= 1;

        // Chirp sequences
        std::vector<complex_t> chirp(n);
        for (size_t i = 0; i < n; ++i) {
            T angle = -PI * i * i / n;
            chirp[i] = complex_t(std::cos(angle), std::sin(angle));
        }

        // Prepare sequences for convolution
        std::vector<complex_t> a_chirp(m, 0), b_chirp(m, 0);

        for (size_t i = 0; i < n; ++i) {
            a_chirp[i] = a[i] * std::conj(chirp[i]);
        }

        b_chirp[0] = chirp[0];
        for (size_t i = 1; i < n; ++i) {
            b_chirp[i] = b_chirp[m - i] = chirp[i];
        }

        // Convolve using FFT
        auto a_fft = complex_fft<T>::fft(a_chirp);
        auto b_fft = complex_fft<T>::fft(b_chirp);

        for (size_t i = 0; i < m; ++i) {
            a_fft[i] *= b_fft[i];
        }

        auto conv = complex_fft<T>::fft(a_fft, true);

        // Extract result
        std::vector<complex_t> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = conv[i] * std::conj(chirp[i]);
        }

        return result;
    }
};

} // namespace stepanov