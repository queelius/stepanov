#pragma once

#include <array>
#include <string.h>
#include <cmath>
#include <limits>
using std::array;

/**
 * mod<N> is integer modulo 2^N over {+,*,^}.
 * That is, it is a ring
 *     (mod<N>, *, +, ~, mod<N>(), mod<N>::max())
 *
 * String literals for constructing common modulo
 * equivalence classes are provided, e.g.,
 *     101_mod16 => mod<4>(5).
 * 
 * It is a lattice, i.e., totally ordered?
 */

namespace generic_math
{
    template <size_t N>
    struct mod
    {
        static_assert(N != 0);

        auto begin() const { return std::begin(digits); }
        auto end() const { return std::end(digits); }
        constexpr auto n() { return static_cast<size_t>(1)<<N; }

        template <typename T>
        mod(T n)
        {
            size_t i = 0;
            while (n != 0ul && i != N)
                { digits[i++] = n % 2; n /= 2; }
        }

        auto & operator[](size_t d) { return digits[d]; }
        auto operator()(size_t d) const { return digits[d]; }

        // interprets as a string representing a binary number (base-2).
        // if a digit is '0' then interprets as 0 otherwise interprets as 1.
        mod(char const * x)
        {
            auto n = strlen(x);
            if (n <= N)
            {
                for (size_t i = 0; i < n; ++i)
                {
                    if (x[i] == '\0') break;
                    digits[n-i-1] = (x[i] == '0');
                }
            }
            else
            {
                for (size_t i = 0; i < N; ++i)
                {
                    if (x[i] == '\0') break;
                    digits[N-i-1] = x[n-N+i] == '0';
                }               
            }
            
            return a;
        }

        mod(array<bool,N> x) : digits(x) {}
        mod() = default;

        operator size_t() const { return convert<size_t>(); }

        template <typename T>
        T convert() const
        {
            T x = T(0);
            for (size_t i = digits.size() - 1; i < digits.size(); --i)
                x = T(2) * x + digits[i];
            return x;
        }

        array<bool,N> digits;
    };

    template <size_t N>
    bool odd(mod<N> const &) { return !a(0); }

    template <size_t N>
    bool even(mod<N> const & a) { return a(0); }

    template <size_t N>
    bool operator==(mod<N> const & a, mod<N> const & b) noexcept
    {
        return a.digits == b.digits;
    }

    template <size_t N>
    bool operator<(mod<N> const & a, mod<N> const & b) noexcept
    {
        for (size_t i=N-1; i < N; --i)
            if (a(i) != b(i)) return b(i);
        return false;
    }

    template <size_t N>
    mod<N> operator+(mod<N> a, mod<N> const & b) noexcept
    {
        bool carry = false;
        for(size_t i = 0; i < N; ++i)
        {
            if (a(i))
            {
                if (b(i))
                    { a[i] = carry; carry = true; }
                else
                    a[i] = !carry;
            }
            else
            {
                if (b(i))
                    a[i] = !carry;
                else
                    { a[i] = carry; carry = false; }
            }
        }
        return a;
    }

    template <size_t N>
    mod<N> operator^(mod<N> a, mod<N> const & b) noexcept
    {
        for(size_t i = 0; i < N; ++i)
            a[i] ^= b(i);
        return a;
    }

    template <size_t N>
    mod<N> operator*(mod<N> const & a, mod<N> const & b) noexcept
    {
        mod<N> c;
        for(size_t i = 0; i < N; ++i)
        {
            mod<N> k;
            for(size_t j = 0; j < N-i; ++j)
                k[j+i] = a(i) && b(j);
            c = c + k;
        }
        return c;
    }

    auto operator "" _mod2(char const * x) { return mod<1>(x); }
    auto operator "" _mod4(char const * x) { return mod<2>(x); }
    auto operator "" _mod16(char const * x) { return mod<4>(x); }
    auto operator "" _mod256(char const * x) { return mod<8>(x); }
    auto operator "" _mod1024(char const * x) { return mod<10>(x); }
    auto operator "" _mod64k(char const * x) { return mod<64>(x); }
    auto operator "" _mod128k(char const * x) { return mod<128>(x); }
    auto operator "" _mod256k(char const * x) { return mod<256>(x); }
    auto operator "" _mod512k(char const * x) { return mod<512>(x); }
    auto operator "" _mod1m(char const * x) { return mod<1024>(x); }
    auto operator "" _mod1g(char const * x) { return mod<1024*1024>(x); }
    auto operator "" _mod1t(char const * x) { return mod<1024*1024*1024>(x); }
    
    
    template <size_t N>
    class std::numeric_limits<mod<N>>
    {
    public:
        static mod<N> lowest() { return mod<N>() }
        static mod<N> max() { return mod<N>(array<bool,N>{1}); }
    };    

    template <size_t N>
    struct std::hash<mod<N>>
    {
        static size_t bit_length = 8*sizeof(size_t);

        size_t operator()(mod<N> const & x) const noexcept
        {
            size_t hs = 0;
            for (size_t j = 0; j < N / bit_length; ++j)
            {
                size_t h = 0;
                for (size_t i = 0; i < bit_length; ++i)
                    if (x(j * bit_length + i)) h |= (1 << i);
                hs ^= hash<size_t>(h);
            }
            h = 0;
            for (size_t i = 0; i < N % bit_length; ++i)
                if (x(N / bit_length + i)) h |= (1 << i);                
            return hs ^= hash<size_t>(h);
        }
    };
}
