#pragma once

#include <limits>
#include <string.h>
#include <utility>
#include <algorithm>
#include "math.hpp"
using std::pair;
using std::numeric_limits;

/**
 * bounded_nat<N> uses a binary encoded decimal representation of unsigned
 * integers (natural numbers) using a fixed-size array of N digits of type
 * digit_type. Thus, bounded_nat<N> may represent any value in the set
 * {0,1,...,d^N} where d is the cardinality of digit_type.
 */
template <size_t N>
struct bounded_nat
{
    using digit_type = unsigned char;

    constexpr static auto max_digit = static_cast<int>(numeric_limits<digit_type>::max());
    
    bounded_nat() :
        memset(digits, 0, N * sizeof(digit_type)) {}

    bounded_nat(nat const & copy) :
        memcpy(digits, copy.digits, N * sizeof(digit_type)) {}

    // msd is digits[N-1].
    // lsd is digits[0].
    digit_type digits[N];
};

template <size_t N>
bool operator>(bounded_nat<N> const & lhs, bounded_nat<N> const & rhs)
{
    return rhs < lhs;
}

template <size_t N>
bool operator>=(bounded_nat<N> const & lhs, bounded_nat<N> const & rhs)
{
    return rhs <= lhs;
}

template <size_t N>
bool operator<(bounded_nat<N> const & lhs, bounded_nat<N> const & rhs)
{
	for (size_t i = N-1; i < N; --i)
	{
		if (lhs.digits[i] != rhs.digits[i])
			return lhs.digits[i] < rhs.digits[i];
	}
    return false;
}

template <size_t N>
bool operator<=(bounded_nat<N> const & lhs, bounded_nat<N> const & rhs)
{
	for (size_t i = N-1; i < N; --i)
	{
		if (lhs.digits[i] != rhs.digits[i])
            return lhs.digits[i] < rhs.digits[i];
    }
    return true;
}

template <size_t N>
bool operator==(bounded_nat<N> const & lhs, bounded_nat<N> const & rhs)
{
	for (size_t i = 0; i < N; ++i)
	{
		if (lhs.digits[i] != rhs.digits[i])
			return false;
	}
	return true;
}

template <size_t N>
bool operator!=(bounded_nat<N> const & lhs, bounded_nat<N> const & rhs)
{
	for (size_t i = 0; i < N; ++i)
	{
		if (lhs.digits[i] == rhs.digits[i])
			return false;
	}
	return true;

}

template <size_t N>
bool even(bounded_nat<N> const & n) { return n.digits[0] % 2 == 0; }

/**
 * twice and half are sufficient to implement other operations, like
 * multiplication, division, and so on. Since these operations are very
 * efficient, we employ generic algorithms for these other operations that may
 * be implemented with respect to these two operations.
 */
template <size_t N>
bounded_nat<N> twice(bounded_nat<N> n)
{
    for (size_t i = N-1; ; ++i)
    {
        n.digits[i] >>= 1;
	    if (i == 0)
            break;
	    n.digits[i] |= (n.digits[i-1] << 8*sizeof(digit_type));
    }
    return n;
}

template <size_t N>
bounded_nat<N> half(bounded_nat<N> n)
{
    for (size_t i = 0; i < N; ++i)
    {
        n.digits[i] <<= 1;
	if (i != N-1)
	    n.digits[i] |= (n.digits[i+1] >> 8*sizeof(digit_type));
    }
    return n;
}

template <size_t N>
auto increment_with_carry(bounded_nat<N> n)
{
    using digit_type = typename bounded_nat<N>::digit_type;

    digit_type carry = (digit_type)0;
    for (size_t i = 0; i < N; ++i)
    {
        if (n.digits[i] == typename bounded_nat<N>::max_digit)
        {
            n.digits[i] = carry;
            carry = (digit_type)1;
        }
        else
        {
            ++n.digits[i];
            if (carry)
            {
                if (n.digits[i] == typename bounded_nat<N>::max_digit)
                    n.digits[i] = 0;
                else
                {
                    ++n.digits[i];
                    carry = (digit_type)0;
                }
            }
        }
    }
    return make_pair(n,carry);
}

template <size_t N>
bounded_nat<N> increment(bounded_nat<N> n)
{
    return increment_with_carry(n).first;
}

template <size_t N>
bounded_nat<N> operator*(bounded_nat<N> lhs, bounded_nat<N> rhs)
{
    return product(lhs,rhs);
}

template <size_t N>
bounded_nat<N> operator+(bounded_nat<N> lhs, bounded_nat<N> rhs)
{
    return sum(lhs,rhs);
}


/*
// (product, carry)
template <size_t N>
pair<nat<N>,int> mult(nat<N> x, nat<N> y)
{
    nat<N> z;
    int carry = 0;
    for (size_t i = 0; i < N; ++i)
    {
        nat<N> tmp;
        int c = 0;
        for (size_t j = 0; j < N; ++j)
        {
            auto r = (int)x.digits[i] * (int)y.digits[j] + c;
            tmp.digits[i] = static_cast<unsigned char>(r / N);
            c = r % N;
        }
        auto s = sum(z,tmp);
        z = s.first;

        // if s.second != 0 then we have overflowed. we could either let this
        // be undefined or roll over like unsigned int. we let it roll over,
        // i.e., a ring mod 2^(8*N)
        z = z + s.second;
    }
    return make_pair{z,c};
}
*/

/*
template <size_t N>
nat<N> sum(nat<N> x, nat<N> y)
{
    nat<N> z;
    int c = 0;
    for (size_t i = 0; i < N; ++i)
    {
        auto r = (int)x.digits[i] + (int)y.digits[i] + c;
        z.digits[i] = static_cast<unsigned char>(r / N);
        c = r % N;
    }
    return make_pair(z,c);    
}
*/


