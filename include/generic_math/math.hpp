#pragma once

template <typename T>
T product(T const & lhs, T const & rhs);

template <typename T>
T square(T const & x);

template <typename T>
T power(T const & base, T const & exp);

template <typename T>
T sum(T const & lhs, T const & rhs);

/**
 * T models an object for which the following is in the overload set:
 * 
 *     twice : T -> T (may be a partial function),
 *     half : T -> T (may be a partial function),
 *     even : T -> bool,
 *     decrement : T -> T (may be a partial function)
 *     T(0) constructs additive identity
 *     T(1) constructs multiplicative identity
 */
template <typename T>
T product(T const & lhs, T const & rhs)
{
    if (rhs == T(0))
	    return T(0);

    if (rhs == T(1))
	    return x;

    return even(rhs) ?
        twice(product(lhs,half(rhs))) :
        sum(x,product(lhs,decrement(rhs)));
}

template <typename T>
T square(T const & x)
{
    if (x == T(0))
	    return T(0);

    if (x == T(1))
	    return x;

    return even(x) ?
        twice(twice(square(half(x)))) :
        sum(x,square(decrement(x)));
}

template <typename T>
T power(T const & base, T const & exp)
{
    if (exp == T(0))
	    return T(1);

    if (exp == T(1))
	    return base;

    return even(exp) ?
        square(power(base,half(exp))) :
        product(base,power(base,decrement(exp)));
}

template <typename T>
T sum(T const & lhs, T const & rhs)
{
    if (lhs == T(0))
	    return rhs;

    if (rhs == T(0))
	    return lhs;

    auto s = twice(sum(half(lhs),half(rhs)));
    if (!even(lhs))
	    s = increment(s);

    if (!even(rhs))
	    s = increment(s);

    return s;
}
