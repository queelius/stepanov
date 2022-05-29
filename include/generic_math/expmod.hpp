#pragma once

/**
 * expmod is a trinary function of type
 *     (T,T,T) -> T
 * that models the compound operation
 *     expmod(base,exp,m) := base^exp mod m,
 * where T models the concept of a ring
 *     (T(0),T(1),*,+),
 * such as data types that model the conept of integers
 * equipped with multiplication and addition operators.
 * 
 * In the invokation of the procedure
 *     expmod(base,exp,m),
 * values of T that are much larger than m are not necessary,
 * which is computationally convenient.
 *
 * T has some additional constraints that must be satisfied.
 * 
 * The square function,
 *     square : T -> T,
 * satisfies
 *     assert(square(k) == k*k).
 * 
 * The floor_half function,
 *     floor_half : T -> T,
 * which conceptually models the composition (floor . half).
 * Essentially, if even(x), then floor_half(x) == half(x)
 * and otherwose if odd(x), then floor_half(x) == half(decrement(x))
 * where half(x) := quotient(x,2).
 * 
 * The even function
 *     even : T -> bool.
 * If T is an integer type, then clearly even satisfies
 *     assert(even(2*k) == true && even(2*k+1) == false).
 * However, we must generalize the notion of even over
 * rings. Let I be an ideal of T whose index is 2.
 * Elements of the coset T(0)+I are even, while elements
 * of the coset T(1)+I are odd.
 * 
 * The zero constructor,
 *     T(0),
 * represents the additive identity,
 *     assert(T(0)+T(k) == T(k) && T(k)+T(0) == T(k)).
 * 
 * The unit constructor,
 *     T(1),
 * represents the multiplicative identity,
 *     assert(T(1)*T(k) == T(k) && T(k)*T(1) == T(k)).
 * 
 * The remainder function,
 *     remainder : (T,T) -> T
 * satisfies
 *     assert(remainder(a,b) == a - b * quotient(a,b)).
 * Since computing remainders also generally computes
 * quotients, by the principle that no information
 * should be thrown away by a procedure, there should
 * probably be a function
 *     quotient_remainder : (T,T) -> pair<T,T>
 * that returns the quotient and remainder.
 * 
 * The multiplication binary operator,
 *     operator * : (T,T) - T.
 * 
 * The decrement unary operator,
 *     decrement : T -> T.
 * 
 * If T models an integer type, its existing computational basis most likely
 * may be used to trivially implement the above overload set. However, even if
 * this is the case, the specific functions in this overload set may sometimes
 * be more efficiently computed in some other way, e.g., if T is a binary
 * encoded decimal, then
 *     floor_half : T -> T
 * may simply shift the bits by one to the right.
 */
template <typename T>
T expmod(T base, T exp, T m)
{
    if (exp == T(0))
        return T(1);

    if (even(exp))
        return remainder(
            square(expmod(base, floor_half(exp), m)),
            m)

    return remainder(base * expmod(base, decrement(exp), m))
}
