
/**
 * T models the Euclidean domain if:
 * 
 *     (1) T models an integral domain with additive identity T(0).
 * 
 *     (2) T has operations
 *             quotient : T -> T -> T
 *         and
 *             remainder : T -> T -> T
 *         such that if b != T(0),
 *             a == quotient(a,b)*b + remainder(a,b).
 * 
 *     (3) E has a non-negative norm
 *             norm : T -> N
 *         satisfying the following:
 *               (i) If norm(a) == T(0), then a == T(0).
 *              (ii) If b != T(0), then norm(a*b) >= norm(a).
 *             (iii) norm(remainder(a,b)) <= norm(b).
 * 
 * We can only use concepts to automatically check that T satisfies the
 * syntactic structure. Axiomatic or algebraic structure, such as (i)-(iii),
 * are not possible to autoatmically check using concepts.
 * 
 * We check only for the required overload set
 *     quotient : T -> T -> T,
 *     remainder : T -> T -> T,
 *     norm : T -> N
 *     operator + : T -> T -> T
 *     operator * : T -> T -> T
 * and provide the rest as type traits, i.e., T::integral_domain is defined.
 * 
 * Additionally,
 *     swap : (&T,&T) -> void
 * must be defined.
 */

using std::swap;

/*template<typename T>
concept euclidean_domain = requires(T a)
{
    { T::integral_domain };
    { swap(a,a) };
    { remainder(a,a) } -> T;
    { quotient(a,a) } -> T;
    { T(0) };
    { norm(a) } -> T;
    { a + a } -> T;
    { a * a } -> T; 
};*/

template <euclidean_domain T>
T gcd(T a, T b)
{
    while (b != T(0))
    {
        a = remainder(a, b);
        swap(a, b);
    }
    return a;
}