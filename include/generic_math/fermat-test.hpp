#include <random>
#include <cmath>

using std::uniform_random_bit_generator;

/**
 * urbg models the concept of a uniform random bit generator.
 * T models the concept of a semi-group
 * 
 * Let A denote the event that n is prime, A' denote the event that n is
 * not prime, and T denote the event that all k fermat tests are positive.
 * Then,
 *     P(T|A) = 1
 * and
 *     0 < P(T|A') <= 2^(-k).
 * 
 * The probabilistic primality test returns true (n is prime) if T
 * is observed and otherwise returns false.
 * 
 * The probability P(T|A) is the true positive rate, which is 1, and therefore
 * the false negative rate is 0. The probability P(T|A') is the false positive
 * rate, denoted by fpr, and is bounded by
 *     0 < fpr(k) <= 2^(-k).
 * Thus, the true negative rate is defined as
 *     tnr(k) := 1-fpr(k).
 * 
 * Increasing the number k of trials, we may achieve an arbitrarily small false
 * positive rate, i.e., lim k->infinity fpr(k) -> 0. The smaller the false
 * positive rate for a given k, the more accurate the algorithm. We assume the
 * worst-case scenario, fpr(k) = 2^(-k).
 * 
 * The positive predictive value, denoted by ppv, is the probability that n is
 * prime given that the primarlity test returns true. It is given by
 *     ppv(k) := P(A|T)
 *             = P(A,T)/P(T)
 *             = P(A,T)/(P(A,T) + P(A',T)).
 *             = P(T|A)P(A) / (P(T|A)P(A) + P(T|A')P(A'))
 *             = P(A) / (P(A) + P(A') fpr(k))
 * and therefore
 *      ppv(k) = P(A) / (P(A) + 2^(-k) P(A')).
 * 
 * If we assume that a number n has an a prior probability p(n) = 1 / ln(n) of
 * being prime (replace P(A) with pn(n)), then
 *     ppv(k|n) = p(n) / (p(n) + fpr(k) * (1-p(n))).
 *
 * The limit of ppv(k|n) as k goes to infinity is 0 and the limit as n goes
 * to infinity is 1, which is in agreement with intuition.
 * 
 * The negative predictive value, denoted by npv, is the probability that n is
 * not prime given that the primarlity test returns false. We have established
 * that this occurs with probability 1.
 */
template <typename T, typename UniformSampler>
approximate<bool> primality_test(
    T n,
    size_t ntrials,
    uniform_random_bit_generator & urbg,
    UniformSampler u = UniformSampler{})
{
    if (ntrails == 0) // use the a priori probability that a number n is prime
        return approximate<bool>{false,1/log(n)};
        // observe that ntrails == 0 is a special case which yields a
        // false negative rate of fnr(n) := 1/log(n).
        // we return false since most numbers are not prime.

    for (size_t i = 0; i < ntrials; ++i)
    {
        T a = u(urbg);
        if (!(expmod(a,n,n) == a)) // fermat test: a^n mod n == a
            return approximate<bool>{false,1};
    }

    auto fpr = pow(2,-(long double)ntrials);
    return approximate<bool>{true,fpr};
}

template <typename T, typename UniformSampler>
approximate<bool> primality_test(
    T n,
    long double fpr,
    uniform_random_bit_generator & urbg,
    UniformSampler u = UniformSampler{})
{
    auto ntrials = static_cast<size_t>(ceil(
        log2(1-fpr) - log2(fpr) + log2(log(n)-1)));

    return primality_test(n, ntrials, urbg, u);

}
