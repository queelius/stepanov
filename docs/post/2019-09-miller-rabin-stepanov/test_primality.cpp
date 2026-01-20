#include <gtest/gtest.h>
#include "primality.hpp"

using namespace primality;

// =============================================================================
// mod_pow tests
// =============================================================================

TEST(ModPowTest, Basic) {
    EXPECT_EQ(mod_pow(2, 10, 1000), 24);   // 1024 mod 1000
    EXPECT_EQ(mod_pow(3, 5, 100), 43);     // 243 mod 100
    EXPECT_EQ(mod_pow(7, 0, 13), 1);       // a^0 = 1
    EXPECT_EQ(mod_pow(5, 1, 100), 5);      // a^1 = a
}

TEST(ModPowTest, FermatLittleTheorem) {
    // a^(p-1) ≡ 1 (mod p) for prime p, gcd(a,p) = 1
    EXPECT_EQ(mod_pow(2LL, 6LL, 7LL), 1);
    EXPECT_EQ(mod_pow(3LL, 10LL, 11LL), 1);
    EXPECT_EQ(mod_pow(5LL, 12LL, 13LL), 1);
}

TEST(ModPowTest, LargeExponents) {
    // 2^100 mod 1000000007
    int64_t result = mod_pow(2LL, 100LL, 1000000007LL);
    EXPECT_EQ(result, 976371285);
}

// =============================================================================
// Error bound calculation
// =============================================================================

TEST(ErrorBoundTest, WitnessesForError) {
    // For error ≤ 0.25, need 1 witness
    EXPECT_EQ(witnesses_for_error(0.25), 1);
    EXPECT_EQ(witnesses_for_error(0.5), 1);

    // For error ≤ 1/16 = 0.0625, need 2 witnesses
    EXPECT_EQ(witnesses_for_error(0.0625), 2);

    // For error ≤ 10^-6, need about 10 witnesses
    EXPECT_GE(witnesses_for_error(1e-6), 9);
    EXPECT_LE(witnesses_for_error(1e-6), 11);

    // For error ≤ 10^-12, need about 20 witnesses
    EXPECT_GE(witnesses_for_error(1e-12), 19);
    EXPECT_LE(witnesses_for_error(1e-12), 21);
}

TEST(ErrorBoundTest, ErrorBoundComputation) {
    EXPECT_DOUBLE_EQ(error_bound(1), 0.25);
    EXPECT_DOUBLE_EQ(error_bound(2), 0.0625);
    EXPECT_NEAR(error_bound(10), std::pow(0.25, 10), 1e-15);
}

// =============================================================================
// is_prime tests
// =============================================================================

TEST(IsPrimeTest, SmallPrimes) {
    EXPECT_TRUE(is_prime(2LL));
    EXPECT_TRUE(is_prime(3LL));
    EXPECT_TRUE(is_prime(5LL));
    EXPECT_TRUE(is_prime(7LL));
    EXPECT_TRUE(is_prime(11LL));
    EXPECT_TRUE(is_prime(13LL));
    EXPECT_TRUE(is_prime(17LL));
    EXPECT_TRUE(is_prime(19LL));
    EXPECT_TRUE(is_prime(23LL));
}

TEST(IsPrimeTest, SmallComposites) {
    EXPECT_FALSE(is_prime(0LL));
    EXPECT_FALSE(is_prime(1LL));
    EXPECT_FALSE(is_prime(4LL));
    EXPECT_FALSE(is_prime(6LL));
    EXPECT_FALSE(is_prime(8LL));
    EXPECT_FALSE(is_prime(9LL));
    EXPECT_FALSE(is_prime(10LL));
    EXPECT_FALSE(is_prime(15LL));
}

TEST(IsPrimeTest, MediumPrimes) {
    EXPECT_TRUE(is_prime(101LL));
    EXPECT_TRUE(is_prime(1009LL));
    EXPECT_TRUE(is_prime(10007LL));
    EXPECT_TRUE(is_prime(100003LL));
}

TEST(IsPrimeTest, LargePrimes) {
    // Well-known primes
    EXPECT_TRUE(is_prime(1000000007LL));  // Common modulus in competitive programming
    EXPECT_TRUE(is_prime(1000000009LL));
}

TEST(IsPrimeTest, CarmichaelNumbers) {
    // Carmichael numbers fool Fermat's test but not Miller-Rabin
    // 561 = 3 × 11 × 17 is the smallest Carmichael number
    EXPECT_FALSE(is_prime(561LL));
    EXPECT_FALSE(is_prime(1105LL));  // 5 × 13 × 17
    EXPECT_FALSE(is_prime(1729LL));  // 7 × 13 × 19 (Hardy-Ramanujan number)
}

TEST(IsPrimeTest, MersennePrimes) {
    // 2^p - 1 for prime p
    EXPECT_TRUE(is_prime(3LL));       // 2^2 - 1
    EXPECT_TRUE(is_prime(7LL));       // 2^3 - 1
    EXPECT_TRUE(is_prime(31LL));      // 2^5 - 1
    EXPECT_TRUE(is_prime(127LL));     // 2^7 - 1
    EXPECT_TRUE(is_prime(8191LL));    // 2^13 - 1
    EXPECT_TRUE(is_prime(131071LL));  // 2^17 - 1
}

// =============================================================================
// is_prime_with_error tests
// =============================================================================

TEST(IsPrimeWithErrorTest, ReturnsErrorBound) {
    auto result = is_prime_with_error(1000000007LL);
    EXPECT_TRUE(result.probably_prime);
    EXPECT_GT(result.error_bound, 0.0);
    EXPECT_LT(result.error_bound, 1e-11);  // Should be very small
    EXPECT_GT(result.witnesses_tested, 0);
}

TEST(IsPrimeWithErrorTest, CompositeHasZeroError) {
    auto result = is_prime_with_error(100LL);
    EXPECT_FALSE(result.probably_prime);
    EXPECT_DOUBLE_EQ(result.error_bound, 0.0);  // No error for composite
}

TEST(IsPrimeWithErrorTest, CustomErrorBound) {
    // With higher error tolerance, should use fewer witnesses
    auto loose = is_prime_with_error(1000000007LL, 0.1);
    auto tight = is_prime_with_error(1000000007LL, 1e-20);

    EXPECT_LT(loose.witnesses_tested, tight.witnesses_tested);
    EXPECT_GT(loose.error_bound, tight.error_bound);
}

TEST(IsPrimeWithErrorTest, BoolConversion) {
    auto prime_result = is_prime_with_error(17LL);
    auto composite_result = is_prime_with_error(18LL);

    // Should work in boolean context
    if (prime_result) {
        EXPECT_TRUE(true);
    } else {
        FAIL() << "17 should be prime";
    }

    if (composite_result) {
        FAIL() << "18 should be composite";
    } else {
        EXPECT_TRUE(true);
    }
}
