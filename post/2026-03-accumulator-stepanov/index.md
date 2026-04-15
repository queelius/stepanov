---
title: Streaming Statistics, One Monoid at a Time
date: 2026-03-13
draft: false
tags:
- C++
- generic-programming
- algorithms
- monoids
- statistics
categories:
- Computer Science
- Mathematics
series:
- stepanov
series_weight: 14
math: true
description: Online accumulators are monoids. Default construction is the identity, combination via += is the binary operation, and parallel composition gives the product monoid, computing arbitrary statistics in a single pass.
linked_project:
- stepanov
---
*Accumulators are monoids. The same algebraic structure from the peasant post, in a different domain.*

## Accumulators as Monoids

An accumulator processes a stream of values, maintaining state that can be queried at any point. Write a class with `operator+=` for each statistic you need. Sum, mean, variance, min, max. Five statistics, five classes.

The problem is combinations. Sum *and* min? Write a sixth class. Sum, min, *and* max? A seventh. Every new combination requires new code.

But every accumulator has the same structure:

1. **Process** a value incrementally: `operator+=(value)`
2. **Combine** with another accumulator of the same type: `operator+=(accumulator)`
3. **Extract** a result: `.eval()`

Default construction gives you an empty accumulator: the **identity element**. Combination via `+=` is **associative**. Together, a monoid. The [peasant post]({{< ref "/post/2019-03-peasant-stepanov" >}}) used the same structure for exponentiation. Here we use it for streaming computation.

In C++20 concepts:

```cpp
template<typename A>
concept Accumulator = std::semiregular<A> &&
    requires(A a, A b, typename A::value_type v) {
        typename A::value_type;
        { a += v } -> std::same_as<A&>;   // process one value
        { a += b } -> std::same_as<A&>;   // combine two accumulators
        { a.eval() };                       // extract result
    };
```

## KBN: Compensated Summation

The simplest accumulator is a sum. But naive floating-point summation accumulates O(n) rounding error:

```cpp
double sum = 0.0;
sum += 1.0;
for (int i = 0; i < 1'000'000; ++i)
    sum += 1e-10;
// Expected: 1.0001    Actual: ~1.00009999999...8
```

When you add a tiny number to a large one, the tiny number's low-order bits get dropped. After a million additions, these losses add up.

Kahan-Babuska-Neumaier (KBN) summation tracks what gets lost:

```cpp
template<std::floating_point T>
class kbn_sum {
    T sum_ = T(0);
    T comp_ = T(0);   // compensation for lost bits

public:
    using value_type = T;

    constexpr kbn_sum& operator+=(const T& v) {
        T t = sum_ + v;
        comp_ += abs_(sum_) >= abs_(v) ? (sum_ - t) + v
                                       : (v - t) + sum_;
        sum_ = t;
        return *this;
    }

    constexpr T eval() const { return sum_ + comp_; }
};
```

The correction term `comp_` recovers the bits that floating-point addition drops. O(1) error instead of O(n), regardless of sequence length.

`kbn_sum` is a monoid:
- **Identity**: `kbn_sum{}` (sum=0, compensation=0)
- **Operation**: `a += b` (combine two compensated sums)

## Welford: Online Mean and Variance

Computing the mean is just sum/count. Variance is harder. The textbook formula \\(\sigma^2 = \frac{1}{n}\sum(x_i - \bar{x})^2\\) requires two passes: one for the mean, one for the deviations.

Welford's algorithm computes both in a single pass:

```cpp
welford& operator+=(const T& v) {
    ++n_;
    T delta = v - mean_;
    mean_ += delta / static_cast<T>(n_);
    T delta2 = v - mean_;   // uses *updated* mean
    m2_ += delta * delta2;
    return *this;
}
```

`delta` uses the old mean, `delta2` uses the new mean. Their product accumulates into `m2_`, the sum of squared deviations. At any point, `variance = m2_ / n`.

What makes this a monoid is the combination formula. Given two independent Welford accumulators with means \\(\bar{x}_A, \bar{x}_B\\) and counts \\(n_A, n_B\\), Chan et al. showed how to merge them:

$$\bar{x}\_{AB} = \frac{n_A \bar{x}_A + n_B \bar{x}_B}{n_A + n_B}$$

$$M\_{2,AB} = M\_{2,A} + M\_{2,B} + \delta^2 \frac{n_A n_B}{n_A + n_B}$$

where \\(\delta = \bar{x}_B - \bar{x}_A\\). This makes `welford` a monoid. You can split data across threads, compute partial statistics independently, and merge the results. Correctness follows from the algebra.

## The Pattern

Every accumulator I've built follows the same pattern:

| Accumulator | Identity | Combine operation |
|-------------|----------|-------------------|
| `kbn_sum` | sum=0 | add sums |
| `welford` | n=0, mean=0, m2=0 | parallel merge |
| `min_accumulator` | +infinity | take smaller |
| `max_accumulator` | -infinity | take larger |
| `count_accumulator` | 0 | add counts |

Default construction always gives the identity element. Combination via `+=` is always associative. This isn't a coincidence. Without associativity and an identity element, combination wouldn't be well-defined.

And associativity is what buys us something concrete. It means we can partition data arbitrarily and combine partial results in any order:

```
fold({a, b, c, d, e, f})
    = fold({a, b, c}) += fold({d, e, f})      // split in half
    = fold({a, b}) += fold({c, d, e, f})      // different split
```

All three give the same answer. That's parallelism, not from a threading library, but from the algebra.

## Composition: The Product Monoid

Each accumulator is a monoid. The standard construction: **if A and B are monoids, then A x B is a monoid** with component-wise operations. The product monoid.

In code, this is `parallel<A, B>`. Both accumulators process the same data stream, each maintaining its own state:

```cpp
template<Accumulator A, Accumulator B>
    requires std::same_as<typename A::value_type, typename B::value_type>
class parallel {
    A a_;
    B b_;
public:
    using value_type = typename A::value_type;

    parallel& operator+=(const value_type& v) {
        a_ += v; b_ += v;
        return *this;
    }

    parallel& operator+=(const parallel& other) {
        a_ += other.a_; b_ += other.b_;
        return *this;
    }

    auto eval() const {
        return std::make_tuple(a_.eval(), b_.eval());
    }
};
```

`parallel<A, B>` itself satisfies `Accumulator`, so it nests:

```cpp
// Sum, min, and max in one pass
auto stats = kbn_sum<double>{} | min_accumulator<double>{} | max_accumulator<double>{};
for (double v : data)
    stats += v;

auto [inner, max] = stats.eval();
auto [sum, min] = inner;
```

The `|` operator constructs a `parallel`. Nesting `parallel<parallel<A, B>, C>` computes three statistics with a single loop. Four statistics? Nest again. The structure composes without limit.

You don't write new code for each combination. You compose existing accumulators. The product monoid construction guarantees that the result is itself a valid accumulator. One loop, arbitrary statistics.

## fold: The Algorithm

Every accumulator is a monoid. Every monoid has a natural fold:

```cpp
template<Accumulator Acc, typename It>
Acc fold(It first, It last) {
    Acc acc{};    // identity element
    for (; first != last; ++first)
        acc += *first;
    return acc;
}
```

Compare this to `power()` from the peasant post. `power()` repeats multiplication guided by an exponent. `fold()` repeats accumulation guided by a data stream. Same pattern.

Because `+=` is associative, we could split the range, fold each piece independently, and combine. That's `std::reduce` with parallel execution, and it works correctly for the same reason the partitioning above works.

```cpp
// All statistics in one pass
using stats = parallel<kbn_sum<double>,
                       parallel<welford<double>,
                                parallel<min_accumulator<double>,
                                         max_accumulator<double>>>>;

auto result = fold<stats>(data.begin(), data.end());
```

## The Connection

In the peasant post, the observation was: any monoid supports efficient exponentiation. Here: any monoid supports composable streaming computation. Same structure, different domain.

The product monoid is what turns five independent classes into a composable algebra. Arbitrary combinations, built from parts, with correctness following from the structure. No new code per combination.

Algorithms arise from algebraic structure.

## Further Reading

- Kahan, "Pracniques: Further Remarks on Reducing Truncation Errors"
- Welford, "Note on a Method for Calculating Corrected Sums of Squares and Products"
- Chan, Golub & LeVeque, "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances"
- Stepanov & Rose, *From Mathematics to Generic Programming*
