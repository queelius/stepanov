---
title: "When Lists Become Bits"
date: 2026-04-23
draft: false
tags:
- C++
- generic-programming
- codecs
- information-theory
- category-theory
- monoids
categories:
- Computer Science
- Mathematics
series:
- stepanov
series_weight: 21
math: true
description: "Prefix-freeness is the property that lifts the free-monoid construction into bit space."
linked_project:
- pfc
- stepanov
---
*The free monoid on a type lifts to bit space. It lifts injectively only when the element codec is prefix-free.*

## Prefix-Free Codes and the Free Monoid

You have a list of unsigned integers. Encode the list as a single bit string.

Fixed-width encoding wastes space. If you allocate 64 bits per integer, small values like 1 or 7 cost as much as values near \(2^{64}\). Variable-width encoding recovers that space, but immediately raises a harder question: where does one encoded integer end and the next begin?

Two escape routes. First, prefix each encoded item with its length. That works, but the length headers are overhead, and you now need a codec for the lengths as well. Second, choose a code where the structure of the codewords makes boundaries unambiguous without any headers. These are prefix-free codes, and this is the right answer, in a precise categorical sense.

The "precise categorical sense" is what this post develops. Encoding a list as the concatenation of encoded elements is a monoid homomorphism from the free monoid on \(T\) to the monoid of bit strings under concatenation. The universal property of the free monoid guarantees this homomorphism always exists. The question of whether the decoder can invert it comes down to exactly one property of the element codec: whether it is prefix-free.

## The Free Monoid, Recalled

A monoid is a set with an associative binary operation and an identity element. The free monoid on a set \(S\) is the set of all finite sequences of elements from \(S\), with concatenation as the operation and the empty sequence as the identity.

"Free" means no equations hold except those forced by the monoid axioms. Nothing is identified with anything else. If you need commutativity or idempotency, you quotient the free monoid by additional equations. But the free monoid itself imposes nothing beyond associativity and identity.

The universal property says: given any monoid \(M\) and any function \(f: S \to M\), there is exactly one monoid homomorphism \(\hat{f}: \text{Free}(S) \to M\) that extends \(f\). That unique extension is fold:

$$\hat{f}([x_1, x_2, \ldots, x_n]) = f(x_1) \cdot f(x_2) \cdot \cdots \cdot f(x_n)$$

where \(\cdot\) is the operation in \(M\). The [free-algebra post](/post/2026-03-free-algebra-stepanov/) develops this in full. For this post, the one fact that matters is that fold is canonical: it is the unique way to extend a per-element map to a list-consuming function that respects the monoid structure.

## The Lifting Question

Fix a type \(T\) and a codec \(C\) for \(T\): a function \(\text{encode}: T \to \text{BitString}\). The monoid we care about here is \((\text{BitString}, +\!\!+, \varepsilon)\), the set of all finite bit strings under concatenation with the empty string as identity.

We want to extend \(\text{encode}\) to a function on lists of \(T\):

$$\text{encode\_list}([x_1, \ldots, x_n]) = \text{encode}(x_1) +\!\!+ \cdots +\!\!+ \text{encode}(x_n)$$

By the universal property, this lift always exists. It is fold applied to the encoding function over the free monoid of bit strings. You do not need to do anything clever. The homomorphism exists by construction.

The real question is whether \(\text{encode\_list}\) is injective: given a bit string \(b\), can we recover the unique list \([x_1, \ldots, x_n]\) that encodes to \(b\)?

This is not guaranteed. The lift always exists. It is not always invertible.

## When the Lift Fails: A Counter-Example

Consider a code that is not prefix-free. Assign codewords as follows:

```
A -> "0"
B -> "01"
C -> "10"
D -> "010"
```

The problem: `"0"` is a prefix of `"01"` and `"010"`. `"01"` is a prefix of `"010"`. So when we concatenate codewords to encode a list, the decoder cannot determine where one codeword ends and the next begins.

The bit string `"010"` has three valid parses:

- `[D]`, because `D` maps to `"010"` directly.
- `[A, C]`, because `"0" ++ "10" = "010"`.
- `[B, A]`, because `"01" ++ "0" = "010"`.

This is verified concretely in the test suite:

```cpp
TEST(PrefixFreeTest, NonPrefixFreeIsAmbiguous) {
    // A "code" that is NOT prefix-free.
    // "0"  is a prefix of "01" and "010".
    // "01" is a prefix of "010".
    std::map<char, std::string> code{
        {'A', "0"},
        {'B', "01"},
        {'C', "10"},
        {'D', "010"},
    };

    // [D], [A, C], and [B, A] all encode to the same bit string "010".
    EXPECT_EQ(encode_string(code, "D"),  "010");
    EXPECT_EQ(encode_string(code, "AC"), "010");
    EXPECT_EQ(encode_string(code, "BA"), "010");

    // Greedy left-to-right decoding cannot pick a unique answer; the multi-parse
    // decoder finds all three valid parses.
    auto parses = enumerate_parses(code, "010");
    EXPECT_EQ(parses.size(), 3u);
}
```

The `enumerate_parses` function is in `prefix_free.hpp`. It does a recursive scan: at each position in the bit string, it tries every codeword that matches at that position, and recurses on the remainder. For a prefix-free code it would return exactly one parse for any valid encoding. For this code, it returns three.

The decoder genuinely has no principled way to choose. The encoding function is not injective on lists; it maps distinct lists to the same bit string. The free-monoid lift exists but fails to be invertible.

## Prefix-Free Codes

A code is prefix-free if no codeword is a prefix of any other codeword. The definition rules out the failure mode above: if no codeword is a prefix of another, then when the decoder reads bits left-to-right and finds a complete codeword, it knows it has found one. There is no ambiguity about whether more bits belong to the current codeword or start a new one.

The canonical visualization is a trie (a binary tree for binary codes). Each codeword corresponds to a root-to-leaf path: go left for 0, right for 1, and the path terminates when you reach a leaf. Prefix-freeness means no codeword ends at an internal node. Every codeword's path ends at a leaf, and leaves are not ancestors of other leaves.

The two codecs in this post, Unary and Gamma, are both prefix-free.

Unary encodes positive integers: the codeword for \(n\) is \((n-1)\) zero bits followed by a one bit.

```cpp
struct Unary {
    using value_type = std::uint64_t;

    template<BitSink S>
    static void encode(value_type n, S& sink) {
        assert(n >= 1 && "Unary is undefined for n = 0");
        for (value_type i = 1; i < n; ++i) sink.write(false);
        sink.write(true);
    }

    template<BitSource S>
    static value_type decode(S& source) {
        value_type n = 1;
        while (!source.read()) ++n;
        return n;
    }
};
```

Prefix-freeness of Unary is immediate: every codeword ends with a `1`, and no other codeword is a prefix of that. The decoder reads zeros until it sees a one. There is no ambiguity.

Gamma is restated here from post 1. The codeword for \(n \geq 1\) is \(\lfloor\log_2 n\rfloor\) zero bits, then a one bit, then the \(\lfloor\log_2 n\rfloor\) lower bits of \(n\) in MSB-first order.

```cpp
struct Gamma {
    using value_type = std::uint64_t;

    template<BitSink S>
    static void encode(value_type n, S& sink) {
        assert(n >= 1 && "Gamma is undefined for n = 0");
        std::size_t bits = std::bit_width(n);
        for (std::size_t i = 0; i < bits - 1; ++i) sink.write(false);
        sink.write(true);
        for (std::size_t i = bits - 1; i > 0; --i) {
            sink.write(((n >> (i - 1)) & 1) != 0);
        }
    }

    template<BitSource S>
    static value_type decode(S& source) {
        std::size_t bits = 1;
        while (!source.read()) ++bits;
        value_type result = 1;
        for (std::size_t i = 1; i < bits; ++i) {
            result = (result << 1) | (source.read() ? value_type{1} : value_type{0});
        }
        return result;
    }
};
```

Gamma is prefix-free: the leading-zero count uniquely determines how many data bits follow. Once the decoder reads the `1` bit that terminates the unary prefix, it knows exactly how many more bits to consume. The boundary is unambiguous.

## The Free-Monoid Lift Works for Prefix-Free Codes

Now we can state the main result cleanly.

Fix a type \(T\) and a prefix-free codec \(C\) for \(T\). The encoding function specializes the universal property construction: \(S = T\), \(M = (\text{BitString}, +\!\!+, \varepsilon)\), and \(f = \text{encode}_C\). The lift \(\text{encode\_list}\) always exists as fold, as argued above. The new claim is that it is injective when \(C\) is prefix-free.

The argument in the prefix-free direction: given a bit string \(b\) produced by \(\text{encode\_list}\), a greedy left-to-right decoder works as follows. Start at position 0. Find the unique codeword that matches the bits starting at position 0. This codeword is unique because: (a) some codeword must match (otherwise \(b\) was not a valid encoding), and (b) no two codewords share a prefix, so at most one codeword can match at any position. Advance past that codeword. Repeat. The decoder always terminates with a unique parse.

The argument in the other direction: if \(C\) is not prefix-free, then there exist symbols \(a\) and \(b\) such that \(\text{encode}(a)\) is a prefix of \(\text{encode}(b)\). Write \(\text{encode}(b) = \text{encode}(a) +\!\!+ s\) for some non-empty string \(s\). If \(s\) is itself a prefix of some codeword sequence, then \([b]\) and \([a] +\!\!+ \text{decode}(s)\) produce the same bit string. This is exactly the failure mode in the counter-example above.

The bigger claim is that, when \(C\) is prefix-free, the image of \(\text{encode\_list}\) forms a submonoid of \(\text{BitString}\) that is isomorphic to the free monoid on \(T\). The encoding is a monoid isomorphism. "Lists of \(T\)" and "bit strings built from \(C\)'s codewords" are the same monoid, witnessed by \(\text{encode\_list}\) and its inverse \(\text{decode\_list}\).

`Vec<C>` from [Bits Follow Types](/post/2026-05-codecs-functors-stepanov/) is this isomorphism made concrete in C++.

## Implementation and Round-Trip

Post 2 has its own minimal `Vec<C>` implementation, independent of post 1's `codecs_functors.hpp`. The decision is intentional: each post in this series is self-contained, following the convention that posts can be read in any order. The implementations are identical in spirit.

```cpp
template<typename C>
struct Vec {
    using value_type = std::vector<typename C::value_type>;

    template<BitSink S>
    static void encode(const value_type& v, S& sink) {
        Gamma::encode(v.size() + 1, sink);
        for (const auto& x : v) {
            C::encode(x, sink);
        }
    }

    template<BitSource S>
    static value_type decode(S& source) {
        std::size_t n = static_cast<std::size_t>(Gamma::decode(source) - 1);
        value_type result;
        result.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            result.push_back(C::decode(source));
        }
        return result;
    }
};
```

The length is gamma-coded as `size + 1` because Gamma requires inputs \(\geq 1\) while vectors can be empty. The decoder subtracts 1 to recover the actual size.

The round-trip test:

```cpp
TEST(PrefixFreeTest, VecOfGammaRoundTrip) {
    using Codec = Vec<Gamma>;
    Codec::value_type v{1, 2, 3, 5, 8, 13, 21};
    EXPECT_EQ(round_trip<Codec>(v), v);
}

TEST(PrefixFreeTest, VecEmptyRoundTrip) {
    using Codec = Vec<Gamma>;
    Codec::value_type v;
    EXPECT_EQ(round_trip<Codec>(v), v);
}
```

Both pass. The Fibonacci sequence encodes and decodes cleanly; the empty list handles the edge case. Gamma's prefix-freeness is what makes this possible: each element's boundary is determined by the code structure, not by a length header.

## Kraft's Inequality, Lightly

There is a necessary condition on codeword lengths for a prefix-free code to exist. Kraft's inequality states: for a binary prefix-free code with codeword lengths \(l_1, l_2, \ldots\), we have

$$\sum_i 2^{-l_i} \leq 1$$

The sum counts how much of the "leaf capacity" of an infinite binary tree each codeword consumes. Prefix-freeness requires each codeword to occupy its own leaf without overlap; the total capacity of the tree is 1, so the sum cannot exceed it.

Both Unary and Gamma satisfy Kraft's inequality, verified computationally:

```cpp
TEST(PrefixFreeTest, KraftInequalityHoldsForUnary) {
    // Unary code: codeword for n has length n bits.
    // Sum of 2^-l for n = 1..K is 1 - 2^-K, which is < 1.
    double sum = 0.0;
    constexpr int K = 30;
    for (int n = 1; n <= K; ++n) {
        sum += std::ldexp(1.0, -n);  // 2^-n
    }
    EXPECT_LE(sum, 1.0);
    // Convergence check: sum approaches 1 as K grows.
    EXPECT_GT(sum, 1.0 - std::ldexp(1.0, -(K - 1)));
}

TEST(PrefixFreeTest, KraftInequalityHoldsForGamma) {
    // Gamma code: codeword for n has length 2 * floor(log2(n)) + 1.
    double sum = 0.0;
    constexpr int N = 1 << 14;
    for (int n = 1; n <= N; ++n) {
        int len = 2 * static_cast<int>(std::bit_width(static_cast<unsigned>(n))) - 1;
        sum += std::ldexp(1.0, -len);
    }
    EXPECT_LE(sum, 1.0 + 1e-9);
}
```

Unary sums to \(1 - 2^{-K}\) for the first \(K\) integers, converging to 1 from below. Gamma sums to exactly 1 in the limit: each "block" of integers sharing the same codeword length contributes \(2^{-(k+1)}\), and \(\sum_{k=0}^{\infty} 2^{-(k+1)} = 1\). Both satisfy \(\leq 1\).

The converse of Kraft's inequality is also true: any sequence of lengths satisfying the inequality is realized by some prefix-free code. This characterizes exactly which length vectors are achievable. The forthcoming Information Theory by Construction series develops the proof and connects it to optimal code design.

## The Bridge Claim

Pull back to the structure.

Prefix-freeness is the property that lifts the free-monoid construction into bit space injectively. Without it, the lift exists as a function but cannot be inverted. With it, the lift is an isomorphism of monoids.

The same pattern applies to the other combinators from [Bits Follow Types](/post/2026-05-codecs-functors-stepanov/):

| Combinator | Universal property being lifted |
|---|---|
| `Opt<C>` | Coproduct with unit \(1 + T\) |
| `Either<A, B>` | Binary coproduct \(A + B\) |
| `Pair<A, B>` | Binary product \(A \times B\) (given both factors prefix-free) |
| `Vec<C>` | Free monoid on \(T\) (given \(C\) prefix-free) |

Each combinator lifts a universal property from the type algebra to the bit-string algebra. The tag-bit in `Opt` and `Either` is what makes their coproduct construction injective: the tag selects the branch, and the branch codec handles the payload. The ordering in `Pair` is recoverable because the first codec is prefix-free, so the decoder knows where the first field ends. And `Vec` is recoverable because the element codec is prefix-free, as just argued.

The unifying frame: codecs are structure-preserving embeddings from the type algebra into the algebra of bit strings. When the type algebra involves a free construction (the free monoid for lists, or the free algebra for more general term structures), the lift is canonical by the universal property. When the codec is also prefix-free, the lift is reversible: encode and decode form an isomorphism.

This is the sense in which prefix-freeness is "the categorically right answer" to the boundary problem. It is not just convenient. It is exactly what is needed for the algebraic structure of the encoding to match the algebraic structure of the type.

The [duality post](/post/2026-01-19-duality-stepanov/) noted that encode and decode are duals. Under the combinator framing, that duality decomposes structurally: the encode-decode duality of a compound type is the product of the encode-decode dualities of its parts. Prefix-freeness is the condition that makes the decode half of that duality well-defined.

## Further Reading

**Background in this series:**
- [Bits Follow Types](/post/2026-05-codecs-functors-stepanov/): `Vec<C>` and the other combinators. This post explains why they work.
- [Why Lists and Polynomials Are Universal](/post/2026-03-free-algebra-stepanov/): the free-monoid construction being lifted to bit space here.
- [The Maps Between Structures](/post/2026-03-homomorphism-stepanov/): the lift is a monoid homomorphism; this post covers what that means and why it matters.
- [Structure, Duality, and the Shape of Computation](/post/2026-01-19-duality-stepanov/): prefix-freeness is what makes the decode half of the encode-decode duality well-defined.

**Where this leads:** The forthcoming Information Theory by Construction series develops Kraft's inequality proof, the converse (every Kraft-satisfying length vector is realizable by a prefix-free code), and the full universal-codes treatment connecting prefix-freeness to entropy, Huffman coding, and arithmetic coding.

**The production library:** PFC ([github.com/queelius/pfc](https://github.com/queelius/pfc)) provides the production version with a richer codec library, full STL integration, and 31k+ test assertions. See `include/pfc/algebraic.hpp` for the combinators and `include/pfc/codecs.hpp` for the full catalog of universal codes.
