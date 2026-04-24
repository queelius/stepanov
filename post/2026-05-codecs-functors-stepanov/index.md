---
title: "Bits Follow Types"
date: 2026-04-23
draft: false
tags:
- C++
- generic-programming
- codecs
- information-theory
- category-theory
- functors
categories:
- Computer Science
- Mathematics
series:
- stepanov
series_weight: 20
math: true
description: "Codecs are not ad-hoc bit formats. They are constructions on the algebraic structure of types."
linked_project:
- pfc
- stepanov
---
*Every type decomposes structurally. So does its codec.*

## Codecs as Functors

You have an `optional<vector<pair<int, string>>>`. The type decomposes structurally: it is an optional of a free monoid of products of an integer and a string. That decomposition is not an observation about memory layout. It is a statement about the algebraic structure of the type.

Now ask: does the codec decompose the same way?

If the answer is yes, you stop writing one-off encoders. You build a codec for `optional<T>` from a codec for `T`. You build a codec for `vector<T>` from a codec for `T`. The codec for `optional<vector<pair<int, string>>>` assembles from its parts with no manual layout decisions, no hand-placed length headers, no ad-hoc format negotiation.

This post argues that the answer is always yes, and shows what the machinery looks like. The thesis: codecs are not ad-hoc bit formats. They are constructions on the algebraic structure of types. The algebraic structure of a type determines its codec, the same way it determines its algorithms.

This extends Stepanov's claim. The [peasant algorithm post](/post/2019-03-peasant-stepanov/) showed that algorithms arise from algebraic structure. The [homomorphism post](/post/2026-03-homomorphism-stepanov/) showed that structure-preserving maps are the natural morphisms. Here, we show the codec itself is a structure-preserving map, and that it lifts from leaf types to compound types by the same algebraic logic.

## Bit I/O: The Foundation

Before combinators, we need concrete bit I/O. The approach taken here follows Stepanov's move in the algorithm posts: state the concept first, then provide a model.

Two concepts govern bit-level I/O:

```cpp
template<typename T>
concept BitSink = requires(T& s, bool bit) {
    { s.write(bit) } -> std::same_as<void>;
};

template<typename T>
concept BitSource = requires(T& s) {
    { s.read() } -> std::same_as<bool>;
    { s.peek() } -> std::convertible_to<bool>;
};
```

A `BitSink` accepts bits. A `BitSource` supplies them. A codec is an algorithm parameterized over `BitSink` and `BitSource`, not a class hierarchy. This is Stepanov's move at the bit level: require only what the algorithm needs, let anything that satisfies the concept participate.

The standard models are `BitWriter` and `BitReader`, which pack bits into byte buffers in LSB-first order:

```cpp
class BitWriter {
    std::span<std::uint8_t> buf_;
    std::size_t byte_idx_ = 0;
    std::uint8_t byte_ = 0;
    std::uint8_t bit_pos_ = 0;
public:
    explicit BitWriter(std::span<std::uint8_t> buf) noexcept : buf_(buf) {}

    void write(bool bit) noexcept {
        byte_ |= (bit ? std::uint8_t{1} : std::uint8_t{0}) << bit_pos_;
        if (++bit_pos_ == 8) {
            buf_[byte_idx_++] = byte_;
            byte_ = 0;
            bit_pos_ = 0;
        }
    }

    void align() noexcept {
        if (bit_pos_ > 0) {
            buf_[byte_idx_++] = byte_;
            byte_ = 0;
            bit_pos_ = 0;
        }
    }

    [[nodiscard]] std::size_t bytes_written() const noexcept {
        return byte_idx_ + (bit_pos_ > 0 ? 1 : 0);
    }
};

class BitReader {
    std::span<const std::uint8_t> buf_;
    std::size_t byte_idx_ = 0;
    std::uint8_t bit_pos_ = 0;
public:
    explicit BitReader(std::span<const std::uint8_t> buf) noexcept : buf_(buf) {}

    bool read() noexcept {
        bool bit = ((buf_[byte_idx_] >> bit_pos_) & 1) != 0;
        if (++bit_pos_ == 8) {
            ++byte_idx_;
            bit_pos_ = 0;
        }
        return bit;
    }

    [[nodiscard]] bool peek() const noexcept {
        return byte_idx_ < buf_.size();
    }
};
```

A codec concept rounds out the three-concept core:

```cpp
template<typename C>
concept Codec = requires {
    typename C::value_type;
};
```

The minimal form checks that `C` names a `value_type`. The full behavioral requirement, that `C` provides `encode` and `decode` templates matching `BitSink` and `BitSource` respectively, is enforced by the template constraints on each codec's methods rather than by the concept itself. This keeps the concept definition lean while the actual requirements are checked at each call site.

All code in this post lives in `codecs_functors.hpp`. The full suite of round-trip tests is in `test_codecs_functors.cpp`.

## A Leaf Codec: Elias Gamma

Before combinators, establish what a leaf codec looks like. Elias gamma encodes positive integers \(n \geq 1\).

The codeword for \(n\) is: \(\lfloor\log_2 n\rfloor\) zero bits, then a one bit, then the bits of \(n\) below its leading 1, written MSB-first. The total length is \(2\lfloor\log_2 n\rfloor + 1\) bits.

Examples: 1 encodes as `1`, 2 as `010`, 3 as `011`, 4 as `00100`, 7 as `00111`.

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

Gamma is asymptotically optimal under a prior proportional to \(1/n^2\). The full story, including the Kraft inequality, optimal code lengths, and the connection to universal codes, belongs to a forthcoming Information Theory by Construction series. For now: gamma is the code you use when you know values are small integers but not their distribution.

## Combinator: Opt

`Opt<C>` encodes `std::optional<T>` given a codec `C` for `T`. The wire format is one tag bit (1 = present, 0 = absent) followed by `C`'s encoding if present.

```cpp
template<typename C>
struct Opt {
    using value_type = std::optional<typename C::value_type>;

    template<BitSink S>
    static void encode(const value_type& v, S& sink) {
        if (v) {
            sink.write(true);
            C::encode(*v, sink);
        } else {
            sink.write(false);
        }
    }

    template<BitSource S>
    static value_type decode(S& source) {
        if (source.read()) {
            return C::decode(source);
        }
        return std::nullopt;
    }
};
```

The type-algebra reading: `optional<T>` is the coproduct of `T` with the unit type (a single absent value). In type algebra, this is written \(1 + T\). The tag bit selects the branch; the payload is present only in the right branch.

This is the first sign of functoriality. `Opt` takes a codec and returns a codec. It acts on codecs the same way `optional` acts on types. The structure of the operation is preserved: the algebraic decomposition of the type is mirrored exactly in the bit encoding.

## Combinator: Either

`Either<A, B>` encodes `std::variant<TA, TB>` given codecs `A` for `TA` and `B` for `TB`. One tag bit selects the branch; the payload follows.

```cpp
template<typename A, typename B>
struct Either {
    using value_type = std::variant<typename A::value_type, typename B::value_type>;

    template<BitSink S>
    static void encode(const value_type& v, S& sink) {
        if (v.index() == 0) {
            sink.write(false);
            A::encode(std::get<0>(v), sink);
        } else {
            sink.write(true);
            B::encode(std::get<1>(v), sink);
        }
    }

    template<BitSource S>
    static value_type decode(S& source) {
        if (source.read()) {
            return value_type{std::in_place_index<1>, B::decode(source)};
        }
        return value_type{std::in_place_index<0>, A::decode(source)};
    }
};
```

Type algebra: \(A + B\), the binary coproduct. For the worked example in this post, we need a ternary variant, so the implementation also provides `Either3`:

```cpp
template<typename A, typename B, typename C>
struct Either3 {
    using value_type = std::variant<typename A::value_type,
                                    typename B::value_type,
                                    typename C::value_type>;

    template<BitSink S>
    static void encode(const value_type& v, S& sink) {
        std::size_t idx = v.index();
        sink.write((idx & 0x1) != 0);
        sink.write((idx & 0x2) != 0);
        if (idx == 0) A::encode(std::get<0>(v), sink);
        else if (idx == 1) B::encode(std::get<1>(v), sink);
        else C::encode(std::get<2>(v), sink);
    }

    template<BitSource S>
    static value_type decode(S& source) {
        std::size_t idx = (source.read() ? 1u : 0u);
        idx |= (source.read() ? 2u : 0u);
        if (idx == 0) return value_type{std::in_place_index<0>, A::decode(source)};
        if (idx == 1) return value_type{std::in_place_index<1>, B::decode(source)};
        return value_type{std::in_place_index<2>, C::decode(source)};
    }
};
```

The tag for `Either3` uses two bits, encoding the branch index as a 2-bit integer (LSB first). This costs \(\lceil\log_2 N\rceil\) bits for N branches regardless of how often each branch occurs. When branch probabilities differ significantly, this is wasteful: Huffman-coded tags or arithmetic coding over the branch distribution would do better. That improvement belongs in the forthcoming Information Theory by Construction series. The flat tag scheme is correct and simple; optimality can be layered on separately.

## Combinator: Pair

`Pair<A, B>` encodes `std::pair<TA, TB>`: `A`'s encoding followed immediately by `B`'s encoding, with no delimiter.

```cpp
template<typename A, typename B>
struct Pair {
    using value_type = std::pair<typename A::value_type, typename B::value_type>;

    template<BitSink S>
    static void encode(const value_type& v, S& sink) {
        A::encode(v.first, sink);
        B::encode(v.second, sink);
    }

    template<BitSource S>
    static value_type decode(S& source) {
        auto a = A::decode(source);
        auto b = B::decode(source);
        return value_type{std::move(a), std::move(b)};
    }
};
```

Type algebra: \(A \times B\), the binary product. Two bit streams concatenated, recovered in sequence. No length header, no delimiter bit.

The reason this works is prefix-freeness: because `A`'s codec is prefix-free, the decoder always knows exactly when `A`'s payload ends and `B`'s payload begins. The boundary is implicit in the structure of the code. If the codecs were not prefix-free, concatenation would be ambiguous.

See [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/) for why prefix-freeness makes concatenation reversible.

## Combinator: Vec

`Vec<C>` encodes `std::vector<T>` given a codec `C` for `T`. The wire format is a gamma-coded length followed by `C`'s encoding of each element.

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

The `+1` bias in `encode` handles the mismatch between Gamma (which requires inputs \(\geq 1\)) and vector sizes (which can be 0). The decoded length subtracts 1 to recover the original size.

Type algebra: `vector<T>` is `List(T)`, the free monoid on `T`. The [free-algebra post](/post/2026-03-free-algebra-stepanov/) showed that `fold` is the unique monoid homomorphism from lists to any monoid: that uniqueness is the universal property of the free monoid. `Vec<C>` is the lift of that universal property to bit space: it extends a per-element encoding `C` to the entire list by applying `C` to each element in sequence.

That lift always exists (it is just fold). The deeper question is whether it is invertible: can the decoder recover the list from the bit string? The answer depends on prefix-freeness, and the analysis belongs in [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/). The short version: `Vec<C>` works when `C` is prefix-free, and Gamma is prefix-free.

## The Functorial Framing

Each combinator corresponds to an operation in the algebra of types:

| Combinator | Type algebra |
|---|---|
| `Opt<C>` | \(1 + T\) (coproduct with unit) |
| `Either<A, B>` | \(A + B\) (binary coproduct) |
| `Pair<A, B>` | \(A \times B\) (binary product) |
| `Vec<C>` | \(\text{List}(T)\) (free monoid on T) |

This table is not a coincidence. Each combinator takes codecs as inputs and returns a codec as output, mirroring the way the corresponding type constructor takes types and returns a type. The mapping from types to codecs respects the type algebra: if you know how to encode `A` and `B`, the combinators give you, for free, how to encode `A + B`, `A \times B`, `1 + A`, and `\text{List}(A)`.

This is the bridge thesis: **the algebraic structure of a type determines its codec, the same way it determines its algorithms.** Stepanov's original claim covered algorithms. These two posts extend it to encodings.

Each combinator is functorial in a precise sense. `Vec`, for instance, maps a morphism between codec targets (a function `f: TA -> TB` liftable to `f*: vector<TA> -> vector<TB>`) to a corresponding transformation between codecs. The formal categorical statement is deferred. For the purposes of this post, the table is the claim: structure-in, structure-out.

Contrast this with the alternative: one-off serialization code written directly for each compound type. That code is not wrong, but it is redundant. The structure of the type already tells you how to encode it. If your serialization code reflects that structure, it is derivable and correct by construction. If it does not, you have extra maintenance burden, a second place to update when the type changes.

The homomorphism post showed that structure-preserving maps are the natural morphisms between algebraic structures. Codecs, under this framing, are structure-preserving maps from the type algebra to the bit-string algebra. The combinators are the proof that these maps exist and compose.

## The Composed Example

To make this concrete, three auxiliary codecs appear in the worked example. Each is short enough to state plainly.

`Bool` encodes a boolean as a single bit:

```cpp
struct Bool {
    using value_type = bool;

    template<BitSink S>
    static void encode(value_type v, S& sink) { sink.write(v); }

    template<BitSource S>
    static value_type decode(S& source) { return source.read(); }
};
```

`Signed<C>` wraps an unsigned codec using zigzag encoding, mapping signed integers to non-negative values and then applying a bias so Gamma (which requires \(\geq 1\)) can handle zero:

```cpp
template<typename C>
struct Signed {
    using value_type = std::int64_t;

    template<BitSink S>
    static void encode(value_type v, S& sink) {
        std::uint64_t z;
        if (v < 0) {
            z = (static_cast<std::uint64_t>(-(v + 1)) << 1) | 1u;
        } else {
            z = static_cast<std::uint64_t>(v) << 1;
        }
        C::encode(z + 1, sink);
    }

    template<BitSource S>
    static value_type decode(S& source) {
        std::uint64_t z = C::decode(source) - 1;
        if (z & 1u) {
            return -static_cast<std::int64_t>(z >> 1) - 1;
        }
        return static_cast<std::int64_t>(z >> 1);
    }
};
```

`String` is gamma-coded length followed by 8 bits per character. It is, conceptually, `Vec<Byte>` where `Byte` is an 8-bit leaf codec. The combinator pattern applies recursively: a string is a list of bytes, and its codec is the list combinator applied to a byte codec.

```cpp
struct String {
    using value_type = std::string;

    template<BitSink S>
    static void encode(const value_type& s, S& sink) {
        Gamma::encode(s.size() + 1, sink);
        for (char c : s) Byte::encode(static_cast<std::uint8_t>(c), sink);
    }

    template<BitSource S>
    static value_type decode(S& source) {
        std::size_t n = static_cast<std::size_t>(Gamma::decode(source) - 1);
        value_type s;
        s.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            s.push_back(static_cast<char>(Byte::decode(source)));
        return s;
    }
};
```

Now the worked example. A configuration record: each entry has a string key and an optional value, where the value can be an integer, a string, or a boolean.

```cpp
using Value  = std::variant<std::int64_t, std::string, bool>;

struct Entry {
    std::string key;
    std::optional<Value> value;
    bool operator==(const Entry&) const = default;
};

using Config = std::vector<Entry>;

using ValueCodec  = Either3<Signed<Gamma>, String, Bool>;
using EntryCodec  = Pair<String, Opt<ValueCodec>>;
using ConfigCodec = Vec<EntryCodec>;
```

The round-trip test from `test_codecs_functors.cpp`:

```cpp
Config c = {
    Entry{"port",    Value{std::in_place_index<0>, std::int64_t{8080}}},
    Entry{"name",    Value{std::in_place_index<1>, std::string{"alpha"}}},
    Entry{"verbose", std::nullopt},
    Entry{"debug",   Value{std::in_place_index<2>, true}},
};

// Convert to the form ConfigCodec expects (vector of pairs).
std::vector<std::pair<std::string, std::optional<Value>>> as_pairs;
for (const auto& e : c) as_pairs.push_back(to_pair(e));

std::array<std::uint8_t, 1024> buf{};
BitWriter w(buf);
ConfigCodec::encode(as_pairs, w);
w.align();

BitReader r({buf.data(), w.bytes_written()});
auto decoded_pairs = ConfigCodec::decode(r);
// ... reconstruct Entry objects and assert equality
```

The codec types mirror the value types exactly. `Config` is `vector<Entry>`; `ConfigCodec` is `Vec<EntryCodec>`. `Entry` is `pair<string, optional<Value>>`; `EntryCodec` is `Pair<String, Opt<ValueCodec>>`. `Value` is `variant<int64_t, string, bool>`; `ValueCodec` is `Either3<Signed<Gamma>, String, Bool>`.

No manual layout. No marshaling. No hand-placed length headers. The encoding emerges from the type structure, because the codecs are constructed from the same algebra as the types.

## Why This Matters

The zero-copy invariant emerges from this construction. Because each codec's wire format is exactly what the structural recursion produces, the bit pattern in the buffer IS the value, recoverable by the same structural recursion in reverse. Encoding and decoding are not two separate processes that maintain a shared contract; they are a single algebraic object viewed from two directions.

This dissolves marshaling rather than solving it. Traditional serialization has two representations of a value: the in-memory layout and the wire format. These representations must be kept in sync, and every time the type changes, the serialization code must change too. Under the combinator approach, there is one representation: the algebraic structure of the type. The codec derives from that structure. Change the type, and the codec follows automatically.

This is what Stepanov meant when he insisted that algorithms should arise from algebraic structure. Here, it is not algorithms but encodings. The point is the same: when you see the structure clearly, the implementation becomes obvious. The hard work is seeing the structure.

The [duality post](/post/2026-01-19-duality-stepanov/) showed that encode and decode are duals. Under this combinator framing, that duality decomposes structurally: the encode-decode duality of a compound type is the product of the encode-decode dualities of its parts.

## Further Reading

**Next in series:** [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/) develops the reason `Vec<C>` works: prefix-freeness is exactly the condition that makes the free-monoid lift to bit space injective.

**Background in this series:**
- [Why Lists and Polynomials Are Universal](/post/2026-03-free-algebra-stepanov/): the free-monoid construction that `Vec` lifts to bit space.
- [The Maps Between Structures](/post/2026-03-homomorphism-stepanov/): homomorphisms as the natural morphisms between algebraic structures; the combinators are homomorphisms from the type algebra to the codec algebra.
- [Structure, Duality, and the Shape of Computation](/post/2026-01-19-duality-stepanov/): encode and decode as dual maps; the duality decomposes structurally under the combinators.

**The production library:** PFC ([github.com/queelius/pfc](https://github.com/queelius/pfc)) provides the production version of this design: a richer codec library, full STL integration, and 31k+ test assertions. See `include/pfc/algebraic.hpp` for the combinators and `include/pfc/codecs.hpp` for the universal-codes catalog.
