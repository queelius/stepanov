# PFC Stepanov Posts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new posts to the Stepanov series (slots 20, 21) that bridge the existing series to the PFC library, plus essential repo wiring and polish edits to three existing posts.

**Architecture:** Two self-contained Stepanov posts, each with a minimal pedagogical C++23 single-header implementation (~295 and ~230 lines) and a single GoogleTest test file. Each post stands alone but cross-references the other and existing series posts. Per the loosely-coupled decision in the spec, each post duplicates BitWriter/BitReader and concepts (~110 lines) so it can be read first.

**Tech Stack:** C++23, GoogleTest v1.14.0 (fetched via `FetchContent` in the parent CMakeLists), mkdocs (local docs site), Hugo (downstream via `mf` toolkit, out of scope for this PR), soul plugin's `check-banned-phrases.sh` PostToolUse hook.

---

## Spec reference

See `docs/superpowers/specs/2026-04-23-pfc-stepanov-posts-design.md` for the full design including section-by-section outlines for both posts, frontmatter conventions, and acceptance criteria.

## File Structure

**New files (created):**
- `post/2026-05-codecs-functors-stepanov/index.md` (~2000 words)
- `post/2026-05-codecs-functors-stepanov/codecs_functors.hpp` (~295 lines)
- `post/2026-05-codecs-functors-stepanov/test_codecs_functors.cpp` (~200 lines)
- `post/2026-05-prefix-free-stepanov/index.md` (~2000 words)
- `post/2026-05-prefix-free-stepanov/prefix_free.hpp` (~230 lines)
- `post/2026-05-prefix-free-stepanov/test_prefix_free.cpp` (~150 lines)

**Modified files:**
- `post/CMakeLists.txt` (add two `add_executable` blocks)
- `docs/index.md` (add posts under "Algebraic Foundations")
- `README.md` (add two rows to posts table)
- `mkdocs.yml` (verify; update if posts are explicitly listed)
- `stepanov-explorer.html` and `docs/stepanov-explorer.html` (add posts to explorer data, or defer with note)
- `post/2026-01-19-duality-stepanov/index.md` (forward-link to new posts)
- `post/2026-03-free-algebra-stepanov/index.md` (closing line about bit-space lift)
- `post/2026-01-18-synthesis-stepanov/index.md` (codec-pattern bullet/paragraph)

---

## Task 1: Reconnaissance

**Files:**
- Read: `mkdocs.yml`
- Read: `stepanov-explorer.html` and `docs/stepanov-explorer.html`
- Read: `post/CMakeLists.txt` (already inspected during spec; re-read to confirm)

- [ ] **Step 1: Inspect mkdocs.yml**

```bash
cat ~/github/metafunctor-series/stepanov/mkdocs.yml
```

Expected: see whether `nav:` enumerates posts explicitly or auto-discovers them. Note the result.

- [ ] **Step 2: Inspect the explorer HTML**

```bash
head -100 ~/github/metafunctor-series/stepanov/stepanov-explorer.html
```

Expected: identify whether the explorer reads structured data (a JSON or YAML block) or has hand-coded post entries. If structured, note the schema. If hand-coded, plan to defer the explorer update to a follow-up commit.

- [ ] **Step 3: Confirm slot weights 20, 21 are free**

```bash
grep -h "^series_weight:" ~/github/metafunctor-series/stepanov/post/*/index.md | sort -V | uniq
```

Expected: weights 0 through 19 listed, no 20 or 21. If 20 or 21 are taken, switch to the next free slots and update the spec.

- [ ] **Step 4: Record findings in a notes file (optional)**

If any finding requires deviation from the plan (e.g., explorer is hand-coded and needs deferral), note it in `docs/superpowers/notes/2026-04-23-recon.md`. No commit needed for notes.

---

## Task 2: Scaffold post 1 directory and wire CMakeLists

**Files:**
- Create: `post/2026-05-codecs-functors-stepanov/index.md` (placeholder frontmatter only)
- Create: `post/2026-05-codecs-functors-stepanov/codecs_functors.hpp` (header guards only)
- Create: `post/2026-05-codecs-functors-stepanov/test_codecs_functors.cpp` (gtest skeleton only)
- Modify: `post/CMakeLists.txt` (append `add_executable` block)

- [ ] **Step 1: Create the post directory and skeleton files**

```bash
mkdir -p ~/github/metafunctor-series/stepanov/post/2026-05-codecs-functors-stepanov
```

Create `index.md` with placeholder frontmatter (full prose later in Task 11):

```markdown
---
title: "Bits Follow Types"
date: 2026-05-01
draft: true
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

(Draft in progress. See plan Task 11 for full prose.)
```

Create `codecs_functors.hpp` with header guards and namespace skeleton:

```cpp
// codecs_functors.hpp
// Pedagogical implementation for the post "Bits Follow Types"
// (Stepanov series, slot 20). For the production version, see PFC:
// https://github.com/queelius/pfc

#pragma once

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace codecs_functors {

// Implementation arrives in subsequent plan tasks.

}  // namespace codecs_functors
```

Create `test_codecs_functors.cpp` with a minimal placeholder so CMake can find it:

```cpp
#include <gtest/gtest.h>
#include "codecs_functors.hpp"

TEST(CodecsFunctorsTest, Placeholder) {
    EXPECT_TRUE(true);
}
```

- [ ] **Step 2: Add the test executable to the parent CMakeLists**

Open `~/github/metafunctor-series/stepanov/post/CMakeLists.txt`. After the existing `test_free_algebra` block, append:

```cmake
# =============================================================================
# Codecs as Functors: Bits Follow Types
# =============================================================================
add_executable(test_codecs_functors 2026-05-codecs-functors-stepanov/test_codecs_functors.cpp)
target_link_libraries(test_codecs_functors GTest::gtest_main)
target_include_directories(test_codecs_functors PRIVATE 2026-05-codecs-functors-stepanov)
add_test(NAME test_codecs_functors COMMAND test_codecs_functors)
```

- [ ] **Step 3: Build to verify the wiring**

```bash
cd ~/github/metafunctor-series/stepanov && make build
```

Expected: build succeeds, including `test_codecs_functors`. If the build directory is fresh, `make build` configures CMake first.

- [ ] **Step 4: Run the placeholder test**

```bash
cd ~/github/metafunctor-series/stepanov && ctest --test-dir build -R test_codecs_functors --output-on-failure
```

Expected: `1 test passed (test_codecs_functors)`.

- [ ] **Step 5: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-codecs-functors-stepanov post/CMakeLists.txt
git commit -m "scaffold: add codecs-functors post directory and CMake wiring"
```

---

## Task 3: Implement BitWriter, BitReader, concepts (TDD)

**Files:**
- Modify: `post/2026-05-codecs-functors-stepanov/codecs_functors.hpp`
- Modify: `post/2026-05-codecs-functors-stepanov/test_codecs_functors.cpp`

- [ ] **Step 1: Write failing tests for BitWriter and BitReader**

Replace the placeholder test in `test_codecs_functors.cpp` with:

```cpp
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <span>
#include "codecs_functors.hpp"

using namespace codecs_functors;

TEST(CodecsFunctorsTest, BitWriterWritesIndividualBits) {
    std::array<std::uint8_t, 2> buf{};
    BitWriter w(std::span(buf));
    // Pattern: 1 0 1 1 0 0 0 1  ->  byte 0xA8 -- wait, LSB-first
    // Writing bits in order: 1, 0, 1, 1, 0, 0, 0, 1 with LSB-first packing
    // gives byte = 0b10001011 = 0x8B
    w.write(true);
    w.write(false);
    w.write(true);
    w.write(true);
    w.write(false);
    w.write(false);
    w.write(false);
    w.write(true);
    EXPECT_EQ(w.bytes_written(), 1u);
    EXPECT_EQ(buf[0], 0x8Bu);
}

TEST(CodecsFunctorsTest, BitWriterAlignsPartialByte) {
    std::array<std::uint8_t, 2> buf{};
    BitWriter w(std::span(buf));
    w.write(true);
    w.write(true);
    w.write(true);
    w.align();
    EXPECT_EQ(w.bytes_written(), 1u);
    EXPECT_EQ(buf[0], 0x07u);
}

TEST(CodecsFunctorsTest, BitReaderReadsBackBitWriterOutput) {
    std::array<std::uint8_t, 2> buf{};
    BitWriter w(std::span(buf));
    const std::vector<bool> pattern{true, false, true, true, false, false, false, true,
                                    false, true, false, true};
    for (bool b : pattern) w.write(b);
    w.align();

    BitReader r(std::span<const std::uint8_t>(buf.data(), w.bytes_written()));
    for (bool b : pattern) {
        EXPECT_EQ(r.read(), b);
    }
}

TEST(CodecsFunctorsTest, ConceptsAreSatisfied) {
    static_assert(BitSink<BitWriter>);
    static_assert(BitSource<BitReader>);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/github/metafunctor-series/stepanov && make build 2>&1 | head -30
```

Expected: compilation FAILS with "BitWriter not declared," "BitReader not declared," "BitSink not declared," etc.

- [ ] **Step 3: Implement BitWriter, BitReader, and the three concepts**

In `codecs_functors.hpp`, replace the comment "Implementation arrives in subsequent plan tasks." with:

```cpp
// ---- Bit I/O ----------------------------------------------------------------

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

// ---- Concepts ---------------------------------------------------------------

template<typename T>
concept BitSink = requires(T& s, bool bit) {
    { s.write(bit) } -> std::same_as<void>;
};

template<typename T>
concept BitSource = requires(T& s) {
    { s.read() } -> std::same_as<bool>;
    { s.peek() } -> std::convertible_to<bool>;
};

template<typename C>
concept Codec = requires {
    typename C::value_type;
};
// Note: a fuller Codec concept would require encode and decode functions
// against any BitSink and BitSource. The minimal form above is sufficient
// for the pedagogical examples in this post; the prose discusses the
// fuller requirements.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R test_codecs_functors --output-on-failure
```

Expected: 4 tests pass (`BitWriterWritesIndividualBits`, `BitWriterAlignsPartialByte`, `BitReaderReadsBackBitWriterOutput`, `ConceptsAreSatisfied`).

- [ ] **Step 5: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-codecs-functors-stepanov
git commit -m "feat(codecs-functors): add BitWriter, BitReader, and BitSink/BitSource concepts"
```

---

## Task 4: Implement Gamma codec (TDD)

**Files:**
- Modify: `post/2026-05-codecs-functors-stepanov/codecs_functors.hpp`
- Modify: `post/2026-05-codecs-functors-stepanov/test_codecs_functors.cpp`

- [ ] **Step 1: Write failing tests for Gamma**

Append to `test_codecs_functors.cpp`:

```cpp
namespace {

template<typename Codec, typename T>
T round_trip(const T& v) {
    std::array<std::uint8_t, 64> buf{};
    BitWriter w(std::span(buf));
    Codec::encode(v, w);
    w.align();
    BitReader r(std::span<const std::uint8_t>(buf.data(), w.bytes_written()));
    return Codec::decode(r);
}

}  // namespace

TEST(CodecsFunctorsTest, GammaRoundTrip) {
    for (std::uint64_t n : {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3},
                            std::uint64_t{100}, std::uint64_t{1024},
                            std::uint64_t{1} << 30}) {
        EXPECT_EQ(round_trip<Gamma>(n), n) << "n = " << n;
    }
}
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd ~/github/metafunctor-series/stepanov && make build 2>&1 | head -10
```

Expected: compilation fails because `Gamma` is undefined.

- [ ] **Step 3: Implement Gamma**

Append to `codecs_functors.hpp` (inside the namespace):

```cpp
// ---- Leaf codec: Elias gamma -----------------------------------------------
// Encodes positive integers (>= 1).
// Codeword for n: (bit_width(n) - 1) zero bits, then a one bit,
// then the bits of n minus the leading 1, MSB first.
// Length: 2 * floor(log2(n)) + 1 bits.

struct Gamma {
    using value_type = std::uint64_t;

    template<BitSink S>
    static void encode(value_type n, S& sink) {
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

- [ ] **Step 4: Run tests, expect pass**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R "test_codecs_functors.GammaRoundTrip" --output-on-failure
```

Expected: `GammaRoundTrip` passes.

- [ ] **Step 5: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-codecs-functors-stepanov
git commit -m "feat(codecs-functors): add Elias gamma leaf codec"
```

---

## Task 5: Implement Opt combinator (TDD)

**Files:**
- Modify: `post/2026-05-codecs-functors-stepanov/codecs_functors.hpp`
- Modify: `post/2026-05-codecs-functors-stepanov/test_codecs_functors.cpp`

- [ ] **Step 1: Write failing tests for Opt**

Append to `test_codecs_functors.cpp`:

```cpp
TEST(CodecsFunctorsTest, OptRoundTripPresent) {
    using Codec = Opt<Gamma>;
    Codec::value_type v = std::uint64_t{42};
    EXPECT_EQ(round_trip<Codec>(v), v);
}

TEST(CodecsFunctorsTest, OptRoundTripAbsent) {
    using Codec = Opt<Gamma>;
    Codec::value_type v = std::nullopt;
    EXPECT_EQ(round_trip<Codec>(v), v);
}
```

- [ ] **Step 2: Run, expect failure**

```bash
cd ~/github/metafunctor-series/stepanov && make build 2>&1 | head -10
```

Expected: `Opt` undefined.

- [ ] **Step 3: Implement Opt**

Append to `codecs_functors.hpp`:

```cpp
// ---- Combinator: Opt -- coproduct with unit (1 + T) -------------------------

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

- [ ] **Step 4: Run, expect pass**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R "test_codecs_functors.Opt" --output-on-failure
```

Expected: 2 Opt tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-codecs-functors-stepanov
git commit -m "feat(codecs-functors): add Opt combinator (coproduct with unit)"
```

---

## Task 6: Implement Either and Either3 combinators (TDD)

**Files:**
- Modify: `post/2026-05-codecs-functors-stepanov/codecs_functors.hpp`
- Modify: `post/2026-05-codecs-functors-stepanov/test_codecs_functors.cpp`

`Either3` is a 3-way variant used by the worked example (config parser). The post prose presents `Either<A, B>` (binary) as the canonical case and notes that the n-ary generalization is left as exercise; `Either3` is shown as the concrete instantiation needed for the example.

- [ ] **Step 1: Write failing tests for Either and Either3**

Append to `test_codecs_functors.cpp`:

```cpp
TEST(CodecsFunctorsTest, EitherRoundTripLeft) {
    using Codec = Either<Gamma, Gamma>;
    Codec::value_type v{std::in_place_index<0>, std::uint64_t{7}};
    auto out = round_trip<Codec>(v);
    EXPECT_EQ(out.index(), 0u);
    EXPECT_EQ(std::get<0>(out), 7u);
}

TEST(CodecsFunctorsTest, EitherRoundTripRight) {
    using Codec = Either<Gamma, Gamma>;
    Codec::value_type v{std::in_place_index<1>, std::uint64_t{13}};
    auto out = round_trip<Codec>(v);
    EXPECT_EQ(out.index(), 1u);
    EXPECT_EQ(std::get<1>(out), 13u);
}

TEST(CodecsFunctorsTest, Either3RoundTripAllBranches) {
    using Codec = Either3<Gamma, Gamma, Gamma>;
    for (std::size_t branch = 0; branch < 3; ++branch) {
        Codec::value_type v;
        if (branch == 0) v.template emplace<0>(std::uint64_t{1});
        if (branch == 1) v.template emplace<1>(std::uint64_t{2});
        if (branch == 2) v.template emplace<2>(std::uint64_t{3});
        auto out = round_trip<Codec>(v);
        EXPECT_EQ(out.index(), branch);
    }
}
```

- [ ] **Step 2: Run, expect failure**

```bash
cd ~/github/metafunctor-series/stepanov && make build 2>&1 | head -10
```

- [ ] **Step 3: Implement Either and Either3**

Append to `codecs_functors.hpp`:

```cpp
// ---- Combinator: Either -- binary coproduct (A + B) -------------------------

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

// ---- Combinator: Either3 -- ternary coproduct (A + B + C) -------------------
// 2 tag bits encode the branch index in the order (A=0b00, B=0b01, C=0b10).

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

- [ ] **Step 4: Run, expect pass**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R "test_codecs_functors.Either" --output-on-failure
```

Expected: 3 Either/Either3 tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-codecs-functors-stepanov
git commit -m "feat(codecs-functors): add Either and Either3 combinators (binary and ternary coproducts)"
```

---

## Task 7: Implement Pair combinator (TDD)

**Files:**
- Modify: `post/2026-05-codecs-functors-stepanov/codecs_functors.hpp`
- Modify: `post/2026-05-codecs-functors-stepanov/test_codecs_functors.cpp`

- [ ] **Step 1: Write failing tests for Pair**

Append to `test_codecs_functors.cpp`:

```cpp
TEST(CodecsFunctorsTest, PairRoundTrip) {
    using Codec = Pair<Gamma, Gamma>;
    Codec::value_type v{std::uint64_t{17}, std::uint64_t{42}};
    auto out = round_trip<Codec>(v);
    EXPECT_EQ(out.first, 17u);
    EXPECT_EQ(out.second, 42u);
}

TEST(CodecsFunctorsTest, PairOfDifferentCodecsRoundTrip) {
    using Codec = Pair<Gamma, Opt<Gamma>>;
    Codec::value_type v{std::uint64_t{5}, std::optional<std::uint64_t>{std::uint64_t{99}}};
    auto out = round_trip<Codec>(v);
    EXPECT_EQ(out.first, 5u);
    ASSERT_TRUE(out.second.has_value());
    EXPECT_EQ(*out.second, 99u);
}
```

- [ ] **Step 2: Run, expect failure**

```bash
cd ~/github/metafunctor-series/stepanov && make build 2>&1 | head -10
```

- [ ] **Step 3: Implement Pair**

Append to `codecs_functors.hpp`:

```cpp
// ---- Combinator: Pair -- binary product (A x B) -----------------------------

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

- [ ] **Step 4: Run, expect pass**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R "test_codecs_functors.Pair" --output-on-failure
```

Expected: 2 Pair tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-codecs-functors-stepanov
git commit -m "feat(codecs-functors): add Pair combinator (binary product)"
```

---

## Task 8: Implement Vec combinator (TDD)

**Files:**
- Modify: `post/2026-05-codecs-functors-stepanov/codecs_functors.hpp`
- Modify: `post/2026-05-codecs-functors-stepanov/test_codecs_functors.cpp`

- [ ] **Step 1: Write failing tests for Vec**

Append to `test_codecs_functors.cpp`:

```cpp
TEST(CodecsFunctorsTest, VecEmpty) {
    using Codec = Vec<Gamma>;
    Codec::value_type v;
    auto out = round_trip<Codec>(v);
    EXPECT_TRUE(out.empty());
}

TEST(CodecsFunctorsTest, VecSingleton) {
    using Codec = Vec<Gamma>;
    Codec::value_type v{std::uint64_t{7}};
    auto out = round_trip<Codec>(v);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], 7u);
}

TEST(CodecsFunctorsTest, VecManyElements) {
    using Codec = Vec<Gamma>;
    Codec::value_type v{1, 2, 3, 5, 8, 13, 21, 34, 55};
    auto out = round_trip<Codec>(v);
    EXPECT_EQ(out, v);
}
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement Vec**

Append to `codecs_functors.hpp`:

```cpp
// ---- Combinator: Vec -- free monoid on T (List<T>) --------------------------
// Length is encoded as gamma(size + 1) since gamma requires a positive integer.

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

- [ ] **Step 4: Run, expect pass**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R "test_codecs_functors.Vec" --output-on-failure
```

Expected: 3 Vec tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-codecs-functors-stepanov
git commit -m "feat(codecs-functors): add Vec combinator (free monoid lift)"
```

---

## Task 9: Implement auxiliary codecs (Bool, Signed, Byte, String) (TDD)

**Files:**
- Modify: `post/2026-05-codecs-functors-stepanov/codecs_functors.hpp`
- Modify: `post/2026-05-codecs-functors-stepanov/test_codecs_functors.cpp`

- [ ] **Step 1: Write failing tests**

Append to `test_codecs_functors.cpp`:

```cpp
TEST(CodecsFunctorsTest, BoolRoundTrip) {
    EXPECT_EQ(round_trip<Bool>(true), true);
    EXPECT_EQ(round_trip<Bool>(false), false);
}

TEST(CodecsFunctorsTest, SignedRoundTrip) {
    using Codec = Signed<Gamma>;
    for (std::int64_t n : {std::int64_t{0}, std::int64_t{1}, std::int64_t{-1},
                           std::int64_t{42}, std::int64_t{-42},
                           std::int64_t{1} << 30, -(std::int64_t{1} << 30)}) {
        EXPECT_EQ(round_trip<Codec>(n), n) << "n = " << n;
    }
}

TEST(CodecsFunctorsTest, ByteRoundTrip) {
    for (int b = 0; b < 256; ++b) {
        EXPECT_EQ(round_trip<Byte>(static_cast<std::uint8_t>(b)), static_cast<std::uint8_t>(b));
    }
}

TEST(CodecsFunctorsTest, StringRoundTrip) {
    EXPECT_EQ(round_trip<String>(std::string{}), std::string{});
    EXPECT_EQ(round_trip<String>(std::string{"hello"}), std::string{"hello"});
    EXPECT_EQ(round_trip<String>(std::string{"with\0null", 9}), std::string("with\0null", 9));
}
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement the auxiliary codecs**

Append to `codecs_functors.hpp`:

```cpp
// ---- Auxiliary leaf codec: Bool ---------------------------------------------

struct Bool {
    using value_type = bool;

    template<BitSink S>
    static void encode(value_type v, S& sink) { sink.write(v); }

    template<BitSource S>
    static value_type decode(S& source) { return source.read(); }
};

// ---- Auxiliary adapter: Signed ----------------------------------------------
// Wraps an unsigned codec via zigzag encoding:
//   0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
// Then biased by 1 because the underlying codec (gamma) requires positives.

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

// ---- Auxiliary leaf codec: Byte ---------------------------------------------

struct Byte {
    using value_type = std::uint8_t;

    template<BitSink S>
    static void encode(value_type v, S& sink) {
        for (int i = 0; i < 8; ++i) sink.write(((v >> i) & 1) != 0);
    }

    template<BitSource S>
    static value_type decode(S& source) {
        std::uint8_t v = 0;
        for (int i = 0; i < 8; ++i) {
            if (source.read()) v |= static_cast<std::uint8_t>(1 << i);
        }
        return v;
    }
};

// ---- Auxiliary codec: String -- Vec<Byte> wrapping std::string -------------

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
        for (std::size_t i = 0; i < n; ++i) s.push_back(static_cast<char>(Byte::decode(source)));
        return s;
    }
};
```

- [ ] **Step 4: Run, expect pass**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R "test_codecs_functors.(Bool|Signed|Byte|String)" --output-on-failure
```

Expected: 4 auxiliary codec tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-codecs-functors-stepanov
git commit -m "feat(codecs-functors): add Bool, Signed, Byte, String auxiliary codecs"
```

---

## Task 10: Composed config-parser test

**Files:**
- Modify: `post/2026-05-codecs-functors-stepanov/test_codecs_functors.cpp`

This is the worked example from the spec. It exercises Either3 + Pair + Opt + Vec + Signed + String + Bool together.

- [ ] **Step 1: Write the composed-example test**

Append to `test_codecs_functors.cpp`:

```cpp
// ---- Composed worked example: a config record ------------------------------

namespace config_example {

using Value = std::variant<std::int64_t, std::string, bool>;

struct Entry {
    std::string key;
    std::optional<Value> value;
    bool operator==(const Entry&) const = default;
};

using Config = std::vector<Entry>;

using ValueCodec  = Either3<Signed<Gamma>, String, Bool>;
using EntryCodec  = Pair<String, Opt<ValueCodec>>;
using ConfigCodec = Vec<EntryCodec>;

// Adapter so EntryCodec (which encodes std::pair<std::string, std::optional<...>>)
// can round-trip an Entry.
inline std::pair<std::string, std::optional<Value>> to_pair(const Entry& e) {
    return {e.key, e.value};
}
inline Entry to_entry(std::pair<std::string, std::optional<Value>> p) {
    return Entry{std::move(p.first), std::move(p.second)};
}

}  // namespace config_example

TEST(CodecsFunctorsTest, ConfigParserRoundTrip) {
    using namespace config_example;

    Config c = {
        Entry{"port", Value{std::in_place_index<0>, std::int64_t{8080}}},
        Entry{"name", Value{std::in_place_index<1>, std::string{"alpha"}}},
        Entry{"verbose", std::nullopt},
        Entry{"debug", Value{std::in_place_index<2>, true}},
    };

    // Convert to the form ConfigCodec expects (vector of pairs).
    std::vector<std::pair<std::string, std::optional<Value>>> as_pairs;
    as_pairs.reserve(c.size());
    for (const auto& e : c) as_pairs.push_back(to_pair(e));

    std::array<std::uint8_t, 256> buf{};
    BitWriter w(std::span(buf));
    ConfigCodec::encode(as_pairs, w);
    w.align();

    BitReader r(std::span<const std::uint8_t>(buf.data(), w.bytes_written()));
    auto decoded_pairs = ConfigCodec::decode(r);

    Config decoded;
    decoded.reserve(decoded_pairs.size());
    for (auto& p : decoded_pairs) decoded.push_back(to_entry(std::move(p)));

    ASSERT_EQ(decoded.size(), c.size());
    for (std::size_t i = 0; i < c.size(); ++i) {
        EXPECT_EQ(decoded[i].key, c[i].key) << "i=" << i;
        EXPECT_EQ(decoded[i].value.has_value(), c[i].value.has_value()) << "i=" << i;
        if (c[i].value) {
            EXPECT_EQ(decoded[i].value->index(), c[i].value->index()) << "i=" << i;
        }
    }
}
```

- [ ] **Step 2: Run, expect pass (no implementation change needed)**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R "test_codecs_functors.ConfigParserRoundTrip" --output-on-failure
```

Expected: pass.

- [ ] **Step 3: Run the entire test suite to confirm nothing regressed**

```bash
cd ~/github/metafunctor-series/stepanov && ctest --test-dir build -R test_codecs_functors --output-on-failure
```

Expected: ~17 tests pass (4 BitIO + 1 concept + 1 gamma + 2 opt + 3 either + 2 pair + 3 vec + 4 aux + 1 composed).

- [ ] **Step 4: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-codecs-functors-stepanov
git commit -m "test(codecs-functors): add config-parser composed example round-trip"
```

---

## Task 11: Draft post 1 prose (`index.md`)

**Files:**
- Modify: `post/2026-05-codecs-functors-stepanov/index.md`

This task drafts the full ~2000-word article. Each step writes one section per the spec's Section 4 outline (sections A through K). Reference the spec at `docs/superpowers/specs/2026-04-23-pfc-stepanov-posts-design.md` for full per-section guidance (word counts, key claims, transitions).

When this task starts, change `draft: true` to `draft: false` in the frontmatter and set the `date` to today's date.

- [ ] **Step 1: Write opening matter (lead-in italic + Section A)**

Replace the placeholder body in `index.md` with:

- Lead-in italic line directly under the frontmatter: a one-sentence thesis statement, in the voice of the existing series (e.g., "Codecs are not ad-hoc bit formats. They are constructions on the algebraic structure of types.")
- Section A "The motivating problem" (~150 words). Per spec: optional<vector<pair<int, string>>> example, decompose-the-type question, thesis statement. End with a forward-pointing sentence about the rest of the post.

- [ ] **Step 2: Write Section B (concepts and bit I/O)**

Section B (~250 words + ~110 lines code). Introduce BitWriter, BitReader, the three concepts. Include the C++ code blocks verbatim from `codecs_functors.hpp` (the exact code from Task 3 step 3). Frame: a codec is an algorithm parameterized over BitSink/BitSource, not a class hierarchy.

- [ ] **Step 3: Write Section C (Gamma)**

Section C (~150 words + ~25 lines code). Include the Gamma code verbatim from Task 4 step 3. Brief discussion of the 1/n^2 prior; defer details to the new series.

- [ ] **Step 4: Write Section D (Opt combinator)**

Section D (~200 words + ~25 lines code). Include the Opt code verbatim from Task 5. Type-algebra interpretation: 1 + T. First sign of functoriality.

- [ ] **Step 5: Write Section E (Either combinator)**

Section E (~250 words + ~30 lines code). Show binary Either, mention Either3 used by the worked example, mention n-ary cost is log2(N) tag bits and forward-pointer to Huffman/arithmetic coding in the new series.

- [ ] **Step 6: Write Section F (Pair combinator)**

Section F (~150 words + ~25 lines code). Concatenation of bit streams. Include the forward-pointer to post 2: "the encode/decode order is recoverable for a deeper reason; see [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/) for why prefix-freeness makes concatenation reversible."

- [ ] **Step 7: Write Section G (Vec combinator)**

Section G (~200 words + ~40 lines code). Include the Vec code. Include the forward-pointer to post 2: "the lifting from a codec for T to a codec for vector<T> is the free-monoid universal property at the bit level; [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/) explains why it works."

- [ ] **Step 8: Write Section H (the functorial framing)**

Section H (~250 words, no code). Include the table of combinators and their type-algebra correspondences (verbatim from spec). State the bridge thesis: "the algebraic structure of a type determines its codec, the same way it determines its algorithms." Brief paragraph on functoriality with pointer to category theory.

- [ ] **Step 9: Write Section I (composed example with auxiliary codecs)**

Section I (~200 words + ~50 lines code).

- Brief preamble: introduce the three auxiliary codecs (Bool, Signed, String). Use the exact code blocks from Task 9 step 3.
- Then the composed config-parser example. Use the type definitions and codec-composition code from Task 10 step 1 (the part inside `namespace config_example`).
- Closing prose: "The codec types mirror the value types exactly. No manual layout, no marshaling, no hand-placed length headers. Encoding emerges from the type structure."

- [ ] **Step 10: Write Section J (why this matters)**

Section J (~200 words, no code). Zero-copy invariant emerges from the construction. Wire format IS the bit pattern. Serialization and the value stop being two separate things.

- [ ] **Step 11: Write Section K (cross-references and footnote)**

Section K (~120 words). Cross-references with Markdown links:

- Forward to [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/)
- Back to [free-algebra](/post/2026-03-free-algebra-stepanov/), [duality](/post/2026-01-19-duality-stepanov/), [homomorphism](/post/2026-03-homomorphism-stepanov/)
- Footnote: "PFC ([github.com/queelius/pfc](https://github.com/queelius/pfc)) provides the production version of this design with a richer codec library, full STL integration, and 31k+ test assertions. See `include/pfc/algebraic.hpp` for the combinators and `include/pfc/codecs.hpp` for the universal-codes catalog."

- [ ] **Step 12: Verify the soul check passes (it runs automatically on Write but verify)**

```bash
cd ~/github/metafunctor-series/stepanov
bash $CLAUDE_PLUGIN_ROOT/hooks/check-banned-phrases.sh post/2026-05-codecs-functors-stepanov/index.md
```

Expected: no errors. If em-dashes or other banned phrases are flagged, edit the prose to use commas/colons/parens/periods instead.

- [ ] **Step 13: Render via mkdocs and visually inspect**

```bash
cd ~/github/metafunctor-series/stepanov && mkdocs serve --dev-addr 127.0.0.1:8001 &
sleep 3
curl -s http://127.0.0.1:8001/post/2026-05-codecs-functors-stepanov/ | head -50
kill %1
```

Expected: HTML renders without 404. If using Hugo, the equivalent local-render check applies.

- [ ] **Step 14: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-codecs-functors-stepanov/index.md
git commit -m "docs(codecs-functors): draft post 1 prose (Bits Follow Types)"
```

---

## Task 12: Scaffold post 2 directory and wire CMakeLists

**Files:**
- Create: `post/2026-05-prefix-free-stepanov/index.md` (placeholder frontmatter only)
- Create: `post/2026-05-prefix-free-stepanov/prefix_free.hpp` (header guards only)
- Create: `post/2026-05-prefix-free-stepanov/test_prefix_free.cpp` (gtest skeleton only)
- Modify: `post/CMakeLists.txt` (append `add_executable` block)

- [ ] **Step 1: Create skeleton files (analogous to Task 2 step 1)**

```bash
mkdir -p ~/github/metafunctor-series/stepanov/post/2026-05-prefix-free-stepanov
```

Create `index.md`:

```markdown
---
title: "When Lists Become Bits"
date: 2026-05-08
draft: true
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

(Draft in progress. See plan Task 17 for full prose.)
```

Create `prefix_free.hpp`:

```cpp
// prefix_free.hpp
// Pedagogical implementation for the post "When Lists Become Bits"
// (Stepanov series, slot 21). For the production version, see PFC:
// https://github.com/queelius/pfc

#pragma once

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace prefix_free {

// Implementation arrives in subsequent plan tasks.

}  // namespace prefix_free
```

Create `test_prefix_free.cpp`:

```cpp
#include <gtest/gtest.h>
#include "prefix_free.hpp"

TEST(PrefixFreeTest, Placeholder) {
    EXPECT_TRUE(true);
}
```

- [ ] **Step 2: Add the test executable to the parent CMakeLists**

Append to `~/github/metafunctor-series/stepanov/post/CMakeLists.txt`:

```cmake
# =============================================================================
# Prefix-Free Codes: When Lists Become Bits
# =============================================================================
add_executable(test_prefix_free 2026-05-prefix-free-stepanov/test_prefix_free.cpp)
target_link_libraries(test_prefix_free GTest::gtest_main)
target_include_directories(test_prefix_free PRIVATE 2026-05-prefix-free-stepanov)
add_test(NAME test_prefix_free COMMAND test_prefix_free)
```

- [ ] **Step 3: Build and verify**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R test_prefix_free --output-on-failure
```

Expected: 1 placeholder test passes.

- [ ] **Step 4: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-prefix-free-stepanov post/CMakeLists.txt
git commit -m "scaffold: add prefix-free post directory and CMake wiring"
```

---

## Task 13: Implement BitWriter, BitReader, concepts in post 2 (TDD)

**Files:**
- Modify: `post/2026-05-prefix-free-stepanov/prefix_free.hpp`
- Modify: `post/2026-05-prefix-free-stepanov/test_prefix_free.cpp`

This duplicates the bit I/O and concepts from post 1 (per the loosely-coupled decision). The implementations are intentionally identical so post 2 can be read first.

- [ ] **Step 1: Write failing tests**

Replace the placeholder body in `test_prefix_free.cpp` with:

```cpp
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <span>
#include "prefix_free.hpp"

using namespace prefix_free;

TEST(PrefixFreeTest, BitIORoundTrip) {
    std::array<std::uint8_t, 2> buf{};
    BitWriter w(std::span(buf));
    const std::vector<bool> pattern{true, false, true, true, false, false, true};
    for (bool b : pattern) w.write(b);
    w.align();

    BitReader r(std::span<const std::uint8_t>(buf.data(), w.bytes_written()));
    for (bool b : pattern) {
        EXPECT_EQ(r.read(), b);
    }
}

TEST(PrefixFreeTest, ConceptsAreSatisfied) {
    static_assert(BitSink<BitWriter>);
    static_assert(BitSource<BitReader>);
}
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement (copy the BitWriter/BitReader/concepts from post 1)**

Append to `prefix_free.hpp` inside the namespace (the code is identical to Task 3 step 3):

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

- [ ] **Step 4: Run, expect pass**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R test_prefix_free --output-on-failure
```

- [ ] **Step 5: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-prefix-free-stepanov
git commit -m "feat(prefix-free): add BitWriter, BitReader, concepts (mirrors post 1 per loose coupling)"
```

---

## Task 14: Implement Unary, Gamma, Vec in post 2 (TDD)

**Files:**
- Modify: `post/2026-05-prefix-free-stepanov/prefix_free.hpp`
- Modify: `post/2026-05-prefix-free-stepanov/test_prefix_free.cpp`

- [ ] **Step 1: Write failing tests**

Append to `test_prefix_free.cpp`:

```cpp
namespace {

template<typename Codec, typename T>
T round_trip(const T& v) {
    std::array<std::uint8_t, 64> buf{};
    BitWriter w(std::span(buf));
    Codec::encode(v, w);
    w.align();
    BitReader r(std::span<const std::uint8_t>(buf.data(), w.bytes_written()));
    return Codec::decode(r);
}

}  // namespace

TEST(PrefixFreeTest, UnaryRoundTrip) {
    for (std::uint64_t n : {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{5},
                            std::uint64_t{10}, std::uint64_t{32}}) {
        EXPECT_EQ(round_trip<Unary>(n), n) << "n = " << n;
    }
}

TEST(PrefixFreeTest, GammaRoundTrip) {
    for (std::uint64_t n : {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{100},
                            std::uint64_t{1024}, std::uint64_t{1} << 20}) {
        EXPECT_EQ(round_trip<Gamma>(n), n) << "n = " << n;
    }
}

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

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement Unary, Gamma, Vec**

Append to `prefix_free.hpp`:

```cpp
// ---- Leaf codec: Unary ------------------------------------------------------
// Encodes positive integers (>= 1).
// Codeword for n: (n - 1) zero bits, then a one bit. Length: n bits.

struct Unary {
    using value_type = std::uint64_t;

    template<BitSink S>
    static void encode(value_type n, S& sink) {
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

// ---- Leaf codec: Elias gamma -- (re-stated from post 1) ---------------------

struct Gamma {
    using value_type = std::uint64_t;

    template<BitSink S>
    static void encode(value_type n, S& sink) {
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

// ---- Combinator: Vec -- the free monoid lift (re-stated from post 1) -------

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

- [ ] **Step 4: Run, expect pass**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R "test_prefix_free.(Unary|Gamma|Vec)" --output-on-failure
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-prefix-free-stepanov
git commit -m "feat(prefix-free): add Unary, Gamma, Vec (Vec mirrors post 1 per loose coupling)"
```

---

## Task 15: Implement multi-parse decoder for the ambiguity demo (TDD)

**Files:**
- Modify: `post/2026-05-prefix-free-stepanov/prefix_free.hpp`
- Modify: `post/2026-05-prefix-free-stepanov/test_prefix_free.cpp`

The ambiguity demo needs:
1. A representation of a "code" as a `std::map<char, std::string>` (symbol to bit-pattern as 0/1 chars).
2. An `encode_string` function that maps a sequence of symbols to a concatenated bit-string.
3. An `enumerate_parses` function that returns ALL valid prefix-matched parses of a bit-string under the code.

These are not part of the codec abstraction; they exist purely to demonstrate the failure mode in section D of the post.

- [ ] **Step 1: Write failing tests**

Append to `test_prefix_free.cpp`:

```cpp
TEST(PrefixFreeTest, EncodeStringConcatenatesCodewords) {
    std::map<char, std::string> code{
        {'A', "0"},
        {'B', "01"},
        {'C', "10"},
        {'D', "010"},
    };
    EXPECT_EQ(encode_string(code, "DA"), "0100");
    EXPECT_EQ(encode_string(code, "AC"), "010");
    EXPECT_EQ(encode_string(code, "BA"), "010");
    EXPECT_EQ(encode_string(code, "D"), "010");
}

TEST(PrefixFreeTest, EnumerateParsesReturnsAllValidDecodings) {
    std::map<char, std::string> code{
        {'A', "0"},
        {'B', "01"},
        {'C', "10"},
        {'D', "010"},
    };
    auto parses = enumerate_parses(code, "010");
    // Expected three parses: "D", "AC", "BA"
    std::vector<std::string> as_strings;
    for (const auto& p : parses) {
        as_strings.emplace_back(p.begin(), p.end());
    }
    std::sort(as_strings.begin(), as_strings.end());
    ASSERT_EQ(as_strings.size(), 3u);
    EXPECT_EQ(as_strings[0], "AC");
    EXPECT_EQ(as_strings[1], "BA");
    EXPECT_EQ(as_strings[2], "D");
}
```

Add `#include <algorithm>` to `test_prefix_free.cpp` if not already present.

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement encode_string and enumerate_parses**

Append to `prefix_free.hpp`:

```cpp
// ---- Helpers for the ambiguity demonstration -------------------------------
// These are NOT part of the codec abstraction. They are included here only
// to support the demonstration in test_prefix_free.cpp showing that a
// non-prefix-free code admits multiple parses for the same bit string.

inline std::string encode_string(const std::map<char, std::string>& code,
                                 std::string_view symbols) {
    std::string out;
    for (char c : symbols) {
        auto it = code.find(c);
        if (it == code.end()) return {};  // unknown symbol
        out += it->second;
    }
    return out;
}

namespace detail {

inline void enumerate_parses_recursive(
    const std::map<char, std::string>& code,
    std::string_view bits,
    std::size_t pos,
    std::vector<char>& current,
    std::vector<std::vector<char>>& results)
{
    if (pos == bits.size()) {
        results.push_back(current);
        return;
    }
    for (const auto& [sym, codeword] : code) {
        if (pos + codeword.size() <= bits.size() &&
            bits.compare(pos, codeword.size(), codeword) == 0) {
            current.push_back(sym);
            enumerate_parses_recursive(code, bits, pos + codeword.size(),
                                       current, results);
            current.pop_back();
        }
    }
}

}  // namespace detail

inline std::vector<std::vector<char>> enumerate_parses(
    const std::map<char, std::string>& code,
    std::string_view bits)
{
    std::vector<std::vector<char>> results;
    std::vector<char> current;
    detail::enumerate_parses_recursive(code, bits, 0, current, results);
    return results;
}
```

Add `#include <string_view>` to the prefix_free.hpp includes if not already present.

- [ ] **Step 4: Run, expect pass**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R "test_prefix_free.(Encode|Enumerate)" --output-on-failure
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-prefix-free-stepanov
git commit -m "feat(prefix-free): add encode_string and enumerate_parses for ambiguity demo"
```

---

## Task 16: Ambiguity demo and Kraft verification tests

**Files:**
- Modify: `post/2026-05-prefix-free-stepanov/test_prefix_free.cpp`

These two tests are the demo content from sections D and H of the post. They use the helpers added in Task 15 plus the codecs added in Task 14.

- [ ] **Step 1: Write the ambiguity demo test**

Append to `test_prefix_free.cpp`:

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

- [ ] **Step 2: Write the Kraft verification test**

Append:

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
    // For each "block" of integers with the same length 2k+1, there are 2^k
    // such integers (from 2^k to 2^(k+1) - 1). Their contribution is
    // 2^k * 2^-(2k+1) = 2^-(k+1).
    // Summing over k = 0, 1, 2, ... gives 1/2 + 1/4 + 1/8 + ... = 1.
    double sum = 0.0;
    constexpr int N = 1 << 14;
    for (int n = 1; n <= N; ++n) {
        int len = 2 * static_cast<int>(std::bit_width(static_cast<unsigned>(n))) - 1;
        sum += std::ldexp(1.0, -len);
    }
    EXPECT_LE(sum, 1.0 + 1e-9);  // small tolerance for floating-point sum
}
```

Add `#include <cmath>` to the test file includes if not already present.

- [ ] **Step 3: Run, expect pass (no implementation change needed)**

```bash
cd ~/github/metafunctor-series/stepanov && make build && ctest --test-dir build -R "test_prefix_free.(NonPrefixFree|Kraft)" --output-on-failure
```

Expected: 3 tests pass.

- [ ] **Step 4: Run the entire post 2 test suite to confirm nothing regressed**

```bash
cd ~/github/metafunctor-series/stepanov && ctest --test-dir build -R test_prefix_free --output-on-failure
```

Expected: ~10 tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-prefix-free-stepanov
git commit -m "test(prefix-free): add ambiguity demo and Kraft inequality verification"
```

---

## Task 17: Draft post 2 prose (`index.md`)

**Files:**
- Modify: `post/2026-05-prefix-free-stepanov/index.md`

This task drafts the full ~2000-word post 2. Each step writes one section per the spec's Section 5 outline (sections A through J). Reference the spec at `docs/superpowers/specs/2026-04-23-pfc-stepanov-posts-design.md` for full per-section guidance.

When this task starts, change `draft: true` to `draft: false` in the frontmatter and set the `date` to today's date (or post-1's date + a few days).

- [ ] **Step 1: Write opening matter (lead-in italic + Section A)**

Lead-in italic: a one-sentence thesis (e.g., "Prefix-freeness is the property that lifts the free-monoid construction from types into bits.")

Section A "Motivating problem" (~200 words). Per spec: list of unsigned integers, fixed-width wastes, variable-width loses boundaries, two escape routes, prefix-free is the categorically right one.

- [ ] **Step 2: Write Section B (free monoid recap)**

Section B (~200 words). Condensed recap from the free-algebra post: monoid axioms, free monoid as lists, universal property, fold as the unique homomorphism. Link back to [free-algebra](/post/2026-03-free-algebra-stepanov/).

- [ ] **Step 3: Write Section C (the lifting question)**

Section C (~200 words). Specialize the universal property. The lift always exists. Question: is it invertible?

- [ ] **Step 4: Write Section D (the counter-example)**

Section D (~250 words + ~40 lines code). Include the non-prefix-free code map literally. Walk through the encoding of `[D]`, `[A, C]`, `[B, A]`, all yielding "010". Walk through the decoding ambiguity. The code block can be the encode_string + enumerate_parses snippet from Task 15 step 3, or a simplified inline-pseudocode version showing the three parses.

- [ ] **Step 5: Write Section E (prefix-free codes)**

Section E (~200 words + ~50 lines code). Define prefix-freeness. Show Unary and Gamma are prefix-free (use the code blocks from Task 14 step 3 verbatim). Trie-view explanation: each codeword is root-to-leaf, prefix-freeness means no codeword ends at an internal node, greedy decoding is unique.

- [ ] **Step 6: Write Section F (the lift works for prefix-free codes)**

Section F (~250 words, no code). Main theoretical payoff. Sketch both directions of the iff. Use the {A → "0", B → "00"} example for the negative direction. Conclude: the prefix-free-image submonoid is isomorphic to the free monoid; Vec from post 1 is this isomorphism made concrete. Cross-link to [post 1](/post/2026-05-codecs-functors-stepanov/).

- [ ] **Step 7: Write Section G (implementation and round-trip)**

Section G (~200 words + ~70 lines code). Note the loose-coupling decision: post 2 has its own minimal Vec. Include the Vec code block. Show a minimal round-trip demo using `Vec<Gamma>` for a list of integers. Cross-link back to post 1 for the full combinator catalogue.

- [ ] **Step 8: Write Section H (Kraft's inequality, light)**

Section H (~150 words + ~30 lines code). State Kraft. Verify computationally for unary and gamma using the test code from Task 16 step 2 (or a simplified version). Mention the converse and forward-pointer: "the forthcoming Information Theory by Construction series develops the proof and the converse."

- [ ] **Step 9: Write Section I (the bridge claim)**

Section I (~250 words, no code). Consolidate. Include the table from the spec mapping each combinator to its universal property. Frame: codecs are structure-preserving embeddings from the type algebra into the algebra of bit strings.

- [ ] **Step 10: Write Section J (cross-references and footnote)**

Section J (~120 words). Cross-references with Markdown links:

- Back to [Bits Follow Types](/post/2026-05-codecs-functors-stepanov/), [free-algebra](/post/2026-03-free-algebra-stepanov/), [duality](/post/2026-01-19-duality-stepanov/), [homomorphism](/post/2026-03-homomorphism-stepanov/)
- Forward (no hyperlink): "the forthcoming Information Theory by Construction series for the Kraft proof and the full universal-codes treatment"
- Footnote: PFC's `algebraic.hpp` and `codecs.hpp` for the production version.

- [ ] **Step 11: Soul check**

```bash
cd ~/github/metafunctor-series/stepanov
bash $CLAUDE_PLUGIN_ROOT/hooks/check-banned-phrases.sh post/2026-05-prefix-free-stepanov/index.md
```

Expected: no errors. Fix em-dashes if flagged.

- [ ] **Step 12: Render check**

```bash
cd ~/github/metafunctor-series/stepanov && mkdocs serve --dev-addr 127.0.0.1:8001 &
sleep 3
curl -s http://127.0.0.1:8001/post/2026-05-prefix-free-stepanov/ | head -50
kill %1
```

Expected: HTML renders. Cross-reference links to post 1 should resolve once both posts are rendered.

- [ ] **Step 13: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-05-prefix-free-stepanov/index.md
git commit -m "docs(prefix-free): draft post 2 prose (When Lists Become Bits)"
```

---

## Task 18: Update `docs/index.md`

**Files:**
- Modify: `docs/index.md`

Add the two new posts under the "Algebraic Foundations" section, immediately after the free-algebra entry.

- [ ] **Step 1: Find the Algebraic Foundations section**

```bash
grep -n "Algebraic Foundations" ~/github/metafunctor-series/stepanov/docs/index.md
```

Expected: locate the heading and the two existing entries (homomorphism, free-algebra).

- [ ] **Step 2: Add the two new entries**

In `docs/index.md`, find the bullet for free-algebra:

```markdown
- [Why Lists and Polynomials Are Universal](post/2026-03-free-algebra-stepanov/index.md): Free algebras, universal property
```

Append two new bullets immediately after:

```markdown
- [Bits Follow Types](post/2026-05-codecs-functors-stepanov/index.md): Codecs as functors; algebraic types lift to bit representations
- [When Lists Become Bits](post/2026-05-prefix-free-stepanov/index.md): Prefix-freeness as the free-monoid universal property in bit space
```

- [ ] **Step 3: Verify rendering**

```bash
cd ~/github/metafunctor-series/stepanov && mkdocs serve --dev-addr 127.0.0.1:8002 &
sleep 3
curl -s http://127.0.0.1:8002/ | grep -i "Bits Follow\|Lists Become" | head
kill %1
```

Expected: both new entries appear in the rendered index.

- [ ] **Step 4: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add docs/index.md
git commit -m "docs: add codecs-functors and prefix-free posts to Algebraic Foundations track"
```

---

## Task 19: Update `README.md`

**Files:**
- Modify: `README.md`

Add two rows to the posts table.

- [ ] **Step 1: Find the posts table**

```bash
grep -n "2026-03-free-algebra-stepanov" ~/github/metafunctor-series/stepanov/README.md
```

Expected: locate the row for the most recent post in the table.

- [ ] **Step 2: Add two new rows**

In `README.md`, after the row for `2026-03-free-algebra-stepanov`, append:

```markdown
| `2026-05-codecs-functors-stepanov/` | Codecs as functors; algebraic types lift to bit representations |
| `2026-05-prefix-free-stepanov/` | Prefix-freeness as the free-monoid universal property in bit space |
```

- [ ] **Step 3: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add README.md
git commit -m "docs: add codecs-functors and prefix-free posts to README table"
```

---

## Task 20: Update `mkdocs.yml` (verify and update if needed)

**Files:**
- Modify: `mkdocs.yml` (only if posts are explicitly listed)

This depends on the recon finding from Task 1.

- [ ] **Step 1: Check whether mkdocs.yml lists posts explicitly**

```bash
grep -n "stepanov" ~/github/metafunctor-series/stepanov/mkdocs.yml
```

If the result shows post-by-post entries (e.g., `- "2019-03-peasant-stepanov/index.md"`), proceed to step 2. If posts are auto-discovered (no per-post entries), skip to step 3.

- [ ] **Step 2: Add the two new posts to the nav (only if needed)**

Find the entry for free-algebra in `mkdocs.yml` and add two entries directly after it. Mirror the existing format exactly. Example pattern (verify against the actual file):

```yaml
- "2026-05-codecs-functors-stepanov/index.md"
- "2026-05-prefix-free-stepanov/index.md"
```

- [ ] **Step 3: Verify the build**

```bash
cd ~/github/metafunctor-series/stepanov && mkdocs build
```

Expected: build succeeds, both new posts appear in `site/`.

- [ ] **Step 4: Commit (only if mkdocs.yml was modified)**

```bash
cd ~/github/metafunctor-series/stepanov
git add mkdocs.yml
git commit -m "docs(mkdocs): list new codecs-functors and prefix-free posts in nav"
```

If `mkdocs.yml` was not modified (auto-discovery in use), skip the commit.

---

## Task 21: Update `stepanov-explorer.html` (or defer)

**Files:**
- Modify: `stepanov-explorer.html` and `docs/stepanov-explorer.html` (if structured data is straightforward)
- OR: defer with a note in the PR description

This depends on the recon finding from Task 1.

- [ ] **Step 1: Check explorer data structure**

If the explorer reads a JSON/YAML data block embedded in the HTML, locate it and add two entries for the new posts. The schema typically includes title, slug, weight, and a brief description; mirror the existing entries.

If the explorer is hand-coded with no structured data block, skip the update and note in the PR description: "stepanov-explorer.html update deferred to a follow-up commit; see plan Task 21."

- [ ] **Step 2: Add new entries (if structured)**

In the explorer's data block, after the entry for free-algebra, add two entries mirroring the schema. Verify by opening `stepanov-explorer.html` in a browser and confirming the two new posts appear in the visualization.

- [ ] **Step 3: Mirror to `docs/stepanov-explorer.html`**

If the docs copy is a separate file, apply the same edit. Verify they remain in sync.

- [ ] **Step 4: Commit (only if explorer was modified)**

```bash
cd ~/github/metafunctor-series/stepanov
git add stepanov-explorer.html docs/stepanov-explorer.html
git commit -m "docs(explorer): add new codecs-functors and prefix-free posts"
```

---

## Task 22: Polish edits to existing posts

**Files:**
- Modify: `post/2026-01-19-duality-stepanov/index.md`
- Modify: `post/2026-03-free-algebra-stepanov/index.md`
- Modify: `post/2026-01-18-synthesis-stepanov/index.md`

Three short forward-reference edits. All three pieces are independent and can be done in any order.

- [ ] **Step 1: Duality post forward-link**

Open `post/2026-01-19-duality-stepanov/index.md`. Find the encode/decode section. Locate the existing line:

> "While not explicitly covered, the type-erasure post implies this pattern: erasing a type is encoding its operations; invoking through the erased interface is decoding back to behavior."

Either replace or extend with a sentence explicitly forward-pointing to the new posts. For example, append after that paragraph:

> "The codec posts develop the encode/decode duality structurally: the [Bits Follow Types](/post/2026-05-codecs-functors-stepanov/) and [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/) posts show how the duality decomposes along the algebraic structure of types."

- [ ] **Step 2: Free-algebra post closing line**

Open `post/2026-03-free-algebra-stepanov/index.md`. Find the section that connects polynomials back to the free monoid (near the end of the body, before "Further Reading"). Append a closing line:

> "The free-monoid construction also extends to bit space via prefix-free codes; see [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/)."

- [ ] **Step 3: Synthesis post bullet**

Open `post/2026-01-18-synthesis-stepanov/index.md`. Find the list of post-summary bullets (the place that recaps the meta-pattern across the series). Add a bullet:

> "Codecs as functors: the algebraic structure of a type determines its codec, the same way it determines its algorithms. See [Bits Follow Types](/post/2026-05-codecs-functors-stepanov/) and [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/)."

- [ ] **Step 4: Soul check on all three modified files**

```bash
cd ~/github/metafunctor-series/stepanov
for f in post/2026-01-19-duality-stepanov/index.md post/2026-03-free-algebra-stepanov/index.md post/2026-01-18-synthesis-stepanov/index.md; do
  bash $CLAUDE_PLUGIN_ROOT/hooks/check-banned-phrases.sh "$f" || echo "FAILED: $f"
done
```

Expected: no failures.

- [ ] **Step 5: Verify rendering of the three posts**

```bash
cd ~/github/metafunctor-series/stepanov && mkdocs build
```

Expected: build succeeds without warnings.

- [ ] **Step 6: Commit**

```bash
cd ~/github/metafunctor-series/stepanov
git add post/2026-01-19-duality-stepanov/index.md \
        post/2026-03-free-algebra-stepanov/index.md \
        post/2026-01-18-synthesis-stepanov/index.md
git commit -m "docs: forward-link duality, free-algebra, synthesis posts to new codec posts"
```

---

## Task 23: Final verification and PR

**Files:**
- All modified files (final cross-check)

- [ ] **Step 1: Clean build from scratch**

```bash
cd ~/github/metafunctor-series/stepanov && make clean && make build
```

Expected: clean build succeeds with no warnings (or only warnings already present in main).

- [ ] **Step 2: Run the full test suite**

```bash
cd ~/github/metafunctor-series/stepanov && ctest --test-dir build --output-on-failure
```

Expected: all tests pass, including the existing series tests and the two new ones.

- [ ] **Step 3: Render the full mkdocs site**

```bash
cd ~/github/metafunctor-series/stepanov && mkdocs build --strict
```

Expected: build succeeds with `--strict` (no warnings, no broken links). If broken links are reported (cross-references to posts that don't exist yet), fix the links or add the posts.

- [ ] **Step 4: Soul-check every touched file**

```bash
cd ~/github/metafunctor-series/stepanov
for f in $(git diff --name-only main); do
  if [[ "$f" == *.md ]]; then
    bash $CLAUDE_PLUGIN_ROOT/hooks/check-banned-phrases.sh "$f" || echo "FAILED: $f"
  fi
done
```

Expected: no failures.

- [ ] **Step 5: Review the diff against main**

```bash
cd ~/github/metafunctor-series/stepanov && git diff --stat main
```

Expected files changed:
- 6 new files (3 per post: index.md, .hpp, test_*.cpp)
- 5 modified essential files (post/CMakeLists.txt, docs/index.md, README.md, mkdocs.yml, stepanov-explorer.html)
- 3 modified polish files (duality, free-algebra, synthesis index.md)

Total: ~14 files. If significantly different, investigate.

- [ ] **Step 6: Confirm with user before pushing or opening PR**

Per the user's standing convention, ask explicitly before pushing or opening a PR. Do NOT push or run `gh pr create` without confirmation. Provide the user with: the spec path, the plan path, the file diff stat, and a one-line summary of what the PR will say. Wait for the explicit go-ahead.

---

## Self-review (run after all tasks complete)

This is a checklist for the implementer at the END of execution, not part of the per-task work.

1. Spec coverage: every section from the spec has a corresponding task. Specifically:
   - Spec File layout, Frontmatter, Voice/code/test sections → Tasks 2, 11, 12, 17 (frontmatter set in scaffolding tasks; voice/code conventions enforced throughout)
   - Spec Post 1 outline (sections A-K) → Task 11 (one step per section)
   - Spec Post 2 outline (sections A-J) → Task 17 (one step per section)
   - Spec cross-references → Tasks 11, 17 (in-prose links), 22 (back-edits to existing posts)
   - Spec essential repo edits (5 items) → Tasks 2 + 12 (CMakeLists), 18, 19, 20, 21
   - Spec polish edits (3 items) → Task 22
   - Spec acceptance criteria → Task 23

2. Type consistency: codec types use `value_type`, `encode(const value_type&, S&)`, `decode(S&) -> value_type`. The Vec codec name and signature match between post 1 and post 2 (intentional duplication per loose-coupling).

3. Suite naming: post 1 uses `CodecsFunctorsTest`, post 2 uses `PrefixFreeTest`. All tests follow this convention.

4. No placeholders in the plan above (verified by scanning for the patterns listed in the writing-plans skill: deferred-implementation markers, vague directives, references to undefined items).
