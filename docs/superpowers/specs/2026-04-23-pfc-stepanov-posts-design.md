# Design: PFC bridge posts for the Stepanov series

**Date:** 2026-04-23
**Status:** Approved (awaiting plan)
**Project:** Two new Stepanov-series posts that bridge the existing series to the PFC library
**Source material:** `~/github/released/pfc/` (PFC, the prefix-free codecs library)
**Deliverable location:** `~/github/metafunctor-series/stepanov/post/`
**Series weights:** 20 (post 1), 21 (post 2)

## Goal

Bridge the existing Stepanov series to the PFC library by extracting two of PFC's strongest theoretical claims into pedagogical posts that fit the established series convention. Each post is a small self-contained world (~200-400 lines of code, ~2000 words of prose) that takes one categorical structure and shows what falls out.

The two posts together establish a thesis that extends Stepanov's original claim: **the algebraic structure of a type determines its codec, the same way it determines its algorithms.** The first post develops the bridge claim. The second post explains why one specific instance (the free-monoid lift to `Vec<C>`) actually works.

## Scope and decomposition

This spec covers **sub-project 1 of 3** in the broader PFC content arc:

1. **(this spec)** Two Stepanov-series posts: "Bits Follow Types" and "When Lists Become Bits"
2. **(separate brainstorm)** New series outline plus its first post: "Information Theory by Construction" (or equivalent title), starting with a full Kraft's inequality treatment
3. **(separate brainstorms, future)** Subsequent posts in the new series (universal codes, Huffman, arithmetic coding, succinct data structures, etc.)

This spec does NOT design the new series. It assumes the new series will exist and uses forward-references like `"...developed in depth in [Information Theory by Construction]"` as placeholders.

## Architectural decisions (locked in during brainstorming)

| Decision | Choice | Reasoning |
|---|---|---|
| Number of posts | Two | Each thesis (functorial composition; free-monoid lift) deserves its own post |
| Code source | Reimplement minimum from scratch in each post | Matches every other Stepanov post; PFC linked via `linked_project` and footnote |
| Categorical depth | Medium-light | Use functor, free monoid, universal property precisely; anchor every concept in concrete code first; avoid adjunctions, natural transformations, monads |
| Post 1 worked example | Config parser (variant + struct + optional + vector) | Hits all four combinators naturally |
| Post 2 Kraft treatment | Light | Full Kraft proof and converse belong in the new series; here Kraft is stated and verified for our codes |
| Post coupling | Loosely coupled, each standalone | Matches series convention "Start anywhere. Each post is self-contained." |

## File layout

For each new post, the directory follows the recent series convention (2026-03 batch: `YYYY-MM-<topic>-stepanov`), which is FLAT: no `examples/` or `tests/` subdirectories, no per-post CMakeLists. Each post is three files:

```
post/
├── 2026-05-codecs-functors-stepanov/
│   ├── index.md                   # Hugo article
│   ├── codecs_functors.hpp        # ~295 lines, minimal pedagogical implementation
│   └── test_codecs_functors.cpp   # GoogleTest suite + worked-example tests
└── 2026-05-prefix-free-stepanov/
    ├── index.md
    ├── prefix_free.hpp            # ~230 lines
    └── test_prefix_free.cpp       # GoogleTest suite + ambiguity demo + Kraft verification
```

The worked examples (config-parser for post 1; ambiguity demo and Kraft verification for post 2) are implemented as test cases in the corresponding `test_*.cpp` file. The same code appears as illustrative blocks in the prose. The recent series convention uses tests-as-demos rather than separate runnable executables.

Each test file is registered with `add_executable` directly in `post/CMakeLists.txt`, matching the pattern used for the 2026-03 batch.

Note: each post has its OWN `BitWriter`/`BitReader` and concept declarations. Per the loosely-coupled decision, the posts do not share code. The duplication is small (~80 lines of bit I/O + ~30 lines of concepts) and preserves the standalone property.

## Frontmatter convention

Both posts use this template:

```yaml
---
title: "Bits Follow Types"             # or "When Lists Become Bits"
date: 2026-05-XX                       # exact date set at draft time
draft: false
tags:
- C++
- generic-programming
- codecs
- information-theory
- category-theory
- functors                              # post 1 only
- monoids                               # post 2 only
categories:
- Computer Science
- Mathematics
series:
- stepanov
series_weight: 20                       # 21 for post 2
math: true
description: "..."
linked_project:
- pfc
- stepanov
---
```

Tags `codecs` and `information-theory` are new entries in the series taxonomy; they need to be added wherever the site's tag taxonomy is materialized.

## Voice, math, code, testing

- **Voice:** soul (Alex's voice). All prose drafts run through the soul check before commit. Em-dashes banned (active hook). Substitute commas, periods, colons, parentheses.
- **Math:** LaTeX. Display math (`$$...$$`) for important equations. Inline (`\(...\)`) for short references. Match the existing posts' density and notation conventions.
- **Code style:** C++23 (matches the parent `post/CMakeLists.txt` setting), headers-only. Use the `template<BitSink S>` shorthand for concept-constrained template parameters (matches PFC and modern C++20-and-later idiom; cleaner than the verbose `template<typename S> requires BitSink<S>`). Codecs expose a `value_type` typedef and `static` `encode`/`decode` member functions (matches PFC's style and the Stepanov pedagogical convention). Use `std::span` for byte buffers. Naming: leaf codecs in `PascalCase` (`Gamma`, `Unary`, `Bool`); combinators in `PascalCase` matching their type-algebra meaning (`Opt`, `Either`, `Pair`, `Vec`).
- **Test framework:** GoogleTest v1.14.0 (matches the rest of the Stepanov series; fetched via `FetchContent` in `post/CMakeLists.txt`). Use the `TEST(SuiteName, TestName)` macro and `EXPECT_EQ` / `EXPECT_TRUE` style assertions. Suite name should be `CodecsFunctorsTest` for post 1 and `PrefixFreeTest` for post 2.

## Post 1: "Bits Follow Types"

**Slot weight:** 20
**Code budget:** ~295 lines
**Prose budget:** ~2000 words
**Opening H2 inside the article:** "Codecs as Functors"

### Sections

#### A. Motivating problem (~150 words, no code)

You have an `optional<vector<pair<int, string>>>`. The TYPE decomposes structurally. Does the CODEC decompose the same way? If yes, you stop writing one-off encoders. State the post's thesis: codecs aren't ad-hoc bit formats; they're constructions on the algebraic structure of types.

#### B. The three concepts and bit I/O (~110 lines code, ~250 words)

`BitWriter`, `BitReader` (~60 lines combined). Concepts `BitSink`, `BitSource`, `Codec` (~30 lines). Key prose: a codec is an algorithm parameterized over BitSink/BitSource, not a class hierarchy. This is Stepanov's move applied to bit-level I/O.

#### C. A leaf codec: gamma (~25 lines code, ~150 words)

Implement Elias gamma. Brief mention of asymptotic optimality under a `1/n^2` prior; defer details to the new series. Establishes "what a leaf codec looks like" before introducing combinators.

#### D. Combinator 1: Opt (~25 lines code, ~200 words)

Signature:
```cpp
template<typename C>
struct Opt {
    using value_type = std::optional<typename C::value_type>;
    template<BitSink S>   static void encode(const value_type&, S&);
    template<BitSource S> static value_type decode(S&);
};
```

One tag bit + payload. Type-algebra interpretation: `Opt` corresponds to `1 + T`. First sign of functoriality: `Opt` takes a *codec*, returns a *codec*.

#### E. Combinator 2: Either (~30 lines code, ~250 words)

Binary version (n-ary mentioned as "PFC's `Variant` does this"). Tag bit selects branch. Type-algebra: `A + B`. Brief note that the n-ary tag-encoding cost (`log2(N)` bits) is suboptimal when branches have unequal probabilities; forward-pointer to Huffman and arithmetic coding in the new series.

#### F. Combinator 3: Pair (~25 lines code, ~150 words)

Concatenation of bit streams. Type-algebra: `A × B`. The encode/decode order matters and depends on a deeper property: prefix-freeness. **Forward-pointer to post 2:** this is what makes the concatenation recoverable.

#### G. Combinator 4: Vec (~40 lines code, ~200 words)

Length (gamma-coded) + concatenation. Type-algebra: `Vec<T>` is the free monoid on T. **Forward-pointer to post 2:** this lifting is the free-monoid universal property at the bit level.

#### H. The functorial framing (~250 words, no code)

Each combinator is a construction on codecs that mirrors a type-algebra operation:

| Combinator | Type algebra |
|---|---|
| `Opt<C>` | `1 + T` (coproduct with unit) |
| `Either<A, B>` | `A + B` (binary coproduct) |
| `Pair<A, B>` | `A × B` (binary product) |
| `Vec<C>` | `List<T>` (free monoid on T) |

The bridge thesis (the reason this post belongs in the series): **the algebraic structure of a type determines its codec, the same way it determines its algorithms.** Stepanov's claim was the first half. The second half is what these two posts add.

Brief paragraph noting each combinator is functorial in a precise sense; pointer to category theory for the formal version.

#### I. The composed example (~50 lines code, ~200 words)

Before the worked example, briefly introduce three auxiliary codecs it depends on:

- `Bool`: 1-bit codec for booleans (~5 lines).
- `Signed<C>`: zigzag adapter that wraps an unsigned codec to handle signed integers (~10 lines).
- `String`: alias for `Vec<Byte>` plus an 8-bit `Byte` leaf codec (~15 lines).

`String` is itself an instance of the combinator pattern: a string IS a vector of bytes encoded by a leaf codec. The auxiliary codecs reinforce that the combinators apply recursively to themselves.

Then the worked example:

```cpp
using Value  = std::variant<int64_t, std::string, bool>;
struct Entry { std::string key; std::optional<Value> value; };
using Config = std::vector<Entry>;

using ValueCodec  = Either<Signed<Gamma>, String, Bool>;
using EntryCodec  = Pair<String, Opt<ValueCodec>>;
using ConfigCodec = Vec<EntryCodec>;

ConfigCodec::encode(c, sink);
Config c2 = ConfigCodec::decode(source);
assert(c == c2);
```

The codec types mirror the value types exactly. No manual layout, no marshaling, no hand-placed length headers. Encoding emerges from the type structure.

#### J. Why this matters (~200 words, no code)

Zero-copy invariant emerges from the construction. Wire format IS the bit pattern, recoverable by structural recursion. "Serialization" and "the value" stop being two separate things; they're two views of one algebraic object. This dissolves marshaling rather than solves it.

#### K. Cross-references and footnote (~120 words)

Forward to post 2. Back to free-algebra post (free-monoid construction being lifted), duality post (encode/decode duality decomposes structurally), homomorphism post (the lifting is a homomorphism from the type algebra to the codec algebra). Footnote pointing at PFC for production use, with specific header references (`include/pfc/algebraic.hpp`, `include/pfc/codecs.hpp`).

## Post 2: "When Lists Become Bits"

**Slot weight:** 21
**Code budget:** ~230 lines
**Prose budget:** ~2000 words
**Opening H2 inside the article:** "Prefix-Free Codes and the Free Monoid"

### Sections

#### A. Motivating problem (~200 words, no code)

You have a list of unsigned integers. Encode them as a single bit string. Fixed-width wastes space on small values. Variable-width loses the boundaries. Two escape routes: length-prefix each item, or choose a code where the *structure* makes boundaries unambiguous (a prefix-free code). The second route is the categorically right one: it corresponds exactly to lifting the free-monoid construction into bit space.

#### B. The free monoid, recap (~200 words, minimal code)

Condensed recap from the free-algebra post. Monoid = associative op + identity. The free monoid on S = "lists over S" with concatenation and `[]`. Universal property: for any monoid M and function `f: S → M`, there's a unique monoid homomorphism `Free(S) → M` extending f, namely fold.

#### C. The lifting question (~200 words, minimal code)

We want to extend a per-element encoding `code: T → bit_string` to `vector<T> → bit_string` by `[x1, ..., xn] → code(x1) ++ ... ++ code(xn)`. By the universal property, the lift always exists (fold is always well-defined). The question is whether it's *invertible*: can we recover the original vector from the bit string?

#### D. The counter-example (~40 lines code, ~250 words)

Use the rich three-way ambiguity:

```cpp
// A "code" that is not prefix-free:
//   A → "0"
//   B → "01"
//   C → "10"
//   D → "010"
//
// "0"  is a prefix of "01" and "010".
// "01" is a prefix of "010".
//
// Consider the bit string "010". Three valid parses:
//   [D]    = "010"
//   [A, C] = "0" + "10"  = "010"
//   [B, A] = "01" + "0"  = "010"
//
// The decoder has no way to choose. The lift fails to be injective.
```

Demo (a test case `TEST(PrefixFreeTest, NonPrefixFreeIsAmbiguous)` in `test_prefix_free.cpp`): given the non-prefix-free code above, encode each of the sequences `[D]`, `[A, C]`, and `[B, A]` and assert they all produce the same bit string `"010"`. Then run a multi-parse decoder that enumerates all valid prefix matches at each position and assert it returns three distinct sequences for the input `"010"`. The test makes the failure mode concrete: the decoder genuinely has no principled way to choose. The same code appears as an illustrative block in the prose.

#### E. Prefix-free codes (~50 lines code, ~200 words)

Definition: no codeword is a prefix of another. Implement unary and gamma; show both are prefix-free. The trie view: each codeword is a root-to-leaf path; prefix-freeness means no codeword ends at an internal node. Greedy left-to-right decoding always produces a unique parse because every bit sequence has a unique path.

#### F. The free-monoid lift works for prefix-free codes (~250 words, informal proof sketch)

The main theoretical payoff. Specialize the universal property:
- S = T, M = (bit strings, ++, ""), f = the encoding function
- Lift `encode_list: vector<T> → bit_string` always exists (as fold)
- *Injective iff f is prefix-free*

Sketch both directions:
- If prefix-free: greedy decode is always unique (trie argument). Therefore `encode_list` has a left inverse.
- If not prefix-free: exhibit a concrete ambiguity. If `code(b) = code(a) ++ s` and `s` lies in the image of fold (as with `{A → "0", B → "00"}` where `s = "0" = code(A)`), then `encode_list([b]) = encode_list([a, ...])`.

The bigger claim: the prefix-free-image submonoid of BitStrings is *isomorphic* to the free monoid on T via the lift. When the code is prefix-free, "lists of T" and "bit strings built from its codewords" are literally the same monoid. `Vec<C>` from post 1 is this isomorphism made concrete.

#### G. Implementation and round-trip (~70 lines code, ~200 words)

Per the loosely-coupled decision, post 2's `prefix_free.hpp` contains its own minimal `Vec<C>` implementation (independent of post 1's `codecs.hpp`). The implementation is identical in spirit but the file is standalone so post 2 can be read first. Demo: a list of integers, encode with `Vec<Gamma>`, decode, show equality. The post mentions in passing that this is the SAME `Vec` introduced in post 1 (with a back-reference for readers who want the full combinator catalogue).

#### H. Kraft's inequality, lightly (~30 lines code, ~150 words)

State Kraft: for codeword lengths `l_1, l_2, ...` of a prefix-free code, `∑ 2^{-l_i} ≤ 1`. Verify computationally:
- Unary: lengths 1, 2, 3, ... → `1/2 + 1/4 + 1/8 + ... = 1`
- Gamma: lengths 1, 3, 3, 5, 5, 5, 5, 7, ... → `≤ 1`

Mention the converse (every Kraft-satisfying length vector is realizable) is true and characterizes exactly which length-vectors are achievable. Forward-pointer: the forthcoming Information Theory by Construction series develops the proof.

#### I. The bridge claim (~250 words, no code)

Consolidate. Prefix-freeness is exactly the property that lifts the free-monoid construction to bit space injectively. The pattern generalizes:

| Combinator | Universal property being lifted |
|---|---|
| `Opt<C>` | Coproduct with unit `1 + T` |
| `Either<A, B>` | Binary coproduct `A + B` |
| `Pair<A, B>` | Binary product `A × B` (given both factors prefix-free) |
| `Vec<C>` | Free monoid on T (given C prefix-free) |

Unifying frame: codecs are structure-preserving embeddings from the type algebra into the algebra of bit strings. When the type algebra is free, the lift is canonical. When it's free AND prefix-free, the lift is reversible.

#### J. Cross-references and footnote (~120 words)

Back to post 1 (where Vec lives), free-algebra post (the construction being lifted), duality post (prefix-freeness is what makes decode well-defined), homomorphism post (the lift is a monoid homomorphism). Forward to the forthcoming Information Theory by Construction series for the Kraft proof and full universal-codes treatment. Footnote to PFC's `algebraic.hpp` and `codecs.hpp`.

## Cross-references within the new posts

**Post 1 to Post 2 (forward):**
- Pair section: "the encode/decode order is recoverable for a deeper reason; see [When Lists Become Bits](...) for why prefix-freeness makes concatenation reversible."
- Vec section: "the lifting from a codec for T to a codec for `vector<T>` is the free-monoid universal property at the bit level; [When Lists Become Bits](...) explains why it works."

**Post 1 back-references:**
- Free-algebra post (Vec corresponds to the free monoid)
- Duality post (encode/decode are duals; here we see them decompose structurally)
- Homomorphism post (codecs are homomorphisms from type algebra to codec algebra)

**Post 2 back-references:**
- Post 1 (where Vec was introduced)
- Free-algebra post (the construction being lifted)
- Duality post (prefix-freeness is what makes decode well-defined)
- Homomorphism post (the lift is a monoid homomorphism)

**Post 2 forward-references:**
- The forthcoming Information Theory by Construction series for the Kraft proof, converse, and full universal-codes treatment.

### Handling the forthcoming-series references

The new "Information Theory by Construction" series is still being designed (sub-project 2 of this PFC arc). Until that series exists:

- All forward-references to it appear as the phrase "the forthcoming Information Theory by Construction series" or "in a follow-up post," with no hyperlink. This avoids dead links at first publication.
- Once the new series exists, a follow-up edit pass converts these phrases to live cross-references.
- This applies to the references in post 2 sections H and J, and to the post 2 forward-references list.

## Edits to existing repository files (essential)

These must be done as part of the same PR so the new posts are discoverable:

1. **`post/CMakeLists.txt`**: add two `add_executable` blocks for `test_codecs_functors` and `test_prefix_free`, matching the pattern used for the 2026-03 batch (free-algebra, homomorphism, etc.). Each block: `add_executable`, `target_link_libraries(... GTest::gtest_main)`, `target_include_directories(... PRIVATE <postdir>)`, `add_test`.
2. **`docs/index.md`**: add both posts to the post list under the **Algebraic Foundations** section, immediately after free-algebra.
3. **`README.md`**: add two rows to the posts table (lines 22-35).
4. **`mkdocs.yml`**: verify post listing convention; update if posts are explicitly listed rather than auto-discovered. Inspect before editing.
5. **`stepanov-explorer.html`** (and copy in `docs/`): add the two new posts to the explorer's data. Inspect structure before editing; if it requires substantial refactoring, defer to a follow-up commit and note in PR description.

## Edits to existing posts (polish, same PR)

These are short, low-risk, and complete the cross-reference web at first publication:

6. **Duality post** (`post/2026-01-19-duality-stepanov/index.md`), encode/decode section: extend the existing "While not explicitly covered, the type-erasure post implies this pattern" sentence to mention the new codec posts.
7. **Free-algebra post** (`post/2026-03-free-algebra-stepanov/index.md`), near the polynomial-connection section: add a closing line: "The free-monoid construction extends to bit space via prefix-free codes; see [When Lists Become Bits](...)."
8. **Synthesis post** (`post/2026-01-18-synthesis-stepanov/index.md`): add a brief paragraph or bullet noting the codec posts as another instance of the structure-determines-implementation pattern.

## Track placement

After these edits, the **Algebraic Foundations** track on the index page will read:

- The Maps Between Structures (homomorphism)
- Why Lists and Polynomials Are Universal (free-algebra)
- **Bits Follow Types** (NEW, slot 20)
- **When Lists Become Bits** (NEW, slot 21)

This is a coherent four-post arc: structure-preserving maps, free constructions, free constructions in code, free constructions in bits.

## Acceptance criteria

The work is done when:

1. Both post directories exist with `index.md`, the corresponding `.hpp`, `examples/`, `tests/`, and `CMakeLists.txt`.
2. `make build` from the series root builds both posts' code without warnings under the existing CMake configuration.
3. `make test` (or `ctest --test-dir build`) passes all tests for both posts (executables `test_codecs_functors` and `test_prefix_free`). Round-trip property tests cover at minimum: gamma alone, `Opt<Gamma>` (both states), `Either` (both branches), `Pair`, `Vec` (empty, singleton, multi-element), and the composed config-parser example. The ambiguity demo (a test case in `test_prefix_free.cpp`) constructs the encodings of `[D]`, `[A, C]`, and `[B, A]` under the non-prefix-free code and asserts they all produce the same bit string `"010"`, demonstrating the failure mode concretely.
4. Both `index.md` files pass the soul check (no banned phrases including em-dashes).
5. `mkdocs serve` (or the configured site builder) renders both posts cleanly with all cross-reference links resolving.
6. The five essential file edits (1 to 5 above) are applied.
7. The three polish edits (6 to 8 above) are applied.
8. PR description summarizes the bridge thesis and lists the eight changed/added files.

## What is NOT in scope

- Designing or drafting the new "Information Theory by Construction" series.
- Writing the Kraft inequality proof or the converse (deferred to the new series).
- Re-implementing PFC in any larger sense than the per-post `*.hpp` files.
- Adding new combinators to PFC itself.
- Refactoring the Stepanov series beyond the targeted edits listed above.
- Syncing the new posts to the Hugo site at metafunctor.com via the `mf` toolkit. Once the posts merge in the Stepanov repo, a separate `mf` sync pass propagates them to Hugo. The Hugo sync is a follow-up commit (in the metafunctor repo), not part of this PR.

## Open questions and known unknowns

These get resolved during the implementation plan or by inspection at draft time:

- Exact dates for the two posts (currently `2026-05-XX`). Pick during draft.
- Whether `mkdocs.yml` lists posts explicitly or auto-discovers. Inspect first.
- Whether `stepanov-explorer.html` uses a structured data file or hand-coded entries. If structured, add entries; if hand-coded, defer.
- The series weight numbering: confirm 20, 21 don't collide with anything already weighted there. Check `post/*/index.md` for existing weights before final commit.
- ~~Catch2 version and integration pattern.~~ Resolved during reconnaissance: series uses GoogleTest v1.14.0 fetched via `FetchContent`, with flat post directory structure (no `tests/` or `examples/` subdir).

## Implementation sequence (handoff to writing-plans)

The implementation plan should structure the work in approximately this order:

1. **Reconnaissance** (read-only): inspect `mkdocs.yml`, `stepanov-explorer.html`, and `grep -r series_weight post/ | sort` to verify no collisions at 20, 21.
2. **Post 1 scaffolding**: create `post/2026-05-codecs-functors-stepanov/` with empty `codecs_functors.hpp`, `test_codecs_functors.cpp`, `index.md`. Add the `add_executable` block to `post/CMakeLists.txt`.
3. **Post 1 implementation (TDD)**: build up `codecs_functors.hpp` one component at a time (BitWriter/BitReader + concepts; Gamma; Opt; Either; Pair; Vec; Bool/Signed/String aux), with corresponding test cases in `test_codecs_functors.cpp`.
4. **Post 1 composed example**: add `TEST(CodecsFunctorsTest, ConfigParserRoundTrip)` exercising the full stack.
5. **Post 1 prose**: write `index.md`, sections A through K. Run soul check.
6. **Post 2 scaffolding**: create `post/2026-05-prefix-free-stepanov/` with empty `prefix_free.hpp`, `test_prefix_free.cpp`, `index.md`. Add `add_executable` block to `post/CMakeLists.txt`.
7. **Post 2 implementation (TDD)**: build up `prefix_free.hpp` (BitWriter/BitReader + concepts; Unary; Gamma; Vec; the multi-parse decoder used by the ambiguity test).
8. **Post 2 ambiguity and Kraft tests**: `TEST(PrefixFreeTest, NonPrefixFreeIsAmbiguous)` and `TEST(PrefixFreeTest, KraftInequalityHolds)`.
9. **Post 2 prose**: write `index.md`, sections A through J. Run soul check.
10. **Series integration**: update `docs/index.md`, `README.md`, `mkdocs.yml` (if posts are explicitly listed), `stepanov-explorer.html` (or defer with note).
11. **Polish edits**: update duality, free-algebra, and synthesis posts with forward-references.
12. **Final verification**: `make build`, `make test`, `mkdocs serve`, soul check on all touched files, link check.
13. **Commit and PR**: single PR with all changes; description summarizes the bridge thesis.

The writing-plans skill takes over here to convert this sequence into a detailed task plan with explicit checkpoints.
