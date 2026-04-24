# Design: Algebra over Wire Formats (series bootstrap, posts 1 and 2)

**Date:** 2026-04-24
**Status:** Awaiting approval
**Project:** New blog series at `~/github/metafunctor-series/wire-formats/`
**Source material:** `~/github/released/pfc/` (PFC, the prefix-free codecs library)
**Companion series:** Stepanov (`~/github/metafunctor-series/stepanov/`)
**Spec home (after bootstrap):** This file will be moved to
`~/github/metafunctor-series/wire-formats/docs/superpowers/specs/2026-04-24-bootstrap-and-posts-1-2-design.md`
during Task 1 of the implementation plan. It lives in the stepanov repo for
now because the wire-formats repo does not yet exist (HARD-GATE: no scaffolding
before approval).

## Goal

Bootstrap a new pedagogical blog series, "Algebra over Wire Formats," that
sits alongside the Stepanov series and develops information theory by
construction. The Stepanov bridge posts ("Bits Follow Types," "When Lists
Become Bits") promise this series exists and forward-reference it. This
project makes good on that promise by:

1. Creating the new repo with the same scaffold pattern as stepanov.
2. Shipping the first two posts (Kraft and McMillan) as a pair.
3. Updating the Stepanov bridge posts' forward-references to point at the
   new series's first post (Kraft) by live link.

The series's central thesis: **a code is a hypothesis about the source.**
Each universal code corresponds to a different prior over the integers; each
optimization (Huffman, arithmetic) is best under different assumptions; each
data structure (succinct bit vectors, RoaringBitmap) is the right answer
under different access patterns. The series develops these as one coherent
algebraic story, with PFC as the production reference.

## Scope and decomposition

This spec covers **sub-project 2 of 3** in the broader PFC content arc:

1. **(done, sub-project 1)** Two Stepanov bridge posts establishing the
   bridge thesis (completed and pushed).
2. **(this spec, sub-project 2)** Bootstrap the new series repo + ship the
   first two posts (Kraft and McMillan) as a pair.
3. **(separate brainstorms, sub-project 3)** Subsequent posts in the new
   series: posts 3-13. Each batch (or each post individually) gets its own
   brainstorm/spec/plan cycle.

This spec does NOT design the full content of posts 3-13 in detail. It
records the planned arc with dates so future brainstorms have a reference
schedule, but the per-post outlines for 3-13 are out of scope here.

## Architectural decisions (locked in during brainstorming)

| Decision | Choice | Reasoning |
|---|---|---|
| Series scope | Medium (10-15 posts) | MacKay-shaped; matches Alex's book-shaped paradigm work; PFC has roughly the right amount of code |
| Series shape | Theory-first arc | Matches Stepanov-series voice; honors the bridge posts' Kraft forward-reference |
| Series location | New repo `~/github/metafunctor-series/wire-formats/` | Mirrors stepanov pattern; standalone build/test discipline; sibling to stepanov |
| Series title | "Algebra over Wire Formats" | "Over" positions wire formats as the ground structure on which algebra is built; matches the Stepanov-bridge framing |
| First sub-project posts | 1 (Kraft) and 2 (McMillan) | Iff theorem ships as a unit; series launches with the foundational result intact |
| Backdating | 2020-2025 across 12 posts; synthesis at 2026-05 | Matches Stepanov's pacing (~2/year); reflects that the work IS that old intellectually; chronologically consistent with the bridge-posts' "long-running" framing once published |

## Repo structure

The new repo at `~/github/metafunctor-series/wire-formats/` mirrors the
stepanov scaffold exactly:

```
wire-formats/
├── post/                                   # Per-post directories
│   ├── 2020-03-kraft-wire-formats/
│   │   ├── index.md
│   │   ├── kraft.hpp
│   │   └── test_kraft.cpp
│   ├── 2020-09-mcmillan-wire-formats/
│   │   ├── index.md
│   │   ├── mcmillan.hpp
│   │   └── test_mcmillan.cpp
│   └── CMakeLists.txt                      # Top-level CMake; FetchContent GoogleTest
├── docs/                                   # mkdocs site source
│   ├── index.md
│   └── about.md
├── mkdocs.yml
├── Makefile                                # WITH the FIXED sync target (not the bug)
├── CLAUDE.md                               # AI assistant instructions
├── README.md
├── LICENSE                                 # MIT, matching stepanov
└── .gitignore                              # Scoped Makefile rule (NOT global)
```

Key differences from stepanov's current state:
- **`Makefile`**: contains the FIXED sync target from the start (per-directory
  rsync, no global `--delete`). This is the bug we hit during the Stepanov
  Hugo sync; no other clone of any repo should hit it again.
- **`.gitignore`**: the `Makefile` exclude rule is scoped to `**/build/Makefile`
  (or equivalent) so the top-level Makefile is trackable. Stepanov's repo
  excludes the Makefile globally, which means it never made it into version
  control. We do better here.

## Frontmatter convention

Both posts use this template (mirrors the Stepanov frontmatter style):

```yaml
---
title: "Kraft's Inequality"            # or "McMillan's Converse"
date: 2020-03-22                       # backdated; see arc below
draft: false
tags:
- C++
- information-theory
- coding-theory
- prefix-free
- combinatorics                         # post 1 only
- constructive-proof                    # post 2 only
categories:
- Computer Science
- Mathematics
series:
- wire-formats                         # NEW series slug
series_weight: 1                       # 2 for post 2; ascending through arc
math: true
description: "..."
linked_project:
- pfc
- wire-formats
---
```

The `wire-formats` series slug needs to be added to the metafunctor `mf` series database
(`mf series create wire-formats ...`), and the `_index.md` in
`content/series/wire-formats/` needs to exist on the metafunctor side. Both are
bootstrap tasks.

## Voice, code, test conventions

These are inherited from the Stepanov pattern (no need to re-design):

- **Voice:** Alex's voice via the soul plugin. Em-dashes blocked by hook.
- **Math:** LaTeX (`$$...$$` for display, `\(...\)` for inline).
- **Code style:** C++23, headers-only, GoogleTest, flat per-post directory
  layout. Each post's `*.hpp` is ~100-400 lines pedagogical implementation.
- **Categorical depth:** medium-light. Use precise terms (prefix-free,
  Kraft inequality, McMillan converse, Shannon-Fano construction) but
  anchor every concept in concrete code first. Avoid heavy CT machinery.

## The 13-post arc (forward-looking; only posts 1-2 specified in detail)

Backdating across 2020-2025, ~2 posts/year. Specific days chosen to avoid
collisions with existing metafunctor posts (verified by grep against
`content/post/*/index.md`).

| # | Slot | Title (working) | Date | Directory |
|---|------|---|------|---|
| 1 | wt 1 | Kraft's Inequality | 2020-03-22 | `2020-03-kraft-wire-formats/` |
| 2 | wt 2 | McMillan's Converse | 2020-09-13 | `2020-09-mcmillan-wire-formats/` |
| 3 | wt 3 | Universal Codes as Priors | 2021-03-29 | `2021-03-priors-wire-formats/` |
| 4 | wt 4 | Unary and Elias Gamma | 2021-08-08 | `2021-08-elias-gamma-wire-formats/` |
| 5 | wt 5 | Elias Delta and Omega | 2022-02-13 | `2022-02-elias-delta-omega-wire-formats/` |
| 6 | wt 6 | Fibonacci Coding | 2022-07-17 | `2022-07-fibonacci-wire-formats/` |
| 7 | wt 7 | Rice / Golomb | 2023-02-19 | `2023-02-rice-golomb-wire-formats/` |
| 8 | wt 8 | VByte / Varint | 2023-08-14 | `2023-08-vbyte-wire-formats/` |
| 9 | wt 9 | Huffman | 2024-03-25 | `2024-03-huffman-wire-formats/` |
| 10 | wt 10 | Arithmetic Coding | 2024-09-04 | `2024-09-arithmetic-coding-wire-formats/` |
| 11 | wt 11 | Succinct Bit Vectors and Rank/Select | 2025-03-09 | `2025-03-succinct-wire-formats/` |
| 12 | wt 12 | RoaringBitmap | 2025-08-10 | `2025-08-roaring-bitmap-wire-formats/` |
| 13 | wt 13 | Synthesis: Codecs as Structure | 2026-05-15 | `2026-05-synthesis-wire-formats/` |

Posts 3-13 are scope for sub-project 3 (each gets its own brainstorm).

## Post 1: "Kraft's Inequality"

**Slot weight:** 1
**Date:** 2020-03-22
**Code budget:** ~200 lines
**Prose budget:** ~2000 words
**Opening H2 inside the article:** "Kraft's Inequality"

### Sections

#### A. Motivating problem (~200 words, no code)

You want to assign codewords to symbols. The codewords must be uniquely
decodable (you can recover the symbol sequence from the bit string). The
question: which length vectors `(l_1, l_2, ..., l_n)` are achievable for
prefix-free codes? Kraft's inequality is the answer.

#### B. The trie view (~250 words + ~50 lines code)

Visualize codewords as paths in a binary tree. Each codeword is a
root-to-leaf path; each internal node is a partial codeword. Prefix-freeness
means: no codeword's path passes through another codeword's leaf. Show this
by drawing (in ASCII or a small image) the tree for the code
`{A → "0", B → "10", C → "110", D → "111"}`.

Implement a tiny `BinaryTree` data structure for codewords (~50 lines)
and verify the prefix-free property mechanically.

#### C. The inequality (~250 words + ~30 lines code)

Statement: for a prefix-free code with codeword lengths `l_1, l_2, ..., l_n`,
$$\sum_{i=1}^{n} 2^{-l_i} \leq 1.$$

Geometric intuition: each codeword "claims" a fraction `2^{-l_i}` of the
unit interval (or equivalently, of the leaves in a depth-`l_max` complete
binary tree). The total claimed fraction is at most 1 because the
fractions are disjoint.

Implement a function `kraft_sum(lengths)` that returns the sum, and
verify Kraft on Unary, Gamma, the example code from B, and a few others.

#### D. The proof (~300 words + ~30 lines code)

Sketch the binary-tree proof rigorously enough for a careful reader:

1. Embed the prefix-free code in the depth-`l_max` complete binary tree
   (where `l_max = max l_i`).
2. Each codeword of length `l_i` corresponds to a subtree of size
   `2^{l_max - l_i}` leaves.
3. Prefix-freeness means these subtrees are disjoint.
4. The total number of leaves used is `sum 2^{l_max - l_i}`, which is
   at most the total number of leaves `2^{l_max}`.
5. Dividing by `2^{l_max}` gives Kraft.

Implement the embedding computationally for small examples to make the
proof concrete. The code should print: "code X has length-l_max tree of
size N leaves; codewords occupy {S_1, S_2, ...} subtrees of sizes
{n_1, n_2, ...}, total {sum} ≤ {N}."

#### E. What Kraft gives us (~250 words, no code)

Three immediate consequences:

1. **A budget.** A codeword of length 3 costs 1/8 of the budget. Once you
   spend the budget, you cannot add more codewords without violating
   prefix-freeness.
2. **A trade-off.** Shorter codewords for some symbols means longer for
   others. Information theory's optimal trade-off (assigning length
   `-log_2 p_i` to a symbol with probability `p_i`) saturates Kraft when
   the lengths are achievable.
3. **A diagnostic.** If your length vector violates Kraft, no prefix-free
   code with those lengths exists. This is the converse direction
   (McMillan), which the next post develops.

#### F. The converse, foreshadowed (~150 words, no code)

State McMillan's theorem (1956): if a length vector satisfies Kraft, then
a prefix-free code with those lengths exists. This is the reverse
direction. Forward-pointer to post 2: "the next post in this series,
McMillan's Converse, proves this and gives a constructive recipe."

#### G. Cross-references and footnote (~100 words)

- Forward: [McMillan's Converse](/post/2020-09-mcmillan-wire-formats/)
- Cross-series: [Bits Follow Types](/post/2026-05-codecs-functors-stepanov/)
  introduces codec combinators on type algebra, where Kraft is what
  underwrites their compositional correctness.
- Cross-series: [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/)
  develops the consequence: prefix-freeness lifts the free-monoid
  construction into bit space.
- Footnote to PFC: production codecs that all satisfy Kraft live at
  `include/pfc/codecs.hpp`. The whole library is structured around
  Kraft-satisfying universal codes.

## Post 2: "McMillan's Converse"

**Slot weight:** 2
**Date:** 2020-09-13
**Code budget:** ~200 lines
**Prose budget:** ~2000 words
**Opening H2 inside the article:** "McMillan's Converse"

### Sections

#### A. The promise (~150 words, no code)

Recap from post 1: Kraft says prefix-free codes have length vectors
satisfying `∑ 2^{-l_i} ≤ 1`. McMillan says the converse: any length vector
satisfying Kraft is realizable by SOME prefix-free code. This post proves
it constructively (you get an algorithm) and discusses the consequences.

#### B. The Shannon-Fano-like construction (~300 words + ~80 lines code)

Algorithm:
1. Sort lengths in non-decreasing order: `l_1 ≤ l_2 ≤ ... ≤ l_n`.
2. For each `i`, assign codeword `c_i` = the binary representation of
   `floor(c_{i-1} + 2^{l_max - l_i})` left-padded to `l_i` bits, where
   `c_0 = 0`.
3. Equivalently: walk the trie left-to-right, assigning the next available
   leaf at depth `l_i` to symbol `i`.

Implement the construction (`build_prefix_free_code(lengths) -> code_map`).
Show it produces a valid prefix-free code for any Kraft-satisfying input
and verify with the Kraft sum from post 1.

#### C. Why it works (~250 words, no code)

The trie-walking interpretation: at each step, the next codeword "fits"
because Kraft guarantees enough budget remains. Sketch:

- After placing codewords `1..i-1`, the remaining budget is
  `1 - ∑_{j<i} 2^{-l_j}`.
- We need to place a codeword of length `l_i`, which costs `2^{-l_i}`.
- Kraft on the original lengths says `∑_j 2^{-l_j} ≤ 1`, so the remaining
  budget is at least `2^{-l_i}` plus whatever's left for the rest.

The construction never "runs out" because Kraft pre-certifies that the
budget suffices.

#### D. The deeper version of the converse (~250 words, no code)

McMillan actually proved something stronger: any **uniquely decodable**
code (not necessarily prefix-free) has length vector satisfying Kraft.
That is, prefix-freeness is not strictly necessary for decodability, but
the length vector must still satisfy Kraft. So you might as well use a
prefix-free code: it has the same length efficiency, and decoding is
simpler (no lookahead).

State the theorem; sketch the slick proof using the L-th power of the
code (the Plotkin / standard textbook proof). The full proof is left to
McMillan's original paper or Cover & Thomas; a sketch suffices here.

#### E. Implications (~250 words, no code)

Two big implications:

1. **There is no advantage to non-prefix-free uniquely decodable codes.**
   Anything you can do with a uniquely decodable code, you can do with a
   prefix-free code of the same length efficiency. So we can restrict
   attention to prefix-free codes without loss of generality.

2. **The optimal lengths exist.** Given any probability distribution
   `(p_1, ..., p_n)`, the lengths `l_i = ceil(-log_2 p_i)` satisfy Kraft
   (Kraft's inequality holds), so a prefix-free code achieving them
   exists. This is what underwrites Huffman coding (which finds
   *integer-length* optimal codes) and arithmetic coding (which achieves
   *non-integer* lengths in the limit).

#### F. The constructive recipe in PFC (~150 words + ~30 lines code)

Reference the construction code in PFC (`include/pfc/huffman.hpp`).
Note that Huffman's algorithm IS a McMillan-style construction that
additionally minimizes the expected length under a given probability
distribution. This post's `build_prefix_free_code` is the simpler
predecessor that Huffman optimizes.

#### G. Cross-references and footnote (~100 words)

- Back to [Kraft's Inequality](/post/2020-03-kraft-wire-formats/)
- Forward: post 3 (Universal Codes as Priors, forthcoming) develops
  the prior-as-hypothesis framing; post 9 (Huffman, forthcoming) uses
  McMillan's construction as the foundation.
- Cross-series back to [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/)
  for the bit-level consequences.
- Footnote to PFC.

## Stepanov bridge updates

The two Stepanov bridge posts currently say "the forthcoming Information
Theory by Construction series" in a few places. Once the new series exists
with a real first post, update those references to:

1. **Post 1 (Bits Follow Types) section E** (the Either combinator's
   forward-pointer to "Huffman/arithmetic coding in the new series"):
   change to "[Huffman](/post/2024-03-huffman-wire-formats/) and
   [Arithmetic Coding](/post/2024-09-arithmetic-coding-wire-formats/) in
   the [Algebra over Wire Formats](/series/wire-formats/) series."

   Note: links to posts 9 and 10 are placeholder until those exist
   (subsequent sub-project 3 work). Could also use a simpler "[Algebra
   over Wire Formats](/series/wire-formats/)" series link without
   per-post hooks.

2. **Post 2 (When Lists Become Bits) section H** (Kraft forward-pointer):
   change "the forthcoming Information Theory by Construction series
   develops the proof" to "see [Kraft's Inequality](/post/2020-03-kraft-wire-formats/)
   and [McMillan's Converse](/post/2020-09-mcmillan-wire-formats/) in
   the Algebra over Wire Formats series."

3. **Post 2 (When Lists Become Bits) section J** (Further Reading):
   replace "the forthcoming Information Theory by Construction series"
   with "[Algebra over Wire Formats](/series/wire-formats/)".

These edits happen as the final step of sub-project 2's implementation,
after the new posts exist and Hugo sync is complete.

## Hugo sync strategy

Same pattern as the stepanov sync, but using the FIXED Makefile target
from the start:

```bash
cd ~/github/metafunctor-series/wire-formats && \
  BLOG_POST_DIR=~/github/repos/metafunctor/content/post make sync
```

The fixed sync target (per-directory rsync, no global `--delete`) is
identical to what we patched into the local stepanov Makefile, but
written into wire-formats from the start.

After sync:
1. Verify both new posts appear in `content/post/`.
2. Run `mf series scan` to confirm `wire-formats` series is detected with
   2 posts.
3. Commit metafunctor changes (just the two new post directories).
4. Push.

## Acceptance criteria

The work is done when:

1. `wire-formats/` exists at `~/github/metafunctor-series/wire-formats/`
   with the full scaffold (Makefile, CMakeLists, mkdocs.yml, README, .gitignore,
   LICENSE, CLAUDE.md).
2. `make build` succeeds, `make test` passes for both new post test files
   (the GoogleTest suites for Kraft and McMillan).
3. Both `index.md` files pass the soul check (no em-dashes, no other
   banned phrases).
4. `mkdocs serve` (or `make docs`) renders both posts cleanly with
   cross-reference links resolving (or appropriately marked as forthcoming).
5. The stepanov bridge posts are updated to point at the new series's
   first posts.
6. The Hugo sync places both new post directories in
   `~/github/repos/metafunctor/content/post/`.
7. `mf series scan` detects `wire-formats` as a tracked series with 2 posts.
8. Both repos (wire-formats and metafunctor) are committed and pushed.

## What is NOT in scope

- Writing posts 3 through 13 (sub-project 3).
- Designing the per-post outlines for posts 3-13 in detail (only their
  titles, dates, and directory names are recorded above for scheduling).
- Creating any GitHub remote for the wire-formats repo. Initial commits
  go to local main; the user can add a remote and push later if desired.
- Hugo theme/template work to render the new series specially.
- Updating PFC (the source-of-truth library) in any way; it is
  reference-only here.

## Open questions and known unknowns

These get resolved during the implementation plan or by inspection:

- Exact day-of-month for posts 1 and 2 (currently 2020-03-22 and
  2020-09-13). Both verified to not collide with existing
  metafunctor posts. Could be shifted by a few days if Alex prefers
  specific dates.
- Whether the `wire-formats` series should have a featured artifact in
  `mf` (like Stepanov has its `stepanov-explorer.html`). Likely a
  follow-up after the series is established.
- Whether the new repo should have a remote (GitHub) at creation, or
  remain local until a few more posts ship. Default: local-only at first.
- Whether `mf series create` is needed before `mf series scan` will
  recognize the new series, or if scan auto-discovers it from
  frontmatter. Inspect during bootstrap.

## Implementation sequence (handoff to writing-plans)

Approximate ordering for the implementation plan:

1. **Bootstrap repo**: `git init` at wire-formats/, create top-level
   files (Makefile, CMakeLists, README, LICENSE, mkdocs.yml, CLAUDE.md,
   .gitignore with PROPERLY-SCOPED Makefile rule).
2. **Move this spec** to `wire-formats/docs/superpowers/specs/`.
3. **Bootstrap docs/**: index.md, about.md, javascripts/ if needed.
4. **mf series create**: register `wire-formats` series in the metafunctor
   `mf` database (or verify auto-discovery works once posts exist).
5. **Post 1 scaffold**: post directory + placeholder files + CMakeLists
   wiring.
6. **Post 1 implementation (TDD)**: build the Kraft pedagogical code
   incrementally (`BinaryTree` data structure, `kraft_sum`, the
   trie-embedding visualization helper).
7. **Post 1 prose**: write `index.md` sections A through G.
8. **Post 2 scaffold**: same shape as post 1.
9. **Post 2 implementation (TDD)**: `build_prefix_free_code`
   construction, McMillan-via-L-th-power proof helpers if needed.
10. **Post 2 prose**: write `index.md` sections A through G.
11. **Stepanov bridge updates**: edit the two Stepanov posts to convert
    forward-references from "the forthcoming Information Theory by
    Construction series" to live links pointing into the new series.
12. **Hugo sync**: rsync both new posts into `metafunctor/content/post/`
    using the FIXED sync target.
13. **Final verification**: `make build`, `make test`, `make docs --strict`,
    soul check on all new files, `mf series scan` confirmation.
14. **Commit + push (with confirmation)**: wire-formats commits + push;
    stepanov bridge-update commit + push; metafunctor sync commits + push.

The writing-plans skill takes over here to convert this sequence into a
detailed task plan with explicit checkpoints.
