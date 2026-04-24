# Algebra over Wire Formats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap a new pedagogical blog series, "Algebra over Wire Formats," at `~/github/metafunctor-series/wire-formats/` mirroring the stepanov scaffold pattern, ship its first two posts (Kraft 2020-03-22, McMillan 2020-09-13) as a pair, update the Stepanov bridge posts' forward-references to point at it, and sync to the metafunctor.com Hugo site.

**Architecture:** A new repo at `~/github/metafunctor-series/wire-formats/` with C++23 + GoogleTest scaffolding mirroring stepanov, but with two corrections built in from day 1: a fixed `Makefile` `sync` target (per-directory rsync, no global `--delete`) and a properly-scoped `.gitignore` so the top-level Makefile is trackable. The two posts develop Kraft's inequality (post 1) and its constructive converse (post 2) using minimal pedagogical C++ implementations. Both posts back-date to 2020 to fit the series's planned chronological arc.

**Tech Stack:** C++23, GoogleTest v1.14.0 (fetched via FetchContent), mkdocs (docs site), Hugo (downstream via metafunctor repo), bash (sync Makefile target), the soul plugin's `check-banned-phrases.sh` PostToolUse hook (em-dash blocking on prose files).

---

## Spec reference

See `docs/superpowers/specs/2026-04-24-algebra-over-wire-formats-design.md` for the full design including the 13-post arc with backdated dates, the per-section outlines for posts 1 and 2, voice/code/test conventions, repo structure, and the lessons-learned items the new repo must encode (fixed sync target, scoped Makefile gitignore).

## File Structure

**New files (created in this plan):**

In the new wire-formats repo (`~/github/metafunctor-series/wire-formats/`):
- `.gitignore` (with PROPERLY-SCOPED Makefile rule)
- `LICENSE` (MIT, copied verbatim from stepanov)
- `README.md`
- `CLAUDE.md`
- `Makefile` (with the FIXED sync target)
- `CMakeLists.txt` is at `post/CMakeLists.txt` (not top-level)
- `mkdocs.yml`
- `docs/index.md`
- `docs/about.md`
- `docs/superpowers/specs/2026-04-24-bootstrap-and-posts-1-2-design.md` (moved from stepanov)
- `docs/superpowers/plans/2026-04-24-algebra-over-wire-formats.md` (moved from stepanov)
- `post/CMakeLists.txt`
- `post/2020-03-kraft-wire-formats/index.md`
- `post/2020-03-kraft-wire-formats/kraft.hpp`
- `post/2020-03-kraft-wire-formats/test_kraft.cpp`
- `post/2020-09-mcmillan-wire-formats/index.md`
- `post/2020-09-mcmillan-wire-formats/mcmillan.hpp`
- `post/2020-09-mcmillan-wire-formats/test_mcmillan.cpp`

In the metafunctor repo (`~/github/repos/metafunctor/`):
- `content/post/2020-03-kraft-wire-formats/` (synced from wire-formats)
- `content/post/2020-09-mcmillan-wire-formats/` (synced from wire-formats)
- `content/series/wire-formats/_index.md` (series landing page)
- Possibly `.mf/series_db.json` updated by `mf series scan`

**Modified files (in stepanov repo):**
- `post/2026-05-codecs-functors-stepanov/index.md` (Either section forward-pointer)
- `post/2026-05-prefix-free-stepanov/index.md` (sections H and J Kraft references)

---

## Task 1: Reconnaissance

**Files:** read-only inspection of stepanov scaffold to mirror.

- [ ] **Step 1: Inventory stepanov's scaffold files**

```bash
ls -la /home/spinoza/github/metafunctor-series/stepanov/
ls /home/spinoza/github/metafunctor-series/stepanov/docs/
cat /home/spinoza/github/metafunctor-series/stepanov/CLAUDE.md | head -40
cat /home/spinoza/github/metafunctor-series/stepanov/CONTRIBUTING.md 2>/dev/null | head -20
cat /home/spinoza/github/metafunctor-series/stepanov/LICENSE | head -3
```

Expected: identify the canonical files to mirror (Makefile, gitignore, README, CLAUDE, LICENSE, mkdocs.yml, docs/index.md, docs/about.md, post/CMakeLists.txt skeleton).

- [ ] **Step 2: Inspect stepanov's mkdocs.yml structure for nav-by-section**

```bash
cat /home/spinoza/github/metafunctor-series/stepanov/mkdocs.yml
```

Expected: note the theme, the plugin list, the markdown_extensions, and the nav structure (organized by section like "Foundations", "Applied Mathematics", "Algebraic Foundations").

- [ ] **Step 3: Inspect stepanov's docs/index.md framing**

```bash
cat /home/spinoza/github/metafunctor-series/stepanov/docs/index.md | head -40
```

Expected: see the series-level thesis statement, table-of-posts pattern, and "principles" section. The new series's `docs/index.md` mirrors this shape with wire-formats-specific content.

- [ ] **Step 4: Verify the metafunctor mf series state**

```bash
mf series list 2>&1 | head -15
ls /home/spinoza/github/repos/metafunctor/content/series/ 2>&1 | head
```

Expected: confirm wire-formats does NOT yet exist in the series db or as a `_index.md`. Note the format used by other series's `_index.md` files (`stepanov/_index.md` is the closest reference).

- [ ] **Step 5: Verify date-collision check passes for posts 1 and 2**

```bash
grep -h "^date:" /home/spinoza/github/repos/metafunctor/content/post/*/index.md 2>/dev/null | grep -E "^date: 2020-03-22|^date: 2020-09-13" | sort -u
```

Expected: empty output (no collisions). If collisions appear, pick adjacent unused days and update the spec before proceeding.

No commit for this task; it produces only findings.

---

## Task 2: Bootstrap the wire-formats repo

**Files:**
- Create: `~/github/metafunctor-series/wire-formats/` (directory)
- Create: `~/github/metafunctor-series/wire-formats/LICENSE`
- Create: `~/github/metafunctor-series/wire-formats/.gitignore`
- Create: `~/github/metafunctor-series/wire-formats/README.md`
- Create: `~/github/metafunctor-series/wire-formats/CLAUDE.md`
- Create: `~/github/metafunctor-series/wire-formats/Makefile`

- [ ] **Step 1: Initialize the directory and git repo**

```bash
mkdir -p /home/spinoza/github/metafunctor-series/wire-formats
cd /home/spinoza/github/metafunctor-series/wire-formats
git init
git config user.email "lex@metafunctor.com"
git config user.name "Alexander Towell"
```

Expected: an empty initialized git repo at the new location.

- [ ] **Step 2: Copy LICENSE from stepanov**

```bash
cp /home/spinoza/github/metafunctor-series/stepanov/LICENSE /home/spinoza/github/metafunctor-series/wire-formats/LICENSE
```

Expected: MIT license file at the new location, identical to stepanov's.

- [ ] **Step 3: Create .gitignore with the SCOPED Makefile rule**

Create `/home/spinoza/github/metafunctor-series/wire-formats/.gitignore` with this content:

```gitignore
# Prerequisites
*.d

# Build directories
build/
cmake-build-*/
out/
.build/

# mkdocs build artifacts
site/
docs/post/

# Compiled Object files
*.slo
*.lo
*.o
*.obj

# Precompiled Headers
*.gch
*.pch

# Compiled Dynamic libraries
*.so
*.dylib
*.dll

# Fortran module files
*.mod
*.smod

# Compiled Static libraries
*.lai
*.la
*.a
*.lib

# Executables
*.exe
*.out
*.app

# CMake
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
CTestTestfile.cmake
compile_commands.json

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store

# Local overrides (gitignored intentionally)
Makefile.local

# Generated build Makefiles only (NOT the top-level Makefile)
**/build/Makefile
**/cmake-build-*/Makefile

# Generated docs site
site/

# Hugo sync drop point (if accidentally left behind)
docs/post/
```

The critical line is `**/build/Makefile` and `**/cmake-build-*/Makefile` (scoped to build directories). NOT the bare `Makefile` rule that stepanov has at line 59 of its .gitignore.

- [ ] **Step 4: Create README.md**

Create `/home/spinoza/github/metafunctor-series/wire-formats/README.md` with this content:

```markdown
# Algebra over Wire Formats

Pedagogical blog posts exploring information theory by construction in C++23.

**Companion to** the [Stepanov series](https://github.com/queelius/stepanov), which develops algorithms from algebraic structure on type. This series develops algorithms from algebraic structure on bit strings: how prefix-free codes work, why each universal code is optimal under a different prior, and how succinct data structures achieve their bounds.

## The Core Thesis

**A code is a hypothesis about the source.** Each universal code corresponds to a different prior over the integers; each entropy-optimal code (Huffman, arithmetic) is best under different assumptions; each succinct data structure (rank/select bit vectors, RoaringBitmap) is the right answer under different access patterns. The series develops these as one coherent algebraic story.

The production reference for the code in this series is [PFC](https://github.com/queelius/pfc), a header-only C++20 prefix-free codecs library.

## Posts

Each post in `post/` is self-contained:
- `index.md` (the article, Hugo-compatible with YAML frontmatter)
- `*.hpp` (minimal pedagogical implementation, ~100-400 lines)
- `test_*.cpp` (GoogleTest suite verifying the implementation)

| Post | Date | Topic |
|------|------|-------|
| `2020-03-kraft-wire-formats/` | 2020-03-22 | Kraft's inequality: which length vectors are achievable for prefix-free codes |
| `2020-09-mcmillan-wire-formats/` | 2020-09-13 | McMillan's converse: any Kraft-satisfying length vector has a prefix-free code |

(More posts forthcoming; the series arc spans 13 posts through 2026.)

## Building

```bash
make build   # Configure and build all post code
make test    # Run all GoogleTest suites
make clean   # Remove build artifacts
make docs    # Build the mkdocs site locally
```

## Hugo Sync (for the canonical metafunctor.com site)

```bash
BLOG_POST_DIR=~/github/repos/metafunctor/content/post make sync
```

The `sync` target rsyncs each post directory into the BLOG_POST_DIR, preserving any unrelated content there. (The `--delete` is scoped per-post, NOT global.)

## License

MIT

## Author

**Alexander Towell** -- [metafunctor.com](https://metafunctor.com) -- [lex@metafunctor.com](mailto:lex@metafunctor.com)
```

- [ ] **Step 5: Create CLAUDE.md**

Create `/home/spinoza/github/metafunctor-series/wire-formats/CLAUDE.md` with this content:

```markdown
# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Algebra over Wire Formats is a pedagogical blog series exploring information theory by construction in C++23. Companion to the Stepanov series (`~/github/metafunctor-series/stepanov/`).

**Series thesis**: a code is a hypothesis about the source. Each universal code corresponds to a different prior; each compression scheme is optimal under different assumptions.

The production reference is PFC (`~/github/released/pfc/`).

## Repo conventions

- Each post lives in `post/YYYY-MM-name-wire-formats/` (flat directory, no subdirs)
- Each post has exactly three files: `index.md`, `*.hpp`, `test_*.cpp`
- Test executables are registered in `post/CMakeLists.txt` (no per-post CMakeLists)
- C++23, GoogleTest v1.14.0, fetched via FetchContent
- Voice: Alex Towell's, soul-plugin checked. NO em-dashes (use commas, periods, colons, parens)
- Math: LaTeX (`$$...$$` for display, `\(...\)` for inline)

## Build

```bash
make build         # Configure and build via post/CMakeLists.txt -> post/build/
make test          # Run all tests via ctest
make clean         # Remove post/build/
make docs          # Build mkdocs site
make docs-serve    # Serve mkdocs locally
```

## Hugo sync

```bash
BLOG_POST_DIR=~/github/repos/metafunctor/content/post make sync
```

The `sync` target uses per-directory rsync with `--delete` scoped to each
destination subdirectory. Unrelated content in BLOG_POST_DIR is preserved.

## Series ordering

The series's planned 13-post arc is documented in
`docs/superpowers/specs/2026-04-24-bootstrap-and-posts-1-2-design.md`. Posts
are backdated 2020-2026 (~2/year) to reflect the long developmental story.

## Cross-references

- Stepanov bridge posts (slot 20 "Bits Follow Types", slot 21 "When Lists
  Become Bits") forward-reference this series. Once posts are shipped,
  update those references to live links pointing here.
```

- [ ] **Step 6: Create the FIXED Makefile**

Create `/home/spinoza/github/metafunctor-series/wire-formats/Makefile` with this content:

```makefile
# Algebra over Wire Formats - Pedagogical C++ Blog Posts
#
# Usage:
#   make build       - Configure and build all posts
#   make test        - Run all tests
#   make clean       - Remove build artifacts
#   make docs        - Build mkdocs site
#   make docs-serve  - Serve docs locally
#   make help        - Show this help

.PHONY: build test clean help sync docs docs-serve docs-clean

# Directory containing posts
POST_DIR := post

help:
	@echo "Algebra over Wire Formats - Pedagogical C++ Blog Posts"
	@echo ""
	@echo "Targets:"
	@echo "  make build       - Configure and build all posts"
	@echo "  make test        - Run all tests"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make docs        - Build mkdocs site"
	@echo "  make docs-serve  - Serve docs locally at localhost:8000"
	@echo "  make docs-clean  - Remove docs build artifacts"
	@echo ""
	@echo "For Hugo sync, set BLOG_POST_DIR and run: make sync"

build:
	cmake -B $(POST_DIR)/build -S $(POST_DIR)
	cmake --build $(POST_DIR)/build

test: build
	ctest --test-dir $(POST_DIR)/build --output-on-failure

clean:
	rm -rf $(POST_DIR)/build

# Documentation
docs: docs-clean
	mkdir -p docs/post
	cp -r $(POST_DIR)/*-wire-formats/ docs/post/
	mkdocs build

docs-serve:
	mkdir -p docs/post
	cp -r $(POST_DIR)/*-wire-formats/ docs/post/
	mkdocs serve

docs-clean:
	rm -rf site docs/post

# Sync to external blog (requires BLOG_POST_DIR to be set).
# Per-directory rsync with --delete scoped to each post's destination.
# Unrelated content in BLOG_POST_DIR is preserved.
#
# Example:
#   BLOG_POST_DIR=~/github/repos/metafunctor/content/post make sync
sync:
ifndef BLOG_POST_DIR
	@echo "Error: BLOG_POST_DIR not set"
	@echo "Usage: BLOG_POST_DIR=/path/to/blog/post make sync"
	@exit 1
endif
	@echo "Syncing to $(BLOG_POST_DIR)..."
	@for dir in $(POST_DIR)/*-wire-formats; do \
		name=$$(basename $$dir); \
		echo "  -> $$name"; \
		rsync -a --delete \
			--exclude='build/' \
			--exclude='CMakeLists.txt' \
			--exclude='README.md' \
			"$$dir/" "$(BLOG_POST_DIR)/$$name/"; \
	done
	@echo "Done. Synced wire-formats posts to $(BLOG_POST_DIR)"

# Include local overrides if present (gitignored)
-include Makefile.local
```

- [ ] **Step 7: Verify the Makefile sync target is safe via tmp-dir test**

```bash
mkdir -p /tmp/wire-formats-sync-test
mkdir -p /home/spinoza/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats
echo "test placeholder" > /home/spinoza/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/index.md
cd /home/spinoza/github/metafunctor-series/wire-formats && BLOG_POST_DIR=/tmp/wire-formats-sync-test make sync 2>&1 | tail -5
ls /tmp/wire-formats-sync-test/
echo "Should show: 2020-03-kraft-wire-formats (the directory itself, NOT flattened files)"
rm -rf /tmp/wire-formats-sync-test
rm /home/spinoza/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/index.md
rmdir /home/spinoza/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats
```

Expected: the dir `2020-03-kraft-wire-formats` appears in `/tmp/wire-formats-sync-test/`, not the flattened content.

- [ ] **Step 8: Verify .gitignore does NOT exclude top-level Makefile**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && git check-ignore Makefile && echo "BAD: Makefile is gitignored" || echo "GOOD: Makefile is trackable"
```

Expected: "GOOD: Makefile is trackable" (exit code from check-ignore is non-zero when the path is NOT ignored).

- [ ] **Step 9: Initial commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add LICENSE README.md CLAUDE.md .gitignore Makefile
git status
git commit -m "scaffold: initialize wire-formats repo with fixed sync and scoped gitignore

The Makefile sync target uses per-directory rsync with --delete scoped to each
post's destination subdirectory; unrelated content in BLOG_POST_DIR is preserved.
The .gitignore excludes Makefile only inside build/ subdirectories, so the
top-level Makefile is trackable.

These two corrections were learned from the stepanov repo, where a global
--delete flattened all post contents into Hugo's content/post/ root and
deleted unrelated blog posts. Recovery there was via git, but those mistakes
do not need to repeat here."
```

Expected: a single commit with the scaffold files. Verify with `git log --oneline -1`.

---

## Task 3: Move spec and plan into the new repo

**Files:**
- Create: `~/github/metafunctor-series/wire-formats/docs/superpowers/specs/2026-04-24-bootstrap-and-posts-1-2-design.md`
- Create: `~/github/metafunctor-series/wire-formats/docs/superpowers/plans/2026-04-24-algebra-over-wire-formats.md`
- Delete (in stepanov): `docs/superpowers/specs/2026-04-24-algebra-over-wire-formats-design.md`
- Delete (in stepanov): `docs/superpowers/plans/2026-04-24-algebra-over-wire-formats.md`

The spec was written into the stepanov repo as a temporary home (HARD-GATE: no scaffolding before approval). Now that the wire-formats repo exists, move it.

- [ ] **Step 1: Create the docs/superpowers tree in wire-formats**

```bash
mkdir -p /home/spinoza/github/metafunctor-series/wire-formats/docs/superpowers/specs
mkdir -p /home/spinoza/github/metafunctor-series/wire-formats/docs/superpowers/plans
```

- [ ] **Step 2: Move the spec from stepanov to wire-formats (renamed)**

```bash
mv /home/spinoza/github/metafunctor-series/stepanov/docs/superpowers/specs/2026-04-24-algebra-over-wire-formats-design.md \
   /home/spinoza/github/metafunctor-series/wire-formats/docs/superpowers/specs/2026-04-24-bootstrap-and-posts-1-2-design.md
```

The new filename matches the spec's own preference: `2026-04-24-bootstrap-and-posts-1-2-design.md`.

- [ ] **Step 3: Move the plan from stepanov to wire-formats**

```bash
mv /home/spinoza/github/metafunctor-series/stepanov/docs/superpowers/plans/2026-04-24-algebra-over-wire-formats.md \
   /home/spinoza/github/metafunctor-series/wire-formats/docs/superpowers/plans/2026-04-24-algebra-over-wire-formats.md
```

- [ ] **Step 4: Commit the move in wire-formats**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add docs/superpowers/
git commit -m "docs: relocate spec and plan from temporary stepanov home"
```

- [ ] **Step 5: Commit the deletion in stepanov**

```bash
cd /home/spinoza/github/metafunctor-series/stepanov
git add docs/superpowers/specs/2026-04-24-algebra-over-wire-formats-design.md \
        docs/superpowers/plans/2026-04-24-algebra-over-wire-formats.md
git status
git commit -m "chore(planning): relocate wire-formats spec and plan to the wire-formats repo"
```

The `git add` here records the deletions (since `mv` moved the files out). The commit captures the deletion.

---

## Task 4: Bootstrap docs/ in wire-formats

**Files:**
- Create: `~/github/metafunctor-series/wire-formats/mkdocs.yml`
- Create: `~/github/metafunctor-series/wire-formats/docs/index.md`
- Create: `~/github/metafunctor-series/wire-formats/docs/about.md`

- [ ] **Step 1: Create mkdocs.yml**

Create `/home/spinoza/github/metafunctor-series/wire-formats/mkdocs.yml` with this content:

```yaml
site_name: Algebra over Wire Formats
site_description: Pedagogical blog posts exploring information theory by construction in C++23
site_author: Alexander Towell
site_url: https://queelius.github.io/wire-formats/

theme:
  name: material
  features:
    - navigation.sections
    - navigation.expand
    - content.code.copy
  palette:
    - scheme: default
      primary: indigo
      accent: indigo

markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.superfences
  - pymdownx.highlight:
      anchor_linenums: true
  - admonition
  - pymdownx.details
  - tables
  - footnotes
  - attr_list

extra_javascript:
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

nav:
  - Home: index.md
  - About: about.md
  - "Foundations":
      - "Kraft's Inequality": "post/2020-03-kraft-wire-formats/index.md"
      - "McMillan's Converse": "post/2020-09-mcmillan-wire-formats/index.md"
```

- [ ] **Step 2: Create docs/index.md (the series landing page)**

Create `/home/spinoza/github/metafunctor-series/wire-formats/docs/index.md` with this content:

```markdown
---
title: "A Code Is a Hypothesis About the Source"
description: "A pedagogical series exploring information theory by construction in C++23"
---

# A Code Is a Hypothesis About the Source

This series develops information theory the same way the [Stepanov series](https://github.com/queelius/stepanov) develops algorithms: by construction, with minimal pedagogical code, and with the algebraic structure made explicit.

The Stepanov series argued that the algebraic structure of a type determines its algorithms. This companion series argues the analogous claim for bit-level encodings: **the algebraic structure of a code determines its compression efficiency, its decoding cost, and its compositional behavior.** Each universal code corresponds to a different prior over the integers; each entropy-optimal code (Huffman, arithmetic) is best under different assumptions; each succinct data structure is the right answer under different access patterns.

## The Principle

A prefix-free code assigns codeword lengths `(l_1, ..., l_n)` to symbols. Kraft's inequality characterizes which length vectors are achievable: $$\sum_i 2^{-l_i} \leq 1.$$ Within this budget, every code is a different way of spending the budget. Allocating more bits to symbol `i` (longer `l_i`) means less budget for the others. Information theory's optimum (assigning `-\log_2 p_i` bits to a symbol with probability `p_i`) saturates Kraft when the probabilities are dyadic.

This series walks through the codes that have shown up in practice, treating each one as a different hypothesis about the integer distribution it expects to encode. The choices are not arbitrary: each code is optimal somewhere, and recognizing where reframes "compression algorithm" as "model selection."

## Posts

### Foundations

- [Kraft's Inequality](post/2020-03-kraft-wire-formats/index.md): the budget constraint on prefix-free codes
- [McMillan's Converse](post/2020-09-mcmillan-wire-formats/index.md): every Kraft-satisfying length vector has a prefix-free code

(More posts forthcoming; see the [arc plan](about.md) for the full schedule.)

## The Production Reference

The code in this series is pedagogical. The production version (with full STL integration, 31k+ test assertions, and the rich combinator library these posts only sketch) lives in [PFC](https://github.com/queelius/pfc).

## Companion Series

- [Stepanov: Generic Programming in C++](https://queelius.github.io/stepanov/): algorithms from algebra on type. The Stepanov bridge posts ([Bits Follow Types](https://queelius.github.io/stepanov/post/2026-05-codecs-functors-stepanov/), [When Lists Become Bits](https://queelius.github.io/stepanov/post/2026-05-prefix-free-stepanov/)) connect that series to this one.

## Author

**Alexander Towell** -- [metafunctor.com](https://metafunctor.com) -- [lex@metafunctor.com](mailto:lex@metafunctor.com)
```

- [ ] **Step 3: Create docs/about.md (the series arc page)**

Create `/home/spinoza/github/metafunctor-series/wire-formats/docs/about.md` with this content:

```markdown
---
title: "About"
description: "About the Algebra over Wire Formats series"
---

# About

## The Series Arc

The series is planned for 13 posts spanning 2020 through 2026, roughly 2 posts per year. The arc moves from foundational results (Kraft, McMillan) through the universal codes (Elias, Fibonacci, Rice/Golomb, VByte) to the entropy-optimal codes (Huffman, arithmetic) and finally to succinct data structures (rank/select, RoaringBitmap), closing with a synthesis post that ties everything back to the codecs-as-functors framing introduced in the Stepanov bridge posts.

| # | Title | Date | Status |
|---|-------|------|--------|
| 1 | Kraft's Inequality | 2020-03-22 | Published |
| 2 | McMillan's Converse | 2020-09-13 | Published |
| 3 | Universal Codes as Priors | 2021-03-29 | Forthcoming |
| 4 | Unary and Elias Gamma | 2021-08-08 | Forthcoming |
| 5 | Elias Delta and Omega | 2022-02-13 | Forthcoming |
| 6 | Fibonacci Coding | 2022-07-17 | Forthcoming |
| 7 | Rice / Golomb | 2023-02-19 | Forthcoming |
| 8 | VByte / Varint | 2023-08-14 | Forthcoming |
| 9 | Huffman | 2024-03-25 | Forthcoming |
| 10 | Arithmetic Coding | 2024-09-04 | Forthcoming |
| 11 | Succinct Bit Vectors and Rank/Select | 2025-03-09 | Forthcoming |
| 12 | RoaringBitmap | 2025-08-10 | Forthcoming |
| 13 | Synthesis: Codecs as Structure | 2026-05-15 | Forthcoming |

## Voice and Style

- Each post is self-contained and stands alone (start anywhere)
- Code is C++23, headers-only, with a minimal pedagogical implementation per post (~100-400 lines)
- Tests use GoogleTest v1.14.0
- Math is rendered with MathJax via mkdocs-arithmatex

## Further Reading

- Cover and Thomas, *Elements of Information Theory* (2006)
- MacKay, *Information Theory, Inference, and Learning Algorithms* (2003)
- Knuth, *The Art of Computer Programming, Volume 4A* (2011)
- Sayood, *Introduction to Data Compression* (2017)
- McMillan, "Two Inequalities Implied by Unique Decipherability," 1956
```

- [ ] **Step 4: Verify the docs build cleanly**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && mkdocs build 2>&1 | tail -10
```

Expected: build succeeds. Warnings about post links in nav (the post directories don't exist yet) are acceptable; we'll create them in subsequent tasks.

If mkdocs is not installed, document this and skip the verify step. Build will be re-verified after posts exist.

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add mkdocs.yml docs/index.md docs/about.md
git commit -m "docs: add mkdocs config and series landing pages (index, about)"
```

---

## Task 5: Bootstrap post/CMakeLists.txt

**Files:**
- Create: `~/github/metafunctor-series/wire-formats/post/CMakeLists.txt`

The CMakeLists is at `post/CMakeLists.txt` (matches stepanov's structure). It fetches GoogleTest and contains placeholder for per-post test executables (added in subsequent tasks).

- [ ] **Step 1: Create post/CMakeLists.txt**

Create `/home/spinoza/github/metafunctor-series/wire-formats/post/CMakeLists.txt` with this content:

```cmake
cmake_minimum_required(VERSION 3.16)
project(wire-formats LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# =============================================================================
# Per-post test executables are added below as posts are written.
# =============================================================================
```

- [ ] **Step 2: Verify the build configures (no posts yet, but FetchContent should work)**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make build 2>&1 | tail -10
```

Expected: cmake configures successfully. The build may produce no executables (no post test executables yet), but the configuration step must succeed.

- [ ] **Step 3: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add post/CMakeLists.txt
git commit -m "build: add post/CMakeLists.txt (C++23 + GoogleTest via FetchContent)"
```

---

## Task 6: Scaffold post 1 directory and wire CMakeLists

**Files:**
- Create: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/index.md` (placeholder frontmatter only)
- Create: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/kraft.hpp` (header guards only)
- Create: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/test_kraft.cpp` (gtest placeholder)
- Modify: `~/github/metafunctor-series/wire-formats/post/CMakeLists.txt` (append `add_executable` block)

- [ ] **Step 1: Create the post directory and skeleton files**

```bash
mkdir -p /home/spinoza/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats
```

Create `index.md` with placeholder frontmatter:

```markdown
---
title: "Kraft's Inequality"
date: 2020-03-22
draft: true
tags:
- C++
- information-theory
- coding-theory
- prefix-free
- combinatorics
categories:
- Computer Science
- Mathematics
series:
- wire-formats
series_weight: 1
math: true
description: "Which codeword-length vectors are achievable by prefix-free codes? Kraft's inequality is the answer."
linked_project:
- pfc
- wire-formats
---

(Draft in progress. See plan Task 13 for full prose.)
```

Create `kraft.hpp` with header guards:

```cpp
// kraft.hpp
// Pedagogical implementation for the post "Kraft's Inequality" in the
// "Algebra over Wire Formats" series. For the production version, see PFC:
// https://github.com/queelius/pfc

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace kraft {

// Implementation arrives in subsequent plan tasks.

}  // namespace kraft
```

Create `test_kraft.cpp` with a placeholder so CMake can find it:

```cpp
#include <gtest/gtest.h>
#include "kraft.hpp"

TEST(KraftTest, Placeholder) {
    EXPECT_TRUE(true);
}
```

- [ ] **Step 2: Add the test executable to post/CMakeLists.txt**

In `~/github/metafunctor-series/wire-formats/post/CMakeLists.txt`, after the `# Per-post test executables are added below as posts are written.` comment, append:

```cmake

# =============================================================================
# Kraft's Inequality (post 1, 2020-03-22)
# =============================================================================
add_executable(test_kraft 2020-03-kraft-wire-formats/test_kraft.cpp)
target_link_libraries(test_kraft GTest::gtest_main)
target_include_directories(test_kraft PRIVATE 2020-03-kraft-wire-formats)
add_test(NAME test_kraft COMMAND test_kraft)
```

- [ ] **Step 3: Build and verify**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make test 2>&1 | tail -8
```

Expected: 1 test passes (`test_kraft.Placeholder`).

- [ ] **Step 4: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add post/2020-03-kraft-wire-formats post/CMakeLists.txt
git commit -m "scaffold(kraft): add post 1 directory and CMake wiring"
```

---

## Task 7: Implement BinaryTree data structure for codewords (TDD)

**Files:**
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/kraft.hpp`
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/test_kraft.cpp`

The `BinaryTree` data structure represents a code as a binary tree of partial codewords. Each codeword is a leaf at a particular depth; internal nodes are partial codewords. We will use it in subsequent tasks for the trie-view, the prefix-freeness check, and the Kraft sum.

- [ ] **Step 1: Write failing tests for BinaryTree**

Replace `test_kraft.cpp` with:

```cpp
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>
#include "kraft.hpp"

using namespace kraft;

TEST(KraftTest, BinaryTreeInsertCodewordAtRoot) {
    BinaryTree t;
    t.insert("0");
    EXPECT_TRUE(t.contains("0"));
    EXPECT_FALSE(t.contains("1"));
}

TEST(KraftTest, BinaryTreeInsertMultipleCodewords) {
    BinaryTree t;
    t.insert("0");
    t.insert("10");
    t.insert("110");
    t.insert("111");
    EXPECT_TRUE(t.contains("0"));
    EXPECT_TRUE(t.contains("10"));
    EXPECT_TRUE(t.contains("110"));
    EXPECT_TRUE(t.contains("111"));
}

TEST(KraftTest, BinaryTreeIsPrefixFreeReturnsTrueForValidCode) {
    BinaryTree t;
    t.insert("0");
    t.insert("10");
    t.insert("110");
    t.insert("111");
    EXPECT_TRUE(t.is_prefix_free());
}

TEST(KraftTest, BinaryTreeIsPrefixFreeReturnsFalseWhenCodewordIsPrefix) {
    BinaryTree t;
    t.insert("0");
    t.insert("01");  // "0" is a prefix of "01"
    EXPECT_FALSE(t.is_prefix_free());
}

TEST(KraftTest, BinaryTreeIsPrefixFreeReturnsFalseWhenLongerCodewordContainsAnother) {
    BinaryTree t;
    t.insert("010");
    t.insert("01");  // "01" is a prefix of "010"
    EXPECT_FALSE(t.is_prefix_free());
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make build 2>&1 | head -20
```

Expected: compile failure with "BinaryTree was not declared in this scope" or similar.

- [ ] **Step 3: Implement BinaryTree**

In `kraft.hpp`, replace `// Implementation arrives in subsequent plan tasks.` with:

```cpp
// ---- BinaryTree -- the trie view of a prefix-free code ----------------------
//
// A code is a set of codewords (each a string over {0, 1}). We represent the
// code as a binary tree where each codeword traces a root-to-leaf path:
// '0' goes left, '1' goes right. Codewords end at terminal nodes (is_codeword
// = true). A code is prefix-free iff no terminal node lies on the path to
// another terminal node.

class BinaryTree {
public:
    BinaryTree() : root_(std::make_unique<Node>()) {}

    // Insert a codeword (a string of '0' and '1' characters).
    // Idempotent: inserting the same codeword twice is a no-op.
    void insert(const std::string& codeword) {
        Node* cur = root_.get();
        for (char c : codeword) {
            assert((c == '0' || c == '1') && "codeword must be over {0,1}");
            std::unique_ptr<Node>& child = (c == '0') ? cur->left : cur->right;
            if (!child) child = std::make_unique<Node>();
            cur = child.get();
        }
        cur->is_codeword = true;
    }

    // Returns true iff the codeword is in the tree.
    bool contains(const std::string& codeword) const {
        const Node* cur = root_.get();
        for (char c : codeword) {
            const std::unique_ptr<Node>& child = (c == '0') ? cur->left : cur->right;
            if (!child) return false;
            cur = child.get();
        }
        return cur->is_codeword;
    }

    // Returns true iff no codeword in the tree is a prefix of another.
    // Equivalently: no terminal node has any descendants that are terminal.
    [[nodiscard]] bool is_prefix_free() const {
        return is_prefix_free_recursive(root_.get(), false);
    }

private:
    struct Node {
        std::unique_ptr<Node> left;   // '0' branch
        std::unique_ptr<Node> right;  // '1' branch
        bool is_codeword = false;
    };

    std::unique_ptr<Node> root_;

    static bool is_prefix_free_recursive(const Node* node, bool ancestor_is_codeword) {
        if (!node) return true;
        // If this node is a codeword AND we passed through a codeword on the
        // way down, the ancestor codeword is a prefix of this one. Violation.
        // Equivalently: if this node is a codeword AND has any children that
        // are also codewords, this codeword is a prefix of those.
        if (node->is_codeword && ancestor_is_codeword) return false;
        bool below = node->is_codeword;
        if (!is_prefix_free_recursive(node->left.get(), ancestor_is_codeword || below)) return false;
        if (!is_prefix_free_recursive(node->right.get(), ancestor_is_codeword || below)) return false;
        return true;
    }
};
```

Also add `#include <memory>` to the includes block (right after `<map>`).

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make test 2>&1 | tail -8
```

Expected: 5 tests pass (BinaryTreeInsertCodewordAtRoot, BinaryTreeInsertMultipleCodewords, BinaryTreeIsPrefixFreeReturnsTrueForValidCode, BinaryTreeIsPrefixFreeReturnsFalseWhenCodewordIsPrefix, BinaryTreeIsPrefixFreeReturnsFalseWhenLongerCodewordContainsAnother).

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add post/2020-03-kraft-wire-formats
git commit -m "feat(kraft): add BinaryTree with insert, contains, is_prefix_free"
```

---

## Task 8: Implement kraft_sum (TDD)

**Files:**
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/kraft.hpp`
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/test_kraft.cpp`

`kraft_sum(lengths)` returns `sum_i 2^{-l_i}`. Used to verify Kraft's inequality computationally.

- [ ] **Step 1: Write failing tests**

Append to `test_kraft.cpp`:

```cpp
TEST(KraftTest, KraftSumOfEmptyVectorIsZero) {
    EXPECT_DOUBLE_EQ(kraft_sum({}), 0.0);
}

TEST(KraftTest, KraftSumOfSingletonLengthOne) {
    EXPECT_DOUBLE_EQ(kraft_sum({1}), 0.5);
}

TEST(KraftTest, KraftSumOfBalancedBinaryCode) {
    // Code with all four 2-bit codewords saturates Kraft.
    // Lengths 2, 2, 2, 2 give 4 * 2^-2 = 1.
    EXPECT_DOUBLE_EQ(kraft_sum({2, 2, 2, 2}), 1.0);
}

TEST(KraftTest, KraftSumOfExampleCode) {
    // Example from the post: A=0 (1 bit), B=10 (2 bits), C=110 (3 bits), D=111 (3 bits).
    // Sum: 1/2 + 1/4 + 1/8 + 1/8 = 1.
    EXPECT_DOUBLE_EQ(kraft_sum({1, 2, 3, 3}), 1.0);
}

TEST(KraftTest, KraftSumOfUnaryFirstFewIsLessThanOne) {
    // Unary lengths: 1, 2, 3, 4, 5. Sum: 1/2 + 1/4 + 1/8 + 1/16 + 1/32 = 31/32.
    EXPECT_NEAR(kraft_sum({1, 2, 3, 4, 5}), 31.0/32.0, 1e-12);
}

TEST(KraftTest, KraftSumExceedsOneForOverlongCode) {
    // 3 codewords of length 1 cannot fit prefix-freely (only 2 leaves at depth 1).
    // Sum: 3 * 1/2 = 1.5 > 1.
    EXPECT_GT(kraft_sum({1, 1, 1}), 1.0);
}
```

- [ ] **Step 2: Verify compile failure**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make build 2>&1 | head -10
```

Expected: `kraft_sum` not declared.

- [ ] **Step 3: Implement kraft_sum**

In `kraft.hpp`, append before the closing `}  // namespace kraft`:

```cpp

// ---- kraft_sum -- the Kraft inequality's left-hand side ---------------------
//
// For codeword lengths (l_1, ..., l_n), returns sum_i 2^{-l_i}.
// Kraft's inequality says this sum is <= 1 for any prefix-free code.

inline double kraft_sum(const std::vector<std::size_t>& lengths) {
    double sum = 0.0;
    for (std::size_t l : lengths) {
        sum += std::ldexp(1.0, -static_cast<int>(l));
    }
    return sum;
}
```

- [ ] **Step 4: Run tests, expect pass**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make test 2>&1 | tail -8
```

Expected: 6 new tests pass (11 codec tests total).

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add post/2020-03-kraft-wire-formats
git commit -m "feat(kraft): add kraft_sum (the inequality's left-hand side)"
```

---

## Task 9: Connect BinaryTree and kraft_sum (Kraft holds for prefix-free codes)

**Files:**
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/test_kraft.cpp`

This task adds tests that verify the Kraft inequality on actual prefix-free codes built via `BinaryTree`. No new implementation; just integration tests.

- [ ] **Step 1: Add integration tests**

Append to `test_kraft.cpp`:

```cpp
namespace {

// Helper: extract the lengths of all codewords in a binary tree.
// Walks the tree depth-first and records (depth -> count) for each terminal node.
void collect_lengths_recursive(const std::map<std::string, std::string>& code,
                               std::vector<std::size_t>& out) {
    for (const auto& [sym, codeword] : code) {
        out.push_back(codeword.size());
    }
}

}  // namespace

TEST(KraftTest, KraftHoldsForExampleCode) {
    // Code: A=0, B=10, C=110, D=111.
    BinaryTree t;
    t.insert("0");
    t.insert("10");
    t.insert("110");
    t.insert("111");
    ASSERT_TRUE(t.is_prefix_free());

    std::map<std::string, std::string> code{
        {"A", "0"}, {"B", "10"}, {"C", "110"}, {"D", "111"}
    };
    std::vector<std::size_t> lengths;
    collect_lengths_recursive(code, lengths);
    EXPECT_DOUBLE_EQ(kraft_sum(lengths), 1.0);
}

TEST(KraftTest, KraftHoldsForUnaryCodeUpToK) {
    // Unary: codeword for n is (n-1) zeros followed by a one. Lengths 1, 2, 3, ...
    constexpr std::size_t K = 10;
    BinaryTree t;
    std::vector<std::size_t> lengths;
    std::string codeword;
    for (std::size_t n = 1; n <= K; ++n) {
        // Unary codeword for n: (n-1) zeros then a one.
        codeword.assign(n - 1, '0');
        codeword += '1';
        t.insert(codeword);
        lengths.push_back(n);
    }
    ASSERT_TRUE(t.is_prefix_free());
    EXPECT_LT(kraft_sum(lengths), 1.0);  // strict for finite K
    EXPECT_NEAR(kraft_sum(lengths), 1.0 - std::ldexp(1.0, -static_cast<int>(K)), 1e-12);
}

TEST(KraftTest, NonPrefixFreeCodeStillSatisfiesKraftIfLengthsAllow) {
    // The lengths 1, 2 satisfy Kraft (sum = 0.75 <= 1), but the specific
    // assignment "0", "01" is NOT prefix-free. This verifies that Kraft is
    // about the LENGTH VECTOR, not the specific assignment.
    BinaryTree t;
    t.insert("0");
    t.insert("01");
    EXPECT_FALSE(t.is_prefix_free());

    std::vector<std::size_t> lengths{1, 2};
    EXPECT_DOUBLE_EQ(kraft_sum(lengths), 0.75);
}
```

- [ ] **Step 2: Run, expect pass (no implementation change)**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make test 2>&1 | tail -8
```

Expected: 3 new tests pass (14 total).

- [ ] **Step 3: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add post/2020-03-kraft-wire-formats/test_kraft.cpp
git commit -m "test(kraft): integration tests connecting BinaryTree and kraft_sum"
```

---

## Task 10: Implement trie-embedding diagnostic (TDD)

**Files:**
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/kraft.hpp`
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/test_kraft.cpp`

The trie-embedding diagnostic computes, for each codeword, the size of the subtree it occupies in the depth-`l_max` complete binary tree. Used in section D of the prose to make the proof concrete.

- [ ] **Step 1: Write failing tests**

Append to `test_kraft.cpp`:

```cpp
TEST(KraftTest, TrieEmbeddingComputesSubtreeSizes) {
    // Code with lengths {1, 2, 3, 3}, l_max = 3.
    // Codeword of length 1 occupies 2^(3-1) = 4 leaves.
    // Codeword of length 2 occupies 2^(3-2) = 2 leaves.
    // Codewords of length 3 occupy 2^(3-3) = 1 leaf each.
    // Total: 4 + 2 + 1 + 1 = 8 = 2^3 (saturates).
    auto info = trie_embedding({1, 2, 3, 3});
    EXPECT_EQ(info.l_max, 3u);
    EXPECT_EQ(info.total_leaves, 8u);  // 2^3
    ASSERT_EQ(info.subtree_sizes.size(), 4u);
    EXPECT_EQ(info.subtree_sizes[0], 4u);
    EXPECT_EQ(info.subtree_sizes[1], 2u);
    EXPECT_EQ(info.subtree_sizes[2], 1u);
    EXPECT_EQ(info.subtree_sizes[3], 1u);
    EXPECT_EQ(info.occupied_leaves, 8u);
}

TEST(KraftTest, TrieEmbeddingForUnsaturatedCode) {
    // Code with lengths {2, 2, 3}, l_max = 3.
    // Subtree sizes: 2, 2, 1 = 5 leaves.
    // Total: 8. Occupied: 5. Spare: 3.
    auto info = trie_embedding({2, 2, 3});
    EXPECT_EQ(info.l_max, 3u);
    EXPECT_EQ(info.total_leaves, 8u);
    EXPECT_EQ(info.occupied_leaves, 5u);
}

TEST(KraftTest, TrieEmbeddingForEmptyCode) {
    auto info = trie_embedding({});
    EXPECT_EQ(info.l_max, 0u);
    EXPECT_EQ(info.total_leaves, 1u);  // 2^0 = 1 (a single root, no codewords)
    EXPECT_EQ(info.occupied_leaves, 0u);
    EXPECT_TRUE(info.subtree_sizes.empty());
}
```

- [ ] **Step 2: Verify compile failure**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make build 2>&1 | head -10
```

- [ ] **Step 3: Implement trie_embedding**

In `kraft.hpp`, append before the closing `}  // namespace kraft`:

```cpp

// ---- trie_embedding -- the binary-tree proof made concrete ------------------
//
// For codeword lengths (l_1, ..., l_n), embed the code in the depth-l_max
// complete binary tree. Each codeword of length l_i corresponds to a subtree
// of size 2^{l_max - l_i} leaves. The Kraft inequality is the statement that
// the sum of these subtree sizes is at most the total number of leaves
// (2^l_max).

struct TrieEmbeddingInfo {
    std::size_t l_max;
    std::size_t total_leaves;            // 2^l_max
    std::size_t occupied_leaves;         // sum of subtree_sizes
    std::vector<std::size_t> subtree_sizes;  // 2^{l_max - l_i} for each codeword
};

inline TrieEmbeddingInfo trie_embedding(const std::vector<std::size_t>& lengths) {
    TrieEmbeddingInfo info;
    info.l_max = 0;
    for (std::size_t l : lengths) {
        if (l > info.l_max) info.l_max = l;
    }
    info.total_leaves = std::size_t{1} << info.l_max;
    info.subtree_sizes.reserve(lengths.size());
    info.occupied_leaves = 0;
    for (std::size_t l : lengths) {
        std::size_t size = std::size_t{1} << (info.l_max - l);
        info.subtree_sizes.push_back(size);
        info.occupied_leaves += size;
    }
    return info;
}
```

- [ ] **Step 4: Run, expect pass**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make test 2>&1 | tail -8
```

Expected: 3 new tests pass (17 total).

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add post/2020-03-kraft-wire-formats
git commit -m "feat(kraft): add trie_embedding (binary-tree proof made concrete)"
```

---

## Task 11: Implement is_kraft_satisfying convenience (TDD)

**Files:**
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/kraft.hpp`
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/test_kraft.cpp`

A small wrapper: returns true iff the lengths satisfy Kraft (allowing tiny floating-point tolerance). Used in the prose's converse-direction discussion.

- [ ] **Step 1: Write failing tests**

Append to `test_kraft.cpp`:

```cpp
TEST(KraftTest, IsKraftSatisfyingTrueForValidLengths) {
    EXPECT_TRUE(is_kraft_satisfying({1, 2, 3, 3}));
    EXPECT_TRUE(is_kraft_satisfying({2, 2, 2, 2}));  // saturates
    EXPECT_TRUE(is_kraft_satisfying({}));  // empty: sum is 0
}

TEST(KraftTest, IsKraftSatisfyingFalseForOverlongLengths) {
    EXPECT_FALSE(is_kraft_satisfying({1, 1, 1}));  // sum 1.5
    EXPECT_FALSE(is_kraft_satisfying({1, 1}));     // sum 1.0 + 1.0 wait, this is sum 1.0
}
```

Wait, `{1, 1}` gives sum 0.5 + 0.5 = 1.0, which IS satisfying. Fix the second test:

Replace the second test with:

```cpp
TEST(KraftTest, IsKraftSatisfyingFalseForOverlongLengths) {
    EXPECT_FALSE(is_kraft_satisfying({1, 1, 1}));  // sum 1.5
    EXPECT_FALSE(is_kraft_satisfying({1, 2, 2, 2, 2}));  // 0.5 + 4*0.25 = 1.5
}
```

- [ ] **Step 2: Verify compile failure**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make build 2>&1 | head -10
```

- [ ] **Step 3: Implement is_kraft_satisfying**

In `kraft.hpp`, append before the closing namespace:

```cpp

// ---- is_kraft_satisfying ----------------------------------------------------
//
// Returns true iff the given length vector satisfies Kraft's inequality
// (sum of 2^{-l_i} <= 1). Allows a small floating-point tolerance.

inline bool is_kraft_satisfying(const std::vector<std::size_t>& lengths) {
    constexpr double kTolerance = 1e-9;
    return kraft_sum(lengths) <= 1.0 + kTolerance;
}
```

- [ ] **Step 4: Run, expect pass**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make test 2>&1 | tail -8
```

Expected: 2 new tests pass (19 total).

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add post/2020-03-kraft-wire-formats
git commit -m "feat(kraft): add is_kraft_satisfying convenience predicate"
```

---

## Task 12: Verify post 1 implementation passes full series suite

**Files:** none modified; verification only.

- [ ] **Step 1: Run the full wire-formats test suite from a clean build**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make clean && make test 2>&1 | tail -10
```

Expected: 1 test executable (`test_kraft`), all KraftTest cases pass.

- [ ] **Step 2: Verify no warnings during build**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make clean && make build 2>&1 | grep -iE "warning|error" | head -20
```

Expected: empty output (no warnings or errors). If there are warnings, file an inline note for follow-up but do not block this task.

No commit for this task.

---

## Task 13: Draft post 1 prose (`index.md`)

**Files:**
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/index.md`

The full ~2000-word article per the spec's Section A-G outline for post 1. This task drafts the prose; one step per section. Reference the spec for the per-section guidance.

When this task starts, change `draft: true` to `draft: false`. The frontmatter `date: 2020-03-22` stays as-is (the post is intentionally backdated).

- [ ] **Step 1: Read the committed `kraft.hpp` to know what code to embed**

```bash
cat /home/spinoza/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/kraft.hpp
```

The code blocks in the prose must MATCH this file (allowing for namespace/header-line trimming). The committed code is the source of truth.

- [ ] **Step 2: Write the opening matter (italic lead-in + Section A)**

Replace the placeholder body in `index.md` with:

- Frontmatter unchanged, but flip `draft: true` to `draft: false`.
- One italic lead-in line directly under the closing `---` of frontmatter. Suggested: *"You want to assign codewords. Which lengths can you actually use?"*
- Section A "The Question" (~200 words): the motivating problem. You want a uniquely decodable code. Which length vectors `(l_1, l_2, ..., l_n)` are achievable? Kraft is the answer. Foreshadow: "the inequality you cannot escape."

- [ ] **Step 3: Write Section B (The Trie View)**

Section B (~250 words + ~50 lines code). Visualize codewords as paths in a binary tree. Show the example code `{A=0, B=10, C=110, D=111}` as a tree. Include the BinaryTree code block (insert, contains, is_prefix_free) verbatim from `kraft.hpp`.

Include an ASCII tree diagram for the example code:

```
         root
        /    \
       0      1
      [A]    / \
            0   1
           [B] / \
              0   1
             [C] [D]
```

(Or a similar clear ASCII rendering.)

- [ ] **Step 4: Write Section C (The Inequality)**

Section C (~250 words + ~30 lines code).

State Kraft formally:
$$\sum_{i=1}^{n} 2^{-l_i} \leq 1$$

Geometric intuition: each codeword of length `l_i` claims a fraction `2^{-l_i}` of the unit budget; prefix-free means the fractions are disjoint.

Include the `kraft_sum` code verbatim. Show the Kraft sum for the example code: 1/2 + 1/4 + 1/8 + 1/8 = 1.

- [ ] **Step 5: Write Section D (The Proof)**

Section D (~300 words + ~30 lines code).

Sketch the binary-tree proof:
1. Embed the code in the depth-`l_max` complete binary tree.
2. Each codeword of length `l_i` corresponds to a subtree of `2^{l_max - l_i}` leaves.
3. Prefix-freeness means these subtrees are disjoint.
4. Total occupied leaves <= total leaves (2^l_max).
5. Dividing by 2^l_max gives Kraft.

Include the `trie_embedding` code verbatim. Show a concrete example trace using lengths {1, 2, 3, 3}: l_max = 3, total_leaves = 8, subtree_sizes = {4, 2, 1, 1}, occupied = 8, ratio = 1 (saturating Kraft).

- [ ] **Step 6: Write Section E (What Kraft Gives Us)**

Section E (~250 words, no code).

Three consequences:
1. **A budget**: a codeword of length 3 costs 1/8. Spend the budget, and you cannot add more codewords.
2. **A trade-off**: shorter for some symbols means longer for others. The optimum (assigning `-log_2 p_i` lengths) saturates Kraft when probabilities are dyadic.
3. **A diagnostic**: a length vector violating Kraft has no prefix-free code. The converse direction (McMillan) says the iff holds.

- [ ] **Step 7: Write Section F (The Converse, Foreshadowed)**

Section F (~150 words, no code). State McMillan briefly: any length vector satisfying Kraft has a prefix-free code. Constructive (you can build the code). Forward-pointer to post 2: [McMillan's Converse](/post/2020-09-mcmillan-wire-formats/) develops the proof and the construction.

- [ ] **Step 8: Write Section G (Cross-references and footnote)**

Section G (~120 words). Cross-reference links:

- Forward to McMillan's Converse (post 2)
- Cross-series back to the Stepanov bridge posts: [Bits Follow Types](/post/2026-05-codecs-functors-stepanov/) and [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/) (which forward-reference Kraft from the type-algebra side)
- Footnote to PFC (`include/pfc/codecs.hpp`)

- [ ] **Step 9: Verify the soul check passes**

The hook runs automatically on Write. If a write was rejected with em-dash detection, fix the offending text using commas/periods/colons/parens before continuing. Verify the final state explicitly:

```bash
grep -c $'\xE2\x80\x94' /home/spinoza/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/index.md && echo "EM-DASHES FOUND" || echo "clean"
```

Expected: "clean" (output 0 followed by the message).

- [ ] **Step 10: Verify mkdocs builds the post**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make docs 2>&1 | tail -5
```

Expected: build succeeds. The cross-reference to post 2 (McMillan, doesn't exist yet) may produce a warning; this is acceptable until post 2 is committed.

- [ ] **Step 11: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add post/2020-03-kraft-wire-formats/index.md
git commit -m "docs(kraft): draft post 1 prose (Kraft's Inequality, 2020-03-22)"
```

---

## Task 14: Scaffold post 2 directory and wire CMakeLists

**Files:**
- Create: `~/github/metafunctor-series/wire-formats/post/2020-09-mcmillan-wire-formats/index.md`
- Create: `~/github/metafunctor-series/wire-formats/post/2020-09-mcmillan-wire-formats/mcmillan.hpp`
- Create: `~/github/metafunctor-series/wire-formats/post/2020-09-mcmillan-wire-formats/test_mcmillan.cpp`
- Modify: `~/github/metafunctor-series/wire-formats/post/CMakeLists.txt`

Same shape as Task 6.

- [ ] **Step 1: Create the post directory and skeleton files**

```bash
mkdir -p /home/spinoza/github/metafunctor-series/wire-formats/post/2020-09-mcmillan-wire-formats
```

Create `index.md`:

```markdown
---
title: "McMillan's Converse"
date: 2020-09-13
draft: true
tags:
- C++
- information-theory
- coding-theory
- prefix-free
- constructive-proof
categories:
- Computer Science
- Mathematics
series:
- wire-formats
series_weight: 2
math: true
description: "Any length vector satisfying Kraft has a prefix-free code. Here is the construction."
linked_project:
- pfc
- wire-formats
---

(Draft in progress. See plan Task 17 for full prose.)
```

Create `mcmillan.hpp`:

```cpp
// mcmillan.hpp
// Pedagogical implementation for the post "McMillan's Converse" in the
// "Algebra over Wire Formats" series. For the production version, see PFC:
// https://github.com/queelius/pfc

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace mcmillan {

// Implementation arrives in subsequent plan tasks.

}  // namespace mcmillan
```

Create `test_mcmillan.cpp`:

```cpp
#include <gtest/gtest.h>
#include "mcmillan.hpp"

TEST(McMillanTest, Placeholder) {
    EXPECT_TRUE(true);
}
```

- [ ] **Step 2: Add the test executable to post/CMakeLists.txt**

Append to `post/CMakeLists.txt` (after the test_kraft block):

```cmake

# =============================================================================
# McMillan's Converse (post 2, 2020-09-13)
# =============================================================================
add_executable(test_mcmillan 2020-09-mcmillan-wire-formats/test_mcmillan.cpp)
target_link_libraries(test_mcmillan GTest::gtest_main)
target_include_directories(test_mcmillan PRIVATE 2020-09-mcmillan-wire-formats)
add_test(NAME test_mcmillan COMMAND test_mcmillan)
```

- [ ] **Step 3: Build and verify**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make test 2>&1 | tail -8
```

Expected: 2 test executables (`test_kraft`, `test_mcmillan`), all tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add post/2020-09-mcmillan-wire-formats post/CMakeLists.txt
git commit -m "scaffold(mcmillan): add post 2 directory and CMake wiring"
```

---

## Task 15: Implement build_prefix_free_code construction (TDD)

**Files:**
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-09-mcmillan-wire-formats/mcmillan.hpp`
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-09-mcmillan-wire-formats/test_mcmillan.cpp`

The Shannon-Fano-like construction: given a Kraft-satisfying length vector, produce a prefix-free code with those lengths.

- [ ] **Step 1: Write failing tests**

Replace `test_mcmillan.cpp` with:

```cpp
#include <gtest/gtest.h>
#include <algorithm>
#include <set>
#include <string>
#include <vector>
#include "mcmillan.hpp"

using namespace mcmillan;

TEST(McMillanTest, BuildPrefixFreeCodeForBalancedLengths) {
    // Lengths 2, 2, 2, 2: should produce a 4-codeword 2-bit code.
    auto code = build_prefix_free_code({2, 2, 2, 2});
    ASSERT_EQ(code.size(), 4u);
    for (const auto& cw : code) {
        EXPECT_EQ(cw.size(), 2u);
    }
    // All codewords distinct.
    std::set<std::string> distinct(code.begin(), code.end());
    EXPECT_EQ(distinct.size(), 4u);
}

TEST(McMillanTest, BuildPrefixFreeCodeForExampleLengths) {
    // Lengths 1, 2, 3, 3 (the example code from post 1): should produce
    // a prefix-free code with those exact lengths.
    auto code = build_prefix_free_code({1, 2, 3, 3});
    ASSERT_EQ(code.size(), 4u);
    EXPECT_EQ(code[0].size(), 1u);
    EXPECT_EQ(code[1].size(), 2u);
    EXPECT_EQ(code[2].size(), 3u);
    EXPECT_EQ(code[3].size(), 3u);
}

TEST(McMillanTest, BuildPrefixFreeCodeProducesPrefixFreeCode) {
    // For any Kraft-satisfying lengths, the constructed code must be prefix-free.
    auto code = build_prefix_free_code({1, 2, 3, 3});
    // No codeword is a prefix of any other.
    for (std::size_t i = 0; i < code.size(); ++i) {
        for (std::size_t j = 0; j < code.size(); ++j) {
            if (i == j) continue;
            const std::string& a = code[i];
            const std::string& b = code[j];
            if (a.size() > b.size()) continue;
            EXPECT_NE(b.substr(0, a.size()), a)
                << "Codeword " << a << " (idx " << i
                << ") is a prefix of " << b << " (idx " << j << ")";
        }
    }
}

TEST(McMillanTest, BuildPrefixFreeCodeForEmptyLengthsReturnsEmpty) {
    auto code = build_prefix_free_code({});
    EXPECT_TRUE(code.empty());
}
```

- [ ] **Step 2: Verify compile failure**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make build 2>&1 | head -15
```

Expected: `build_prefix_free_code` not declared.

- [ ] **Step 3: Implement build_prefix_free_code**

In `mcmillan.hpp`, replace `// Implementation arrives in subsequent plan tasks.` with:

```cpp
// ---- build_prefix_free_code -- the McMillan construction --------------------
//
// Given codeword lengths (l_1, ..., l_n) satisfying Kraft's inequality, this
// function produces a prefix-free code with those exact lengths.
//
// Algorithm (Shannon-Fano-like, walking the trie left-to-right):
//   1. Sort lengths in non-decreasing order.
//   2. Start with the integer counter c = 0.
//   3. For each length l_i in the sorted order:
//      - Emit the binary representation of c, left-padded to l_i bits.
//      - Increment c by 2^{l_max - l_i}, where l_max is the maximum length.
//   4. Return the code (in the original length order).
//
// The increment ensures the next codeword starts at the next available leaf
// in the depth-l_max trie. Kraft's inequality guarantees the increments fit
// within 2^l_max.

inline std::string to_binary(std::uint64_t value, std::size_t width) {
    std::string s(width, '0');
    for (std::size_t i = 0; i < width; ++i) {
        if (value & (std::uint64_t{1} << (width - 1 - i))) {
            s[i] = '1';
        }
    }
    return s;
}

inline std::vector<std::string> build_prefix_free_code(
    const std::vector<std::size_t>& lengths)
{
    if (lengths.empty()) return {};

    // Pair each length with its original index so we can return codewords
    // in the input order after sorting.
    std::vector<std::pair<std::size_t, std::size_t>> indexed;
    indexed.reserve(lengths.size());
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        indexed.emplace_back(lengths[i], i);
    }
    std::sort(indexed.begin(), indexed.end());

    std::size_t l_max = indexed.back().first;
    std::vector<std::string> code(lengths.size());
    std::uint64_t counter = 0;
    for (const auto& [l, original_idx] : indexed) {
        std::uint64_t shifted = counter >> (l_max - l);
        code[original_idx] = to_binary(shifted, l);
        counter += std::uint64_t{1} << (l_max - l);
    }
    return code;
}
```

- [ ] **Step 4: Run tests, expect pass**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make test 2>&1 | tail -8
```

Expected: 4 tests pass for McMillanTest.

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add post/2020-09-mcmillan-wire-formats
git commit -m "feat(mcmillan): add build_prefix_free_code (Shannon-Fano-like construction)"
```

---

## Task 16: Implement is_prefix_free verification helper (TDD)

**Files:**
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-09-mcmillan-wire-formats/mcmillan.hpp`
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-09-mcmillan-wire-formats/test_mcmillan.cpp`

A free-standing verifier so tests can verify codes regardless of construction. Mirrors the BinaryTree::is_prefix_free in post 1 but operates on a vector of codewords.

- [ ] **Step 1: Write failing tests**

Append to `test_mcmillan.cpp`:

```cpp
TEST(McMillanTest, IsPrefixFreeReturnsTrueForValidCode) {
    EXPECT_TRUE(is_prefix_free({"0", "10", "110", "111"}));
}

TEST(McMillanTest, IsPrefixFreeReturnsFalseWhenViolating) {
    EXPECT_FALSE(is_prefix_free({"0", "01"}));
    EXPECT_FALSE(is_prefix_free({"01", "010"}));
}

TEST(McMillanTest, IsPrefixFreeReturnsTrueForEmptyCode) {
    EXPECT_TRUE(is_prefix_free({}));
}

TEST(McMillanTest, BuildPrefixFreeCodeOutputIsActuallyPrefixFree) {
    // End-to-end: construction must produce a prefix-free code.
    for (const std::vector<std::size_t>& lengths : {
        std::vector<std::size_t>{1, 2, 3, 3},
        std::vector<std::size_t>{2, 2, 2, 2},
        std::vector<std::size_t>{3, 3, 3, 3, 3, 3, 3, 3},  // 8 length-3 codewords (saturates)
        std::vector<std::size_t>{1, 3, 3, 3, 3},  // 1/2 + 4 * 1/8 = 1
    }) {
        auto code = build_prefix_free_code(lengths);
        EXPECT_TRUE(is_prefix_free(code))
            << "Constructed code is not prefix-free for lengths above";
    }
}
```

- [ ] **Step 2: Verify compile failure**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make build 2>&1 | head -10
```

- [ ] **Step 3: Implement is_prefix_free**

In `mcmillan.hpp`, append before the closing namespace:

```cpp

// ---- is_prefix_free -- verification helper ----------------------------------
//
// Returns true iff no codeword in the input vector is a prefix of any other.
// Useful for verifying that build_prefix_free_code produces valid output.

inline bool is_prefix_free(const std::vector<std::string>& code) {
    for (std::size_t i = 0; i < code.size(); ++i) {
        for (std::size_t j = 0; j < code.size(); ++j) {
            if (i == j) continue;
            const std::string& a = code[i];
            const std::string& b = code[j];
            if (a.size() > b.size()) continue;
            if (b.compare(0, a.size(), a) == 0) return false;
        }
    }
    return true;
}
```

- [ ] **Step 4: Run, expect pass**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make test 2>&1 | tail -8
```

Expected: 4 new tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add post/2020-09-mcmillan-wire-formats
git commit -m "feat(mcmillan): add is_prefix_free verification helper"
```

---

## Task 17: Draft post 2 prose (`index.md`)

**Files:**
- Modify: `~/github/metafunctor-series/wire-formats/post/2020-09-mcmillan-wire-formats/index.md`

The full ~2000-word article per the spec's Section A-G outline for post 2. Reference the spec for per-section guidance.

When this task starts, change `draft: true` to `draft: false`.

- [ ] **Step 1: Read the committed `mcmillan.hpp` to know what to embed**

```bash
cat /home/spinoza/github/metafunctor-series/wire-formats/post/2020-09-mcmillan-wire-formats/mcmillan.hpp
```

- [ ] **Step 2: Read post 1's prose for voice continuity**

```bash
cat /home/spinoza/github/metafunctor-series/wire-formats/post/2020-03-kraft-wire-formats/index.md
```

- [ ] **Step 3: Write opening matter (italic lead-in + Section A "The Promise")**

Replace the placeholder in `index.md` with frontmatter (flip `draft: true` to `draft: false`) plus:

- One italic lead-in. Suggested: *"Kraft told us which lengths are achievable. McMillan tells us how to achieve them."*
- Section A "The Promise" (~150 words): recap from post 1 (Kraft says length vectors of prefix-free codes satisfy `sum 2^-l_i <= 1`). McMillan says the converse: any Kraft-satisfying length vector has a prefix-free code. This post proves it constructively.

- [ ] **Step 4: Write Section B (The Construction)**

Section B (~300 words + ~80 lines code).

State the algorithm:
1. Sort lengths in non-decreasing order.
2. Walk a trie left-to-right, assigning the next available leaf at each depth.
3. Equivalently: maintain a counter; for each length `l_i`, emit the binary representation of the counter (right-shifted to width `l_i`), then increment the counter by `2^{l_max - l_i}`.

Include the full `build_prefix_free_code` code (and `to_binary`) verbatim from `mcmillan.hpp`. Walk through a concrete example: lengths {1, 2, 3, 3} produces codewords {"0", "10", "110", "111"} (the same code from post 1).

- [ ] **Step 5: Write Section C (Why It Works)**

Section C (~250 words, no code).

Trie-walking interpretation: at each step, the next codeword "fits" because Kraft guarantees enough budget remains.

Sketch:
- After placing codewords 1..i-1, the remaining budget is `1 - sum_{j<i} 2^{-l_j}`.
- Placing a codeword of length `l_i` costs `2^{-l_i}`.
- Kraft on the original lengths says the total cost is at most 1, so the remaining budget covers `2^{-l_i}` plus everything still to come.

The construction never runs out of budget because Kraft pre-certifies it.

- [ ] **Step 6: Write Section D (The Deeper Converse)**

Section D (~250 words, no code).

McMillan actually proved something stronger: any UNIQUELY DECODABLE code (not necessarily prefix-free) has length vector satisfying Kraft. State the theorem; sketch the slick proof using the L-th power of the code:

- Consider extending the code by concatenating L codewords. This produces an "L-th-power code" with `n^L` codewords whose lengths range from `L * l_min` to `L * l_max`.
- For the original code to be uniquely decodable, the L-th-power code must have all distinct codewords.
- Counting: at each length `m`, there are at most `2^m` distinct binary strings, so the count of length-`m` codewords in the L-th-power is bounded.
- Algebra (see Cover & Thomas Theorem 5.5.1 for the full chain of inequalities) gives `(sum 2^{-l_i})^L <= L * (l_max - l_min) + 1` for all L.
- The right side grows polynomially in L; the left side grows exponentially in L if `sum > 1`. So `sum <= 1`.

This shows: prefix-freeness gives no length advantage over unique decodability. We can restrict to prefix-free codes without loss.

- [ ] **Step 7: Write Section E (Implications)**

Section E (~250 words, no code).

Two consequences:
1. **No advantage to non-prefix-free uniquely-decodable codes**: any uniquely-decodable code with given lengths can be replaced by a prefix-free code with the same lengths. So we restrict to prefix-free codes.
2. **The optimal lengths exist**: given any probability distribution `(p_1, ..., p_n)`, the lengths `l_i = ceil(-log_2 p_i)` satisfy Kraft (this is straightforward to verify), so a prefix-free code achieving them exists. This underwrites Huffman (post 9, forthcoming) and arithmetic coding (post 10, forthcoming).

- [ ] **Step 8: Write Section F (The Construction in PFC)**

Section F (~150 words + ~30 lines code).

Reference the construction in PFC (`include/pfc/huffman.hpp`). Note that Huffman's algorithm IS a McMillan-style construction that additionally minimizes the expected length under a given probability distribution. The construction here is the simpler predecessor that Huffman optimizes.

If you want to show Huffman as inheriting the construction in code, embed a small snippet from PFC; otherwise just describe the connection in prose.

- [ ] **Step 9: Write Section G (Cross-references and footnote)**

Section G (~120 words). Cross-reference links:

- Back to [Kraft's Inequality](/post/2020-03-kraft-wire-formats/) (post 1)
- Forward (no link yet, plain text): "Universal Codes as Priors (post 3, forthcoming) develops the prior-as-hypothesis framing; Huffman (post 9, forthcoming) uses McMillan's construction."
- Cross-series back to [When Lists Become Bits](/post/2026-05-prefix-free-stepanov/) for the bit-level consequences of prefix-freeness.
- Footnote to PFC.

- [ ] **Step 10: Verify the soul check**

```bash
grep -c $'\xE2\x80\x94' /home/spinoza/github/metafunctor-series/wire-formats/post/2020-09-mcmillan-wire-formats/index.md && echo "EM-DASHES FOUND" || echo "clean"
```

Expected: "clean".

- [ ] **Step 11: Verify mkdocs builds**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make docs 2>&1 | tail -5
```

Expected: build succeeds. The cross-link to post 1 should resolve cleanly.

- [ ] **Step 12: Commit**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git add post/2020-09-mcmillan-wire-formats/index.md
git commit -m "docs(mcmillan): draft post 2 prose (McMillan's Converse, 2020-09-13)"
```

---

## Task 18: Verify both posts pass full series suite from clean build

**Files:** none modified; verification only.

- [ ] **Step 1: Run the full wire-formats test suite from clean**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make clean && make test 2>&1 | tail -10
```

Expected: 2 test executables (`test_kraft`, `test_mcmillan`), all tests pass (~26 individual tests across both).

- [ ] **Step 2: Verify mkdocs renders both posts cleanly**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make docs 2>&1 | tail -10
```

Expected: build succeeds. Cross-links between post 1 and post 2 resolve. Cross-links to forthcoming posts (3-13) and the Stepanov bridge posts may produce warnings (acceptable).

- [ ] **Step 3: Verify both prose files pass the soul check**

```bash
for f in /home/spinoza/github/metafunctor-series/wire-formats/post/*-wire-formats/index.md; do
    grep -c $'\xE2\x80\x94' "$f" >/dev/null && echo "EM-DASHES in $f" || true
done
echo "scan complete (no output above means clean)"
```

Expected: no em-dash detections. (The hook would have already blocked any em-dashes during writes.)

No commit for this task.

---

## Task 19: Update Stepanov bridge posts' forward-references

**Files:**
- Modify: `~/github/metafunctor-series/stepanov/post/2026-05-codecs-functors-stepanov/index.md`
- Modify: `~/github/metafunctor-series/stepanov/post/2026-05-prefix-free-stepanov/index.md`

The Stepanov bridge posts forward-reference "the forthcoming Information Theory by Construction series." Now that wire-formats exists with posts 1 and 2, update these references to point at the new series and/or its specific posts.

- [ ] **Step 1: Find the forthcoming-series references in post 1 (codecs-functors)**

```bash
grep -n -i "forthcoming\|information theory by construction\|new series" /home/spinoza/github/metafunctor-series/stepanov/post/2026-05-codecs-functors-stepanov/index.md
```

Expected: a small number of matches (likely 1-2). Note line numbers and surrounding context.

- [ ] **Step 2: Update post 1's references**

For each match, replace the placeholder phrasing with a live link. The most likely existing phrasings (per the post 1 spec) are around the Either combinator section and the Vec section, where the post forward-pointed to "Huffman/arithmetic coding" or to "the Kraft proof." Replace these with:

For the Huffman/arithmetic forward-pointer (in section E, the Either combinator):
- Before: "the forthcoming Information Theory by Construction series develops [or "covers"] Huffman and arithmetic coding"
- After: "[Huffman](/post/2024-03-huffman-wire-formats/) and [Arithmetic Coding](/post/2024-09-arithmetic-coding-wire-formats/), forthcoming in the [Algebra over Wire Formats](/series/wire-formats/) series, develop these"

(Note: links to posts 9 and 10 will resolve once those posts exist; until then, a series-level link `/series/wire-formats/` resolves on the live site.)

For other generic "forthcoming series" mentions, replace with: "the [Algebra over Wire Formats](/series/wire-formats/) series."

Use the Edit tool with the exact existing string as `old_string`. Read the file first to get the exact wording.

- [ ] **Step 3: Find and update post 2 (prefix-free) references**

```bash
grep -n -i "forthcoming\|information theory by construction\|new series" /home/spinoza/github/metafunctor-series/stepanov/post/2026-05-prefix-free-stepanov/index.md
```

Per the spec, post 2 sections H and J reference the forthcoming series. Update them:

For section H (Kraft mention):
- Before: "the forthcoming Information Theory by Construction series develops the proof"
- After: "[Kraft's Inequality](/post/2020-03-kraft-wire-formats/) and [McMillan's Converse](/post/2020-09-mcmillan-wire-formats/) in the [Algebra over Wire Formats](/series/wire-formats/) series develop the proof and the constructive converse"

For section J (Further Reading):
- Before: "the forthcoming Information Theory by Construction series for the Kraft proof and the full universal-codes treatment"
- After: "the [Algebra over Wire Formats](/series/wire-formats/) series for the Kraft proof, the constructive converse (McMillan), and the full universal-codes treatment"

- [ ] **Step 4: Verify the soul check on both edited stepanov posts**

```bash
for f in /home/spinoza/github/metafunctor-series/stepanov/post/2026-05-codecs-functors-stepanov/index.md \
         /home/spinoza/github/metafunctor-series/stepanov/post/2026-05-prefix-free-stepanov/index.md; do
    grep -c $'\xE2\x80\x94' "$f" >/dev/null 2>&1 && echo "EM-DASHES in $f" || true
done
echo "scan complete (no output = clean)"
```

Expected: no em-dash detections.

- [ ] **Step 5: Verify mkdocs in stepanov still builds**

```bash
cd /home/spinoza/github/metafunctor-series/stepanov && make docs 2>&1 | tail -5
```

Expected: build succeeds. New cross-references to wire-formats posts may produce warnings until those posts are synced into the metafunctor Hugo site (Task 21); this is acceptable for a local mkdocs preview.

- [ ] **Step 6: Commit (in stepanov)**

```bash
cd /home/spinoza/github/metafunctor-series/stepanov
git add post/2026-05-codecs-functors-stepanov/index.md post/2026-05-prefix-free-stepanov/index.md
git commit -m "docs(stepanov-bridge): replace 'forthcoming series' with live links to wire-formats posts"
```

---

## Task 20: Hugo sync (wire-formats posts into metafunctor)

**Files:**
- Create (in metafunctor): `content/post/2020-03-kraft-wire-formats/`
- Create (in metafunctor): `content/post/2020-09-mcmillan-wire-formats/`
- (Possibly modify) `~/github/repos/metafunctor/.mf/series_db.json` via `mf series scan` if needed.
- Create (if needed): `content/series/wire-formats/_index.md`

- [ ] **Step 1: Run the (FIXED) sync target**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && \
  BLOG_POST_DIR=/home/spinoza/github/repos/metafunctor/content/post make sync 2>&1 | tail -10
```

Expected output: per-directory rsync output for both posts ("-> 2020-03-kraft-wire-formats" then "-> 2020-09-mcmillan-wire-formats"), then "Done. Synced wire-formats posts to ...".

- [ ] **Step 2: Verify both posts are now in metafunctor's content/post/ as DIRECTORIES**

```bash
ls /home/spinoza/github/repos/metafunctor/content/post/ | grep wire-formats
ls /home/spinoza/github/repos/metafunctor/content/post/2020-03-kraft-wire-formats/
ls /home/spinoza/github/repos/metafunctor/content/post/2020-09-mcmillan-wire-formats/
```

Expected: two directories, each containing `index.md`, `kraft.hpp` or `mcmillan.hpp`, and `test_kraft.cpp` or `test_mcmillan.cpp`. NO flattened files at the top of `content/post/`.

- [ ] **Step 3: Verify metafunctor's content/post/ wasn't otherwise damaged**

```bash
cd /home/spinoza/github/repos/metafunctor && git status --short content/post/ | head -20
```

Expected: only the two new untracked directories appear. NO mass deletions or unexpected modifications.

- [ ] **Step 4: Check whether mf auto-discovers wire-formats**

```bash
cd /home/spinoza/github/repos/metafunctor && mf series scan 2>&1 | grep -i wire-formats
```

Expected one of:
(a) "wire-formats ... 2 posts ... Yes ... Yes" -- already auto-discovered, no further action needed.
(b) "Orphaned series (used in content but not in DB): - wire-formats (2 posts)" -- need to add to series_db.json.

- [ ] **Step 5: If wire-formats is orphaned, add it to mf**

If step 4 reported wire-formats as orphaned, the simplest fix is to add it via mf:

```bash
cd /home/spinoza/github/repos/metafunctor && mf series create wire-formats --title "Algebra over Wire Formats" 2>&1 | tail -5
```

(If `mf series create` doesn't support `--title` or has different flags, run `mf series create --help` to see the actual interface.)

If step 4 already showed the series in the DB, skip this step.

- [ ] **Step 6: Verify mf series scan now lists wire-formats correctly**

```bash
cd /home/spinoza/github/repos/metafunctor && mf series scan 2>&1 | grep -E "stepanov|wire-formats" | head -5
```

Expected: both stepanov and wire-formats appear in the table with correct post counts.

- [ ] **Step 7: Optionally create content/series/wire-formats/_index.md**

If the Hugo site requires a series landing page (check whether `content/series/stepanov/_index.md` exists for comparison), create one for wire-formats:

```bash
mkdir -p /home/spinoza/github/repos/metafunctor/content/series/wire-formats
cat > /home/spinoza/github/repos/metafunctor/content/series/wire-formats/_index.md <<'EOF'
---
title: "Algebra over Wire Formats"
description: "A pedagogical series exploring information theory by construction in C++23. Companion to the Stepanov series."
---

A pedagogical series exploring information theory by construction in C++23. Companion to the [Stepanov series](/series/stepanov/), which develops algorithms from algebra on type. This series develops algorithms from algebra on bit strings.

The central thesis: a code is a hypothesis about the source. Each universal code corresponds to a different prior over the integers; each entropy-optimal code (Huffman, arithmetic) is best under different assumptions; each succinct data structure is the right answer under different access patterns.
EOF
```

If `content/series/stepanov/_index.md` does NOT exist, skip this step (Hugo handles series taxonomies automatically without needing a landing page).

- [ ] **Step 8: Confirm the metafunctor working tree is clean other than the new content**

```bash
cd /home/spinoza/github/repos/metafunctor && git status --short content/post/ content/series/ 2>&1 | head -10
```

Expected: only the new wire-formats post directories (and possibly the new `_index.md`) appear as untracked.

No commit in metafunctor for this task; the user will commit in Task 22 after final verification.

---

## Task 21: Wire-formats final verification and commits

**Files:** verification across both repos.

- [ ] **Step 1: Verify wire-formats clean rebuild**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && make clean && make test 2>&1 | tail -5
```

Expected: 2 test executables, all tests pass.

- [ ] **Step 2: Verify wire-formats mkdocs --strict if available, else regular build**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && mkdocs build --strict 2>&1 | tail -10
```

Expected: succeeds with no warnings, OR fails on cross-references to forthcoming posts (3-13) that don't exist. If --strict fails, run regular `make docs` and verify no errors (warnings about unresolved /series/wire-formats/ links are acceptable; that resolves on Hugo, not on mkdocs).

- [ ] **Step 3: Verify wire-formats git log is clean and descriptive**

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats && git log --oneline
```

Expected: a sequence of small, well-described commits (from Task 2 onward). Each task should have produced 1-2 commits with clear conventional-commit-style messages.

- [ ] **Step 4: Verify stepanov repo's bridge edits are clean**

```bash
cd /home/spinoza/github/metafunctor-series/stepanov && git log --oneline -3
```

Expected: top commit is the Task 19 forward-reference update.

- [ ] **Step 5: Pause for user confirmation before pushing**

Per the user's standing convention, do not push or open PRs without explicit approval. Report:
- Wire-formats repo: N commits ahead of (no remote yet OR origin/main if a remote was set up)
- Stepanov repo: 1 new commit ahead of origin/main
- Metafunctor repo: 2 new untracked directories in content/post/ and possibly 1 new file in content/series/

Wait for the user to say "push" before proceeding to Task 22.

No commit for this task.

---

## Task 22: Push and finalize

**Files:** push commits to remotes (with user confirmation).

- [ ] **Step 1: Wait for user push confirmation (gate from Task 21)**

If the user has not approved, do not proceed.

- [ ] **Step 2: Decide on wire-formats remote**

If a wire-formats GitHub remote exists, push to it:

```bash
cd /home/spinoza/github/metafunctor-series/wire-formats
git remote -v
# If a remote exists:
git push -u origin main 2>&1 | tail -5
```

If no remote exists, ask the user whether to (a) leave it local-only or (b) wait for them to create the remote and push later. Either is acceptable; the wire-formats posts already live in the metafunctor Hugo site after sync.

- [ ] **Step 3: Push the stepanov bridge update**

```bash
cd /home/spinoza/github/metafunctor-series/stepanov && git push origin main 2>&1 | tail -5
```

Expected: 1 commit pushed (the Task 19 bridge-reference update).

- [ ] **Step 4: Commit and push the metafunctor sync**

```bash
cd /home/spinoza/github/repos/metafunctor
git add content/post/2020-03-kraft-wire-formats content/post/2020-09-mcmillan-wire-formats
# If content/series/wire-formats/ was created in Task 20 step 7, also add:
# git add content/series/wire-formats/
git commit -m "feat(posts): add Kraft's Inequality and McMillan's Converse (Algebra over Wire Formats series, posts 1 and 2)

Two new posts launching the Algebra over Wire Formats series, which sits
alongside the Stepanov series and develops information theory by construction.

- Kraft's Inequality (2020-03-22): which length vectors are achievable for
  prefix-free codes
- McMillan's Converse (2020-09-13): every Kraft-satisfying length vector
  has a prefix-free code

Companion code lives in the canonical wire-formats repo
(github.com/queelius/wire-formats, slot 1-2) once published. The series
forward-references from the Stepanov bridge posts now point here."
git push origin master 2>&1 | tail -5
```

Note: the metafunctor repo uses `master` as the default branch (per Task 20 verification on a similar repo).

- [ ] **Step 5: Final report**

Summarize for the user:
- Wire-formats repo state (local + push status)
- Stepanov bridge update pushed
- Metafunctor repo state (commits made + push status)
- Hugo site state: both new posts available at the planned URLs once the next site deploy runs

Sub-project 2 complete.

---

## Self-review

After writing the complete plan, look at the spec with fresh eyes and check the plan against it.

**1. Spec coverage:** Each spec section has at least one corresponding task:
- Spec "Repo structure" -> Tasks 2 (LICENSE, gitignore, README, CLAUDE, Makefile), 4 (mkdocs.yml, docs/), 5 (post/CMakeLists.txt)
- Spec "Frontmatter convention" -> Tasks 6 and 14 (placeholder frontmatter when scaffolding) and 13 and 17 (final frontmatter when drafting prose)
- Spec "Voice, code, test conventions" -> Tasks 7-11 (TDD code), 13 (post 1 prose with soul check), 15-16 (TDD code), 17 (post 2 prose with soul check)
- Spec "Post 1: Kraft's Inequality" sections A-G -> Task 13 (one step per section)
- Spec "Post 2: McMillan's Converse" sections A-G -> Task 17 (one step per section)
- Spec "Stepanov bridge updates" -> Task 19
- Spec "Hugo sync strategy" -> Task 20
- Spec "Acceptance criteria" -> Task 21 (verification) and Task 22 (push)

**2. Placeholder scan:** No placeholder text patterns appear ("TBD", "TODO", "fill in details", etc.).

**3. Type consistency:** Function names used in tests match function names in implementations:
- `BinaryTree::insert`, `BinaryTree::contains`, `BinaryTree::is_prefix_free` (Task 7)
- `kraft_sum` (Task 8)
- `trie_embedding` returning `TrieEmbeddingInfo{l_max, total_leaves, occupied_leaves, subtree_sizes}` (Task 10)
- `is_kraft_satisfying` (Task 11)
- `to_binary`, `build_prefix_free_code` (Task 15)
- `is_prefix_free` (Task 16)

All names consistent across tasks.
