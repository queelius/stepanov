# Stepanov

Pedagogical blog posts exploring generic programming and algorithmic mathematics in C++20.

**Named for Alex Stepanov**, whose work on the STL and *Elements of Programming* showed that algorithms arise from algebraic structure, not ad-hoc coding.

## The Core Thesis

**Algorithms arise from algebraic structure.** The Russian peasant algorithm isn't about integers—it's about monoids. The same `power()` function computes integer exponentiation, Fibonacci numbers (via matrices), 3D rotation composition (via quaternions), and shortest paths (via tropical semirings).

See structure first, algorithm second.

## Posts

Each post in `post/` is self-contained with:
- `index.md` — The article (Hugo-compatible with YAML frontmatter)
- `*.hpp` — Minimal implementation (~100-400 lines)
- Tests

| Post | Topic |
|------|-------|
| `2019-03-peasant-stepanov/` | Russian peasant algorithm, exponentiation, 15 monoid examples |
| `2019-06-modular-stepanov/` | Integers mod N as rings and fields |
| `2019-09-miller-rabin-stepanov/` | Probabilistic primality testing |
| `2020-02-rational-stepanov/` | Exact fraction arithmetic, GCD |
| `2020-07-polynomials-stepanov/` | Polynomial arithmetic, Euclidean domains |
| `2021-03-elementa-stepanov/` | Pedagogical linear algebra, Matrix concept |
| `2021-09-dual-stepanov/` | Forward-mode autodiff via dual numbers |
| `2022-04-finite-diff-stepanov/` | Numerical differentiation |
| `2023-01-autodiff-stepanov/` | Reverse-mode autodiff (backpropagation) |
| `2023-08-integration-stepanov/` | Numerical quadrature |
| `2024-02-type-erasure-stepanov/` | Sean Parent's value-semantic polymorphism |
| `2025-01-differentiation-stepanov/` | Differentiation techniques compared |
| `2026-01-18-synthesis-stepanov/` | Synthesis: seeing structure first |
| `2026-01-19-duality-stepanov/` | Duality: forward/reverse, push/pull, encode/decode |

## Building

```bash
make build   # Configure and build
make test    # Run tests
make clean   # Remove build artifacts
```

Or manually:

```bash
cd post
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Directory Structure

```
stepanov/
├── post/                        # Blog posts (Hugo/mkdocs compatible)
│   ├── 2019-03-peasant-stepanov/
│   │   ├── index.md            # Article with YAML frontmatter
│   │   ├── peasant.hpp         # Implementation
│   │   └── examples/           # Monoid examples
│   ├── 2019-06-modular-stepanov/
│   │   └── ...
│   ├── CMakeLists.txt
│   └── build/                  # Generated (gitignored)
├── docs/                        # mkdocs site source
│   ├── index.md                # Landing page
│   └── about.md
├── mkdocs.yml                   # mkdocs configuration
├── CLAUDE.md                    # AI assistant instructions
├── Makefile
└── README.md
```

## Reading Order

**Number theory track:**
1. `peasant` → `modular` → `miller-rabin`

**Calculus track:**
1. `dual` → `finite-diff` → `integration` → `autodiff`

**Linear algebra track:**
1. `elementa` → `autodiff`

**Algebraic structures track:**
1. `rational` → `polynomials`

**Start anywhere.** Each post is self-contained. The `synthesis` post ties everything together.

## Philosophy

- **Minimal**: ~100-400 lines per implementation
- **Pedagogical**: Code teaches principles, not production patterns
- **Algebraic**: Structure determines algorithms
- **Self-contained**: Each post stands alone

## Blog

These posts are published at [metafunctor.com](https://metafunctor.com) in the Stepanov series.

## Further Reading

- Stepanov & McJones, *Elements of Programming* (2009)
- Stepanov & Rose, *From Mathematics to Generic Programming* (2014)
- [Stepanov's collected papers](http://stepanovpapers.com/)

## Author

**Alex Towell** — [metafunctor.com](https://metafunctor.com) — [queelius@gmail.com](mailto:queelius@gmail.com)

## License

MIT
