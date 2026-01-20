---
title: "Duality: The Hidden Structure of Opposites"
date: 2026-01-19
draft: false
tags:
  - C++
  - duality
  - category-theory
  - automatic-differentiation
  - iterators
  - generic-programming
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 13
math: true
description: "Many structures come in pairs: forward/reverse AD, push/pull iteration, encode/decode. Recognizing duality lets you transfer theorems and insights between domains."
---

*Many structures come in pairs. Recognizing duality lets you transfer insights between domains.*

## The Motivating Example

This collection includes two approaches to automatic differentiation:

- **Forward mode** (in [dual](/post/2021-09-dual-stepanov/)): Propagate derivatives alongside values, from inputs toward outputs
- **Reverse mode** (in [autodiff](/post/2023-01-autodiff-stepanov/)): Build a graph during forward evaluation, then propagate gradients backward from outputs toward inputs

These aren't just two implementations of the same idea. They're *duals*---mirror images with complementary strengths.

Forward mode computes one column of the Jacobian per pass. If \(f: \mathbb{R}^n \to \mathbb{R}^m\), computing the full Jacobian takes \(n\) passes. Reverse mode computes one row per pass---\(m\) passes for the full Jacobian.

For neural network training, we have many inputs (millions of parameters) and one output (the loss). Reverse mode wins overwhelmingly: one backward pass gives all gradients. This is why backpropagation dominates deep learning.

For sensitivity analysis with few parameters and many outputs, forward mode wins. Same algorithm structure, opposite traversal direction, complementary use cases.

The mathematical explanation: forward mode computes Jacobian-vector products (\(Jv\)); reverse mode computes vector-Jacobian products (\(v^T J\)). These are transposes of each other. **Duality is transposition.**

## Push vs Pull

Consider two ways to traverse a sequence:

**Pull (iterator/consumer controls):**
```cpp
for (auto it = seq.begin(); it != seq.end(); ++it) {
    process(*it);  // Consumer pulls each element
}
```

**Push (producer controls):**
```cpp
seq.for_each([](auto x) {
    process(x);  // Producer pushes each element
});
```

Same traversal. Same elements processed. But control flow is reversed:

| Aspect | Pull (Iterator) | Push (Generator) |
|--------|-----------------|------------------|
| Who controls pace? | Consumer | Producer |
| Suspend/resume? | Consumer decides when to call `++` | Producer decides when to yield |
| Backpressure | Natural (just stop pulling) | Must be designed in |
| Composition | Chain iterators | Chain callbacks |

C++ ranges are pull-based: `view | filter | transform` creates an iterator that pulls through the pipeline. Reactive streams (Rx) are push-based: events flow through a pipeline of observers.

These are duals. Given a pull-based algorithm, you can mechanically derive its push-based counterpart by reversing who initiates each step. The transformation preserves correctness because it's just changing *direction*, not *content*.

## Encode vs Decode

Compression algorithms come in pairs:

```cpp
// Encoder: structure -> bits
auto encode(const Document& doc) -> Bitstream;

// Decoder: bits -> structure
auto decode(const Bitstream& bits) -> Document;
```

These must be inverses: `decode(encode(x)) == x`. But their implementations are often strikingly different:

| Aspect | Encoder | Decoder |
|--------|---------|---------|
| Sees | Full structure | Linear stream |
| Can do | Multiple passes, global analysis | Single pass, local decisions |
| Optimizes for | Small output | Fast reconstruction |

Huffman coding illustrates this asymmetry. Encoding requires building a frequency table (scan all symbols), constructing a tree, then traversing the tree for each symbol. Decoding walks the tree bit-by-bit---simpler, often faster.

The same pattern appears in:
- Serialization (object -> bytes) vs deserialization (bytes -> object)
- Compilation (source -> bytecode) vs interpretation (bytecode -> execution)
- Encryption vs decryption

Recognizing encode/decode as duals suggests questions: if encoding is hard, can decoding be easy? If we have a fast encoder, what does its dual look like?

## Covariance and Contravariance

Functions have a peculiar duality in their arguments:

```cpp
// If Dog <: Animal (Dog is a subtype of Animal)
// Then:
//   vector<Dog> is NOT a subtype of vector<Animal>     (invariant)
//   function<Dog> is NOT a subtype of function<Animal>  (contravariant in input)
//   function returning Dog IS a subtype of function returning Animal (covariant in output)
```

A function `Animal -> String` can substitute for `Dog -> String` because any code expecting to pass a Dog can pass it to something expecting any Animal. The subtyping relationship *reverses* for inputs.

This is contravariance: inputs are dual to outputs. The reversal happens because inputs flow *into* the function while outputs flow *out*. Direction reversal is duality.

In C++, this manifests in template variance:
- `const T*` is covariant in T (can convert `const Dog*` to `const Animal*`)
- Function parameters are contravariant (accepting broader types is safe)

Understanding this duality prevents type errors and clarifies API design.

## The Category-Theoretic View

Category theory makes duality precise. A category has objects and arrows (morphisms) between them:

```
    f        g
A -----> B -----> C
```

The **opposite category** \(C^{op}\) has the same objects but all arrows reversed:

```
    f^op     g^op
A <----- B <----- C
```

Every theorem in \(C\) has a dual theorem in \(C^{op}\), obtained by reversing all arrows. This mechanical transformation---"reverse the arrows"---generates valid mathematics for free.

Some dualities this reveals:

| Concept | Dual Concept |
|---------|--------------|
| Product (\(A \times B\)) | Coproduct (\(A + B\), sum type) |
| Terminal object (unit, `void`) | Initial object (empty, `never`) |
| Monomorphism (injection) | Epimorphism (surjection) |
| Limit | Colimit |
| Algebra | Coalgebra |

A **product** is an object \(P\) with projections to \(A\) and \(B\) such that any object with maps to both \(A\) and \(B\) factors uniquely through \(P\):

```
        X
       /|\
      / | \
     /  |  \
    v   v   v
    A<--P-->B
```

A **coproduct** is the dual: an object \(S\) with injections from \(A\) and \(B\), such that any object receiving maps from both factors uniquely through \(S\):

```
    A-->S<--B
        |
        v
        X
```

In types: product is `struct { A a; B b; }` (you have both). Coproduct is `variant<A, B>` (you have one or the other). Same diagram, arrows reversed.

## Why Duality Matters

**Transfer theorems.** Prove something about products; the dual theorem about coproducts follows automatically. Prove forward-mode AD is correct; reverse-mode correctness follows by transposition. Work done once yields two results.

**Design insight.** When you build X, ask: what is dual(X)? If you have a parser (string -> tree), consider the printer (tree -> string). If you have a fold (consume structure), consider an unfold (produce structure). The dual often fills a complementary need.

**Debugging.** If forward mode works but reverse mode doesn't, examine where arrow reversal might have gone wrong. Duality localizes errors: the bug is where the dual transformation wasn't applied correctly.

**API symmetry.** Push/pull, encode/decode, serialize/deserialize---recognizing these as duals suggests they should have symmetric interfaces. Asymmetry in dual pairs often indicates missing functionality or awkward design.

## Examples in This Collection

The duality theme runs through several posts:

**Forward vs reverse AD** ([dual](/post/2021-09-dual-stepanov/) and [autodiff](/post/2023-01-autodiff-stepanov/)): The clearest example. Same derivatives, opposite traversal directions, complementary efficiency profiles.

**Iterators and ranges**: The [elementa](/post/2021-03-elementa-stepanov/) matrix iteration follows pull semantics. The dual push semantics would be a visitor pattern over matrix elements.

**Fold and unfold in power()**: The [peasant](/post/2019-03-peasant-stepanov/) algorithm is a fold: it consumes the exponent bit-by-bit, accumulating the result. The dual unfold would generate the sequence of squares from a base---which is exactly what repeated squaring does implicitly.

**Encode/decode**: While not explicitly covered, the type-erasure post implies this pattern: erasing a type is encoding its operations; invoking through the erased interface is decoding back to behavior.

## The Deeper Pattern

Duality arises whenever there's a notion of *direction* that can be reversed:

- Data flow direction (forward/backward)
- Control flow direction (push/pull)
- Type hierarchy direction (covariant/contravariant)
- Logical implication direction (modus ponens/modus tollens)
- Arrow direction in diagrams

Each reversal is a duality. Each duality doubles your theorems, doubles your designs, and halves your blind spots.

The next time you build something, ask: what happens if I reverse the arrows? The answer might be exactly what you need.

---

## Further Reading

- Wadler, "Theorems for Free!" (1989) --- parametricity as a source of dual theorems
- Gibbons, "Calculating Functional Programs" --- unfolds as duals of folds
- Awodey, *Category Theory* --- the formal treatment of opposite categories
- Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" --- forward vs reverse mode
