---
author:
  name: "Alex Towell"
  email: "queelius@gmail.com"
  url: "https://metafunctor.com"
title: "Runtime Polymorphism Without Inheritance"
date: 2024-02-05
draft: false
tags:
  - C++
  - type-erasure
  - polymorphism
  - design-patterns
  - generic-programming
categories:
  - Computer Science
series: ["stepanov"]
series_weight: 11
math: true
description: "Sean Parent's type erasure technique provides value-semantic polymorphism without inheritance. Combined with Stepanov's algebraic thinking, we can type-erase entire algebraic structures."
---

*Sean Parent's technique for value-semantic polymorphism, and algebraic extensions*

## The Problem

You want a container that holds different types:

```cpp
std::vector<???> items;
items.push_back(42);
items.push_back(std::string("hello"));
items.push_back(MyCustomType{});
```

Traditional OOP says: make everything inherit from a base class. But that's intrusive---you can't add `int` to your hierarchy.

## The Solution: Type Erasure

Store the interface in an abstract class, but wrap any type that provides the needed operations:

```cpp
class any_regular {
    struct concept_t {
        virtual ~concept_t() = default;
        virtual std::unique_ptr<concept_t> clone() const = 0;
        virtual bool equals(concept_t const&) const = 0;
    };

    template<typename T>
    struct model_t : concept_t {
        T value;
        // Implement concept_t using T's operations
    };

    std::unique_ptr<concept_t> self_;
};
```

The magic: `model_t<int>`, `model_t<string>`, `model_t<MyType>` are all different types, but they all inherit from `concept_t`. We've moved the inheritance *inside* the wrapper.

## The Pattern

1. **concept_t**: Abstract base defining required operations
2. **model_t<T>**: Template that wraps any type T and implements concept_t
3. **Wrapper class**: Holds `unique_ptr<concept_t>`, provides value semantics

The wrapper looks like a value but can hold any type:

```cpp
any_regular a = 42;           // Stores model_t<int>
any_regular b = "hello"s;     // Stores model_t<string>
any_regular c = a;            // Deep copy via clone()
```

## Value Semantics

The key is `clone()`. Copying the wrapper deep-copies the held value:

```cpp
any_regular(any_regular const& other)
    : self_(other.self_ ? other.self_->clone() : nullptr) {}
```

This gives us value semantics: copies are independent, assignment replaces content, no shared state surprises.

## Beyond Regular: Algebraic Interfaces

Here's where Stepanov's insight applies. What if we type-erase *algebraic structure*?

### any_ring: Erasing Ring Operations

```cpp
class any_ring {
    struct concept_t {
        virtual std::unique_ptr<concept_t> clone() const = 0;
        virtual std::unique_ptr<concept_t> add(concept_t const&) const = 0;
        virtual std::unique_ptr<concept_t> multiply(concept_t const&) const = 0;
        virtual std::unique_ptr<concept_t> negate() const = 0;
        virtual std::unique_ptr<concept_t> zero() const = 0;
        virtual std::unique_ptr<concept_t> one() const = 0;
        virtual bool equals(concept_t const&) const = 0;
    };
    // ...
};
```

Now `any_ring` can hold integers, polynomials, matrices, or any ring---and you can do arithmetic on it at runtime:

```cpp
any_ring a = 42;
any_ring b = polynomial{1, 2, 3};  // Different types!
// a + b would throw (type mismatch), but:
any_ring c = a + any_ring(10);     // Works: 52
```

### any_vector_space: Linear Algebra Abstraction

A vector space over a field F has:
- Vector addition: v + w
- Scalar multiplication: alpha * v
- Zero vector: 0

```cpp
template<typename Scalar>
class any_vector {
    struct concept_t {
        virtual std::unique_ptr<concept_t> clone() const = 0;
        virtual std::unique_ptr<concept_t> add(concept_t const&) const = 0;
        virtual std::unique_ptr<concept_t> scale(Scalar) const = 0;
        virtual std::unique_ptr<concept_t> zero() const = 0;
        virtual Scalar inner_product(concept_t const&) const = 0;
    };
    // ...
};
```

This lets you write algorithms that work on *any* vector space:

```cpp
// Gram-Schmidt orthogonalization - works on any inner product space
template<typename Scalar>
std::vector<any_vector<Scalar>> gram_schmidt(
    std::vector<any_vector<Scalar>> const& basis
) {
    std::vector<any_vector<Scalar>> orthonormal;
    for (auto const& v : basis) {
        auto u = v;
        for (auto const& e : orthonormal) {
            // u = u - <u,e>*e
            u = u - e.scale(u.inner_product(e));
        }
        // Normalize u
        Scalar norm = std::sqrt(u.inner_product(u));
        orthonormal.push_back(u.scale(1.0 / norm));
    }
    return orthonormal;
}
```

## The Stepanov Connection

This is Stepanov's approach at the type level:
1. **Identify the algebraic structure** (ring, vector space, etc.)
2. **Define operations axiomatically** (not by inheritance)
3. **Let any type that provides the operations participate**

The concept/model pattern makes this work at runtime. It's the same philosophy as C++20 concepts, but with runtime dispatch instead of compile-time.

## Trade-offs

**Costs:**
- Heap allocation
- Virtual dispatch (~20ns overhead)
- Can't mix different concrete types in one operation

**Benefits:**
- Works with existing types (no inheritance required)
- Value semantics
- Algorithm code is simple and generic
- Easy to add new types

## When to Use This

- Plugin systems where concrete types aren't known at compile time
- Numerical libraries that support multiple matrix representations
- Document systems with heterogeneous content
- Undo/redo stacks with different action types

## Further Reading

- Sean Parent, "Inheritance Is The Base Class of Evil" (GoingNative 2013)
- Sean Parent, "Better Code: Runtime Polymorphism" (NDC 2017)
- Stepanov & McJones, *Elements of Programming*, Chapter 1
