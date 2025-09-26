# Stepanov Best Practices Guide

## Philosophy First

Before writing any code with Stepanov, understand: this is not just another library. It is a different way of thinking about programming. These practices emerge from mathematical principles, not arbitrary conventions.

## Fundamental Principles

### 1. Think in Terms of Concepts, Not Types

**Wrong:**
```cpp
template<typename T>
T multiply_by_two(T value) {
    return value * 2;  // Assumes T supports * with int
}
```

**Right:**
```cpp
template<typename T>
    requires stepanov::Multiplicative<T>
T twice(T value) {
    return value + value;  // Use the fundamental operation
}
```

**Best:**
```cpp
template<typename T>
    requires stepanov::Regular<T> && requires(T x) { x + x; }
constexpr T twice(T value) noexcept(noexcept(value + value)) {
    return value + value;
}
```

### 2. Prefer Compile-Time to Runtime

**Wrong:**
```cpp
auto compute(int algorithm_type, auto data) {
    if (algorithm_type == 1)
        return algorithm1(data);
    else
        return algorithm2(data);
}
```

**Right:**
```cpp
template<int AlgorithmType>
auto compute(auto data) {
    if constexpr (AlgorithmType == 1)
        return algorithm1(data);
    else
        return algorithm2(data);
}
```

**Best:**
```cpp
template<typename Algorithm>
auto compute(auto data) {
    return Algorithm{}(data);  // Strategy pattern at compile time
}
```

### 3. Composition Over Inheritance

**Wrong:**
```cpp
class SortedContainer : public Container {
    void insert(T value) override {
        Container::insert(value);
        sort();
    }
};
```

**Right:**
```cpp
template<typename Container>
class sorted_adaptor {
    Container c;
    void insert(auto value) {
        c.insert(std::upper_bound(c.begin(), c.end(), value), value);
    }
};
```

**Best:**
```cpp
template<typename Container, typename Compare = std::less<>>
class sorted_view {
    Container* c;
    Compare cmp;
public:
    auto insert(auto&& value) {
        return c->insert(
            stepanov::partition_point(c->begin(), c->end(),
                [&](auto& x) { return cmp(x, value); }),
            std::forward<decltype(value)>(value)
        );
    }
};
```

## Algorithm Design

### 4. Express Algorithms in Terms of Fundamental Operations

**Example: Power Algorithm**
```cpp
// Don't hardcode operations
template<typename T>
T power_bad(T base, int n) {
    T result = 1;  // Assumes multiplicative identity is 1
    while (n > 0) {
        result *= base;  // Assumes * operator
        --n;
    }
    return result;
}

// Do express in terms of operations
template<typename T, typename N, typename Op>
T power(T base, N n, Op op, T identity) {
    if (n == 0) return identity;

    while ((n & 1) == 0) {
        base = op(base, base);
        n >>= 1;
    }

    T result = base;
    n >>= 1;

    while (n != 0) {
        base = op(base, base);
        if ((n & 1) != 0)
            result = op(result, base);
        n >>= 1;
    }

    return result;
}

// Best: with concepts and optimal operations
template<typename T, typename N>
    requires stepanov::Regular<T> &&
             stepanov::Integer<N> &&
             stepanov::Multiplicative<T>
constexpr T power(T base, N n) {
    return stepanov::power(base, n,
        [](T x, T y) { return x * y; }, T{1});
}
```

### 5. Make Invalid States Unrepresentable

**Wrong:**
```cpp
class optional {
    bool has_value;
    T value;  // Uninitialized if !has_value
};
```

**Right:**
```cpp
template<typename T>
class optional {
    alignas(T) unsigned char storage[sizeof(T)];
    bool has_value = false;

public:
    T& value() {
        if (!has_value) throw bad_optional_access{};
        return *std::launder(reinterpret_cast<T*>(storage));
    }
};
```

**Best:**
```cpp
template<typename T>
using optional = stepanov::variant<stepanov::monostate, T>;

// Or with monadic interface
template<typename T>
class maybe {
    stepanov::variant<nothing, just<T>> data;
public:
    template<typename F>
    auto bind(F&& f) const {
        return std::visit(
            stepanov::overloaded{
                [](nothing) { return maybe<decltype(f(T{}))>{}; },
                [&](const just<T>& j) { return f(j.value); }
            }, data);
    }
};
```

## Performance Considerations

### 6. Understand the Cost Model

**Lazy Evaluation Cost:**
```cpp
// Understand when laziness pays off
auto lazy_range = stepanov::lazy_sequence([n = 0]() mutable {
    return n++;
});

// Good: Only compute what's needed
auto first_10_primes = lazy_range
    | stepanov::filter(is_prime)
    | stepanov::take(10);  // Stops after 10 primes found

// Bad: Computing everything defeats laziness
auto all_primes = lazy_range
    | stepanov::filter(is_prime)
    | stepanov::to_vector();  // Infinite loop!
```

**Persistent Structure Cost:**
```cpp
// Understand persistence overhead
stepanov::persistent_vector<int> v1;
auto v2 = v1.push_back(42);  // O(log n) - path copying

// Good: When you need history
auto undo_buffer = stepanov::persistent_stack<State>();

// Bad: When you don't need persistence
for (int i = 0; i < 1000000; ++i) {
    v = v.push_back(i);  // O(n log n) total
}
```

### 7. Cache-Conscious Programming

**Optimize for Cache:**
```cpp
// Bad: Random access pattern
for (auto& key : keys) {
    result += map[key];  // Cache misses
}

// Good: Sequential access
stepanov::btree<Key, Value, 64> map;  // B-tree with cache-line nodes
for (auto& [k, v] : map) {  // Sequential traversal
    result += v;
}

// Best: Cache-oblivious
auto result = stepanov::cache_oblivious::reduce(
    data.begin(), data.end(), 0,
    [](auto a, auto b) { return a + b; }
);
```

### 8. Compile-Time Computation

**Move Work to Compile Time:**
```cpp
// Runtime computation
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

// Compile-time computation
template<int N>
constexpr int fibonacci = fibonacci<N-1> + fibonacci<N-2>;

template<> constexpr int fibonacci<0> = 0;
template<> constexpr int fibonacci<1> = 1;

// Best: With memoization
template<int N>
struct fib {
    static constexpr int value = fib<N-1>::value + fib<N-2>::value;
};

template<> struct fib<0> { static constexpr int value = 0; };
template<> struct fib<1> { static constexpr int value = 1; };

// Or using Stepanov's compile-time facilities
constexpr auto fib_10 = stepanov::meta::fibonacci<10>;
```

## Error Handling Patterns

### 9. Monadic Error Handling

**Don't Use Exceptions for Control Flow:**
```cpp
// Bad
try {
    auto result = risky_operation();
    process(result);
} catch (const error& e) {
    handle_error(e);
}

// Good: Explicit error handling
auto result = risky_operation();  // Returns result<T, Error>
if (result) {
    process(result.value());
} else {
    handle_error(result.error());
}

// Best: Monadic composition
risky_operation()
    .and_then(process)
    .or_else(handle_error)
    .value_or(default_value);
```

### 10. Compile-Time Error Detection

**Detect Errors at Compile Time:**
```cpp
// Runtime error
template<typename T>
T divide(T a, T b) {
    if (b == 0) throw division_by_zero{};
    return a / b;
}

// Compile-time safety
template<typename T, T B>
    requires (B != 0)
constexpr T divide(T a, stepanov::constant<T, B>) {
    return a / B;
}

// Best: Type-safe dimensions
using meters = stepanov::dimension<1, 0, 0>;
using seconds = stepanov::dimension<0, 1, 0>;
using velocity = stepanov::dimension<1, -1, 0>;

auto v = stepanov::quantity<double, velocity>(10.0);  // 10 m/s
auto t = stepanov::quantity<double, seconds>(5.0);    // 5 s
auto d = v * t;  // Type is quantity<double, meters>
```

## Concurrency Patterns

### 11. Lock-Free When Possible

```cpp
// Bad: Mutex for simple counter
std::mutex m;
int counter = 0;

void increment() {
    std::lock_guard lock(m);
    ++counter;
}

// Good: Atomic operations
std::atomic<int> counter{0};

void increment() {
    counter.fetch_add(1, std::memory_order_relaxed);
}

// Best: Lock-free data structures
stepanov::lock_free_queue<Task> tasks;

void producer() {
    tasks.push(create_task());
}

void consumer() {
    if (auto task = tasks.try_pop()) {
        process(*task);
    }
}
```

### 12. Software Transactional Memory for Complex State

```cpp
// Complex concurrent updates
struct Account {
    stepanov::stm_var<double> balance;
    stepanov::stm_var<std::vector<Transaction>> history;
};

void transfer(Account& from, Account& to, double amount) {
    stepanov::atomic_transaction([&] {
        auto from_bal = from.balance.read();
        if (from_bal < amount) {
            stepanov::retry();  // Wait for balance
        }

        from.balance.write(from_bal - amount);
        to.balance.write(to.balance.read() + amount);

        auto timestamp = std::chrono::steady_clock::now();
        from.history.modify([&](auto& h) {
            h.push_back({-amount, timestamp});
        });
        to.history.modify([&](auto& h) {
            h.push_back({amount, timestamp});
        });
    });
}
```

## Testing Strategies

### 13. Property-Based Testing

```cpp
// Don't just test examples
TEST(sort, specific_cases) {
    EXPECT_EQ(sort({3,1,2}), {1,2,3});
    EXPECT_EQ(sort({1}), {1});
    // Missing edge cases...
}

// Do test properties
PROPERTY_TEST(sort, preserves_elements) {
    stepanov::property([](std::vector<int> v) {
        auto sorted = stepanov::sort(v);
        std::sort(v.begin(), v.end());
        return sorted == v;
    });
}

PROPERTY_TEST(sort, idempotent) {
    stepanov::property([](std::vector<int> v) {
        auto once = stepanov::sort(v);
        auto twice = stepanov::sort(once);
        return once == twice;
    });
}

// Best: Verify mathematical properties
PROPERTY_TEST(power, mathematical_laws) {
    stepanov::property([](int base, int m, int n) {
        // Power laws
        return stepanov::power(base, m + n) ==
               stepanov::power(base, m) * stepanov::power(base, n);
    });
}
```

### 14. Axiom Verification

```cpp
// Verify type models concept correctly
template<typename T>
void verify_group() {
    using stepanov::verify;

    // Identity element
    verify::identity<T>([](T a, T e) {
        return a * e == a && e * a == a;
    });

    // Associativity
    verify::associativity<T>([](T a, T b, T c) {
        return (a * b) * c == a * (b * c);
    });

    // Inverse
    verify::inverse<T>([](T a) {
        T inv = inverse(a);
        return a * inv == identity<T>() && inv * a == identity<T>();
    });
}

// Use in tests
TEST(quaternion, is_group) {
    verify_group<stepanov::quaternion<double>>();
}
```

## Compression and Learning

### 15. Compression for Intelligence

```cpp
// Pattern detection through compression
bool has_pattern(const std::string& data) {
    auto compressed = stepanov::compress(data);
    double ratio = stepanov::compression_ratio(data, compressed);
    return ratio > 0.5;  // High compression = pattern exists
}

// Classification without explicit features
class universal_classifier {
    std::map<std::string, std::string> training_data;

public:
    void train(const std::string& example, const std::string& label) {
        training_data[label] += example;
    }

    std::string classify(const std::string& input) {
        std::string best_label;
        double best_distance = std::numeric_limits<double>::max();

        for (const auto& [label, data] : training_data) {
            double dist = stepanov::normalized_compression_distance(
                input, data
            );
            if (dist < best_distance) {
                best_distance = dist;
                best_label = label;
            }
        }

        return best_label;
    }
};
```

## Advanced Patterns

### 16. Algebraic Effects for Control Flow

```cpp
// Complex control flow with effects
using namespace stepanov::effects;

template<typename T>
auto safe_divide(T a, T b) {
    return b != 0
        ? pure(a / b)
        : throw_error(division_by_zero{});
}

auto computation =
    pure(10)
    >>= [](int x) { return safe_divide(x, 2); }
    >>= [](int x) { return safe_divide(x, 0); }  // Will fail
    >>= [](int x) { return pure(x + 1); };       // Never reached

auto result = handle_error(computation,
    [](auto error) { return pure(-1); });  // Returns -1 on error
```

### 17. Lazy Infinite Computations

```cpp
// Working with infinity
auto naturals = stepanov::lazy_sequence([]() {
    static int n = 0;
    return n++;
});

// Infinite prime generator
auto primes = naturals
    | stepanov::filter(stepanov::is_prime);

// Compute as needed
for (auto p : primes | stepanov::take(100)) {
    process_prime(p);
}

// Infinite recursive structures
struct infinite_tree {
    int value;
    stepanov::lazy<infinite_tree> left;
    stepanov::lazy<infinite_tree> right;

    infinite_tree(int v) : value(v),
        left([v] { return infinite_tree(2*v); }),
        right([v] { return infinite_tree(2*v + 1); }) {}
};
```

### 18. Differentiable Programming

```cpp
// Automatic optimization
template<typename F>
auto optimize(F f, double initial, double learning_rate = 0.01) {
    stepanov::dual<double> x(initial, 1.0);

    for (int i = 0; i < 1000; ++i) {
        auto y = f(x);
        x = stepanov::dual<double>(
            x.value() - learning_rate * y.derivative(),
            1.0
        );

        if (std::abs(y.derivative()) < 1e-6) break;
    }

    return x.value();
}

// Use with any differentiable function
auto minimum = optimize([](auto x) {
    return x*x*x*x - 2*x*x + x;  // Find minimum
}, 0.0);
```

## Integration Patterns

### 19. Composing Modules

```cpp
// Compression + Lazy + Parallel
auto compressed_computation =
    stepanov::lazy_sequence(data_generator)
    | stepanov::parallel_map([](auto chunk) {
        return stepanov::compress(chunk);
      })
    | stepanov::lazy_transform([](auto compressed) {
        return analyze(compressed);
      });

// Effects + STM + Persistence
auto transactional_history =
    stepanov::atomic_transaction([&] {
        auto current = state.read();
        auto new_state = transform(current);

        history.modify([&](auto& h) {
            h = h.push_back(current);  // Persistent vector
        });

        state.write(new_state);
        return stepanov::pure(new_state);
    });
```

### 20. Building Domain-Specific Languages

```cpp
// Create DSL using Stepanov primitives
namespace trading {
    using price = stepanov::quantity<decimal, dollars>;
    using volume = stepanov::quantity<int, shares>;

    struct order {
        stepanov::variant<buy, sell> side;
        price limit;
        volume size;
        stepanov::lazy<bool> filled;
    };

    auto execute = stepanov::monadic()
        .bind([](order o) { return validate(o); })
        .bind([](order o) { return check_balance(o); })
        .bind([](order o) { return place_order(o); })
        .recover([](error e) { return cancel_order(e); });
}
```

## Common Pitfalls to Avoid

### Don't Mix Paradigms Unnecessarily
- Use functional style consistently within a module
- Don't mix lazy and eager evaluation without clear boundaries
- Keep effect systems isolated from pure code

### Don't Overuse Template Metaprogramming
- Prefer concepts over SFINAE
- Use if constexpr over template specialization when possible
- Keep compile times reasonable

### Don't Ignore Mathematical Properties
- If a type models a mathematical structure, verify the axioms
- Document mathematical requirements clearly
- Use concepts to enforce properties

### Don't Sacrifice Clarity for Cleverness
- Elegant doesn't mean obscure
- Mathematical doesn't mean incomprehensible
- Generic doesn't mean unreadable

## Summary Checklist

Before committing code using Stepanov:

- [ ] Are concepts used instead of raw templates?
- [ ] Is the algorithm expressed in terms of fundamental operations?
- [ ] Are mathematical properties documented and verified?
- [ ] Is error handling explicit and composable?
- [ ] Are compile-time computations preferred where possible?
- [ ] Is the code cache-conscious where performance matters?
- [ ] Are concurrent operations lock-free where feasible?
- [ ] Are effects properly isolated and composed?
- [ ] Have properties been tested, not just examples?
- [ ] Is the code elegant, correct, and understandable?

## Final Wisdom

> "Make it work, make it right, make it beautiful, make it fast—in that order."

In Stepanov, beautiful and fast often coincide, because mathematical elegance leads to optimal algorithms. Never compromise on correctness. Never settle for ugly. And always remember: we're not just writing code, we're discovering the mathematics of computation.

---

*Use Stepanov not because it's easy, but because it's right.*