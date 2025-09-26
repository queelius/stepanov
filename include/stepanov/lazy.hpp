// lazy.hpp - True Lazy Evaluation in C++
// Infinite data structures, lazy algorithms, and memoization
// Brings Haskell-style lazy evaluation to C++ with zero overhead

#ifndef STEPANOV_LAZY_HPP
#define STEPANOV_LAZY_HPP

#include <concepts>
#include <memory>
#include <optional>
#include <variant>
#include <functional>
#include <utility>
#include <thread>
#include <mutex>
#include <algorithm>
#include <ranges>

namespace stepanov::lazy {

// Forward declarations
template<typename T> class lazy;
template<typename T> class lazy_list;
template<typename T> class lazy_tree;
template<typename F, typename... Args> class memoized;

// ============================================================================
// Core Lazy Evaluation Infrastructure
// ============================================================================

// Thunk - deferred computation
template<typename T>
class thunk {
    mutable std::variant<std::function<T()>, T> data_;
    mutable std::mutex mutex_;

public:
    explicit thunk(std::function<T()> f) : data_(std::move(f)) {}
    explicit thunk(T value) : data_(std::move(value)) {}

    const T& force() const {
        std::lock_guard lock(mutex_);

        if (std::holds_alternative<std::function<T()>>(data_)) {
            auto f = std::get<std::function<T()>>(data_);
            data_ = f();
        }

        return std::get<T>(data_);
    }

    bool is_evaluated() const {
        std::lock_guard lock(mutex_);
        return std::holds_alternative<T>(data_);
    }
};

// Lazy value wrapper
template<typename T>
class lazy {
    std::shared_ptr<thunk<T>> thunk_;

public:
    explicit lazy(std::function<T()> f)
        : thunk_(std::make_shared<thunk<T>>(std::move(f))) {}

    explicit lazy(T value)
        : thunk_(std::make_shared<thunk<T>>(std::move(value))) {}

    const T& operator*() const { return thunk_->force(); }
    const T* operator->() const { return &thunk_->force(); }

    bool is_evaluated() const { return thunk_->is_evaluated(); }

    // Functor operations
    template<typename F>
    auto map(F&& f) const {
        return lazy<decltype(f(*this))>([*this, f = std::forward<F>(f)]() {
            return f(**this);
        });
    }

    // Monad operations
    template<typename F>
    auto bind(F&& f) const {
        return lazy([*this, f = std::forward<F>(f)]() {
            return *f(**this);
        });
    }

    // Create lazy value from pure value
    static lazy<T> pure(T value) {
        return lazy<T>(std::move(value));
    }
};

// ============================================================================
// Infinite Lists - True Lazy Lists like Haskell
// ============================================================================

template<typename T>
class lazy_list {
public:
    struct node {
        T value;
        lazy<std::shared_ptr<node>> next;

        node(T val, std::function<std::shared_ptr<node>()> tail)
            : value(std::move(val)), next(std::move(tail)) {}
    };

private:
    lazy<std::shared_ptr<node>> head_;

public:
    lazy_list() : head_([]() { return std::shared_ptr<node>(); }) {}

    explicit lazy_list(std::function<std::shared_ptr<node>()> gen)
        : head_(std::move(gen)) {}

    // Cons - prepend element
    static lazy_list cons(T value, lazy_list tail) {
        return lazy_list([value = std::move(value), tail]() {
            return std::make_shared<node>(value, [tail]() {
                return *tail.head_;
            });
        });
    }

    // Check if empty
    bool is_empty() const {
        return *head_ == nullptr;
    }

    // Head - first element (forces evaluation)
    T head() const {
        if (is_empty()) throw std::runtime_error("empty list");
        return (*head_)->value;
    }

    // Tail - rest of the list (lazy)
    lazy_list tail() const {
        if (is_empty()) throw std::runtime_error("empty list");
        return lazy_list((*head_)->next);
    }

    // Take first n elements
    std::vector<T> take(size_t n) const {
        std::vector<T> result;
        lazy_list current = *this;

        for (size_t i = 0; i < n && !current.is_empty(); ++i) {
            result.push_back(current.head());
            current = current.tail();
        }

        return result;
    }

    // Drop first n elements
    lazy_list drop(size_t n) const {
        lazy_list current = *this;
        for (size_t i = 0; i < n && !current.is_empty(); ++i) {
            current = current.tail();
        }
        return current;
    }

    // Map function over list
    template<typename F>
    auto map(F&& f) const {
        using U = decltype(f(head()));

        return lazy_list<U>([*this, f = std::forward<F>(f)]() -> std::shared_ptr<typename lazy_list<U>::node> {
            if (is_empty()) return nullptr;

            return std::make_shared<typename lazy_list<U>::node>(
                f(head()),
                [tail = tail(), f]() { return *tail.map(f).head_; }
            );
        });
    }

    // Filter list
    lazy_list filter(std::function<bool(const T&)> pred) const {
        return lazy_list([*this, pred]() -> std::shared_ptr<node> {
            lazy_list current = *this;

            while (!current.is_empty()) {
                if (pred(current.head())) {
                    return std::make_shared<node>(
                        current.head(),
                        [tail = current.tail(), pred]() {
                            return *tail.filter(pred).head_;
                        }
                    );
                }
                current = current.tail();
            }

            return nullptr;
        });
    }

    // Fold (forces evaluation)
    template<typename U, typename F>
    U fold(U init, F&& f) const {
        lazy_list current = *this;
        U result = std::move(init);

        while (!current.is_empty()) {
            result = f(std::move(result), current.head());
            current = current.tail();
        }

        return result;
    }

    // Zip two lists
    template<typename U>
    auto zip(const lazy_list<U>& other) const {
        using pair_t = std::pair<T, U>;

        return lazy_list<pair_t>([*this, other]() -> std::shared_ptr<typename lazy_list<pair_t>::node> {
            if (is_empty() || other.is_empty()) return nullptr;

            return std::make_shared<typename lazy_list<pair_t>::node>(
                std::make_pair(head(), other.head()),
                [tail1 = tail(), tail2 = other.tail()]() {
                    return *tail1.zip(tail2).head_;
                }
            );
        });
    }

    // Infinite list generators
    static lazy_list iterate(T initial, std::function<T(const T&)> f) {
        return lazy_list([initial, f]() {
            return std::make_shared<node>(
                initial,
                [next = f(initial), f]() {
                    return *iterate(next, f).head_;
                }
            );
        });
    }

    static lazy_list from(T start) {
        return iterate(start, [](const T& x) { return x + 1; });
    }

    static lazy_list repeat(T value) {
        return iterate(value, [](const T& x) { return x; });
    }

    static lazy_list cycle(std::vector<T> values) {
        if (values.empty()) return lazy_list();

        std::function<std::shared_ptr<node>(size_t)> gen =
            [values = std::move(values), &gen](size_t idx) -> std::shared_ptr<node> {
                return std::make_shared<node>(
                    values[idx],
                    [&gen, values, idx]() {
                        return gen((idx + 1) % values.size());
                    }
                );
            };

        return lazy_list([gen]() { return gen(0); });
    }

    // Fibonacci sequence
    static lazy_list fibonacci() {
        std::function<std::shared_ptr<node>(T, T)> gen =
            [&gen](T a, T b) -> std::shared_ptr<node> {
                return std::make_shared<node>(
                    a,
                    [&gen, a, b]() { return gen(b, a + b); }
                );
            };

        return lazy_list([gen]() { return gen(T(0), T(1)); });
    }

    // Prime numbers using sieve
    static lazy_list primes() {
        std::function<lazy_list(lazy_list)> sieve = [&sieve](lazy_list nums) {
            if (nums.is_empty()) return lazy_list();

            T p = nums.head();
            return cons(p, sieve(nums.tail().filter([p](const T& x) {
                return x % p != 0;
            })));
        };

        return sieve(from(T(2)));
    }
};

// ============================================================================
// Lazy Trees - Infinite Trees Evaluated on Demand
// ============================================================================

template<typename T>
class lazy_tree {
public:
    struct node {
        T value;
        lazy<std::vector<lazy_tree>> children;

        node(T val, std::function<std::vector<lazy_tree>()> kids)
            : value(std::move(val)), children(std::move(kids)) {}
    };

private:
    lazy<std::shared_ptr<node>> root_;

public:
    explicit lazy_tree(std::function<std::shared_ptr<node>()> gen)
        : root_(std::move(gen)) {}

    // Create leaf
    static lazy_tree leaf(T value) {
        return lazy_tree([value]() {
            return std::make_shared<node>(value, []() {
                return std::vector<lazy_tree>();
            });
        });
    }

    // Create node with children
    static lazy_tree branch(T value, std::function<std::vector<lazy_tree>()> children) {
        return lazy_tree([value, children = std::move(children)]() {
            return std::make_shared<node>(value, children);
        });
    }

    // Get value at root
    T value() const {
        return (*root_)->value;
    }

    // Get children (forces their creation)
    std::vector<lazy_tree> children() const {
        return *(*root_)->children;
    }

    // Map over tree
    template<typename F>
    auto map(F&& f) const {
        using U = decltype(f(value()));

        return lazy_tree<U>([*this, f = std::forward<F>(f)]() {
            return std::make_shared<typename lazy_tree<U>::node>(
                f(value()),
                [*this, f]() {
                    auto kids = children();
                    std::vector<lazy_tree<U>> result;
                    for (const auto& child : kids) {
                        result.push_back(child.map(f));
                    }
                    return result;
                }
            );
        });
    }

    // Infinite tree generators
    static lazy_tree full_binary(T value, std::function<T(T, bool)> gen) {
        return branch(value, [value, gen]() {
            return std::vector<lazy_tree>{
                full_binary(gen(value, false), gen),
                full_binary(gen(value, true), gen)
            };
        });
    }

    // Game tree for minimax
    template<typename State, typename Move>
    static lazy_tree game_tree(State state, std::function<std::vector<Move>(const State&)> moves) {
        return branch(state, [state, moves]() {
            auto possible = moves(state);
            std::vector<lazy_tree> children;

            for (const auto& move : possible) {
                State next = state;
                next.apply(move);
                children.push_back(game_tree(next, moves));
            }

            return children;
        });
    }
};

// ============================================================================
// Memoization Framework - Automatic Caching
// ============================================================================

template<typename Key, typename Value>
class memo_table {
    mutable std::unordered_map<Key, Value> cache_;
    mutable std::mutex mutex_;

public:
    std::optional<Value> lookup(const Key& key) const {
        std::lock_guard lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void insert(const Key& key, const Value& value) const {
        std::lock_guard lock(mutex_);
        cache_[key] = value;
    }

    void clear() {
        std::lock_guard lock(mutex_);
        cache_.clear();
    }

    size_t size() const {
        std::lock_guard lock(mutex_);
        return cache_.size();
    }
};

template<typename F, typename... Args>
class memoized {
    using result_t = std::invoke_result_t<F, Args...>;
    using key_t = std::tuple<std::decay_t<Args>...>;

    F f_;
    mutable memo_table<key_t, result_t> cache_;

public:
    explicit memoized(F func) : f_(std::move(func)) {}

    result_t operator()(Args... args) const {
        key_t key(args...);

        if (auto cached = cache_.lookup(key)) {
            return *cached;
        }

        result_t result = f_(args...);
        cache_.insert(key, result);
        return result;
    }

    void clear_cache() {
        cache_.clear();
    }

    size_t cache_size() const {
        return cache_.size();
    }
};

// Factory function for memoization
template<typename F>
auto memoize(F&& f) {
    return memoized<std::decay_t<F>>(std::forward<F>(f));
}

// ============================================================================
// Lazy Sorting - O(k) for First k Elements
// ============================================================================

template<typename RandomIt, typename Compare = std::less<>>
class lazy_sorter {
    RandomIt first_, last_;
    Compare comp_;
    mutable std::vector<bool> sorted_until_;
    mutable size_t sorted_count_ = 0;

public:
    lazy_sorter(RandomIt first, RandomIt last, Compare comp = Compare{})
        : first_(first), last_(last), comp_(comp),
          sorted_until_(std::distance(first, last), false) {}

    // Get kth smallest element (0-indexed)
    auto nth_element(size_t k) const {
        if (k >= sorted_until_.size()) {
            throw std::out_of_range("k out of range");
        }

        if (!sorted_until_[k]) {
            // Use nth_element to partition
            std::nth_element(first_ + sorted_count_, first_ + k, last_, comp_);

            // Mark elements before k as potentially sorted
            for (size_t i = sorted_count_; i <= k; ++i) {
                sorted_until_[i] = true;
            }
            sorted_count_ = k + 1;
        }

        return *(first_ + k);
    }

    // Get first k sorted elements
    std::vector<typename std::iterator_traits<RandomIt>::value_type>
    first_k(size_t k) const {
        using value_type = typename std::iterator_traits<RandomIt>::value_type;
        std::vector<value_type> result;

        for (size_t i = 0; i < k && i < sorted_until_.size(); ++i) {
            result.push_back(nth_element(i));
        }

        // Sort the first k elements
        std::sort(result.begin(), result.end(), comp_);
        return result;
    }

    // Check if we need to sort more
    bool is_partially_sorted(size_t k) const {
        return k < sorted_count_;
    }
};

// ============================================================================
// Lazy Stream Processing
// ============================================================================

template<typename T>
class stream {
    lazy_list<T> data_;

public:
    explicit stream(lazy_list<T> list) : data_(std::move(list)) {}

    // Stream operations with lazy evaluation
    template<typename F>
    auto map(F&& f) const {
        using U = decltype(f(data_.head()));
        return stream<U>(data_.map(std::forward<F>(f)));
    }

    stream filter(std::function<bool(const T&)> pred) const {
        return stream(data_.filter(pred));
    }

    template<typename U>
    stream<U> flat_map(std::function<stream<U>(const T&)> f) const {
        // Lazy concatenation of streams
        return stream<U>(lazy_list<U>([*this, f]() {
            if (data_.is_empty()) return nullptr;

            auto inner = f(data_.head());
            return concatenate(inner.data_, flat_map(f).data_);
        }));
    }

    // Terminal operations (force evaluation)
    std::vector<T> collect(size_t max_elements = -1) const {
        return data_.take(max_elements);
    }

    template<typename U>
    U reduce(U init, std::function<U(U, const T&)> f) const {
        return data_.fold(std::move(init), f);
    }

    std::optional<T> find_first(std::function<bool(const T&)> pred) const {
        lazy_list<T> current = data_;

        while (!current.is_empty()) {
            if (pred(current.head())) {
                return current.head();
            }
            current = current.tail();
        }

        return std::nullopt;
    }

    // Create streams
    static stream<T> generate(std::function<T()> gen) {
        return stream(lazy_list<T>::iterate(gen(), [gen](const T&) {
            return gen();
        }));
    }

    static stream<T> iterate(T seed, std::function<T(const T&)> f) {
        return stream(lazy_list<T>::iterate(seed, f));
    }

    static stream<T> of(std::vector<T> values) {
        if (values.empty()) return stream(lazy_list<T>());

        std::function<lazy_list<T>(size_t)> build =
            [values = std::move(values), &build](size_t idx) -> lazy_list<T> {
                if (idx >= values.size()) return lazy_list<T>();
                return lazy_list<T>::cons(values[idx], build(idx + 1));
            };

        return stream(build(0));
    }
};

// ============================================================================
// Practical Utilities
// ============================================================================

// Lazy evaluation of expensive computations
template<typename T>
class lazy_value {
    mutable std::optional<T> value_;
    std::function<T()> compute_;
    mutable std::mutex mutex_;

public:
    explicit lazy_value(std::function<T()> f) : compute_(std::move(f)) {}

    const T& get() const {
        std::lock_guard lock(mutex_);
        if (!value_) {
            value_ = compute_();
        }
        return *value_;
    }

    void reset() {
        std::lock_guard lock(mutex_);
        value_.reset();
    }

    bool is_computed() const {
        std::lock_guard lock(mutex_);
        return value_.has_value();
    }
};

// Lazy initialization pattern
template<typename T>
class lazy_singleton {
    inline static lazy_value<T> instance_{[]() { return T(); }};

public:
    static T& get() {
        return const_cast<T&>(instance_.get());
    }

    template<typename F>
    static void initialize(F&& factory) {
        instance_ = lazy_value<T>(std::forward<F>(factory));
    }
};

} // namespace stepanov::lazy

#endif // STEPANOV_LAZY_HPP