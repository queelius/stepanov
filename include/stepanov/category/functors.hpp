// functors.hpp
// Category theory constructs: Functors, Monads, Applicatives, and more
// Brings functional programming elegance to C++

#pragma once

#include <type_traits>
#include <functional>
#include <optional>
#include <variant>
#include <vector>
#include <memory>
#include <concepts>

namespace stepanov::category {

// Functor type class
template<template<typename> class F>
struct functor_traits {
    template<typename A, typename B>
    using fmap_t = F<B>(const F<A>&, std::function<B(A)>);
};

// Functor concept
template<template<typename> class F>
concept Functor = requires {
    typename functor_traits<F>::template fmap_t<int, int>;
};

// Maybe monad (similar to std::optional but with monadic interface)
template<typename T>
class maybe {
private:
    std::optional<T> value_;

public:
    // Constructors
    maybe() = default;
    maybe(const T& val) : value_(val) {}
    maybe(T&& val) : value_(std::move(val)) {}
    maybe(std::nullopt_t) : value_(std::nullopt) {}

    // Monadic return (pure/unit)
    static maybe pure(const T& val) {
        return maybe(val);
    }

    static maybe nothing() {
        return maybe(std::nullopt);
    }

    // Functor map
    template<typename F>
    auto fmap(F&& f) const -> maybe<decltype(f(std::declval<T>()))> {
        using U = decltype(f(std::declval<T>()));
        if (value_) {
            return maybe<U>(f(*value_));
        }
        return maybe<U>::nothing();
    }

    // Applicative apply
    template<typename U>
    auto apply(const maybe<std::function<U(T)>>& mf) const -> maybe<U> {
        if (mf.value_ && value_) {
            return maybe<U>((*mf.value_)(*value_));
        }
        return maybe<U>::nothing();
    }

    // Monadic bind (>>=)
    template<typename F>
    auto bind(F&& f) const -> decltype(f(std::declval<T>())) {
        using Result = decltype(f(std::declval<T>()));
        if (value_) {
            return f(*value_);
        }
        return Result::nothing();
    }

    // Kleisli composition (>=>)
    template<typename F, typename G>
    static auto kleisli(F&& f, G&& g) {
        return [f = std::forward<F>(f), g = std::forward<G>(g)](const auto& x) {
            return f(x).bind(g);
        };
    }

    // Monadic operations
    bool has_value() const { return value_.has_value(); }
    const T& value() const { return *value_; }
    T& value() { return *value_; }
    T value_or(const T& default_val) const { return value_.value_or(default_val); }

    // Pattern matching helpers
    template<typename F, typename G>
    auto match(F&& just_case, G&& nothing_case) const {
        if (value_) {
            return just_case(*value_);
        } else {
            return nothing_case();
        }
    }

    // Equality
    bool operator==(const maybe& other) const { return value_ == other.value_; }
};

// Either monad (for error handling)
template<typename L, typename R>
class either {
private:
    std::variant<L, R> value_;

public:
    // Constructors
    static either left(const L& l) {
        either e;
        e.value_ = l;
        return e;
    }

    static either right(const R& r) {
        either e;
        e.value_ = r;
        return e;
    }

    // Check which side
    bool is_left() const { return std::holds_alternative<L>(value_); }
    bool is_right() const { return std::holds_alternative<R>(value_); }

    // Get values
    const L& get_left() const { return std::get<L>(value_); }
    const R& get_right() const { return std::get<R>(value_); }

    // Functor map (only on Right)
    template<typename F>
    auto fmap(F&& f) const -> either<L, decltype(f(std::declval<R>()))> {
        using U = decltype(f(std::declval<R>()));
        if (is_right()) {
            return either<L, U>::right(f(get_right()));
        }
        return either<L, U>::left(get_left());
    }

    // Monadic bind (only on Right)
    template<typename F>
    auto bind(F&& f) const -> decltype(f(std::declval<R>())) {
        using Result = decltype(f(std::declval<R>()));
        if (is_right()) {
            return f(get_right());
        }
        return Result::left(get_left());
    }

    // Bimap (map both sides)
    template<typename F, typename G>
    auto bimap(F&& f, G&& g) const -> either<decltype(f(std::declval<L>())),
                                             decltype(g(std::declval<R>()))> {
        using NewL = decltype(f(std::declval<L>()));
        using NewR = decltype(g(std::declval<R>()));

        if (is_left()) {
            return either<NewL, NewR>::left(f(get_left()));
        }
        return either<NewL, NewR>::right(g(get_right()));
    }

    // Pattern matching
    template<typename F, typename G>
    auto match(F&& left_case, G&& right_case) const {
        if (is_left()) {
            return left_case(get_left());
        }
        return right_case(get_right());
    }
};

// List monad
template<typename T>
class list {
private:
    std::vector<T> items_;

public:
    // Constructors
    list() = default;
    list(std::initializer_list<T> init) : items_(init) {}
    explicit list(const std::vector<T>& vec) : items_(vec) {}

    // Monadic return
    static list pure(const T& val) {
        return list{val};
    }

    // Functor map
    template<typename F>
    auto fmap(F&& f) const -> list<decltype(f(std::declval<T>()))> {
        using U = decltype(f(std::declval<T>()));
        list<U> result;
        for (const auto& item : items_) {
            result.items_.push_back(f(item));
        }
        return result;
    }

    // Applicative apply
    template<typename U>
    auto apply(const list<std::function<U(T)>>& fs) const -> list<U> {
        list<U> result;
        for (const auto& f : fs.items_) {
            for (const auto& item : items_) {
                result.items_.push_back(f(item));
            }
        }
        return result;
    }

    // Monadic bind
    template<typename F>
    auto bind(F&& f) const -> decltype(f(std::declval<T>())) {
        using Result = decltype(f(std::declval<T>()));
        Result result;
        for (const auto& item : items_) {
            auto r = f(item);
            result.items_.insert(result.items_.end(), r.items_.begin(), r.items_.end());
        }
        return result;
    }

    // List operations
    void push_back(const T& val) { items_.push_back(val); }
    size_t size() const { return items_.size(); }
    bool empty() const { return items_.empty(); }

    auto begin() { return items_.begin(); }
    auto end() { return items_.end(); }
    auto begin() const { return items_.begin(); }
    auto end() const { return items_.end(); }

    // Monoid operations
    list concat(const list& other) const {
        list result = *this;
        result.items_.insert(result.items_.end(), other.items_.begin(), other.items_.end());
        return result;
    }

    static list mempty() {
        return list{};
    }
};

// State monad
template<typename S, typename A>
class state {
private:
    std::function<std::pair<A, S>(S)> run_state_;

public:
    // Constructor
    explicit state(std::function<std::pair<A, S>(S)> f) : run_state_(std::move(f)) {}

    // Run the state computation
    std::pair<A, S> run(S s) const {
        return run_state_(s);
    }

    // Evaluate (get value, discard state)
    A eval(S s) const {
        return run_state_(s).first;
    }

    // Execute (get state, discard value)
    S exec(S s) const {
        return run_state_(s).second;
    }

    // Monadic return
    static state pure(const A& a) {
        return state([a](S s) { return std::make_pair(a, s); });
    }

    // Get current state
    static state get() {
        return state([](S s) { return std::make_pair(s, s); });
    }

    // Set state
    static state put(S s) {
        return state([s](S) { return std::make_pair(S{}, s); });
    }

    // Modify state
    static state modify(std::function<S(S)> f) {
        return state([f](S s) { return std::make_pair(S{}, f(s)); });
    }

    // Functor map
    template<typename F>
    auto fmap(F&& f) const -> state<S, decltype(f(std::declval<A>()))> {
        using B = decltype(f(std::declval<A>()));
        return state<S, B>([this, f](S s) {
            auto [a, s2] = run_state_(s);
            return std::make_pair(f(a), s2);
        });
    }

    // Monadic bind
    template<typename F>
    auto bind(F&& f) const -> decltype(f(std::declval<A>())) {
        using Result = decltype(f(std::declval<A>()));
        return Result([this, f](S s) {
            auto [a, s2] = run_state_(s);
            return f(a).run(s2);
        });
    }
};

// Reader monad (dependency injection)
template<typename R, typename A>
class reader {
private:
    std::function<A(R)> run_reader_;

public:
    explicit reader(std::function<A(R)> f) : run_reader_(std::move(f)) {}

    // Run the reader
    A run(R r) const {
        return run_reader_(r);
    }

    // Monadic return
    static reader pure(const A& a) {
        return reader([a](R) { return a; });
    }

    // Ask for the environment
    static reader ask() {
        return reader([](R r) { return r; });
    }

    // Functor map
    template<typename F>
    auto fmap(F&& f) const -> reader<R, decltype(f(std::declval<A>()))> {
        using B = decltype(f(std::declval<A>()));
        return reader<R, B>([this, f](R r) {
            return f(run_reader_(r));
        });
    }

    // Monadic bind
    template<typename F>
    auto bind(F&& f) const -> decltype(f(std::declval<A>())) {
        using Result = decltype(f(std::declval<A>()));
        return Result([this, f](R r) {
            return f(run_reader_(r)).run(r);
        });
    }

    // Local (modify environment locally)
    reader local(std::function<R(R)> f) const {
        return reader([this, f](R r) {
            return run_reader_(f(r));
        });
    }
};

// Writer monad
template<typename W, typename A>
class writer {
private:
    A value_;
    W log_;

public:
    writer(const A& a, const W& w) : value_(a), log_(w) {}

    // Run the writer
    std::pair<A, W> run() const {
        return {value_, log_};
    }

    // Monadic return (requires W to be a monoid)
    static writer pure(const A& a) {
        return writer(a, W{});
    }

    // Tell (write to log)
    static writer tell(const W& w) {
        return writer(A{}, w);
    }

    // Functor map
    template<typename F>
    auto fmap(F&& f) const -> writer<W, decltype(f(std::declval<A>()))> {
        using B = decltype(f(std::declval<A>()));
        return writer<W, B>(f(value_), log_);
    }

    // Monadic bind (requires W to support append operation)
    template<typename F>
    auto bind(F&& f) const -> decltype(f(std::declval<A>())) {
        auto result = f(value_);
        auto [b, w2] = result.run();
        W combined_log = log_;
        // Assuming W has append operation
        if constexpr (requires { combined_log.append(w2); }) {
            combined_log.append(w2);
        } else if constexpr (requires { combined_log + w2; }) {
            combined_log = combined_log + w2;
        }
        return decltype(result)(b, combined_log);
    }

    // Listen (get log)
    writer<W, std::pair<A, W>> listen() const {
        return writer<W, std::pair<A, W>>({value_, log_}, log_);
    }
};

// Continuation monad
template<typename R, typename A>
class cont {
private:
    std::function<R(std::function<R(A)>)> run_cont_;

public:
    explicit cont(std::function<R(std::function<R(A)>)> f) : run_cont_(std::move(f)) {}

    // Run the continuation
    R run(std::function<R(A)> k) const {
        return run_cont_(k);
    }

    // Monadic return
    static cont pure(const A& a) {
        return cont([a](std::function<R(A)> k) { return k(a); });
    }

    // Call with current continuation
    static cont callCC(std::function<cont(std::function<cont(A)>)> f) {
        return cont([f](std::function<R(A)> k) {
            auto escape = [k](A a) { return cont([k, a](std::function<R(A)>) { return k(a); }); };
            return f(escape).run(k);
        });
    }

    // Functor map
    template<typename F>
    auto fmap(F&& f) const -> cont<R, decltype(f(std::declval<A>()))> {
        using B = decltype(f(std::declval<A>()));
        return cont<R, B>([this, f](std::function<R(B)> k) {
            return run_cont_([f, k](A a) { return k(f(a)); });
        });
    }

    // Monadic bind
    template<typename F>
    auto bind(F&& f) const -> decltype(f(std::declval<A>())) {
        using Result = decltype(f(std::declval<A>()));
        return Result([this, f](std::function<R(typename Result::value_type)> k) {
            return run_cont_([f, k](A a) { return f(a).run(k); });
        });
    }

    using value_type = A;
};

// Free monad (for building DSLs)
template<template<typename> class F, typename A>
class free {
private:
    struct pure_t { A value; };
    struct free_t { std::unique_ptr<F<free>> functor; };

    std::variant<pure_t, free_t> data_;

public:
    // Constructors
    explicit free(const A& a) : data_(pure_t{a}) {}
    explicit free(F<free> f) : data_(free_t{std::make_unique<F<free>>(std::move(f))}) {}

    // Monadic return
    static free pure(const A& a) {
        return free(a);
    }

    // Lift functor to free monad
    static free lift_f(const F<A>& fa) {
        // This would need proper functor mapping
        // Simplified version
        return free(A{});
    }

    // Pattern matching
    template<typename PureCase, typename FreeCase>
    auto match(PureCase&& pure_case, FreeCase&& free_case) const {
        return std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, pure_t>) {
                return pure_case(v.value);
            } else {
                return free_case(*v.functor);
            }
        }, data_);
    }

    // Monadic bind
    template<typename G>
    auto bind(G&& g) const -> decltype(g(std::declval<A>())) {
        return match(
            [g](const A& a) { return g(a); },
            [g](const F<free>& f) {
                // Would need proper functor mapping
                return decltype(g(std::declval<A>()))(A{});
            }
        );
    }
};

// Do-notation simulation using coroutines (C++20)
// This is a simplified version - full implementation would use co_await

template<typename M>
class do_notation {
private:
    M monad_;

public:
    explicit do_notation(M m) : monad_(std::move(m)) {}

    template<typename F>
    auto then(F&& f) {
        return do_notation(monad_.bind(std::forward<F>(f)));
    }

    M end() { return monad_; }
};

// Helper for do-notation
template<typename M>
do_notation<M> do_(M m) {
    return do_notation<M>(std::move(m));
}

} // namespace stepanov::category