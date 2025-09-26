#pragma once

#include <concepts>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include "concepts.hpp"

namespace stepanov {

// Forward declarations
template<typename T> class optional;

// Error category for structured error handling (simplified)
class error_category {
public:
    virtual ~error_category() = default;
    virtual std::string message(int code) const = 0;
    virtual std::string name() const = 0;
};

// Generic error type
class error {
private:
    int code_;
    std::string message_;
    std::source_location location_;

public:
    error(int code, std::string_view msg,
          const std::source_location& loc = std::source_location::current())
        : code_(code), message_(msg), location_(loc) {}

    int code() const { return code_; }
    const std::string& message() const { return message_; }
    const std::source_location& location() const { return location_; }

    std::string full_message() const {
        return std::string(location_.file_name()) + ":" +
               std::to_string(location_.line()) + " " +
               location_.function_name() + ": " + message_;
    }
};

// Concepts for monadic operations
template<typename T>
concept monadic = requires(T t) {
    typename T::value_type;
    { t.has_value() } -> std::convertible_to<bool>;
};

// Enhanced optional type with monadic operations
template<typename T>
class optional {
public:
    using value_type = T;

private:
    std::optional<T> storage;

public:
    optional() = default;
    optional(std::nullopt_t) : storage(std::nullopt) {}
    optional(const T& value) : storage(value) {}
    optional(T&& value) : storage(std::move(value)) {}

    template<typename... Args>
    optional(std::in_place_t, Args&&... args)
        : storage(std::in_place, std::forward<Args>(args)...) {}

    // Basic operations
    bool has_value() const { return storage.has_value(); }
    explicit operator bool() const { return has_value(); }

    T& value() & { return storage.value(); }
    const T& value() const& { return storage.value(); }
    T&& value() && { return std::move(storage.value()); }
    const T&& value() const&& { return std::move(storage.value()); }

    T* operator->() { return storage.operator->(); }
    const T* operator->() const { return storage.operator->(); }

    T& operator*() & { return *storage; }
    const T& operator*() const& { return *storage; }
    T&& operator*() && { return std::move(*storage); }
    const T&& operator*() const&& { return std::move(*storage); }

    template<typename U>
    T value_or(U&& default_value) const& {
        return storage.value_or(std::forward<U>(default_value));
    }

    template<typename U>
    T value_or(U&& default_value) && {
        return std::move(storage).value_or(std::forward<U>(default_value));
    }

    // Monadic operations

    // map: optional<T> -> (T -> U) -> optional<U>
    template<typename F>
        requires std::invocable<F, T>
    auto map(F&& f) const -> optional<std::invoke_result_t<F, T>> {
        if (has_value()) {
            return optional<std::invoke_result_t<F, T>>(
                std::invoke(std::forward<F>(f), value()));
        }
        return std::nullopt;
    }

    // flat_map/bind: optional<T> -> (T -> optional<U>) -> optional<U>
    template<typename F>
        requires std::invocable<F, T>
    auto flat_map(F&& f) const -> std::invoke_result_t<F, T> {
        if (has_value()) {
            return std::invoke(std::forward<F>(f), value());
        }
        return std::invoke_result_t<F, T>{};
    }

    // filter: optional<T> -> (T -> bool) -> optional<T>
    template<typename P>
        requires std::predicate<P, T>
    optional filter(P&& pred) const {
        if (has_value() && std::invoke(std::forward<P>(pred), value())) {
            return *this;
        }
        return std::nullopt;
    }

    // or_else: optional<T> -> (() -> optional<T>) -> optional<T>
    template<typename F>
        requires std::invocable<F>
    optional or_else(F&& f) const {
        if (has_value()) {
            return *this;
        }
        return std::invoke(std::forward<F>(f));
    }

    // and_then: optional<T> -> optional<U> -> optional<U>
    template<typename U>
    optional<U> and_then(const optional<U>& other) const {
        if (has_value()) {
            return other;
        }
        return std::nullopt;
    }

    void reset() { storage.reset(); }

    template<typename... Args>
    T& emplace(Args&&... args) {
        return storage.emplace(std::forward<Args>(args)...);
    }
};

// Expected type (like Rust's Result<T, E>)
template<typename T, typename E = error>
    requires (!std::is_same_v<T, E>)
class expected {
public:
    using value_type = T;
    using error_type = E;

private:
    std::variant<T, E> storage;

public:
    // Constructors
    expected(const T& value) : storage(value) {}
    expected(T&& value) : storage(std::move(value)) {}
    expected(const E& error) : storage(error) {}
    expected(E&& error) : storage(std::move(error)) {}

    template<typename... Args>
    static expected success(Args&&... args) {
        return expected(T(std::forward<Args>(args)...));
    }

    template<typename... Args>
    static expected failure(Args&&... args) {
        return expected(E(std::forward<Args>(args)...));
    }

    // Observers
    bool has_value() const { return std::holds_alternative<T>(storage); }
    bool has_error() const { return std::holds_alternative<E>(storage); }
    explicit operator bool() const { return has_value(); }

    T& value() & {
        if (!has_value()) {
            throw std::bad_variant_access();
        }
        return std::get<T>(storage);
    }

    const T& value() const& {
        if (!has_value()) {
            throw std::bad_variant_access();
        }
        return std::get<T>(storage);
    }

    T&& value() && {
        if (!has_value()) {
            throw std::bad_variant_access();
        }
        return std::move(std::get<T>(storage));
    }

    E& error() & {
        if (!has_error()) {
            throw std::bad_variant_access();
        }
        return std::get<E>(storage);
    }

    const E& error() const& {
        if (!has_error()) {
            throw std::bad_variant_access();
        }
        return std::get<E>(storage);
    }

    E&& error() && {
        if (!has_error()) {
            throw std::bad_variant_access();
        }
        return std::move(std::get<E>(storage));
    }

    T* operator->() {
        return has_value() ? &value() : nullptr;
    }

    const T* operator->() const {
        return has_value() ? &value() : nullptr;
    }

    T& operator*() & { return value(); }
    const T& operator*() const& { return value(); }
    T&& operator*() && { return std::move(value()); }

    // Monadic operations

    // map: expected<T, E> -> (T -> U) -> expected<U, E>
    template<typename F>
        requires std::invocable<F, T>
    auto map(F&& f) const -> expected<std::invoke_result_t<F, T>, E> {
        if (has_value()) {
            return expected<std::invoke_result_t<F, T>, E>(
                std::invoke(std::forward<F>(f), value()));
        }
        return expected<std::invoke_result_t<F, T>, E>(error());
    }

    // map_error: expected<T, E> -> (E -> F) -> expected<T, F>
    template<typename F>
        requires std::invocable<F, E>
    auto map_error(F&& f) const -> expected<T, std::invoke_result_t<F, E>> {
        if (has_error()) {
            return expected<T, std::invoke_result_t<F, E>>(
                std::invoke(std::forward<F>(f), error()));
        }
        return expected<T, std::invoke_result_t<F, E>>(value());
    }

    // flat_map/and_then: expected<T, E> -> (T -> expected<U, E>) -> expected<U, E>
    template<typename F>
        requires std::invocable<F, T>
    auto flat_map(F&& f) const -> std::invoke_result_t<F, T> {
        if (has_value()) {
            return std::invoke(std::forward<F>(f), value());
        }
        return std::invoke_result_t<F, T>(error());
    }

    // or_else: expected<T, E> -> (E -> expected<T, F>) -> expected<T, F>
    template<typename F>
        requires std::invocable<F, E>
    auto or_else(F&& f) const -> std::invoke_result_t<F, E> {
        if (has_error()) {
            return std::invoke(std::forward<F>(f), error());
        }
        return std::invoke_result_t<F, E>(value());
    }

    // unwrap_or: expected<T, E> -> T -> T
    template<typename U>
    T unwrap_or(U&& default_value) const& {
        return has_value() ? value() : static_cast<T>(std::forward<U>(default_value));
    }

    template<typename U>
    T unwrap_or(U&& default_value) && {
        return has_value() ? std::move(value()) : static_cast<T>(std::forward<U>(default_value));
    }

    // unwrap_or_else: expected<T, E> -> (E -> T) -> T
    template<typename F>
        requires std::invocable<F, E>
    T unwrap_or_else(F&& f) const {
        if (has_value()) {
            return value();
        }
        return std::invoke(std::forward<F>(f), error());
    }
};

// Try macro for error propagation (similar to Rust's ? operator)
#define TRY(expr) \
    do { \
        auto&& _result = (expr); \
        if (!_result.has_value()) { \
            return std::move(_result).error(); \
        } \
    } while(0)

// Compile-time validation with better error messages
template<typename T>
concept validatable = requires(T t) {
    { T::validate() } -> std::convertible_to<bool>;
};

template<validatable T>
consteval bool validate_type() {
    return T::validate();
}

// Contract assertions
enum class contract_level {
    audit,    // Expensive checks, usually disabled
    default_, // Normal runtime checks
    axiom     // Assumptions that must always hold
};

template<contract_level Level = contract_level::default_>
class contract {
private:
    static constexpr bool enabled() {
        if constexpr (Level == contract_level::audit) {
#ifdef STEPANOV_ENABLE_AUDIT_CONTRACTS
            return true;
#else
            return false;
#endif
        } else if constexpr (Level == contract_level::axiom) {
            return true;  // Always enabled
        } else {
#ifdef NDEBUG
            return false;
#else
            return true;
#endif
        }
    }

public:
    static void require(bool condition, std::string_view msg = "",
                        const std::source_location& loc = std::source_location::current()) {
        if constexpr (enabled()) {
            if (!condition) {
                throw std::logic_error(
                    std::string("Precondition failed at ") +
                    loc.file_name() + ":" + std::to_string(loc.line()) +
                    " in " + loc.function_name() +
                    (msg.empty() ? "" : ": ") + std::string(msg));
            }
        }
    }

    static void ensures(bool condition, std::string_view msg = "",
                       const std::source_location& loc = std::source_location::current()) {
        if constexpr (enabled()) {
            if (!condition) {
                throw std::logic_error(
                    std::string("Postcondition failed at ") +
                    loc.file_name() + ":" + std::to_string(loc.line()) +
                    " in " + loc.function_name() +
                    (msg.empty() ? "" : ": ") + std::string(msg));
            }
        }
    }

    static void invariant(bool condition, std::string_view msg = "",
                         const std::source_location& loc = std::source_location::current()) {
        if constexpr (enabled()) {
            if (!condition) {
                throw std::logic_error(
                    std::string("Invariant violated at ") +
                    loc.file_name() + ":" + std::to_string(loc.line()) +
                    " in " + loc.function_name() +
                    (msg.empty() ? "" : ": ") + std::string(msg));
            }
        }
    }
};

// RAII scope guard for exception safety
template<typename F>
class scope_guard {
private:
    F cleanup;
    bool active;

public:
    explicit scope_guard(F&& f) : cleanup(std::forward<F>(f)), active(true) {}

    scope_guard(scope_guard&& other) noexcept
        : cleanup(std::move(other.cleanup)), active(other.active) {
        other.dismiss();
    }

    ~scope_guard() {
        if (active) {
            cleanup();
        }
    }

    void dismiss() noexcept { active = false; }

    scope_guard(const scope_guard&) = delete;
    scope_guard& operator=(const scope_guard&) = delete;
    scope_guard& operator=(scope_guard&&) = delete;
};

template<typename F>
scope_guard(F) -> scope_guard<F>;

// Helper to create scope guards
template<typename F>
auto make_scope_guard(F&& f) {
    return scope_guard<std::decay_t<F>>(std::forward<F>(f));
}

// Exception-safe resource wrapper
template<typename T, typename Deleter = std::default_delete<T>>
class resource {
private:
    std::unique_ptr<T, Deleter> ptr;

public:
    resource() = default;

    explicit resource(T* p, Deleter d = Deleter{})
        : ptr(p, d) {}

    template<typename... Args>
    static resource create(Args&&... args) {
        return resource(new T(std::forward<Args>(args)...));
    }

    T* get() const { return ptr.get(); }
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr.get(); }
    explicit operator bool() const { return ptr != nullptr; }

    T* release() { return ptr.release(); }
    void reset(T* p = nullptr) { ptr.reset(p); }

    // Move operations
    resource(resource&&) = default;
    resource& operator=(resource&&) = default;

    // No copy
    resource(const resource&) = delete;
    resource& operator=(const resource&) = delete;
};

// Result type combining value and error collection
template<typename T>
class result {
private:
    optional<T> value_;
    std::vector<error> errors_;
    std::vector<std::string> warnings_;

public:
    result() = default;
    result(const T& value) : value_(value) {}
    result(T&& value) : value_(std::move(value)) {}

    bool is_success() const { return value_.has_value() && errors_.empty(); }
    bool has_warnings() const { return !warnings_.empty(); }
    bool has_errors() const { return !errors_.empty(); }

    const optional<T>& value() const { return value_; }
    optional<T>& value() { return value_; }

    const std::vector<error>& errors() const { return errors_; }
    const std::vector<std::string>& warnings() const { return warnings_; }

    result& add_error(error e) {
        errors_.push_back(std::move(e));
        return *this;
    }

    result& add_warning(std::string w) {
        warnings_.push_back(std::move(w));
        return *this;
    }

    // Combine results
    result& merge(const result& other) {
        if (!other.value_.has_value()) {
            value_.reset();
        }
        errors_.insert(errors_.end(), other.errors_.begin(), other.errors_.end());
        warnings_.insert(warnings_.end(), other.warnings_.begin(), other.warnings_.end());
        return *this;
    }
};

// Error accumulator for validation
template<typename E = std::string>
class error_accumulator {
private:
    std::vector<E> errors;
    bool fail_fast;

public:
    explicit error_accumulator(bool fast = false) : fail_fast(fast) {}

    void add(E error) {
        errors.push_back(std::move(error));
        if (fail_fast && !errors.empty()) {
            throw std::runtime_error("Validation failed (fail-fast mode)");
        }
    }

    bool has_errors() const { return !errors.empty(); }
    const std::vector<E>& get_errors() const { return errors; }

    void clear() { errors.clear(); }

    expected<void, std::vector<E>> to_expected() const {
        if (errors.empty()) {
            return expected<void, std::vector<E>>::success();
        }
        return expected<void, std::vector<E>>::failure(errors);
    }
};

} // namespace stepanov