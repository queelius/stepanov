#pragma once

#include <memory>
#include <functional>
#include <typeindex>
#include <any>
#include <concepts>
#include "concepts.hpp"

namespace stepanov {

/**
 * Type erasure for mathematical objects
 *
 * Provides runtime polymorphism while maintaining value semantics
 * Following Sean Parent's "Runtime Polymorphism" techniques
 */

// =============================================================================
// Regular type erasure - for any regular type
// =============================================================================

class any_regular {
private:
    struct concept_t {
        virtual ~concept_t() = default;
        virtual std::unique_ptr<concept_t> clone() const = 0;
        virtual bool equals(const concept_t& other) const = 0;
        virtual std::size_t hash() const = 0;
        virtual std::type_index type() const = 0;
        virtual void* get() = 0;
        virtual const void* get() const = 0;
    };

    template <regular T>
    struct model_t final : concept_t {
        T value;

        explicit model_t(T x) : value(std::move(x)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(value);
        }

        bool equals(const concept_t& other) const override {
            if (type() != other.type()) return false;
            return value == static_cast<const model_t&>(other).value;
        }

        std::size_t hash() const override {
            return std::hash<T>{}(value);
        }

        std::type_index type() const override {
            return typeid(T);
        }

        void* get() override { return &value; }
        const void* get() const override { return &value; }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_regular() = default;

    template <regular T>
    any_regular(T x) : pimpl(std::make_unique<model_t<T>>(std::move(x))) {}

    any_regular(const any_regular& other)
        : pimpl(other.pimpl ? other.pimpl->clone() : nullptr) {}

    any_regular(any_regular&&) noexcept = default;

    any_regular& operator=(const any_regular& other) {
        any_regular temp(other);
        std::swap(pimpl, temp.pimpl);
        return *this;
    }

    any_regular& operator=(any_regular&&) noexcept = default;

    // Comparison
    friend bool operator==(const any_regular& a, const any_regular& b) {
        if (!a.pimpl && !b.pimpl) return true;
        if (!a.pimpl || !b.pimpl) return false;
        return a.pimpl->equals(*b.pimpl);
    }

    friend bool operator!=(const any_regular& a, const any_regular& b) {
        return !(a == b);
    }

    // Type checking
    template <typename T>
    bool holds() const {
        return pimpl && pimpl->type() == typeid(T);
    }

    // Value extraction
    template <typename T>
    T* get() {
        if (!holds<T>()) return nullptr;
        return static_cast<T*>(pimpl->get());
    }

    template <typename T>
    const T* get() const {
        if (!holds<T>()) return nullptr;
        return static_cast<const T*>(pimpl->get());
    }

    bool has_value() const { return pimpl != nullptr; }

    std::size_t hash() const {
        return pimpl ? pimpl->hash() : 0;
    }
};

// =============================================================================
// Algebraic type erasure - for types with algebraic operations
// =============================================================================

class any_algebraic {
private:
    struct concept_t {
        virtual ~concept_t() = default;
        virtual std::unique_ptr<concept_t> clone() const = 0;
        virtual std::unique_ptr<concept_t> add(const concept_t& other) const = 0;
        virtual std::unique_ptr<concept_t> multiply(const concept_t& other) const = 0;
        virtual std::unique_ptr<concept_t> negate() const = 0;
        virtual std::unique_ptr<concept_t> zero() const = 0;
        virtual std::unique_ptr<concept_t> one() const = 0;
        virtual bool equals(const concept_t& other) const = 0;
        virtual std::type_index type() const = 0;
    };

    template <ring T>
    struct model_t final : concept_t {
        T value;

        explicit model_t(T x) : value(std::move(x)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(value);
        }

        std::unique_ptr<concept_t> add(const concept_t& other) const override {
            if (type() != other.type()) {
                throw std::runtime_error("Type mismatch in addition");
            }
            return std::make_unique<model_t>(
                value + static_cast<const model_t&>(other).value);
        }

        std::unique_ptr<concept_t> multiply(const concept_t& other) const override {
            if (type() != other.type()) {
                throw std::runtime_error("Type mismatch in multiplication");
            }
            return std::make_unique<model_t>(
                value * static_cast<const model_t&>(other).value);
        }

        std::unique_ptr<concept_t> negate() const override {
            return std::make_unique<model_t>(-value);
        }

        std::unique_ptr<concept_t> zero() const override {
            return std::make_unique<model_t>(T(0));
        }

        std::unique_ptr<concept_t> one() const override {
            return std::make_unique<model_t>(T(1));
        }

        bool equals(const concept_t& other) const override {
            if (type() != other.type()) return false;
            return value == static_cast<const model_t&>(other).value;
        }

        std::type_index type() const override {
            return typeid(T);
        }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_algebraic() = default;

    template <ring T>
    any_algebraic(T x) : pimpl(std::make_unique<model_t<T>>(std::move(x))) {}

    any_algebraic(const any_algebraic& other)
        : pimpl(other.pimpl ? other.pimpl->clone() : nullptr) {}

    any_algebraic(any_algebraic&&) noexcept = default;

    any_algebraic& operator=(const any_algebraic& other) {
        any_algebraic temp(other);
        std::swap(pimpl, temp.pimpl);
        return *this;
    }

    any_algebraic& operator=(any_algebraic&&) noexcept = default;

    // Algebraic operations
    any_algebraic operator+(const any_algebraic& other) const {
        if (!pimpl || !other.pimpl) {
            throw std::runtime_error("Operation on empty algebraic type");
        }
        any_algebraic result;
        result.pimpl = pimpl->add(*other.pimpl);
        return result;
    }

    any_algebraic operator*(const any_algebraic& other) const {
        if (!pimpl || !other.pimpl) {
            throw std::runtime_error("Operation on empty algebraic type");
        }
        any_algebraic result;
        result.pimpl = pimpl->multiply(*other.pimpl);
        return result;
    }

    any_algebraic operator-() const {
        if (!pimpl) {
            throw std::runtime_error("Operation on empty algebraic type");
        }
        any_algebraic result;
        result.pimpl = pimpl->negate();
        return result;
    }

    any_algebraic operator-(const any_algebraic& other) const {
        return *this + (-other);
    }

    // Identity elements
    any_algebraic zero() const {
        if (!pimpl) {
            throw std::runtime_error("Operation on empty algebraic type");
        }
        any_algebraic result;
        result.pimpl = pimpl->zero();
        return result;
    }

    any_algebraic one() const {
        if (!pimpl) {
            throw std::runtime_error("Operation on empty algebraic type");
        }
        any_algebraic result;
        result.pimpl = pimpl->one();
        return result;
    }

    // Comparison
    friend bool operator==(const any_algebraic& a, const any_algebraic& b) {
        if (!a.pimpl && !b.pimpl) return true;
        if (!a.pimpl || !b.pimpl) return false;
        return a.pimpl->equals(*b.pimpl);
    }

    friend bool operator!=(const any_algebraic& a, const any_algebraic& b) {
        return !(a == b);
    }

    bool has_value() const { return pimpl != nullptr; }
};

// =============================================================================
// Function type erasure - for transformations
// =============================================================================

template <typename Signature>
class any_function;

template <typename R, typename... Args>
class any_function<R(Args...)> {
private:
    struct concept_t {
        virtual ~concept_t() = default;
        virtual std::unique_ptr<concept_t> clone() const = 0;
        virtual R invoke(Args... args) const = 0;
    };

    template <typename F>
    struct model_t final : concept_t {
        F f;

        explicit model_t(F func) : f(std::move(func)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(f);
        }

        R invoke(Args... args) const override {
            return f(args...);
        }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_function() = default;

    template <typename F>
        requires std::invocable<F, Args...> &&
                 std::convertible_to<std::invoke_result_t<F, Args...>, R>
    any_function(F f) : pimpl(std::make_unique<model_t<F>>(std::move(f))) {}

    any_function(const any_function& other)
        : pimpl(other.pimpl ? other.pimpl->clone() : nullptr) {}

    any_function(any_function&&) noexcept = default;

    any_function& operator=(const any_function& other) {
        any_function temp(other);
        std::swap(pimpl, temp.pimpl);
        return *this;
    }

    any_function& operator=(any_function&&) noexcept = default;

    // Invocation
    R operator()(Args... args) const {
        if (!pimpl) {
            throw std::runtime_error("Invoking empty function");
        }
        return pimpl->invoke(args...);
    }

    bool has_value() const { return pimpl != nullptr; }

    explicit operator bool() const { return has_value(); }
};

// =============================================================================
// Helper macros for type erasure boilerplate
// =============================================================================

#define STEPANOV_TYPE_ERASURE_COPY_SEMANTICS(Class) \
    Class(const Class& other) \
        : pimpl(other.pimpl ? other.pimpl->clone() : nullptr) {} \
    Class& operator=(const Class& other) { \
        Class temp(other); \
        std::swap(pimpl, temp.pimpl); \
        return *this; \
    }

#define STEPANOV_TYPE_ERASURE_MOVE_SEMANTICS(Class) \
    Class(Class&&) noexcept = default; \
    Class& operator=(Class&&) noexcept = default;

#define STEPANOV_TYPE_ERASURE_VALUE_SEMANTICS(Class) \
    STEPANOV_TYPE_ERASURE_COPY_SEMANTICS(Class) \
    STEPANOV_TYPE_ERASURE_MOVE_SEMANTICS(Class)

// =============================================================================
// Base concept for cloneable types
// =============================================================================

template<typename Derived>
struct cloneable_concept {
    virtual ~cloneable_concept() = default;
    virtual std::unique_ptr<Derived> clone() const = 0;
};

// =============================================================================
// Codec type erasure - for bit-level encoding/decoding
// =============================================================================

template<typename From, typename To>
class any_codec {
private:
    struct concept_t : cloneable_concept<concept_t> {
        virtual To encode(const From& value) const = 0;
        virtual From decode(const To& encoded) const = 0;
        virtual std::size_t encoded_size(const From& value) const = 0;
        virtual std::string name() const = 0;
    };

    template<typename Codec>
    struct model_t final : concept_t {
        Codec codec;

        explicit model_t(Codec c) : codec(std::move(c)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(codec);
        }

        To encode(const From& value) const override {
            return codec.encode(value);
        }

        From decode(const To& encoded) const override {
            return codec.decode(encoded);
        }

        std::size_t encoded_size(const From& value) const override {
            if constexpr (requires { codec.encoded_size(value); }) {
                return codec.encoded_size(value);
            } else {
                return sizeof(To); // Default estimate
            }
        }

        std::string name() const override {
            if constexpr (requires { codec.name(); }) {
                return codec.name();
            } else {
                return "unknown_codec";
            }
        }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_codec() = default;

    template<typename Codec>
        requires requires(const Codec& c, const From& f, const To& t) {
            { c.encode(f) } -> std::convertible_to<To>;
            { c.decode(t) } -> std::convertible_to<From>;
        }
    any_codec(Codec c) : pimpl(std::make_unique<model_t<Codec>>(std::move(c))) {}

    STEPANOV_TYPE_ERASURE_VALUE_SEMANTICS(any_codec)

    // Operations
    To encode(const From& value) const {
        if (!pimpl) throw std::runtime_error("Empty codec");
        return pimpl->encode(value);
    }

    From decode(const To& encoded) const {
        if (!pimpl) throw std::runtime_error("Empty codec");
        return pimpl->decode(encoded);
    }

    std::size_t encoded_size(const From& value) const {
        if (!pimpl) throw std::runtime_error("Empty codec");
        return pimpl->encoded_size(value);
    }

    std::string name() const {
        return pimpl ? pimpl->name() : "null_codec";
    }

    bool has_value() const noexcept { return pimpl != nullptr; }
    explicit operator bool() const noexcept { return has_value(); }
};

// =============================================================================
// Text processor type erasure - for text algorithms
// =============================================================================

template<typename Input, typename Output>
class any_text_processor {
private:
    struct concept_t : cloneable_concept<concept_t> {
        virtual Output process(const Input& text) const = 0;
        virtual bool can_process(const Input& text) const = 0;
        virtual std::string description() const = 0;
    };

    template<typename Processor>
    struct model_t final : concept_t {
        Processor processor;

        explicit model_t(Processor p) : processor(std::move(p)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(processor);
        }

        Output process(const Input& text) const override {
            return processor(text);
        }

        bool can_process(const Input& text) const override {
            if constexpr (requires { processor.can_process(text); }) {
                return processor.can_process(text);
            } else {
                return true; // Assume it can process by default
            }
        }

        std::string description() const override {
            if constexpr (requires { processor.description(); }) {
                return processor.description();
            } else {
                return "text_processor";
            }
        }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_text_processor() = default;

    template<typename Processor>
        requires std::invocable<Processor, Input> &&
                 std::convertible_to<std::invoke_result_t<Processor, Input>, Output>
    any_text_processor(Processor p)
        : pimpl(std::make_unique<model_t<Processor>>(std::move(p))) {}

    STEPANOV_TYPE_ERASURE_VALUE_SEMANTICS(any_text_processor)

    // Operations
    Output operator()(const Input& text) const {
        if (!pimpl) throw std::runtime_error("Empty text processor");
        return pimpl->process(text);
    }

    Output process(const Input& text) const {
        return (*this)(text);
    }

    bool can_process(const Input& text) const {
        return pimpl && pimpl->can_process(text);
    }

    std::string description() const {
        return pimpl ? pimpl->description() : "null_processor";
    }

    bool has_value() const noexcept { return pimpl != nullptr; }
    explicit operator bool() const noexcept { return has_value(); }

    // Composition
    template<typename OtherOutput>
    any_text_processor<Input, OtherOutput>
    operator|(const any_text_processor<Output, OtherOutput>& next) const {
        auto self = *this;
        return any_text_processor<Input, OtherOutput>(
            [self, next](const Input& input) -> OtherOutput {
                return next(self(input));
            }
        );
    }
};

// =============================================================================
// Differentiable type erasure - for automatic differentiation
// =============================================================================

template<typename T>
class any_differentiable {
private:
    struct concept_t : cloneable_concept<concept_t> {
        virtual T value() const = 0;
        virtual T gradient() const = 0;
        virtual void set_gradient(T grad) = 0;
        virtual std::unique_ptr<concept_t> forward(const concept_t& other,
                                                   const std::string& op) const = 0;
        virtual void backward(T upstream_grad) = 0;
        virtual bool requires_grad() const = 0;
    };

    template<typename Var>
    struct model_t final : concept_t {
        Var variable;

        explicit model_t(Var v) : variable(std::move(v)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(variable);
        }

        T value() const override {
            if constexpr (requires { variable.value(); }) {
                return variable.value();
            } else {
                return static_cast<T>(variable);
            }
        }

        T gradient() const override {
            if constexpr (requires { variable.gradient(); }) {
                return variable.gradient();
            } else {
                return T{0};
            }
        }

        void set_gradient(T grad) override {
            if constexpr (requires { variable.set_gradient(grad); }) {
                variable.set_gradient(grad);
            }
        }

        std::unique_ptr<concept_t> forward(const concept_t& other,
                                          const std::string& op) const override {
            // This would need specific implementation based on the operation
            return clone(); // Placeholder
        }

        void backward(T upstream_grad) override {
            if constexpr (requires { variable.backward(upstream_grad); }) {
                variable.backward(upstream_grad);
            }
        }

        bool requires_grad() const override {
            if constexpr (requires { variable.requires_grad(); }) {
                return variable.requires_grad();
            } else {
                return false;
            }
        }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_differentiable() = default;

    template<typename Var>
    any_differentiable(Var v)
        : pimpl(std::make_unique<model_t<Var>>(std::move(v))) {}

    STEPANOV_TYPE_ERASURE_VALUE_SEMANTICS(any_differentiable)

    // Operations
    T value() const {
        if (!pimpl) throw std::runtime_error("Empty differentiable");
        return pimpl->value();
    }

    T gradient() const {
        if (!pimpl) throw std::runtime_error("Empty differentiable");
        return pimpl->gradient();
    }

    void set_gradient(T grad) {
        if (!pimpl) throw std::runtime_error("Empty differentiable");
        pimpl->set_gradient(grad);
    }

    void backward(T upstream_grad = T{1}) {
        if (!pimpl) throw std::runtime_error("Empty differentiable");
        pimpl->backward(upstream_grad);
    }

    bool requires_grad() const {
        return pimpl && pimpl->requires_grad();
    }

    bool has_value() const noexcept { return pimpl != nullptr; }
    explicit operator bool() const noexcept { return has_value(); }

    // Arithmetic operations
    any_differentiable operator+(const any_differentiable& other) const {
        // Would create a new differentiable representing the sum
        return *this; // Placeholder
    }

    any_differentiable operator*(const any_differentiable& other) const {
        // Would create a new differentiable representing the product
        return *this; // Placeholder
    }
};

// =============================================================================
// Iterator type erasure
// =============================================================================

template <typename T>
class any_iterator {
private:
    struct concept_t {
        virtual ~concept_t() = default;
        virtual std::unique_ptr<concept_t> clone() const = 0;
        virtual T& dereference() = 0;
        virtual const T& dereference() const = 0;
        virtual void increment() = 0;
        virtual bool equals(const concept_t& other) const = 0;
    };

    template <typename It>
    struct model_t final : concept_t {
        It it;

        explicit model_t(It iter) : it(std::move(iter)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(it);
        }

        T& dereference() override {
            return *it;
        }

        const T& dereference() const override {
            return *it;
        }

        void increment() override {
            ++it;
        }

        bool equals(const concept_t& other) const override {
            // This requires RTTI or other mechanism for type checking
            auto* p = dynamic_cast<const model_t*>(&other);
            return p && it == p->it;
        }
    };

    std::shared_ptr<concept_t> pimpl;

public:
    // Iterator traits
    using iterator_category = std::input_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    // Constructors
    any_iterator() = default;

    template <typename It>
        requires std::input_iterator<It> &&
                 std::same_as<T, std::iter_value_t<It>>
    any_iterator(It it) : pimpl(std::make_shared<model_t<It>>(std::move(it))) {}

    // Iterator operations
    T& operator*() {
        if (!pimpl) throw std::runtime_error("Dereferencing null iterator");
        return pimpl->dereference();
    }

    const T& operator*() const {
        if (!pimpl) throw std::runtime_error("Dereferencing null iterator");
        return pimpl->dereference();
    }

    any_iterator& operator++() {
        if (!pimpl) throw std::runtime_error("Incrementing null iterator");
        pimpl->increment();
        return *this;
    }

    any_iterator operator++(int) {
        any_iterator temp = *this;
        ++(*this);
        return temp;
    }

    friend bool operator==(const any_iterator& a, const any_iterator& b) {
        if (!a.pimpl && !b.pimpl) return true;
        if (!a.pimpl || !b.pimpl) return false;
        return a.pimpl->equals(*b.pimpl);
    }

    friend bool operator!=(const any_iterator& a, const any_iterator& b) {
        return !(a == b);
    }
};

// =============================================================================
// Interval set type erasure
// =============================================================================

template<typename T>
class any_interval_set {
private:
    struct concept_t : cloneable_concept<concept_t> {
        virtual void insert(T start, T end) = 0;
        virtual void remove(T start, T end) = 0;
        virtual bool contains(T value) const = 0;
        virtual bool overlaps(T start, T end) const = 0;
        virtual std::size_t size() const = 0;
        virtual void clear() = 0;
        virtual std::vector<std::pair<T, T>> intervals() const = 0;
    };

    template<typename IntervalSet>
    struct model_t final : concept_t {
        IntervalSet set;

        explicit model_t(IntervalSet s) : set(std::move(s)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(set);
        }

        void insert(T start, T end) override {
            set.insert(start, end);
        }

        void remove(T start, T end) override {
            if constexpr (requires { set.remove(start, end); }) {
                set.remove(start, end);
            } else if constexpr (requires { set.erase(start, end); }) {
                set.erase(start, end);
            }
        }

        bool contains(T value) const override {
            return set.contains(value);
        }

        bool overlaps(T start, T end) const override {
            if constexpr (requires { set.overlaps(start, end); }) {
                return set.overlaps(start, end);
            } else {
                // Default implementation
                for (const auto& [s, e] : intervals()) {
                    if (s < end && start < e) return true;
                }
                return false;
            }
        }

        std::size_t size() const override {
            return set.size();
        }

        void clear() override {
            set.clear();
        }

        std::vector<std::pair<T, T>> intervals() const override {
            if constexpr (requires { set.intervals(); }) {
                return set.intervals();
            } else {
                // Try to iterate and collect intervals
                std::vector<std::pair<T, T>> result;
                for (const auto& interval : set) {
                    if constexpr (requires { interval.first; interval.second; }) {
                        result.push_back({interval.first, interval.second});
                    } else if constexpr (requires { interval.start(); interval.end(); }) {
                        result.push_back({interval.start(), interval.end()});
                    }
                }
                return result;
            }
        }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_interval_set() = default;

    template<typename IntervalSet>
    any_interval_set(IntervalSet s)
        : pimpl(std::make_unique<model_t<IntervalSet>>(std::move(s))) {}

    STEPANOV_TYPE_ERASURE_VALUE_SEMANTICS(any_interval_set)

    // Operations
    void insert(T start, T end) {
        if (!pimpl) throw std::runtime_error("Empty interval set");
        pimpl->insert(start, end);
    }

    void remove(T start, T end) {
        if (!pimpl) throw std::runtime_error("Empty interval set");
        pimpl->remove(start, end);
    }

    bool contains(T value) const {
        if (!pimpl) return false;
        return pimpl->contains(value);
    }

    bool overlaps(T start, T end) const {
        if (!pimpl) return false;
        return pimpl->overlaps(start, end);
    }

    std::size_t size() const {
        if (!pimpl) return 0;
        return pimpl->size();
    }

    void clear() {
        if (pimpl) pimpl->clear();
    }

    std::vector<std::pair<T, T>> intervals() const {
        if (!pimpl) return {};
        return pimpl->intervals();
    }

    bool has_value() const noexcept { return pimpl != nullptr; }
    explicit operator bool() const noexcept { return has_value(); }
};

// =============================================================================
// Integrator type erasure
// =============================================================================

template<typename T>
struct integration_result {
    T value;
    T error_estimate;
    std::size_t evaluations;
    bool converged;
};

template<typename T>
class any_integrator {
private:
    struct concept_t : cloneable_concept<concept_t> {
        virtual integration_result<T> integrate(
            const std::function<T(T)>& f, T a, T b, T tolerance) const = 0;
        virtual std::string method_name() const = 0;
    };

    template<typename Integrator>
    struct model_t final : concept_t {
        Integrator integrator;

        explicit model_t(Integrator i) : integrator(std::move(i)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(integrator);
        }

        integration_result<T> integrate(
            const std::function<T(T)>& f, T a, T b, T tolerance) const override {
            if constexpr (requires { integrator.integrate(f, a, b, tolerance); }) {
                return integrator.integrate(f, a, b, tolerance);
            } else if constexpr (requires { integrator(f, a, b, tolerance); }) {
                // Try callable interface
                auto result = integrator(f, a, b, tolerance);
                if constexpr (std::same_as<decltype(result), integration_result<T>>) {
                    return result;
                } else {
                    // Assume it returns just the value
                    return {result, T{0}, 0, true};
                }
            } else {
                return {T{0}, T{0}, 0, false};
            }
        }

        std::string method_name() const override {
            if constexpr (requires { integrator.method_name(); }) {
                return integrator.method_name();
            } else if constexpr (requires { integrator.name(); }) {
                return integrator.name();
            } else {
                return "unknown_integrator";
            }
        }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_integrator() = default;

    template<typename Integrator>
    any_integrator(Integrator i)
        : pimpl(std::make_unique<model_t<Integrator>>(std::move(i))) {}

    STEPANOV_TYPE_ERASURE_VALUE_SEMANTICS(any_integrator)

    // Operations
    integration_result<T> integrate(
        const std::function<T(T)>& f, T a, T b, T tolerance = T{1e-8}) const {
        if (!pimpl) throw std::runtime_error("Empty integrator");
        return pimpl->integrate(f, a, b, tolerance);
    }

    integration_result<T> operator()(
        const std::function<T(T)>& f, T a, T b, T tolerance = T{1e-8}) const {
        return integrate(f, a, b, tolerance);
    }

    std::string method_name() const {
        return pimpl ? pimpl->method_name() : "null_integrator";
    }

    bool has_value() const noexcept { return pimpl != nullptr; }
    explicit operator bool() const noexcept { return has_value(); }
};

// =============================================================================
// Hash function type erasure
// =============================================================================

template<typename T>
class any_hash {
private:
    struct concept_t : cloneable_concept<concept_t> {
        virtual std::size_t hash(const T& value) const = 0;
        virtual std::string algorithm_name() const = 0;
    };

    template<typename Hash>
    struct model_t final : concept_t {
        Hash hasher;

        explicit model_t(Hash h) : hasher(std::move(h)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(hasher);
        }

        std::size_t hash(const T& value) const override {
            return hasher(value);
        }

        std::string algorithm_name() const override {
            if constexpr (requires { hasher.algorithm_name(); }) {
                return hasher.algorithm_name();
            } else if constexpr (requires { hasher.name(); }) {
                return hasher.name();
            } else {
                return "unknown_hash";
            }
        }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_hash() = default;

    template<typename Hash>
        requires std::invocable<Hash, T> &&
                 std::convertible_to<std::invoke_result_t<Hash, T>, std::size_t>
    any_hash(Hash h)
        : pimpl(std::make_unique<model_t<Hash>>(std::move(h))) {}

    STEPANOV_TYPE_ERASURE_VALUE_SEMANTICS(any_hash)

    // Operations
    std::size_t operator()(const T& value) const {
        if (!pimpl) throw std::runtime_error("Empty hash function");
        return pimpl->hash(value);
    }

    std::size_t hash(const T& value) const {
        return (*this)(value);
    }

    std::string algorithm_name() const {
        return pimpl ? pimpl->algorithm_name() : "null_hash";
    }

    bool has_value() const noexcept { return pimpl != nullptr; }
    explicit operator bool() const noexcept { return has_value(); }
};

// =============================================================================
// Accumulator type erasure
// =============================================================================

template<typename T>
class any_accumulator {
private:
    struct concept_t : cloneable_concept<concept_t> {
        virtual void add(const T& value) = 0;
        virtual T result() const = 0;
        virtual void reset() = 0;
        virtual std::size_t count() const = 0;
    };

    template<typename Accumulator>
    struct model_t final : concept_t {
        mutable Accumulator acc;

        explicit model_t(Accumulator a) : acc(std::move(a)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(acc);
        }

        void add(const T& value) override {
            if constexpr (requires { acc.add(value); }) {
                acc.add(value);
            } else if constexpr (requires { acc(value); }) {
                acc(value);
            } else if constexpr (requires { acc += value; }) {
                acc += value;
            }
        }

        T result() const override {
            if constexpr (requires { acc.result(); }) {
                return acc.result();
            } else if constexpr (requires { acc.value(); }) {
                return acc.value();
            } else if constexpr (requires { static_cast<T>(acc); }) {
                return static_cast<T>(acc);
            } else {
                return T{};
            }
        }

        void reset() override {
            if constexpr (requires { acc.reset(); }) {
                acc.reset();
            } else if constexpr (requires { acc.clear(); }) {
                acc.clear();
            } else if constexpr (requires { acc = Accumulator{}; }) {
                acc = Accumulator{};
            }
        }

        std::size_t count() const override {
            if constexpr (requires { acc.count(); }) {
                return acc.count();
            } else if constexpr (requires { acc.size(); }) {
                return acc.size();
            } else {
                return 0;
            }
        }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_accumulator() = default;

    template<typename Accumulator>
    any_accumulator(Accumulator a)
        : pimpl(std::make_unique<model_t<Accumulator>>(std::move(a))) {}

    STEPANOV_TYPE_ERASURE_VALUE_SEMANTICS(any_accumulator)

    // Operations
    void add(const T& value) {
        if (!pimpl) throw std::runtime_error("Empty accumulator");
        pimpl->add(value);
    }

    any_accumulator& operator+=(const T& value) {
        add(value);
        return *this;
    }

    T result() const {
        if (!pimpl) throw std::runtime_error("Empty accumulator");
        return pimpl->result();
    }

    void reset() {
        if (pimpl) pimpl->reset();
    }

    std::size_t count() const {
        if (!pimpl) return 0;
        return pimpl->count();
    }

    bool has_value() const noexcept { return pimpl != nullptr; }
    explicit operator bool() const noexcept { return has_value(); }
};

// =============================================================================
// Matrix type erasure
// =============================================================================

template<typename T>
class any_matrix {
private:
    struct concept_t : cloneable_concept<concept_t> {
        virtual std::size_t rows() const = 0;
        virtual std::size_t cols() const = 0;
        virtual T get(std::size_t i, std::size_t j) const = 0;
        virtual void set(std::size_t i, std::size_t j, T value) = 0;
        virtual std::unique_ptr<concept_t> multiply(const concept_t& other) const = 0;
        virtual std::unique_ptr<concept_t> add(const concept_t& other) const = 0;
        virtual std::unique_ptr<concept_t> transpose() const = 0;
    };

    template<typename Matrix>
    struct model_t final : concept_t {
        mutable Matrix matrix;

        explicit model_t(Matrix m) : matrix(std::move(m)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(matrix);
        }

        std::size_t rows() const override {
            if constexpr (requires { matrix.rows(); }) {
                return matrix.rows();
            } else if constexpr (requires { matrix.size(); }) {
                return matrix.size();
            } else {
                return 0;
            }
        }

        std::size_t cols() const override {
            if constexpr (requires { matrix.cols(); }) {
                return matrix.cols();
            } else if constexpr (requires { matrix[0].size(); }) {
                return matrix[0].size();
            } else {
                return 0;
            }
        }

        T get(std::size_t i, std::size_t j) const override {
            if constexpr (requires { matrix(i, j); }) {
                return matrix(i, j);
            } else if constexpr (requires { matrix[i][j]; }) {
                return matrix[i][j];
            } else if constexpr (requires { matrix.at(i, j); }) {
                return matrix.at(i, j);
            } else {
                return T{};
            }
        }

        void set(std::size_t i, std::size_t j, T value) override {
            if constexpr (requires { matrix(i, j) = value; }) {
                matrix(i, j) = value;
            } else if constexpr (requires { matrix[i][j] = value; }) {
                matrix[i][j] = value;
            } else if constexpr (requires { matrix.at(i, j) = value; }) {
                matrix.at(i, j) = value;
            }
        }

        std::unique_ptr<concept_t> multiply(const concept_t& other) const override {
            // This would need specific implementation
            return clone(); // Placeholder
        }

        std::unique_ptr<concept_t> add(const concept_t& other) const override {
            // This would need specific implementation
            return clone(); // Placeholder
        }

        std::unique_ptr<concept_t> transpose() const override {
            if constexpr (requires { matrix.transpose(); }) {
                return std::make_unique<model_t>(matrix.transpose());
            } else {
                // Default implementation would transpose manually
                return clone(); // Placeholder
            }
        }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_matrix() = default;

    template<typename Matrix>
    any_matrix(Matrix m)
        : pimpl(std::make_unique<model_t<Matrix>>(std::move(m))) {}

    STEPANOV_TYPE_ERASURE_VALUE_SEMANTICS(any_matrix)

    // Operations
    std::size_t rows() const {
        if (!pimpl) return 0;
        return pimpl->rows();
    }

    std::size_t cols() const {
        if (!pimpl) return 0;
        return pimpl->cols();
    }

    T operator()(std::size_t i, std::size_t j) const {
        if (!pimpl) throw std::runtime_error("Empty matrix");
        return pimpl->get(i, j);
    }

    void set(std::size_t i, std::size_t j, T value) {
        if (!pimpl) throw std::runtime_error("Empty matrix");
        pimpl->set(i, j, value);
    }

    any_matrix operator*(const any_matrix& other) const {
        if (!pimpl || !other.pimpl) throw std::runtime_error("Empty matrix");
        any_matrix result;
        result.pimpl = pimpl->multiply(*other.pimpl);
        return result;
    }

    any_matrix operator+(const any_matrix& other) const {
        if (!pimpl || !other.pimpl) throw std::runtime_error("Empty matrix");
        any_matrix result;
        result.pimpl = pimpl->add(*other.pimpl);
        return result;
    }

    any_matrix transpose() const {
        if (!pimpl) throw std::runtime_error("Empty matrix");
        any_matrix result;
        result.pimpl = pimpl->transpose();
        return result;
    }

    bool has_value() const noexcept { return pimpl != nullptr; }
    explicit operator bool() const noexcept { return has_value(); }
};

// =============================================================================
// Range type erasure
// =============================================================================

template<typename T>
class any_range {
private:
    struct concept_t : cloneable_concept<concept_t> {
        virtual any_iterator<T> begin() = 0;
        virtual any_iterator<T> end() = 0;
        virtual std::size_t size() const = 0;
        virtual bool empty() const = 0;
    };

    template<typename Range>
    struct model_t final : concept_t {
        Range range;

        explicit model_t(Range r) : range(std::move(r)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(range);
        }

        any_iterator<T> begin() override {
            return any_iterator<T>(std::begin(range));
        }

        any_iterator<T> end() override {
            return any_iterator<T>(std::end(range));
        }

        std::size_t size() const override {
            if constexpr (requires { range.size(); }) {
                return range.size();
            } else if constexpr (requires { std::size(range); }) {
                return std::size(range);
            } else {
                return std::distance(std::begin(range), std::end(range));
            }
        }

        bool empty() const override {
            if constexpr (requires { range.empty(); }) {
                return range.empty();
            } else if constexpr (requires { std::empty(range); }) {
                return std::empty(range);
            } else {
                return std::begin(range) == std::end(range);
            }
        }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_range() = default;

    template<typename Range>
        requires std::ranges::range<Range>
    any_range(Range r)
        : pimpl(std::make_unique<model_t<Range>>(std::move(r))) {}

    STEPANOV_TYPE_ERASURE_VALUE_SEMANTICS(any_range)

    // Operations
    any_iterator<T> begin() {
        if (!pimpl) throw std::runtime_error("Empty range");
        return pimpl->begin();
    }

    any_iterator<T> end() {
        if (!pimpl) throw std::runtime_error("Empty range");
        return pimpl->end();
    }

    std::size_t size() const {
        if (!pimpl) return 0;
        return pimpl->size();
    }

    bool empty() const {
        if (!pimpl) return true;
        return pimpl->empty();
    }

    bool has_value() const noexcept { return pimpl != nullptr; }
    explicit operator bool() const noexcept { return has_value(); }
};

// =============================================================================
// Allocator type erasure
// =============================================================================

class any_allocator {
private:
    struct concept_t : cloneable_concept<concept_t> {
        virtual void* allocate(std::size_t size, std::size_t alignment) = 0;
        virtual void deallocate(void* ptr, std::size_t size) = 0;
        virtual std::size_t max_size() const = 0;
        virtual std::string name() const = 0;
    };

    template<typename Allocator>
    struct model_t final : concept_t {
        mutable Allocator allocator;

        explicit model_t(Allocator a) : allocator(std::move(a)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(allocator);
        }

        void* allocate(std::size_t size, std::size_t alignment) override {
            if constexpr (requires { allocator.allocate(size, alignment); }) {
                return allocator.allocate(size, alignment);
            } else if constexpr (requires { allocator.allocate(size); }) {
                return allocator.allocate(size);
            } else {
                return std::aligned_alloc(alignment, size);
            }
        }

        void deallocate(void* ptr, std::size_t size) override {
            if constexpr (requires { allocator.deallocate(ptr, size); }) {
                allocator.deallocate(ptr, size);
            } else if constexpr (requires { allocator.deallocate(ptr); }) {
                allocator.deallocate(ptr);
            } else {
                std::free(ptr);
            }
        }

        std::size_t max_size() const override {
            if constexpr (requires { allocator.max_size(); }) {
                return allocator.max_size();
            } else {
                return std::numeric_limits<std::size_t>::max();
            }
        }

        std::string name() const override {
            if constexpr (requires { allocator.name(); }) {
                return allocator.name();
            } else {
                return "unknown_allocator";
            }
        }
    };

    std::unique_ptr<concept_t> pimpl;

public:
    // Constructors
    any_allocator() = default;

    template<typename Allocator>
    any_allocator(Allocator a)
        : pimpl(std::make_unique<model_t<Allocator>>(std::move(a))) {}

    STEPANOV_TYPE_ERASURE_VALUE_SEMANTICS(any_allocator)

    // Operations
    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        if (!pimpl) throw std::runtime_error("Empty allocator");
        return pimpl->allocate(size, alignment);
    }

    void deallocate(void* ptr, std::size_t size) {
        if (!pimpl) throw std::runtime_error("Empty allocator");
        pimpl->deallocate(ptr, size);
    }

    std::size_t max_size() const {
        if (!pimpl) return 0;
        return pimpl->max_size();
    }

    std::string name() const {
        return pimpl ? pimpl->name() : "null_allocator";
    }

    bool has_value() const noexcept { return pimpl != nullptr; }
    explicit operator bool() const noexcept { return has_value(); }
};

// =============================================================================
// Helper utilities for type erasure
// =============================================================================

// Type trait to check if a type is type-erased
template<typename T>
struct is_type_erased : std::false_type {};

template<> struct is_type_erased<any_regular> : std::true_type {};
template<> struct is_type_erased<any_algebraic> : std::true_type {};
template<typename Sig> struct is_type_erased<any_function<Sig>> : std::true_type {};
template<typename T> struct is_type_erased<any_iterator<T>> : std::true_type {};
template<typename T> struct is_type_erased<any_interval_set<T>> : std::true_type {};
template<typename T> struct is_type_erased<any_integrator<T>> : std::true_type {};
template<typename T> struct is_type_erased<any_hash<T>> : std::true_type {};
template<typename T> struct is_type_erased<any_accumulator<T>> : std::true_type {};
template<typename T> struct is_type_erased<any_matrix<T>> : std::true_type {};
template<typename T> struct is_type_erased<any_range<T>> : std::true_type {};
template<> struct is_type_erased<any_allocator> : std::true_type {};
template<typename F, typename T> struct is_type_erased<any_codec<F, T>> : std::true_type {};
template<typename I, typename O> struct is_type_erased<any_text_processor<I, O>> : std::true_type {};
template<typename T> struct is_type_erased<any_differentiable<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_type_erased_v = is_type_erased<T>::value;

} // namespace stepanov