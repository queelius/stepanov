#pragma once

/**
 * @file any_regular.hpp
 * @brief Type Erasure for Regular Types
 *
 * Implements Sean Parent's "Runtime Polymorphism" pattern: value semantics
 * with runtime polymorphism, no inheritance required from the stored type.
 *
 * The pattern:
 * 1. concept_t: Abstract interface defining required operations
 * 2. model_t<T>: Concrete implementation wrapping any type T
 * 3. Wrapper class holding unique_ptr<concept_t>
 *
 * This gives us:
 * - Value semantics (copyable, assignable)
 * - No intrusive base class required
 * - Type-safe heterogeneous containers
 *
 * A "regular type" satisfies: default constructible, copyable, equality
 * comparable. This is the minimal requirement for value semantics.
 */

#include <concepts>
#include <memory>
#include <typeindex>
#include <functional>

namespace type_erasure {

/// Concept for regular types (value semantics)
template<typename T>
concept regular = std::regular<T>;

/**
 * any_regular: Type-erased container for any regular type
 *
 * Can hold any type that supports copy and equality comparison.
 * Provides value semantics: copying an any_regular copies its contents.
 */
class any_regular {
    // ==========================================================================
    // The concept/model pattern
    // ==========================================================================

    struct concept_t {
        virtual ~concept_t() = default;
        virtual std::unique_ptr<concept_t> clone() const = 0;
        virtual bool equals(concept_t const& other) const = 0;
        virtual std::type_index type() const = 0;
        virtual void const* data() const = 0;
    };

    template<regular T>
    struct model_t final : concept_t {
        T value;

        explicit model_t(T v) : value(std::move(v)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(value);
        }

        bool equals(concept_t const& other) const override {
            if (type() != other.type()) return false;
            return value == static_cast<model_t const&>(other).value;
        }

        std::type_index type() const override {
            return typeid(T);
        }

        void const* data() const override {
            return &value;
        }
    };

    std::unique_ptr<concept_t> self_;

public:
    // ==========================================================================
    // Constructors
    // ==========================================================================

    any_regular() = default;

    template<typename T>
        requires (!std::same_as<std::remove_cvref_t<T>, any_regular>) && regular<T>
    any_regular(T value) : self_(std::make_unique<model_t<T>>(std::move(value))) {}

    // Copy (deep copy via clone)
    any_regular(any_regular const& other)
        : self_(other.self_ ? other.self_->clone() : nullptr) {}

    any_regular(any_regular&&) noexcept = default;

    // ==========================================================================
    // Assignment
    // ==========================================================================

    any_regular& operator=(any_regular const& other) {
        if (this != &other) {
            self_ = other.self_ ? other.self_->clone() : nullptr;
        }
        return *this;
    }

    any_regular& operator=(any_regular&&) noexcept = default;

    template<typename T>
        requires (!std::same_as<std::remove_cvref_t<T>, any_regular>) && regular<T>
    any_regular& operator=(T value) {
        self_ = std::make_unique<model_t<T>>(std::move(value));
        return *this;
    }

    // ==========================================================================
    // Observers
    // ==========================================================================

    bool has_value() const { return self_ != nullptr; }
    explicit operator bool() const { return has_value(); }

    std::type_index type() const {
        return self_ ? self_->type() : typeid(void);
    }

    template<typename T>
    bool holds() const {
        return self_ && self_->type() == typeid(T);
    }

    // ==========================================================================
    // Access
    // ==========================================================================

    template<typename T>
    T const* get() const {
        if (!holds<T>()) return nullptr;
        return static_cast<T const*>(self_->data());
    }

    template<typename T>
    T const& get_or(T const& default_value) const {
        auto const* p = get<T>();
        return p ? *p : default_value;
    }

    // ==========================================================================
    // Comparison
    // ==========================================================================

    friend bool operator==(any_regular const& a, any_regular const& b) {
        if (!a.self_ && !b.self_) return true;
        if (!a.self_ || !b.self_) return false;
        return a.self_->equals(*b.self_);
    }

    friend bool operator!=(any_regular const& a, any_regular const& b) {
        return !(a == b);
    }
};

/**
 * any_with<Ops...>: Type erasure with custom operations
 *
 * Extends the pattern to support additional operations beyond equality.
 * Each Op is a function object type that defines an operation.
 */
template<typename... Ops>
class any_with {
    struct concept_t {
        virtual ~concept_t() = default;
        virtual std::unique_ptr<concept_t> clone() const = 0;
        virtual std::type_index type() const = 0;
    };

    template<typename T>
    struct model_t final : concept_t {
        T value;

        explicit model_t(T v) : value(std::move(v)) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(value);
        }

        std::type_index type() const override {
            return typeid(T);
        }
    };

    std::unique_ptr<concept_t> self_;

public:
    any_with() = default;

    template<typename T>
    any_with(T value) : self_(std::make_unique<model_t<T>>(std::move(value))) {}

    any_with(any_with const& other)
        : self_(other.self_ ? other.self_->clone() : nullptr) {}

    any_with(any_with&&) noexcept = default;

    any_with& operator=(any_with const& other) {
        if (this != &other) {
            self_ = other.self_ ? other.self_->clone() : nullptr;
        }
        return *this;
    }

    any_with& operator=(any_with&&) noexcept = default;

    bool has_value() const { return self_ != nullptr; }
};

} // namespace type_erasure
