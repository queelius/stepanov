// persistent_vector_simple.hpp
// A simpler persistent vector implementation that actually compiles
// Uses path copying for structural sharing

#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include <initializer_list>

namespace stepanov::structures {

template<typename T>
class persistent_vector {
private:
    // Simple implementation using shared vector
    std::shared_ptr<std::vector<T>> data_;

    // Private constructor for internal use
    explicit persistent_vector(std::shared_ptr<std::vector<T>> data)
        : data_(std::move(data)) {}

public:
    // Default constructor
    persistent_vector() : data_(std::make_shared<std::vector<T>>()) {}

    // Initializer list constructor
    persistent_vector(std::initializer_list<T> init)
        : data_(std::make_shared<std::vector<T>>(init)) {}

    // Size operations
    [[nodiscard]] size_t size() const noexcept {
        return data_->size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return data_->empty();
    }

    // Element access
    [[nodiscard]] T operator[](size_t index) const {
        if (index >= size()) {
            throw std::out_of_range("Index out of bounds");
        }
        return (*data_)[index];
    }

    [[nodiscard]] T at(size_t index) const {
        return operator[](index);
    }

    // Push back - returns new vector
    [[nodiscard]] persistent_vector push_back(const T& value) const {
        auto new_data = std::make_shared<std::vector<T>>(*data_);
        new_data->push_back(value);
        return persistent_vector(new_data);
    }

    // Pop back - returns new vector
    [[nodiscard]] persistent_vector pop_back() const {
        if (empty()) {
            throw std::runtime_error("Cannot pop from empty vector");
        }
        auto new_data = std::make_shared<std::vector<T>>(*data_);
        new_data->pop_back();
        return persistent_vector(new_data);
    }

    // Functional update - returns new vector with updated value
    [[nodiscard]] persistent_vector assoc(size_t index, const T& value) const {
        if (index >= size()) {
            throw std::out_of_range("Cannot assoc beyond size");
        }
        auto new_data = std::make_shared<std::vector<T>>(*data_);
        (*new_data)[index] = value;
        return persistent_vector(new_data);
    }

    // Functional operations
    template<typename F>
    [[nodiscard]] persistent_vector map(F&& f) const {
        auto new_data = std::make_shared<std::vector<T>>();
        new_data->reserve(size());
        for (const auto& item : *data_) {
            new_data->push_back(f(item));
        }
        return persistent_vector(new_data);
    }

    template<typename F>
    [[nodiscard]] persistent_vector filter(F&& pred) const {
        auto new_data = std::make_shared<std::vector<T>>();
        for (const auto& item : *data_) {
            if (pred(item)) {
                new_data->push_back(item);
            }
        }
        return persistent_vector(new_data);
    }

    template<typename F, typename Init>
    [[nodiscard]] Init reduce(F&& f, Init init) const {
        for (const auto& item : *data_) {
            init = f(std::move(init), item);
        }
        return init;
    }

    // Iteration
    template<typename F>
    void for_each(F&& f) const {
        for (const auto& item : *data_) {
            f(item);
        }
    }

    // Slicing
    [[nodiscard]] persistent_vector slice(size_t start, size_t end) const {
        if (start > end || end > size()) {
            throw std::out_of_range("Invalid slice range");
        }
        auto new_data = std::make_shared<std::vector<T>>(
            data_->begin() + start, data_->begin() + end);
        return persistent_vector(new_data);
    }

    // Concatenation
    [[nodiscard]] persistent_vector concat(const persistent_vector& other) const {
        auto new_data = std::make_shared<std::vector<T>>(*data_);
        new_data->insert(new_data->end(), other.data_->begin(), other.data_->end());
        return persistent_vector(new_data);
    }

    // Transient builder for efficient batch updates
    class transient {
        friend class persistent_vector;
        std::shared_ptr<std::vector<T>> data_;
        bool is_persistent_ = false;

        explicit transient(const persistent_vector& v)
            : data_(std::make_shared<std::vector<T>>(*v.data_)) {}

    public:
        transient& push_back(const T& value) {
            if (is_persistent_) {
                throw std::runtime_error("Cannot modify persistent transient");
            }
            data_->push_back(value);
            return *this;
        }

        transient& assoc(size_t index, const T& value) {
            if (is_persistent_) {
                throw std::runtime_error("Cannot modify persistent transient");
            }
            if (index >= data_->size()) {
                throw std::out_of_range("Index out of bounds");
            }
            (*data_)[index] = value;
            return *this;
        }

        persistent_vector persistent() {
            is_persistent_ = true;
            return persistent_vector(data_);
        }
    };

    [[nodiscard]] transient make_transient() const {
        return transient(*this);
    }
};

// Deduction guide
template<typename T>
persistent_vector(std::initializer_list<T>) -> persistent_vector<T>;

} // namespace stepanov::structures