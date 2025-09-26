// persistent_vector.hpp
// A persistent vector implementation inspired by Clojure's PersistentVector
// Uses a tree structure with wide branching factor for O(log₃₂ n) operations
// All operations return new versions without modifying originals

#pragma once

#include <memory>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <concepts>
#include <bit>

namespace stepanov::structures {

// Configuration
constexpr size_t BITS_PER_LEVEL = 5;  // 32-way branching
constexpr size_t BRANCH_FACTOR = 1 << BITS_PER_LEVEL;
constexpr size_t LEVEL_MASK = BRANCH_FACTOR - 1;

template<typename T>
class persistent_vector {
private:
    struct node {
        using ptr = std::shared_ptr<node>;
        std::array<ptr, BRANCH_FACTOR> children{};
        size_t ref_count = 1;

        node() = default;
        node(const node& other) : children(other.children) {}
    };

    struct leaf {
        using ptr = std::shared_ptr<leaf>;
        std::array<T, BRANCH_FACTOR> values;
        size_t count = 0;

        leaf() = default;
        leaf(const leaf& other) : values(other.values), count(other.count) {}
    };

    using node_ptr = typename node::ptr;
    using leaf_ptr = typename leaf::ptr;

    node_ptr root_;
    std::vector<leaf_ptr> tail_;  // Tail optimization for recent additions
    size_t size_ = 0;
    size_t shift_ = BITS_PER_LEVEL;  // Height of the tree

    // Helper: Calculate tree level for index
    static size_t level_for(size_t index, size_t shift) {
        return (index >> shift) & LEVEL_MASK;
    }

    // Path copying for structural sharing
    template<typename NodeType>
    std::shared_ptr<NodeType> copy_if_shared(std::shared_ptr<NodeType>& node_ptr) {
        if (!node_ptr) {
            node_ptr = std::make_shared<NodeType>();
        } else if (node_ptr.use_count() > 1) {
            node_ptr = std::make_shared<NodeType>(*node_ptr);
        }
        return node_ptr;
    }

    // Recursive helper for getting value at index
    T do_get(size_t index, size_t level, const node_ptr& node) const {
        if (level == 0) {
            // We're at leaf level
            auto leaf_node = std::static_pointer_cast<leaf>(node);
            return leaf_node->values[index & LEVEL_MASK];
        }

        size_t child_idx = level_for(index, level);
        if (!node->children[child_idx]) {
            throw std::out_of_range("Index out of bounds");
        }

        return do_get(index, level - BITS_PER_LEVEL, node->children[child_idx]);
    }

    // Recursive helper for setting value at index (with path copying)
    node_ptr do_assoc(size_t index, const T& value, size_t level, node_ptr node) const {
        node = copy_if_shared(node);

        if (level == 0) {
            // Leaf level
            auto leaf_node = std::static_pointer_cast<leaf>(node);
            leaf_node = std::make_shared<leaf>(*leaf_node);
            leaf_node->values[index & LEVEL_MASK] = value;
            if ((index & LEVEL_MASK) >= leaf_node->count) {
                leaf_node->count = (index & LEVEL_MASK) + 1;
            }
            return std::static_pointer_cast<node>(leaf_node);
        }

        size_t child_idx = level_for(index, level);
        if (!node->children[child_idx]) {
            node->children[child_idx] = std::make_shared<node>();
        }

        node->children[child_idx] = do_assoc(index, value, level - BITS_PER_LEVEL,
                                            node->children[child_idx]);
        return node;
    }

    // Expand tree height when needed
    node_ptr push_tail(size_t level, node_ptr parent, leaf_ptr tail_node) const {
        parent = copy_if_shared(parent);

        size_t subidx = level_for(size_ - 1, level);

        if (level == BITS_PER_LEVEL) {
            // Direct child is a leaf
            parent->children[subidx] = tail_node;
        } else {
            auto child = parent->children[subidx];
            if (child) {
                // Recurse into existing child
                parent->children[subidx] = push_tail(level - BITS_PER_LEVEL, child, tail_node);
            } else {
                // Create new path
                parent->children[subidx] = new_path(level - BITS_PER_LEVEL, tail_node);
            }
        }

        return parent;
    }

    // Create new path to leaf
    node_ptr new_path(size_t level, leaf_ptr leaf_node) const {
        if (level == 0) {
            return leaf_node;
        }

        auto new_node = std::make_shared<node>();
        new_node->children[0] = new_path(level - BITS_PER_LEVEL, leaf_node);
        return new_node;
    }

public:
    // Constructors
    persistent_vector() : root_(std::make_shared<node>()) {}

    persistent_vector(std::initializer_list<T> init) : persistent_vector() {
        for (const auto& val : init) {
            *this = push_back(val);
        }
    }

    // Size operations
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    // Element access (O(log₃₂ n))
    [[nodiscard]] T operator[](size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Index out of bounds");
        }

        // Check if in tail
        if (!tail_.empty() && index >= size_ - tail_.back()->count) {
            size_t tail_idx = index - (size_ - tail_.back()->count);
            return tail_.back()->values[tail_idx];
        }

        return do_get(index, shift_, root_);
    }

    [[nodiscard]] T at(size_t index) const {
        return operator[](index);
    }

    // Functional update - returns new vector with updated value
    [[nodiscard]] persistent_vector assoc(size_t index, const T& value) const {
        if (index >= size_) {
            throw std::out_of_range("Cannot assoc beyond size");
        }

        persistent_vector result(*this);

        // If updating tail
        if (!tail_.empty() && index >= size_ - tail_.back()->count) {
            size_t tail_idx = index - (size_ - tail_.back()->count);
            result.tail_.back() = std::make_shared<leaf>(*tail_.back());
            result.tail_.back()->values[tail_idx] = value;
            return result;
        }

        result.root_ = do_assoc(index, value, shift_, root_);
        return result;
    }

    // Push back - returns new vector with added element
    [[nodiscard]] persistent_vector push_back(const T& value) const {
        persistent_vector result(*this);

        // Tail room optimization
        if (tail_.empty() || tail_.back()->count >= BRANCH_FACTOR) {
            // Need new tail
            auto new_tail = std::make_shared<leaf>();
            new_tail->values[0] = value;
            new_tail->count = 1;

            if (!tail_.empty()) {
                // Push old tail into tree
                if ((size_ & ~LEVEL_MASK) == (1u << (shift_ + BITS_PER_LEVEL))) {
                    // Need to increase tree height
                    auto new_root = std::make_shared<node>();
                    new_root->children[0] = root_;
                    new_root->children[1] = new_path(shift_, tail_.back());
                    result.root_ = new_root;
                    result.shift_ += BITS_PER_LEVEL;
                } else {
                    result.root_ = push_tail(shift_, root_, tail_.back());
                }
            }

            result.tail_.clear();
            result.tail_.push_back(new_tail);
        } else {
            // Add to existing tail
            result.tail_.back() = std::make_shared<leaf>(*tail_.back());
            result.tail_.back()->values[result.tail_.back()->count++] = value;
        }

        result.size_++;
        return result;
    }

    // Pop back - returns new vector without last element
    [[nodiscard]] persistent_vector pop_back() const {
        if (empty()) {
            throw std::runtime_error("Cannot pop from empty vector");
        }

        persistent_vector result(*this);
        result.size_--;

        // Handle tail
        if (!tail_.empty() && tail_.back()->count > 1) {
            result.tail_.back() = std::make_shared<leaf>(*tail_.back());
            result.tail_.back()->count--;
        } else {
            // Need to get new tail from tree or shrink tree
            result.tail_.clear();
            // Tree shrinking logic would go here for completeness
        }

        return result;
    }

    // Transient builder for efficient batch updates
    class transient {
        friend class persistent_vector;
        persistent_vector vec_;
        bool is_persistent_ = false;

        explicit transient(const persistent_vector& v) : vec_(v) {}

    public:
        transient& push_back(const T& value) {
            if (is_persistent_) {
                throw std::runtime_error("Cannot modify persistent transient");
            }
            vec_ = vec_.push_back(value);
            return *this;
        }

        transient& assoc(size_t index, const T& value) {
            if (is_persistent_) {
                throw std::runtime_error("Cannot modify persistent transient");
            }
            vec_ = vec_.assoc(index, value);
            return *this;
        }

        persistent_vector persistent() {
            is_persistent_ = true;
            return vec_;
        }
    };

    [[nodiscard]] transient make_transient() const {
        return transient(*this);
    }

    // Iteration support
    template<typename F>
    void for_each(F&& f) const {
        for (size_t i = 0; i < size_; ++i) {
            f(operator[](i));
        }
    }

    // Functional operations
    template<typename F>
    [[nodiscard]] persistent_vector map(F&& f) const {
        auto trans = make_transient();
        for (size_t i = 0; i < size_; ++i) {
            trans.assoc(i, f(operator[](i)));
        }
        return trans.persistent();
    }

    template<typename F, typename Init>
    [[nodiscard]] Init reduce(F&& f, Init init) const {
        for (size_t i = 0; i < size_; ++i) {
            init = f(std::move(init), operator[](i));
        }
        return init;
    }

    template<typename F>
    [[nodiscard]] persistent_vector filter(F&& pred) const {
        persistent_vector result;
        for (size_t i = 0; i < size_; ++i) {
            if (pred(operator[](i))) {
                result = result.push_back(operator[](i));
            }
        }
        return result;
    }

    // Slicing - returns sub-vector
    [[nodiscard]] persistent_vector slice(size_t start, size_t end) const {
        if (start > end || end > size_) {
            throw std::out_of_range("Invalid slice range");
        }

        persistent_vector result;
        for (size_t i = start; i < end; ++i) {
            result = result.push_back(operator[](i));
        }
        return result;
    }

    // Concatenation
    [[nodiscard]] persistent_vector concat(const persistent_vector& other) const {
        auto trans = make_transient();
        for (size_t i = 0; i < other.size(); ++i) {
            trans.push_back(other[i]);
        }
        return trans.persistent();
    }
};

// Deduction guide
template<typename T>
persistent_vector(std::initializer_list<T>) -> persistent_vector<T>;

} // namespace stepanov::structures