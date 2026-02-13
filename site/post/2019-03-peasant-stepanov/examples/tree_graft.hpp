#pragma once

/**
 * @file tree_graft.hpp
 * @brief Tree grafting monoid - fractal trees via repeated squaring
 *
 * Tree "multiplication": graft tree T₂ at every leaf of T₁
 *
 * This forms a monoid where:
 * - Identity: a single leaf
 * - Multiplication: grafting
 *
 * Mind-blowing insight:
 *   power(binary_branch, n) creates a complete binary tree with 2^n leaves!
 *   power(ternary_branch, n) creates a complete ternary tree with 3^n leaves!
 *
 * Each "squaring" doubles the depth, so we get exponential growth in O(log n).
 */

#include <memory>
#include <vector>
#include <compare>
#include <cstddef>

namespace peasant::examples {

// A simple tree node
template<typename Label = int>
struct tree_node {
    Label label;
    std::vector<std::shared_ptr<tree_node>> children;

    explicit tree_node(Label l) : label(l) {}
    tree_node(Label l, std::vector<std::shared_ptr<tree_node>> ch)
        : label(l), children(std::move(ch)) {}

    bool is_leaf() const { return children.empty(); }

    // Count leaves
    size_t count_leaves() const {
        if (is_leaf()) return 1;
        size_t count = 0;
        for (auto const& child : children) {
            count += child->count_leaves();
        }
        return count;
    }

    // Get depth
    size_t depth() const {
        if (is_leaf()) return 0;
        size_t max_child_depth = 0;
        for (auto const& child : children) {
            max_child_depth = std::max(max_child_depth, child->depth());
        }
        return 1 + max_child_depth;
    }
};

template<typename Label = int>
using tree_ptr = std::shared_ptr<tree_node<Label>>;

// Deep copy a tree
template<typename Label>
tree_ptr<Label> clone_tree(tree_ptr<Label> const& t) {
    if (!t) return nullptr;
    auto copy = std::make_shared<tree_node<Label>>(t->label);
    for (auto const& child : t->children) {
        copy->children.push_back(clone_tree(child));
    }
    return copy;
}

// Graft scion at every leaf of stock
template<typename Label>
tree_ptr<Label> graft(tree_ptr<Label> const& stock, tree_ptr<Label> const& scion) {
    if (!stock) return clone_tree(scion);
    if (!scion) return clone_tree(stock);

    if (stock->is_leaf()) {
        // Replace leaf with a clone of scion
        return clone_tree(scion);
    }

    // Recursively graft at children
    auto result = std::make_shared<tree_node<Label>>(stock->label);
    for (auto const& child : stock->children) {
        result->children.push_back(graft(child, scion));
    }
    return result;
}

// The tree monoid
template<typename Label = int>
struct tree_monoid {
    tree_ptr<Label> root;

    tree_monoid() : root(std::make_shared<tree_node<Label>>(Label{})) {}  // Single leaf
    explicit tree_monoid(tree_ptr<Label> r) : root(std::move(r)) {}

    bool operator==(tree_monoid const& o) const {
        // Compare by structure (simplified: just compare leaf counts and depth)
        if (!root && !o.root) return true;
        if (!root || !o.root) return false;
        return root->count_leaves() == o.root->count_leaves() &&
               root->depth() == o.root->depth();
    }

    auto operator<=>(tree_monoid const& o) const {
        size_t a_leaves = root ? root->count_leaves() : 0;
        size_t b_leaves = o.root ? o.root->count_leaves() : 0;
        if (a_leaves != b_leaves) return a_leaves <=> b_leaves;
        size_t a_depth = root ? root->depth() : 0;
        size_t b_depth = o.root ? o.root->depth() : 0;
        return a_depth <=> b_depth;
    }

    // Multiplication is grafting
    tree_monoid operator*(tree_monoid const& o) const {
        return tree_monoid{graft(root, o.root)};
    }

    // Required for algebraic concept
    tree_monoid operator+(tree_monoid const& o) const { return *this * o; }
    tree_monoid operator-() const { return *this; }

    size_t count_leaves() const { return root ? root->count_leaves() : 0; }
    size_t depth() const { return root ? root->depth() : 0; }
};

// ADL functions
template<typename L>
tree_monoid<L> zero(tree_monoid<L>) {
    // Identity: single leaf
    return tree_monoid<L>{std::make_shared<tree_node<L>>(L{})};
}

template<typename L>
tree_monoid<L> one(tree_monoid<L>) {
    return tree_monoid<L>{std::make_shared<tree_node<L>>(L{})};
}

template<typename L>
tree_monoid<L> twice(tree_monoid<L> const& t) { return t * t; }

template<typename L>
tree_monoid<L> half(tree_monoid<L> const& t) { return t; }

template<typename L>
bool even(tree_monoid<L> const&) { return true; }

template<typename L>
tree_monoid<L> increment(tree_monoid<L> const& t) { return t; }

template<typename L>
tree_monoid<L> decrement(tree_monoid<L> const& t) { return t; }

// Factory: Binary branching node (root with 2 leaf children)
template<typename Label = int>
tree_monoid<Label> binary_branch(Label root_label = Label{}) {
    auto root = std::make_shared<tree_node<Label>>(root_label);
    root->children.push_back(std::make_shared<tree_node<Label>>(Label{}));
    root->children.push_back(std::make_shared<tree_node<Label>>(Label{}));
    return tree_monoid<Label>{root};
}

// Factory: Ternary branching node
template<typename Label = int>
tree_monoid<Label> ternary_branch(Label root_label = Label{}) {
    auto root = std::make_shared<tree_node<Label>>(root_label);
    root->children.push_back(std::make_shared<tree_node<Label>>(Label{}));
    root->children.push_back(std::make_shared<tree_node<Label>>(Label{}));
    root->children.push_back(std::make_shared<tree_node<Label>>(Label{}));
    return tree_monoid<Label>{root};
}

// Factory: N-ary branching node
template<typename Label = int>
tree_monoid<Label> n_ary_branch(size_t n, Label root_label = Label{}) {
    auto root = std::make_shared<tree_node<Label>>(root_label);
    for (size_t i = 0; i < n; ++i) {
        root->children.push_back(std::make_shared<tree_node<Label>>(Label{}));
    }
    return tree_monoid<Label>{root};
}

} // namespace peasant::examples
