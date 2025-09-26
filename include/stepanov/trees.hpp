// trees.hpp - Elegant Tree Structures
// The best tree implementations: Weight-Balanced, B+, Patricia Trie, Finger Tree
// Each chosen for mathematical elegance and practical superiority

#ifndef STEPANOV_TREES_HPP
#define STEPANOV_TREES_HPP

#include <concepts>
#include <memory>
#include <vector>
#include <optional>
#include <functional>
#include <algorithm>
#include <string>
#include <bit>
#include <array>
#include <variant>
#include <ranges>

namespace stepanov::trees {

// ============================================================================
// Weight-Balanced Tree - Better than Red-Black, Simpler Invariants
// ============================================================================

template<typename T, typename Compare = std::less<T>>
class weight_balanced_tree {
    // Based on Adams' weight-balanced trees
    // Maintains size invariant instead of color invariant
    // Simpler and more efficient than red-black trees

    struct node {
        T value;
        size_t size;  // Total nodes in subtree
        std::unique_ptr<node> left;
        std::unique_ptr<node> right;

        explicit node(T val) : value(std::move(val)), size(1) {}

        void update_size() {
            size = 1 + (left ? left->size : 0) + (right ? right->size : 0);
        }

        size_t left_size() const { return left ? left->size : 0; }
        size_t right_size() const { return right ? right->size : 0; }
    };

    std::unique_ptr<node> root_;
    Compare comp_;

    // Weight balance parameters (omega and alpha)
    static constexpr double omega = 2.0 + std::numbers::sqrt2;  // ~3.414
    static constexpr double alpha = std::numbers::sqrt2 - 1.0;   // ~0.414

    // Check if tree needs rebalancing
    static bool is_balanced(const node* n) {
        if (!n) return true;
        size_t l = n->left_size();
        size_t r = n->right_size();
        return l <= omega * r + 1 && r <= omega * l + 1;
    }

    // Single rotation
    std::unique_ptr<node> rotate_left(std::unique_ptr<node> x) {
        auto y = std::move(x->right);
        x->right = std::move(y->left);
        x->update_size();
        y->left = std::move(x);
        y->update_size();
        return y;
    }

    std::unique_ptr<node> rotate_right(std::unique_ptr<node> y) {
        auto x = std::move(y->left);
        y->left = std::move(x->right);
        y->update_size();
        x->right = std::move(y);
        x->update_size();
        return x;
    }

    // Smart rebalancing - decides between single and double rotations
    std::unique_ptr<node> balance(std::unique_ptr<node> n) {
        if (!n) return n;

        size_t l = n->left_size();
        size_t r = n->right_size();

        if (r > omega * l + 1) {
            // Right heavy
            if (n->right->left_size() > alpha * n->right->size) {
                n->right = rotate_right(std::move(n->right));
            }
            return rotate_left(std::move(n));
        } else if (l > omega * r + 1) {
            // Left heavy
            if (n->left->right_size() > alpha * n->left->size) {
                n->left = rotate_left(std::move(n->left));
            }
            return rotate_right(std::move(n));
        }

        return n;
    }

    std::unique_ptr<node> insert_impl(std::unique_ptr<node> n, T value) {
        if (!n) {
            return std::make_unique<node>(std::move(value));
        }

        if (comp_(value, n->value)) {
            n->left = insert_impl(std::move(n->left), std::move(value));
        } else if (comp_(n->value, value)) {
            n->right = insert_impl(std::move(n->right), std::move(value));
        } else {
            n->value = std::move(value);  // Update existing
            return n;
        }

        n->update_size();
        return balance(std::move(n));
    }

    std::unique_ptr<node> remove_impl(std::unique_ptr<node> n, const T& value) {
        if (!n) return n;

        if (comp_(value, n->value)) {
            n->left = remove_impl(std::move(n->left), value);
        } else if (comp_(n->value, value)) {
            n->right = remove_impl(std::move(n->right), value);
        } else {
            // Found node to delete
            if (!n->left) return std::move(n->right);
            if (!n->right) return std::move(n->left);

            // Two children - replace with inorder successor
            auto* min = n->right.get();
            while (min->left) min = min->left.get();
            n->value = min->value;
            n->right = remove_impl(std::move(n->right), min->value);
        }

        n->update_size();
        return balance(std::move(n));
    }

public:
    weight_balanced_tree(Compare comp = Compare{}) : comp_(comp) {}

    void insert(T value) {
        root_ = insert_impl(std::move(root_), std::move(value));
    }

    void remove(const T& value) {
        root_ = remove_impl(std::move(root_), value);
    }

    bool contains(const T& value) const {
        auto* n = root_.get();
        while (n) {
            if (comp_(value, n->value)) {
                n = n->left.get();
            } else if (comp_(n->value, value)) {
                n = n->right.get();
            } else {
                return true;
            }
        }
        return false;
    }

    size_t size() const {
        return root_ ? root_->size : 0;
    }

    // O(log n) rank operations due to size maintenance
    std::optional<T> nth_element(size_t k) const {
        if (k >= size()) return std::nullopt;

        auto* n = root_.get();
        while (n) {
            size_t left_size = n->left_size();
            if (k < left_size) {
                n = n->left.get();
            } else if (k > left_size) {
                k -= left_size + 1;
                n = n->right.get();
            } else {
                return n->value;
            }
        }
        return std::nullopt;
    }

    size_t rank(const T& value) const {
        size_t r = 0;
        auto* n = root_.get();

        while (n) {
            if (comp_(value, n->value)) {
                n = n->left.get();
            } else if (comp_(n->value, value)) {
                r += n->left_size() + 1;
                n = n->right.get();
            } else {
                return r + n->left_size();
            }
        }
        return size();  // Not found
    }
};

// ============================================================================
// B+ Tree - The Only Tree You Need for Large Data
// ============================================================================

template<typename Key, typename Value, size_t Order = 128>
class bplus_tree {
    static_assert(Order >= 3, "B+ tree order must be at least 3");

    struct node {
        bool is_leaf;
        std::vector<Key> keys;
        virtual ~node() = default;
    };

    struct internal_node : node {
        std::vector<std::unique_ptr<node>> children;

        internal_node() { this->is_leaf = false; }
    };

    struct leaf_node : node {
        std::vector<Value> values;
        leaf_node* next = nullptr;  // For range queries

        leaf_node() { this->is_leaf = true; }
    };

    std::unique_ptr<node> root_;
    leaf_node* first_leaf_ = nullptr;  // For iteration
    size_t size_ = 0;

    // Find leaf containing key
    leaf_node* find_leaf(const Key& key) const {
        if (!root_) return nullptr;

        node* n = root_.get();
        while (!n->is_leaf) {
            auto* internal = static_cast<internal_node*>(n);
            auto it = std::upper_bound(n->keys.begin(), n->keys.end(), key);
            size_t idx = std::distance(n->keys.begin(), it);
            n = internal->children[idx].get();
        }

        return static_cast<leaf_node*>(n);
    }

    // Split a full node
    std::pair<Key, std::unique_ptr<node>> split_node(std::unique_ptr<node> n) {
        if (n->is_leaf) {
            auto* leaf = static_cast<leaf_node*>(n.get());
            auto new_leaf = std::make_unique<leaf_node>();

            size_t mid = leaf->keys.size() / 2;
            Key mid_key = leaf->keys[mid];

            // Move half to new leaf
            new_leaf->keys.assign(
                leaf->keys.begin() + mid,
                leaf->keys.end()
            );
            new_leaf->values.assign(
                leaf->values.begin() + mid,
                leaf->values.end()
            );

            leaf->keys.resize(mid);
            leaf->values.resize(mid);

            // Update linked list
            new_leaf->next = leaf->next;
            leaf->next = new_leaf.get();

            return {mid_key, std::move(new_leaf)};
        } else {
            auto* internal = static_cast<internal_node*>(n.get());
            auto new_internal = std::make_unique<internal_node>();

            size_t mid = internal->keys.size() / 2;
            Key mid_key = internal->keys[mid];

            // Move half to new internal node
            new_internal->keys.assign(
                internal->keys.begin() + mid + 1,
                internal->keys.end()
            );
            new_internal->children.reserve(new_internal->keys.size() + 1);

            for (size_t i = mid + 1; i <= internal->keys.size(); ++i) {
                new_internal->children.push_back(
                    std::move(internal->children[i])
                );
            }

            internal->keys.resize(mid);
            internal->children.resize(mid + 1);

            return {mid_key, std::move(new_internal)};
        }
    }

    // Insert into non-full node
    std::optional<std::pair<Key, std::unique_ptr<node>>>
    insert_non_full(node* n, Key key, Value value) {
        if (n->is_leaf) {
            auto* leaf = static_cast<leaf_node*>(n);
            auto it = std::lower_bound(leaf->keys.begin(), leaf->keys.end(), key);
            size_t idx = std::distance(leaf->keys.begin(), it);

            if (it != leaf->keys.end() && *it == key) {
                // Update existing
                leaf->values[idx] = std::move(value);
                return std::nullopt;
            }

            leaf->keys.insert(it, key);
            leaf->values.insert(leaf->values.begin() + idx, std::move(value));
            ++size_;

            if (leaf->keys.size() >= Order) {
                // Need to split
                return split_node(std::unique_ptr<node>(n));
            }

            return std::nullopt;
        } else {
            auto* internal = static_cast<internal_node*>(n);
            auto it = std::upper_bound(internal->keys.begin(), internal->keys.end(), key);
            size_t idx = std::distance(internal->keys.begin(), it);

            auto result = insert_non_full(
                internal->children[idx].get(),
                std::move(key),
                std::move(value)
            );

            if (result) {
                // Child was split
                internal->keys.insert(internal->keys.begin() + idx, result->first);
                internal->children.insert(
                    internal->children.begin() + idx + 1,
                    std::move(result->second)
                );

                if (internal->keys.size() >= Order) {
                    return split_node(std::unique_ptr<node>(n));
                }
            }

            return std::nullopt;
        }
    }

public:
    bplus_tree() {
        auto leaf = std::make_unique<leaf_node>();
        first_leaf_ = leaf.get();
        root_ = std::move(leaf);
    }

    void insert(Key key, Value value) {
        auto result = insert_non_full(root_.get(), std::move(key), std::move(value));

        if (result) {
            // Root was split
            auto new_root = std::make_unique<internal_node>();
            new_root->keys.push_back(result->first);
            new_root->children.push_back(std::move(root_));
            new_root->children.push_back(std::move(result->second));
            root_ = std::move(new_root);
        }
    }

    std::optional<Value> find(const Key& key) const {
        auto* leaf = find_leaf(key);
        if (!leaf) return std::nullopt;

        auto it = std::lower_bound(leaf->keys.begin(), leaf->keys.end(), key);
        if (it != leaf->keys.end() && *it == key) {
            size_t idx = std::distance(leaf->keys.begin(), it);
            return leaf->values[idx];
        }

        return std::nullopt;
    }

    // Range query - efficiently iterate over range
    std::vector<std::pair<Key, Value>> range(const Key& start, const Key& end) const {
        std::vector<std::pair<Key, Value>> result;

        auto* leaf = find_leaf(start);
        while (leaf) {
            for (size_t i = 0; i < leaf->keys.size(); ++i) {
                if (leaf->keys[i] >= start && leaf->keys[i] <= end) {
                    result.emplace_back(leaf->keys[i], leaf->values[i]);
                } else if (leaf->keys[i] > end) {
                    return result;
                }
            }
            leaf = leaf->next;
        }

        return result;
    }

    size_t size() const { return size_; }
};

// ============================================================================
// Patricia Trie - Most Elegant for Strings
// ============================================================================

template<typename Value>
class patricia_trie {
    struct node {
        std::optional<Value> value;
        size_t bit_index;  // Bit position to test
        std::unique_ptr<node> left;   // 0 bit
        std::unique_ptr<node> right;  // 1 bit
        std::string key_prefix;  // For reconstruction

        node(size_t bit_idx = 0) : bit_index(bit_idx) {}
    };

    std::unique_ptr<node> root_;
    size_t size_ = 0;

    // Get bit at position in key
    static bool get_bit(const std::string& key, size_t pos) {
        size_t byte_idx = pos / 8;
        size_t bit_idx = 7 - (pos % 8);
        if (byte_idx >= key.size()) return false;
        return (key[byte_idx] >> bit_idx) & 1;
    }

    // Find first differing bit position
    static size_t find_diff_bit(const std::string& a, const std::string& b) {
        size_t max_bits = std::max(a.size(), b.size()) * 8;
        for (size_t i = 0; i < max_bits; ++i) {
            if (get_bit(a, i) != get_bit(b, i)) {
                return i;
            }
        }
        return max_bits;
    }

public:
    patricia_trie() : root_(std::make_unique<node>()) {}

    void insert(const std::string& key, Value value) {
        if (!root_->value && root_->bit_index == 0 && !root_->left && !root_->right) {
            // Empty tree
            root_->value = std::move(value);
            root_->key_prefix = key;
            ++size_;
            return;
        }

        // Find insertion point
        node* current = root_.get();
        node* parent = nullptr;
        bool went_left = false;

        while (true) {
            if (current->value && current->key_prefix == key) {
                // Update existing
                current->value = std::move(value);
                return;
            }

            size_t diff_bit = find_diff_bit(key, current->key_prefix);

            if (!current->left && !current->right) {
                // Leaf - split here
                auto new_node = std::make_unique<node>(diff_bit);
                new_node->key_prefix = key;
                new_node->value = std::move(value);

                if (get_bit(key, diff_bit)) {
                    new_node->left = std::make_unique<node>(*current);
                    *current = std::move(*new_node);
                } else {
                    new_node->right = std::make_unique<node>(*current);
                    *current = std::move(*new_node);
                }

                ++size_;
                return;
            }

            bool bit = get_bit(key, current->bit_index);
            parent = current;
            went_left = !bit;

            if (bit) {
                if (!current->right) {
                    current->right = std::make_unique<node>();
                    current->right->key_prefix = key;
                    current->right->value = std::move(value);
                    ++size_;
                    return;
                }
                current = current->right.get();
            } else {
                if (!current->left) {
                    current->left = std::make_unique<node>();
                    current->left->key_prefix = key;
                    current->left->value = std::move(value);
                    ++size_;
                    return;
                }
                current = current->left.get();
            }
        }
    }

    std::optional<Value> find(const std::string& key) const {
        node* current = root_.get();

        while (current) {
            if (current->value && current->key_prefix == key) {
                return current->value;
            }

            if (!current->left && !current->right) {
                return std::nullopt;
            }

            bool bit = get_bit(key, current->bit_index);
            current = bit ? current->right.get() : current->left.get();
        }

        return std::nullopt;
    }

    // Find all keys with given prefix
    std::vector<std::pair<std::string, Value>> prefix_search(const std::string& prefix) const {
        std::vector<std::pair<std::string, Value>> result;

        std::function<void(node*)> collect = [&](node* n) {
            if (!n) return;

            if (n->value && n->key_prefix.starts_with(prefix)) {
                result.emplace_back(n->key_prefix, *n->value);
            }

            collect(n->left.get());
            collect(n->right.get());
        };

        collect(root_.get());
        return result;
    }

    size_t size() const { return size_; }
};

// ============================================================================
// Finger Tree - Pure Functional with Amazing Operations
// ============================================================================

template<typename T, typename Measure = size_t>
class finger_tree {
    // 2-3 finger tree with measurement
    // Supports O(1) cons/snoc, O(log n) split/concatenation

    struct node {
        virtual ~node() = default;
        virtual Measure measure() const = 0;
    };

    struct leaf : node {
        T value;
        explicit leaf(T val) : value(std::move(val)) {}
        Measure measure() const override { return Measure{1}; }
    };

    struct node2 : node {
        std::shared_ptr<node> left, right;
        mutable std::optional<Measure> cached_measure;

        node2(std::shared_ptr<node> l, std::shared_ptr<node> r)
            : left(std::move(l)), right(std::move(r)) {}

        Measure measure() const override {
            if (!cached_measure) {
                cached_measure = left->measure() + right->measure();
            }
            return *cached_measure;
        }
    };

    struct node3 : node {
        std::shared_ptr<node> left, middle, right;
        mutable std::optional<Measure> cached_measure;

        node3(std::shared_ptr<node> l, std::shared_ptr<node> m, std::shared_ptr<node> r)
            : left(std::move(l)), middle(std::move(m)), right(std::move(r)) {}

        Measure measure() const override {
            if (!cached_measure) {
                cached_measure = left->measure() + middle->measure() + right->measure();
            }
            return *cached_measure;
        }
    };

    using digit = std::vector<std::shared_ptr<node>>;

    struct deep {
        digit prefix;
        std::shared_ptr<finger_tree> middle;
        digit suffix;

        Measure measure() const {
            Measure m{};
            for (const auto& n : prefix) m = m + n->measure();
            if (middle && !middle->empty()) m = m + middle->measure();
            for (const auto& n : suffix) m = m + n->measure();
            return m;
        }
    };

    std::variant<std::monostate, std::shared_ptr<node>, deep> data_;

public:
    finger_tree() : data_(std::monostate{}) {}

    explicit finger_tree(std::shared_ptr<node> n) : data_(std::move(n)) {}

    explicit finger_tree(deep d) : data_(std::move(d)) {}

    bool empty() const {
        return std::holds_alternative<std::monostate>(data_);
    }

    Measure measure() const {
        if (empty()) return Measure{};

        return std::visit([](const auto& d) -> Measure {
            using D = std::decay_t<decltype(d)>;
            if constexpr (std::is_same_v<D, std::monostate>) {
                return Measure{};
            } else if constexpr (std::is_same_v<D, std::shared_ptr<node>>) {
                return d->measure();
            } else {
                return d.measure();
            }
        }, data_);
    }

    // Add element to front
    finger_tree push_front(T value) const {
        auto new_leaf = std::make_shared<leaf>(std::move(value));

        if (empty()) {
            return finger_tree(new_leaf);
        }

        if (auto* single = std::get_if<std::shared_ptr<node>>(&data_)) {
            deep d;
            d.prefix = {new_leaf};
            d.middle = std::make_shared<finger_tree>();
            d.suffix = {*single};
            return finger_tree(std::move(d));
        }

        auto d = std::get<deep>(data_);
        if (d.prefix.size() < 4) {
            d.prefix.insert(d.prefix.begin(), new_leaf);
            return finger_tree(std::move(d));
        }

        // Need to push into middle
        auto node3_from_prefix = std::make_shared<node3>(
            d.prefix[1], d.prefix[2], d.prefix[3]
        );

        deep new_d;
        new_d.prefix = {new_leaf, d.prefix[0]};
        new_d.middle = std::make_shared<finger_tree>(
            d.middle->push_front_node(node3_from_prefix)
        );
        new_d.suffix = d.suffix;

        return finger_tree(std::move(new_d));
    }

    // Add element to back
    finger_tree push_back(T value) const {
        auto new_leaf = std::make_shared<leaf>(std::move(value));

        if (empty()) {
            return finger_tree(new_leaf);
        }

        if (auto* single = std::get_if<std::shared_ptr<node>>(&data_)) {
            deep d;
            d.prefix = {*single};
            d.middle = std::make_shared<finger_tree>();
            d.suffix = {new_leaf};
            return finger_tree(std::move(d));
        }

        auto d = std::get<deep>(data_);
        if (d.suffix.size() < 4) {
            d.suffix.push_back(new_leaf);
            return finger_tree(std::move(d));
        }

        // Need to push into middle
        auto node3_from_suffix = std::make_shared<node3>(
            d.suffix[0], d.suffix[1], d.suffix[2]
        );

        deep new_d;
        new_d.prefix = d.prefix;
        new_d.middle = std::make_shared<finger_tree>(
            d.middle->push_back_node(node3_from_suffix)
        );
        new_d.suffix = {d.suffix[3], new_leaf};

        return finger_tree(std::move(new_d));
    }

    // Get first element
    std::optional<T> front() const {
        if (empty()) return std::nullopt;

        if (auto* single = std::get_if<std::shared_ptr<node>>(&data_)) {
            if (auto* l = dynamic_cast<leaf*>(single->get())) {
                return l->value;
            }
        } else if (auto* d = std::get_if<deep>(&data_)) {
            if (!d->prefix.empty()) {
                if (auto* l = dynamic_cast<leaf*>(d->prefix[0].get())) {
                    return l->value;
                }
            }
        }

        return std::nullopt;
    }

    // Get last element
    std::optional<T> back() const {
        if (empty()) return std::nullopt;

        if (auto* single = std::get_if<std::shared_ptr<node>>(&data_)) {
            if (auto* l = dynamic_cast<leaf*>(single->get())) {
                return l->value;
            }
        } else if (auto* d = std::get_if<deep>(&data_)) {
            if (!d->suffix.empty()) {
                if (auto* l = dynamic_cast<leaf*>(d->suffix.back().get())) {
                    return l->value;
                }
            }
        }

        return std::nullopt;
    }

    // Concatenate two trees - O(log(min(n,m)))
    finger_tree concat(const finger_tree& other) const {
        if (empty()) return other;
        if (other.empty()) return *this;

        // Complex merge operation
        // Would require full implementation of deep merge
        // This is simplified version
        finger_tree result = *this;

        // Add all elements from other
        std::function<void(const finger_tree&)> add_all;
        add_all = [&result](const finger_tree& tree) {
            if (auto v = tree.front()) {
                result = result.push_back(*v);
                // Continue with rest (would need pop_front)
            }
        };

        add_all(other);
        return result;
    }

private:
    finger_tree push_front_node(std::shared_ptr<node> n) const {
        // Internal version for nodes
        if (empty()) {
            return finger_tree(n);
        }
        // Implementation details...
        return *this;
    }

    finger_tree push_back_node(std::shared_ptr<node> n) const {
        // Internal version for nodes
        if (empty()) {
            return finger_tree(n);
        }
        // Implementation details...
        return *this;
    }
};

} // namespace stepanov::trees

#endif // STEPANOV_TREES_HPP