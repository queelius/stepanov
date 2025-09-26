// persistent_hash_map.hpp
// Hash Array Mapped Trie (HAMT) - A persistent hash map implementation
// Provides O(log₃₂ n) operations with structural sharing
// Based on Phil Bagwell's paper and Clojure's implementation

#pragma once

#include <memory>
#include <array>
#include <optional>
#include <bit>
#include <functional>
#include <concepts>

namespace stepanov::structures {

template<typename K, typename V,
         typename Hash = std::hash<K>,
         typename Equal = std::equal_to<K>>
class persistent_hash_map {
private:
    static constexpr size_t BITS = 5;
    static constexpr size_t WIDTH = 1 << BITS;
    static constexpr size_t MASK = WIDTH - 1;

    struct node_base {
        virtual ~node_base() = default;
    };

    using node_ptr = std::shared_ptr<node_base>;

    // Leaf node containing actual key-value pair
    struct leaf_node : node_base {
        K key;
        V value;
        size_t hash;

        leaf_node(K k, V v, size_t h) : key(std::move(k)), value(std::move(v)), hash(h) {}
    };

    // Collision node for hash collisions
    struct collision_node : node_base {
        std::vector<std::pair<K, V>> pairs;
        size_t hash;

        explicit collision_node(size_t h) : hash(h) {}
    };

    // Internal node with bitmap for sparse array
    struct bitmap_node : node_base {
        uint32_t bitmap = 0;  // Bit set indicates presence of child
        std::vector<node_ptr> children;

        size_t index(uint32_t bit) const {
            return std::popcount(bitmap & (bit - 1));
        }

        bool contains(uint32_t bit) const {
            return bitmap & bit;
        }

        void insert(uint32_t bit, node_ptr child) {
            size_t idx = index(bit);
            bitmap |= bit;
            children.insert(children.begin() + idx, std::move(child));
        }

        void update(uint32_t bit, node_ptr child) {
            children[index(bit)] = std::move(child);
        }

        void remove(uint32_t bit) {
            if (contains(bit)) {
                size_t idx = index(bit);
                bitmap &= ~bit;
                children.erase(children.begin() + idx);
            }
        }

        node_ptr get(uint32_t bit) const {
            return contains(bit) ? children[index(bit)] : nullptr;
        }
    };

    node_ptr root_;
    size_t size_ = 0;
    Hash hasher_;
    Equal equal_;

    // Hash distribution helper
    static uint32_t hash_fragment(size_t hash, size_t shift) {
        return (hash >> shift) & MASK;
    }

    static uint32_t bit_pos(size_t hash, size_t shift) {
        return 1u << hash_fragment(hash, shift);
    }

    // Path copying for structural sharing
    template<typename NodeType>
    std::shared_ptr<NodeType> copy_node(const std::shared_ptr<NodeType>& node) const {
        if (!node) return nullptr;
        return std::make_shared<NodeType>(*node);
    }

    // Core operations
    node_ptr do_assoc(node_ptr node, size_t shift, size_t hash,
                     const K& key, const V& value, bool& added) const {
        if (!node) {
            added = true;
            return std::make_shared<leaf_node>(key, value, hash);
        }

        if (auto leaf = std::dynamic_pointer_cast<leaf_node>(node)) {
            if (hash == leaf->hash) {
                if (equal_(key, leaf->key)) {
                    // Update existing
                    added = false;
                    return std::make_shared<leaf_node>(key, value, hash);
                } else {
                    // Hash collision - create collision node
                    added = true;
                    auto coll = std::make_shared<collision_node>(hash);
                    coll->pairs.push_back({leaf->key, leaf->value});
                    coll->pairs.push_back({key, value});
                    return coll;
                }
            } else {
                // Different hashes - create bitmap node
                added = true;
                auto bitmap = std::make_shared<bitmap_node>();
                bitmap->insert(bit_pos(leaf->hash, shift), node);
                return do_assoc(std::static_pointer_cast<node_base>(bitmap),
                              shift, hash, key, value, added);
            }
        }

        if (auto coll = std::dynamic_pointer_cast<collision_node>(node)) {
            if (hash == coll->hash) {
                auto new_coll = copy_node(coll);
                for (auto& [k, v] : new_coll->pairs) {
                    if (equal_(key, k)) {
                        v = value;
                        added = false;
                        return new_coll;
                    }
                }
                new_coll->pairs.push_back({key, value});
                added = true;
                return new_coll;
            } else {
                // Convert collision to bitmap with collision as child
                added = true;
                auto bitmap = std::make_shared<bitmap_node>();
                bitmap->insert(bit_pos(coll->hash, shift), node);
                return do_assoc(std::static_pointer_cast<node_base>(bitmap),
                              shift, hash, key, value, added);
            }
        }

        if (auto bitmap = std::dynamic_pointer_cast<bitmap_node>(node)) {
            auto new_bitmap = copy_node(bitmap);
            uint32_t bit = bit_pos(hash, shift);

            if (new_bitmap->contains(bit)) {
                // Recurse deeper
                auto child = new_bitmap->get(bit);
                auto new_child = do_assoc(child, shift + BITS, hash, key, value, added);
                new_bitmap->update(bit, new_child);
            } else {
                // Add new branch
                added = true;
                auto leaf = std::make_shared<leaf_node>(key, value, hash);
                new_bitmap->insert(bit, leaf);
            }
            return new_bitmap;
        }

        return node;
    }

    std::optional<V> do_find(node_ptr node, size_t shift, size_t hash, const K& key) const {
        if (!node) return std::nullopt;

        if (auto leaf = std::dynamic_pointer_cast<leaf_node>(node)) {
            if (hash == leaf->hash && equal_(key, leaf->key)) {
                return leaf->value;
            }
            return std::nullopt;
        }

        if (auto coll = std::dynamic_pointer_cast<collision_node>(node)) {
            if (hash == coll->hash) {
                for (const auto& [k, v] : coll->pairs) {
                    if (equal_(key, k)) return v;
                }
            }
            return std::nullopt;
        }

        if (auto bitmap = std::dynamic_pointer_cast<bitmap_node>(node)) {
            uint32_t bit = bit_pos(hash, shift);
            if (bitmap->contains(bit)) {
                return do_find(bitmap->get(bit), shift + BITS, hash, key);
            }
            return std::nullopt;
        }

        return std::nullopt;
    }

    node_ptr do_dissoc(node_ptr node, size_t shift, size_t hash,
                       const K& key, bool& removed) const {
        if (!node) {
            removed = false;
            return nullptr;
        }

        if (auto leaf = std::dynamic_pointer_cast<leaf_node>(node)) {
            if (hash == leaf->hash && equal_(key, leaf->key)) {
                removed = true;
                return nullptr;
            }
            removed = false;
            return node;
        }

        if (auto coll = std::dynamic_pointer_cast<collision_node>(node)) {
            if (hash == coll->hash) {
                auto new_coll = copy_node(coll);
                auto it = std::remove_if(new_coll->pairs.begin(), new_coll->pairs.end(),
                    [&](const auto& p) { return equal_(key, p.first); });

                if (it != new_coll->pairs.end()) {
                    new_coll->pairs.erase(it, new_coll->pairs.end());
                    removed = true;

                    if (new_coll->pairs.size() == 1) {
                        // Collapse to leaf
                        return std::make_shared<leaf_node>(
                            new_coll->pairs[0].first,
                            new_coll->pairs[0].second,
                            hash);
                    }
                    return new_coll;
                }
            }
            removed = false;
            return node;
        }

        if (auto bitmap = std::dynamic_pointer_cast<bitmap_node>(node)) {
            uint32_t bit = bit_pos(hash, shift);
            if (!bitmap->contains(bit)) {
                removed = false;
                return node;
            }

            auto new_bitmap = copy_node(bitmap);
            auto child = new_bitmap->get(bit);
            auto new_child = do_dissoc(child, shift + BITS, hash, key, removed);

            if (removed) {
                if (!new_child) {
                    new_bitmap->remove(bit);
                    if (new_bitmap->children.empty()) {
                        return nullptr;
                    }
                    if (new_bitmap->children.size() == 1) {
                        // Collapse single child
                        return new_bitmap->children[0];
                    }
                } else {
                    new_bitmap->update(bit, new_child);
                }
            }
            return new_bitmap;
        }

        removed = false;
        return node;
    }

    // Visitor for iteration
    template<typename F>
    void visit_node(node_ptr node, F&& f) const {
        if (!node) return;

        if (auto leaf = std::dynamic_pointer_cast<leaf_node>(node)) {
            f(leaf->key, leaf->value);
        } else if (auto coll = std::dynamic_pointer_cast<collision_node>(node)) {
            for (const auto& [k, v] : coll->pairs) {
                f(k, v);
            }
        } else if (auto bitmap = std::dynamic_pointer_cast<bitmap_node>(node)) {
            for (const auto& child : bitmap->children) {
                visit_node(child, f);
            }
        }
    }

public:
    // Constructors
    persistent_hash_map() = default;

    persistent_hash_map(std::initializer_list<std::pair<K, V>> init) {
        for (const auto& [k, v] : init) {
            *this = assoc(k, v);
        }
    }

    // Size
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    // Lookup
    [[nodiscard]] std::optional<V> find(const K& key) const {
        size_t hash = hasher_(key);
        return do_find(root_, 0, hash, key);
    }

    [[nodiscard]] bool contains(const K& key) const {
        return find(key).has_value();
    }

    [[nodiscard]] V get(const K& key, const V& default_value = V{}) const {
        return find(key).value_or(default_value);
    }

    // Functional updates
    [[nodiscard]] persistent_hash_map assoc(const K& key, const V& value) const {
        persistent_hash_map result(*this);
        bool added = false;
        size_t hash = hasher_(key);
        result.root_ = do_assoc(root_, 0, hash, key, value, added);
        if (added) result.size_++;
        return result;
    }

    [[nodiscard]] persistent_hash_map dissoc(const K& key) const {
        persistent_hash_map result(*this);
        bool removed = false;
        size_t hash = hasher_(key);
        result.root_ = do_dissoc(root_, 0, hash, key, removed);
        if (removed) result.size_--;
        return result;
    }

    // Batch updates
    [[nodiscard]] persistent_hash_map update(const K& key, std::function<V(const V&)> f) const {
        auto val = find(key);
        if (val) {
            return assoc(key, f(*val));
        }
        return *this;
    }

    [[nodiscard]] persistent_hash_map update_or_insert(const K& key,
                                                       std::function<V(const V&)> f,
                                                       const V& default_val) const {
        auto val = find(key);
        return assoc(key, val ? f(*val) : default_val);
    }

    // Merge
    [[nodiscard]] persistent_hash_map merge(const persistent_hash_map& other) const {
        persistent_hash_map result(*this);
        other.for_each([&result](const K& k, const V& v) {
            result = result.assoc(k, v);
        });
        return result;
    }

    // Iteration
    template<typename F>
    void for_each(F&& f) const {
        visit_node(root_, std::forward<F>(f));
    }

    // Functional operations
    template<typename F>
    [[nodiscard]] persistent_hash_map map_values(F&& f) const {
        persistent_hash_map result;
        for_each([&result, &f](const K& k, const V& v) {
            result = result.assoc(k, f(v));
        });
        return result;
    }

    template<typename F>
    [[nodiscard]] persistent_hash_map filter(F&& pred) const {
        persistent_hash_map result;
        for_each([&result, &pred](const K& k, const V& v) {
            if (pred(k, v)) {
                result = result.assoc(k, v);
            }
        });
        return result;
    }

    template<typename F, typename Init>
    [[nodiscard]] Init reduce(F&& f, Init init) const {
        for_each([&f, &init](const K& k, const V& v) {
            init = f(std::move(init), k, v);
        });
        return init;
    }

    // Transient for efficient batch operations
    class transient {
        friend class persistent_hash_map;
        persistent_hash_map map_;
        bool is_persistent_ = false;

        explicit transient(const persistent_hash_map& m) : map_(m) {}

    public:
        transient& assoc(const K& key, const V& value) {
            if (is_persistent_) {
                throw std::runtime_error("Cannot modify persistent transient");
            }
            map_ = map_.assoc(key, value);
            return *this;
        }

        transient& dissoc(const K& key) {
            if (is_persistent_) {
                throw std::runtime_error("Cannot modify persistent transient");
            }
            map_ = map_.dissoc(key);
            return *this;
        }

        persistent_hash_map persistent() {
            is_persistent_ = true;
            return map_;
        }
    };

    [[nodiscard]] transient make_transient() const {
        return transient(*this);
    }
};

// Deduction guide
template<typename K, typename V>
persistent_hash_map(std::initializer_list<std::pair<K, V>>) -> persistent_hash_map<K, V>;

} // namespace stepanov::structures