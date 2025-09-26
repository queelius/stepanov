#ifndef STEPANOV_CACHE_OBLIVIOUS_HPP
#define STEPANOV_CACHE_OBLIVIOUS_HPP

#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <bit>
#include <span>
#include <numeric>
#include <queue>
#include <optional>
#include <cmath>

namespace stepanov::cache_oblivious {

// Cache-oblivious algorithms achieve optimal cache performance
// without knowing cache parameters (B = block size, M = cache size).
// Key principle: Recursive divide-and-conquer with the right layout.

// ============================================================================
// Van Emde Boas Layout for Trees
// ============================================================================

template<typename T>
class veb_tree {
    // Van Emde Boas layout: recursive subdivision for optimal cache usage
    // Tree of height h is split into top tree of height h/2 and bottom trees

    std::vector<T> data;
    size_t capacity;
    size_t height;
    size_t size_;

    // Map logical position to physical position in vEB layout
    size_t veb_position(size_t logical_pos, size_t h) const {
        if (h <= 1) return logical_pos;

        size_t half_h = h / 2;
        size_t top_size = (1ull << half_h) - 1;
        size_t bottom_h = h - half_h;
        size_t bottom_size = (1ull << bottom_h) - 1;

        if (logical_pos <= top_size) {
            // In top tree
            return veb_position(logical_pos, half_h);
        } else {
            // In bottom trees
            size_t bottom_tree = (logical_pos - top_size - 1) / bottom_size;
            size_t pos_in_bottom = (logical_pos - top_size - 1) % bottom_size;

            size_t bottom_start = top_size + bottom_tree * bottom_size;
            return bottom_start + veb_position(pos_in_bottom, bottom_h);
        }
    }

    // Recursive search in vEB layout
    size_t search_helper(const T& key, size_t node, size_t h) const {
        if (node >= capacity) return capacity;

        size_t phys_pos = veb_position(node, height);
        if (phys_pos >= size_) return capacity;

        if (data[phys_pos] == key) return node;

        if (h == 0) return capacity;  // Leaf

        size_t left = 2 * node + 1;
        size_t right = 2 * node + 2;

        if (key < data[phys_pos]) {
            return search_helper(key, left, h - 1);
        } else {
            return search_helper(key, right, h - 1);
        }
    }

public:
    veb_tree() : capacity(0), height(0), size_(0) {}

    explicit veb_tree(size_t max_size)
        : height(std::bit_width(max_size)),
          capacity((1ull << height) - 1),
          size_(0) {
        data.resize(capacity);
    }

    void insert(const T& value) {
        if (size_ >= capacity) {
            // Resize and rebuild
            size_t new_capacity = capacity * 2 + 1;
            height++;
            capacity = new_capacity;

            std::vector<T> old_data = std::move(data);
            data.resize(capacity);

            // Rebuild with new layout
            for (size_t i = 0; i < size_; ++i) {
                data[veb_position(i, height)] = old_data[veb_position(i, height - 1)];
            }
        }

        // Binary search tree insertion with vEB layout
        if (size_ == 0) {
            data[veb_position(0, height)] = value;
            size_++;
            return;
        }

        size_t pos = 0;
        size_t h = height;

        while (h > 0) {
            size_t phys_pos = veb_position(pos, height);

            if (value < data[phys_pos]) {
                pos = 2 * pos + 1;
            } else {
                pos = 2 * pos + 2;
            }
            h--;

            if (pos >= capacity || veb_position(pos, height) >= size_) {
                data[veb_position(pos, height)] = value;
                size_++;
                return;
            }
        }
    }

    bool search(const T& key) const {
        return search_helper(key, 0, height) < capacity;
    }

    size_t size() const { return size_; }

    // Range query with cache-oblivious efficiency
    std::vector<T> range_query(const T& low, const T& high) const {
        std::vector<T> result;
        range_query_helper(low, high, 0, height, result);
        return result;
    }

private:
    void range_query_helper(const T& low, const T& high,
                          size_t node, size_t h,
                          std::vector<T>& result) const {
        if (node >= capacity) return;

        size_t phys_pos = veb_position(node, height);
        if (phys_pos >= size_) return;

        const T& value = data[phys_pos];

        if (value >= low && value <= high) {
            result.push_back(value);
        }

        if (h > 0) {
            if (low < value) {
                range_query_helper(low, high, 2 * node + 1, h - 1, result);
            }
            if (high > value) {
                range_query_helper(low, high, 2 * node + 2, h - 1, result);
            }
        }
    }
};

// ============================================================================
// Cache-Oblivious Matrix Multiplication
// ============================================================================

template<typename T>
class matrix {
    std::vector<T> data;
    size_t rows_, cols_;

public:
    matrix(size_t r, size_t c) : rows_(r), cols_(c), data(r * c) {}

    T& operator()(size_t i, size_t j) { return data[i * cols_ + j]; }
    const T& operator()(size_t i, size_t j) const { return data[i * cols_ + j]; }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // Cache-oblivious matrix multiply using recursive blocking
    matrix operator*(const matrix& B) const {
        if (cols_ != B.rows_) {
            throw std::invalid_argument("Incompatible matrix dimensions");
        }

        matrix C(rows_, B.cols_);
        multiply_recursive(*this, B, C,
                         0, rows_, 0, cols_, 0, B.cols_);
        return C;
    }

private:
    // Recursive matrix multiplication with automatic cache adaptation
    static void multiply_recursive(const matrix& A, const matrix& B, matrix& C,
                                  size_t i0, size_t i1,
                                  size_t j0, size_t j1,
                                  size_t k0, size_t k1) {
        static constexpr size_t CUTOFF = 64;  // Base case size

        size_t m = i1 - i0;
        size_t n = k1 - k0;
        size_t p = j1 - j0;

        // Base case: small enough for L1 cache
        if (m <= CUTOFF && n <= CUTOFF && p <= CUTOFF) {
            // Standard triple-loop multiplication
            for (size_t i = i0; i < i1; ++i) {
                for (size_t k = k0; k < k1; ++k) {
                    T sum = 0;
                    for (size_t j = j0; j < j1; ++j) {
                        sum += A(i, j) * B(j, k);
                    }
                    C(i, k) += sum;
                }
            }
            return;
        }

        // Recursive case: divide largest dimension
        if (m >= n && m >= p) {
            // Split A horizontally
            size_t mid = i0 + m / 2;
            multiply_recursive(A, B, C, i0, mid, j0, j1, k0, k1);
            multiply_recursive(A, B, C, mid, i1, j0, j1, k0, k1);
        } else if (n >= p) {
            // Split C and B vertically
            size_t mid = k0 + n / 2;
            multiply_recursive(A, B, C, i0, i1, j0, j1, k0, mid);
            multiply_recursive(A, B, C, i0, i1, j0, j1, mid, k1);
        } else {
            // Split A vertically and B horizontally
            size_t mid = j0 + p / 2;
            multiply_recursive(A, B, C, i0, i1, j0, mid, k0, k1);
            multiply_recursive(A, B, C, i0, i1, mid, j1, k0, k1);
        }
    }
};

// ============================================================================
// Funnel Sort - Cache-Oblivious Sorting
// ============================================================================

template<typename T>
class funnel_sort {
    // K-funnel: merges K sorted sequences cache-obliviously
    struct k_funnel {
        size_t k;  // Number of input streams
        std::vector<std::vector<T>> buffers;
        std::vector<size_t> buffer_pos;
        std::unique_ptr<k_funnel> left_funnel;
        std::unique_ptr<k_funnel> right_funnel;
        std::vector<T> output_buffer;
        size_t output_pos;

        k_funnel(size_t k_val) : k(k_val), buffer_pos(k, 0), output_pos(0) {
            if (k > 2) {
                size_t sqrt_k = std::sqrt(k);
                left_funnel = std::make_unique<k_funnel>(sqrt_k);
                right_funnel = std::make_unique<k_funnel>(k - sqrt_k);
            }
            buffers.resize(k);
        }

        void fill_buffers(std::vector<std::span<T>>& inputs) {
            for (size_t i = 0; i < k && i < inputs.size(); ++i) {
                size_t buffer_size = std::min(size_t(std::sqrt(inputs[i].size())),
                                             inputs[i].size());
                buffers[i].assign(inputs[i].begin(),
                                 inputs[i].begin() + buffer_size);
                inputs[i] = inputs[i].subspan(buffer_size);
            }
        }

        T extract() {
            if (k == 2) {
                // Base case: simple 2-way merge
                if (buffer_pos[0] < buffers[0].size() &&
                    buffer_pos[1] < buffers[1].size()) {
                    if (buffers[0][buffer_pos[0]] <= buffers[1][buffer_pos[1]]) {
                        return buffers[0][buffer_pos[0]++];
                    } else {
                        return buffers[1][buffer_pos[1]++];
                    }
                } else if (buffer_pos[0] < buffers[0].size()) {
                    return buffers[0][buffer_pos[0]++];
                } else {
                    return buffers[1][buffer_pos[1]++];
                }
            }

            // Recursive case: use sub-funnels
            if (output_pos >= output_buffer.size()) {
                // Refill output buffer from sub-funnels
                output_buffer.clear();
                size_t sqrt_k = std::sqrt(k);

                for (size_t i = 0; i < sqrt_k * sqrt_k; ++i) {
                    T val1 = left_funnel->extract();
                    T val2 = right_funnel->extract();
                    output_buffer.push_back(val1 <= val2 ? val1 : val2);
                }
                output_pos = 0;
            }

            return output_buffer[output_pos++];
        }
    };

    // Recursive funnel sort
    static void sort_recursive(std::span<T> data) {
        if (data.size() <= 1) return;

        static constexpr size_t BASE_SIZE = 64;

        if (data.size() <= BASE_SIZE) {
            // Base case: use standard sort
            std::sort(data.begin(), data.end());
            return;
        }

        // Split into √n subarrays
        size_t k = std::sqrt(data.size());
        size_t subarray_size = data.size() / k;

        std::vector<std::span<T>> subarrays;
        for (size_t i = 0; i < k; ++i) {
            size_t start = i * subarray_size;
            size_t end = (i == k - 1) ? data.size() : (i + 1) * subarray_size;
            subarrays.emplace_back(data.subspan(start, end - start));
        }

        // Recursively sort subarrays
        for (auto& subarray : subarrays) {
            sort_recursive(subarray);
        }

        // Merge using k-funnel
        k_funnel funnel(k);
        funnel.fill_buffers(subarrays);

        std::vector<T> sorted;
        sorted.reserve(data.size());

        for (size_t i = 0; i < data.size(); ++i) {
            sorted.push_back(funnel.extract());
        }

        std::copy(sorted.begin(), sorted.end(), data.begin());
    }

public:
    void operator()(std::span<T> data) {
        sort_recursive(data);
    }
};

// ============================================================================
// Cache-Oblivious Priority Queue
// ============================================================================

template<typename T>
class cache_oblivious_priority_queue {
    // Profile-adaptive binary heap with cache-oblivious layout
    std::vector<T> heap;
    size_t size_;

    // Map heap position to cache-oblivious layout position
    size_t co_position(size_t i) const {
        if (i == 0) return 0;

        // Find level and position within level
        size_t level = std::bit_width(i + 1) - 1;
        size_t pos_in_level = i - ((1ull << level) - 1);

        // Use bit-reversal permutation for cache-oblivious layout
        size_t reversed = 0;
        size_t temp = pos_in_level;
        for (size_t j = 0; j < level; ++j) {
            reversed = (reversed << 1) | (temp & 1);
            temp >>= 1;
        }

        return ((1ull << level) - 1) + reversed;
    }

    void sift_up(size_t i) {
        T value = heap[co_position(i)];

        while (i > 0) {
            size_t parent = (i - 1) / 2;
            size_t parent_pos = co_position(parent);

            if (heap[parent_pos] >= value) break;

            heap[co_position(i)] = heap[parent_pos];
            i = parent;
        }

        heap[co_position(i)] = value;
    }

    void sift_down(size_t i) {
        T value = heap[co_position(i)];

        while (2 * i + 1 < size_) {
            size_t child = 2 * i + 1;
            size_t child_pos = co_position(child);

            if (child + 1 < size_) {
                size_t right_pos = co_position(child + 1);
                if (heap[right_pos] > heap[child_pos]) {
                    child++;
                    child_pos = right_pos;
                }
            }

            if (value >= heap[child_pos]) break;

            heap[co_position(i)] = heap[child_pos];
            i = child;
        }

        heap[co_position(i)] = value;
    }

public:
    cache_oblivious_priority_queue() : size_(0) {}

    void push(const T& value) {
        if (size_ >= heap.size()) {
            heap.resize(heap.size() * 2 + 1);
        }

        heap[co_position(size_)] = value;
        sift_up(size_);
        size_++;
    }

    T pop() {
        if (size_ == 0) {
            throw std::runtime_error("Priority queue is empty");
        }

        T result = heap[co_position(0)];
        size_--;

        if (size_ > 0) {
            heap[co_position(0)] = heap[co_position(size_)];
            sift_down(0);
        }

        return result;
    }

    const T& top() const {
        if (size_ == 0) {
            throw std::runtime_error("Priority queue is empty");
        }
        return heap[co_position(0)];
    }

    bool empty() const { return size_ == 0; }
    size_t size() const { return size_; }
};

// ============================================================================
// Cache-Oblivious B-Tree
// ============================================================================

template<typename Key, typename Value>
class cache_oblivious_btree {
    static constexpr size_t MIN_BLOCK = 4;
    static constexpr size_t MAX_BLOCK = 256;

    struct node {
        std::vector<Key> keys;
        std::vector<Value> values;
        std::vector<std::unique_ptr<node>> children;
        bool is_leaf;

        node(bool leaf = true) : is_leaf(leaf) {}

        size_t size() const { return keys.size(); }

        // Binary search for key position
        size_t find_position(const Key& key) const {
            return std::lower_bound(keys.begin(), keys.end(), key) - keys.begin();
        }
    };

    std::unique_ptr<node> root;
    size_t size_;

    // Determine optimal node size based on recursive depth
    size_t optimal_node_size(size_t depth) const {
        // Double node size every two levels for cache adaptation
        size_t size = MIN_BLOCK;
        for (size_t i = 0; i < depth / 2 && size < MAX_BLOCK; ++i) {
            size *= 2;
        }
        return size;
    }

    // Split child node when full
    void split_child(node* parent, size_t index) {
        auto& child = parent->children[index];
        size_t mid = child->size() / 2;

        auto new_node = std::make_unique<node>(child->is_leaf);

        // Move half of keys/values to new node
        new_node->keys.assign(child->keys.begin() + mid + 1, child->keys.end());
        new_node->values.assign(child->values.begin() + mid + 1, child->values.end());

        if (!child->is_leaf) {
            new_node->children.reserve(new_node->keys.size() + 1);
            for (size_t i = mid + 1; i <= child->keys.size(); ++i) {
                new_node->children.push_back(std::move(child->children[i]));
            }
            child->children.resize(mid + 1);
        }

        Key median_key = child->keys[mid];
        Value median_value = child->values[mid];

        child->keys.resize(mid);
        child->values.resize(mid);

        // Insert median into parent
        parent->keys.insert(parent->keys.begin() + index, median_key);
        parent->values.insert(parent->values.begin() + index, median_value);
        parent->children.insert(parent->children.begin() + index + 1,
                              std::move(new_node));
    }

    void insert_non_full(node* n, const Key& key, const Value& value, size_t depth) {
        size_t pos = n->find_position(key);

        if (n->is_leaf) {
            // Insert into leaf
            n->keys.insert(n->keys.begin() + pos, key);
            n->values.insert(n->values.begin() + pos, value);
        } else {
            // Recurse to child
            auto& child = n->children[pos];

            if (child->size() >= optimal_node_size(depth + 1)) {
                split_child(n, pos);
                if (key > n->keys[pos]) {
                    pos++;
                }
            }

            insert_non_full(child.get(), key, value, depth + 1);
        }
    }

public:
    cache_oblivious_btree() : root(std::make_unique<node>()), size_(0) {}

    void insert(const Key& key, const Value& value) {
        if (root->size() >= optimal_node_size(0)) {
            // Split root
            auto new_root = std::make_unique<node>(false);
            new_root->children.push_back(std::move(root));
            root = std::move(new_root);
            split_child(root.get(), 0);
        }

        insert_non_full(root.get(), key, value, 0);
        size_++;
    }

    std::optional<Value> search(const Key& key) const {
        node* current = root.get();

        while (current) {
            size_t pos = current->find_position(key);

            if (pos < current->size() && current->keys[pos] == key) {
                return current->values[pos];
            }

            if (current->is_leaf) {
                return std::nullopt;
            }

            current = current->children[pos].get();
        }

        return std::nullopt;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
};

} // namespace stepanov::cache_oblivious

#endif // STEPANOV_CACHE_OBLIVIOUS_HPP