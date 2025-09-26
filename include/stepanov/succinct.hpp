#ifndef STEPANOV_SUCCINCT_HPP
#define STEPANOV_SUCCINCT_HPP

#include <vector>
#include <bit>
#include <cstdint>
#include <algorithm>
#include <span>
#include <string_view>
#include <memory>
#include <ranges>
#include <numeric>

namespace stepanov::succinct {

// Succinct data structures achieve information-theoretic space lower bounds
// while supporting operations in O(1) or O(log n) time.
// Key insight: Use auxiliary structures of size o(n) to accelerate queries.

// ============================================================================
// Succinct Bit Vector with Rank and Select
// ============================================================================

class bit_vector {
    // Primary structure: the actual bits
    std::vector<uint64_t> blocks;
    size_t n_bits;

    // Rank auxiliary structures (o(n) extra space)
    std::vector<size_t> superblock_ranks;  // Every 512 bits
    std::vector<uint16_t> block_ranks;     // Every 64 bits within superblock

    // Select auxiliary structures
    std::vector<size_t> select1_samples;   // Sample every log²n ones
    std::vector<size_t> select0_samples;   // Sample every log²n zeros

    static constexpr size_t BITS_PER_BLOCK = 64;
    static constexpr size_t BLOCKS_PER_SUPERBLOCK = 8;
    static constexpr size_t BITS_PER_SUPERBLOCK = BITS_PER_BLOCK * BLOCKS_PER_SUPERBLOCK;

public:
    bit_vector() : n_bits(0) {}

    explicit bit_vector(size_t n) : n_bits(n) {
        blocks.resize((n + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK, 0);
    }

    bit_vector(std::initializer_list<bool> bits) : n_bits(bits.size()) {
        blocks.resize((n_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK, 0);
        size_t i = 0;
        for (bool b : bits) {
            if (b) set(i);
            i++;
        }
        build_auxiliary();
    }

    // Constructor from iterators
    template<typename InputIt>
    bit_vector(InputIt first, InputIt last) : n_bits(std::distance(first, last)) {
        blocks.resize((n_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK, 0);
        size_t i = 0;
        for (auto it = first; it != last; ++it) {
            if (*it) set(i);
            i++;
        }
        build_auxiliary();
    }

    // Basic operations
    bool operator[](size_t i) const {
        return (blocks[i / BITS_PER_BLOCK] >> (i % BITS_PER_BLOCK)) & 1;
    }

    void set(size_t i) {
        blocks[i / BITS_PER_BLOCK] |= (1ULL << (i % BITS_PER_BLOCK));
    }

    void clear(size_t i) {
        blocks[i / BITS_PER_BLOCK] &= ~(1ULL << (i % BITS_PER_BLOCK));
    }

    void flip(size_t i) {
        blocks[i / BITS_PER_BLOCK] ^= (1ULL << (i % BITS_PER_BLOCK));
    }

    size_t size() const { return n_bits; }

    // Build auxiliary structures for rank/select
    void build_auxiliary() {
        build_rank_structure();
        build_select_structure();
    }

    // Rank: count of 1s in [0, i)
    size_t rank1(size_t i) const {
        if (i == 0) return 0;
        if (i > n_bits) i = n_bits;

        size_t block_idx = i / BITS_PER_BLOCK;
        size_t bit_idx = i % BITS_PER_BLOCK;

        // Start with superblock rank
        size_t superblock_idx = block_idx / BLOCKS_PER_SUPERBLOCK;
        size_t result = superblock_idx > 0 ? superblock_ranks[superblock_idx - 1] : 0;

        // Add block ranks within superblock
        size_t block_in_super = block_idx % BLOCKS_PER_SUPERBLOCK;
        if (block_in_super > 0) {
            result += block_ranks[block_idx - 1];
        }

        // Add popcount of partial block
        if (bit_idx > 0) {
            uint64_t mask = (1ULL << bit_idx) - 1;
            result += std::popcount(blocks[block_idx] & mask);
        }

        return result;
    }

    size_t rank0(size_t i) const {
        return i - rank1(i);
    }

    // Select: position of the i-th 1 (0-indexed)
    size_t select1(size_t i) const {
        if (i >= rank1(n_bits)) return n_bits;

        // Binary search using rank
        size_t left = 0, right = n_bits;
        while (left < right) {
            size_t mid = left + (right - left) / 2;
            if (rank1(mid + 1) <= i) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    size_t select0(size_t i) const {
        if (i >= rank0(n_bits)) return n_bits;

        // Binary search using rank0
        size_t left = 0, right = n_bits;
        while (left < right) {
            size_t mid = left + (right - left) / 2;
            if (rank0(mid + 1) <= i) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

private:
    void build_rank_structure() {
        superblock_ranks.clear();
        block_ranks.clear();

        size_t cumulative_rank = 0;
        size_t superblock_rank = 0;

        for (size_t i = 0; i < blocks.size(); ++i) {
            if (i % BLOCKS_PER_SUPERBLOCK == 0 && i > 0) {
                superblock_ranks.push_back(cumulative_rank);
                superblock_rank = cumulative_rank;
            }

            size_t block_popcount = std::popcount(blocks[i]);
            cumulative_rank += block_popcount;

            // Store rank relative to superblock
            if (i % BLOCKS_PER_SUPERBLOCK != 0) {
                block_ranks.push_back(cumulative_rank - superblock_rank);
            }
        }

        // Final superblock rank
        if (!blocks.empty() && blocks.size() % BLOCKS_PER_SUPERBLOCK == 0) {
            superblock_ranks.push_back(cumulative_rank);
        }
    }

    void build_select_structure() {
        // Build select samples for faster select queries
        // Sample every log²n positions
        size_t sample_rate = std::max(size_t(1), size_t(std::bit_width(n_bits) * std::bit_width(n_bits)));

        select1_samples.clear();
        select0_samples.clear();

        size_t ones_seen = 0, zeros_seen = 0;
        for (size_t i = 0; i < n_bits; ++i) {
            if ((*this)[i]) {
                if (ones_seen % sample_rate == 0) {
                    select1_samples.push_back(i);
                }
                ones_seen++;
            } else {
                if (zeros_seen % sample_rate == 0) {
                    select0_samples.push_back(i);
                }
                zeros_seen++;
            }
        }
    }
};

// ============================================================================
// Wavelet Tree for Sequence Operations
// ============================================================================

template<typename T>
class wavelet_tree {
    struct node {
        bit_vector bits;
        T split_value;
        std::unique_ptr<node> left;
        std::unique_ptr<node> right;
    };

    std::unique_ptr<node> root;
    std::vector<T> alphabet;
    size_t n;

    // Build tree recursively
    std::unique_ptr<node> build(const std::vector<T>& sequence,
                                T min_val, T max_val) {
        if (min_val == max_val || sequence.empty()) {
            return nullptr;
        }

        auto n = std::make_unique<node>();
        T mid = min_val + (max_val - min_val) / 2;
        n->split_value = mid;

        // Build bit vector and partition sequence
        std::vector<bool> bits;
        std::vector<T> left_seq, right_seq;

        for (T val : sequence) {
            if (val <= mid) {
                bits.push_back(false);
                left_seq.push_back(val);
            } else {
                bits.push_back(true);
                right_seq.push_back(val);
            }
        }

        n->bits = bit_vector(bits.begin(), bits.end());
        n->bits.build_auxiliary();

        // Recursively build children
        if (min_val < mid) {
            n->left = build(left_seq, min_val, mid);
        }
        if (mid + 1 <= max_val) {
            n->right = build(right_seq, mid + 1, max_val);
        }

        return n;
    }

    // Helper for range queries
    size_t rank_helper(const node* n, size_t i, T val, T min_val, T max_val) const {
        if (!n || i == 0) return 0;
        if (min_val == max_val) return i;

        T mid = n->split_value;
        if (val <= mid) {
            size_t left_i = n->bits.rank0(i);
            return rank_helper(n->left.get(), left_i, val, min_val, mid);
        } else {
            size_t right_i = n->bits.rank1(i);
            return rank_helper(n->right.get(), right_i, val, mid + 1, max_val);
        }
    }

public:
    wavelet_tree() : n(0) {}

    wavelet_tree(const std::vector<T>& sequence) : n(sequence.size()) {
        if (sequence.empty()) return;

        // Find alphabet
        alphabet = sequence;
        std::ranges::sort(alphabet);
        alphabet.erase(std::unique(alphabet.begin(), alphabet.end()), alphabet.end());

        // Build tree directly with original values
        root = build(sequence, alphabet.front(), alphabet.back());
    }

    // Access element at position i
    T operator[](size_t i) const {
        return access(i);
    }

    T access(size_t i) const {
        if (i >= n) throw std::out_of_range("Index out of bounds");

        const node* current = root.get();
        T min_val = 0, max_val = alphabet.size() - 1;

        while (current && min_val < max_val) {
            T mid = current->split_value;
            if (current->bits[i]) {
                // Go right
                i = current->bits.rank1(i);
                min_val = mid + 1;
                current = current->right.get();
            } else {
                // Go left
                i = current->bits.rank0(i);
                max_val = mid;
                current = current->left.get();
            }
        }

        return alphabet[min_val];
    }

    // Rank: count occurrences of val in [0, i)
    size_t rank(size_t i, T val) const {
        auto it = std::lower_bound(alphabet.begin(), alphabet.end(), val);
        if (it == alphabet.end() || *it != val) return 0;
        size_t val_rank = std::distance(alphabet.begin(), it);
        return rank_helper(root.get(), i, val_rank, 0, alphabet.size() - 1);
    }

    // Select: position of the i-th occurrence of val
    size_t select(size_t i, T val) const {
        // Binary search using rank
        size_t left = 0, right = n;
        while (left < right) {
            size_t mid = left + (right - left) / 2;
            if (rank(mid + 1, val) <= i) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    // Range quantile: k-th smallest in range [l, r)
    T quantile(size_t l, size_t r, size_t k) const {
        if (l >= r || k >= r - l) throw std::out_of_range("Invalid range or k");

        const node* current = root.get();
        T min_val = 0, max_val = alphabet.size() - 1;

        while (current && min_val < max_val) {
            size_t left_count = current->bits.rank0(r) - current->bits.rank0(l);

            if (k < left_count) {
                // k-th smallest is in left subtree
                l = current->bits.rank0(l);
                r = current->bits.rank0(r);
                max_val = current->split_value;
                current = current->left.get();
            } else {
                // k-th smallest is in right subtree
                k -= left_count;
                l = current->bits.rank1(l);
                r = current->bits.rank1(r);
                min_val = current->split_value + 1;
                current = current->right.get();
            }
        }

        return alphabet[min_val];
    }

    // Count distinct values in range [l, r)
    size_t range_count_distinct(size_t l, size_t r) const {
        // Implementation using wavelet tree traversal
        size_t count = 0;
        for (const T& val : alphabet) {
            if (rank(r, val) > rank(l, val)) {
                count++;
            }
        }
        return count;
    }
};

// ============================================================================
// FM-Index for Compressed Full-Text Indexing
// ============================================================================

class fm_index {
    std::string bwt;                    // Burrows-Wheeler transform
    std::vector<size_t> C;              // Cumulative character counts
    wavelet_tree<char> wavelet;        // Wavelet tree of BWT
    size_t sample_rate;                 // Suffix array sampling rate
    std::vector<size_t> sa_samples;    // Sampled suffix array positions
    std::vector<size_t> sa_sample_pos; // Positions of samples

    // Compute BWT from text
    std::string compute_bwt(const std::string& text) {
        size_t n = text.size();
        std::vector<size_t> sa(n);

        // Simple suffix array construction (should use more efficient algorithm)
        std::iota(sa.begin(), sa.end(), 0);
        std::sort(sa.begin(), sa.end(), [&text](size_t i, size_t j) {
            return std::lexicographical_compare(
                text.begin() + i, text.end(),
                text.begin() + j, text.end());
        });

        // Extract BWT
        std::string bwt;
        bwt.reserve(n);
        for (size_t i : sa) {
            bwt.push_back(i == 0 ? text.back() : text[i - 1]);
        }

        // Sample suffix array
        sample_rate = std::max(size_t(1), size_t(std::bit_width(n)));
        for (size_t i = 0; i < n; ++i) {
            if (sa[i] % sample_rate == 0) {
                sa_samples.push_back(sa[i]);
                sa_sample_pos.push_back(i);
            }
        }

        return bwt;
    }

    // LF-mapping: where would BWT[i] be in F column?
    size_t lf(size_t i) const {
        char c = bwt[i];
        return C[c] + wavelet.rank(i, c);
    }

public:
    fm_index() = default;

    explicit fm_index(const std::string& text) {
        if (text.empty()) return;

        // Add terminator if not present
        std::string t = text;
        if (t.back() != '$') t.push_back('$');

        // Compute BWT
        bwt = compute_bwt(t);

        // Build cumulative counts
        C.resize(256, 0);
        std::vector<size_t> counts(256, 0);
        for (char c : bwt) {
            counts[static_cast<unsigned char>(c)]++;
        }

        size_t cumulative = 0;
        for (size_t i = 0; i < 256; ++i) {
            C[i] = cumulative;
            cumulative += counts[i];
        }

        // Build wavelet tree
        wavelet = wavelet_tree<char>(std::vector<char>(bwt.begin(), bwt.end()));
    }

    // Count occurrences of pattern
    size_t count(const std::string& pattern) const {
        if (pattern.empty() || bwt.empty()) return 0;

        size_t sp = 0, ep = bwt.size();

        // Backward search
        for (auto it = pattern.rbegin(); it != pattern.rend(); ++it) {
            char c = *it;
            size_t c_idx = static_cast<unsigned char>(c);

            sp = C[c_idx] + (sp > 0 ? wavelet.rank(sp, c) : 0);
            ep = C[c_idx] + wavelet.rank(ep, c);

            if (sp >= ep) return 0;  // Pattern not found
        }

        return ep - sp;
    }

    // Find all occurrences of pattern
    std::vector<size_t> locate(const std::string& pattern) const {
        if (pattern.empty() || bwt.empty()) return {};

        size_t sp = 0, ep = bwt.size();

        // Backward search
        for (auto it = pattern.rbegin(); it != pattern.rend(); ++it) {
            char c = *it;
            size_t c_idx = static_cast<unsigned char>(c);

            sp = C[c_idx] + (sp > 0 ? wavelet.rank(sp, c) : 0);
            ep = C[c_idx] + wavelet.rank(ep, c);

            if (sp >= ep) return {};  // Pattern not found
        }

        // Extract positions
        std::vector<size_t> positions;
        for (size_t i = sp; i < ep; ++i) {
            // Walk backward until we hit a sampled position
            size_t pos = i;
            size_t steps = 0;

            while (true) {
                auto it = std::find(sa_sample_pos.begin(), sa_sample_pos.end(), pos);
                if (it != sa_sample_pos.end()) {
                    size_t idx = std::distance(sa_sample_pos.begin(), it);
                    positions.push_back((sa_samples[idx] + steps) % bwt.size());
                    break;
                }
                pos = lf(pos);
                steps++;
                if (steps > bwt.size()) break;  // Avoid infinite loop
            }
        }

        std::sort(positions.begin(), positions.end());
        return positions;
    }

    // Extract substring from compressed text
    std::string extract(size_t from, size_t to) const {
        if (from >= to || bwt.empty()) return "";

        std::string result;
        size_t len = to - from;

        // Find closest sampled position
        size_t best_sample = 0;
        size_t best_dist = bwt.size();

        for (size_t i = 0; i < sa_samples.size(); ++i) {
            if (sa_samples[i] >= from && sa_samples[i] < to) {
                size_t dist = sa_samples[i] - from;
                if (dist < best_dist) {
                    best_dist = dist;
                    best_sample = i;
                }
            }
        }

        // Extract using LF-mapping
        if (best_dist < bwt.size()) {
            size_t pos = sa_sample_pos[best_sample];
            size_t text_pos = sa_samples[best_sample];

            // Move to start position
            while (text_pos > from) {
                pos = lf(pos);
                text_pos--;
            }

            // Extract characters
            for (size_t i = 0; i < len; ++i) {
                result.push_back(bwt[pos]);
                pos = lf(pos);
            }
        }

        return result;
    }

    size_t size() const { return bwt.size(); }
};

// ============================================================================
// Succinct Tree Representations
// ============================================================================

class balanced_parentheses_tree {
    bit_vector bp;  // Balanced parentheses representation
    size_t n_nodes;

    // Helper functions for navigation
    size_t find_close(size_t open) const {
        // Find matching close parenthesis
        int counter = 1;
        size_t pos = open + 1;
        while (counter > 0 && pos < bp.size()) {
            counter += bp[pos] ? 1 : -1;
            if (counter == 0) return pos;
            pos++;
        }
        return bp.size();
    }

    size_t find_open(size_t close) const {
        // Find matching open parenthesis
        int counter = -1;
        size_t pos = close - 1;
        while (counter < 0 && pos < bp.size()) {
            counter += bp[pos] ? 1 : -1;
            if (counter == 0) return pos;
            pos--;
        }
        return 0;
    }

public:
    balanced_parentheses_tree() : n_nodes(0) {}

    // Build from depth-first traversal
    balanced_parentheses_tree(const std::vector<bool>& parentheses)
        : bp(parentheses.begin(), parentheses.end()) {
        bp.build_auxiliary();
        n_nodes = bp.rank1(bp.size()) / 2;
    }

    // Build from tree structure (example with binary tree node)
    template<typename Node>
    balanced_parentheses_tree(const Node* root) {
        std::vector<bool> paren;
        build_from_tree(root, paren);
        bp = bit_vector(paren.begin(), paren.end());
        bp.build_auxiliary();
        n_nodes = bp.rank1(bp.size()) / 2;
    }

    // Tree operations
    size_t root() const { return 0; }

    bool is_leaf(size_t v) const {
        return !bp[v + 1];  // Next bit is close paren
    }

    size_t parent(size_t v) const {
        if (v == 0) return bp.size();  // Root has no parent
        size_t close = find_close(v);
        return bp.select1(bp.rank1(v));  // Enclosing open paren
    }

    size_t first_child(size_t v) const {
        if (is_leaf(v)) return bp.size();
        return v + 1;
    }

    size_t next_sibling(size_t v) const {
        size_t close = find_close(v);
        if (close + 1 >= bp.size() || !bp[close + 1]) {
            return bp.size();  // No next sibling
        }
        return close + 1;
    }

    size_t subtree_size(size_t v) const {
        size_t close = find_close(v);
        return (close - v + 1) / 2;
    }

    size_t depth(size_t v) const {
        // Count excess open parens before v
        return bp.rank1(v) - bp.rank0(v);
    }

    size_t lca(size_t u, size_t v) const {
        // Lowest common ancestor
        if (u > v) std::swap(u, v);

        // Find minimum excess in range [u, v]
        int min_excess = INT32_MAX;
        size_t lca_pos = u;

        for (size_t i = u; i <= v; ++i) {
            int excess = bp.rank1(i) - bp.rank0(i);
            if (excess < min_excess) {
                min_excess = excess;
                lca_pos = i;
            }
        }

        // Find enclosing node
        return bp.select1(min_excess);
    }

private:
    template<typename Node>
    void build_from_tree(const Node* node, std::vector<bool>& paren) {
        if (!node) return;
        paren.push_back(true);  // Open
        if (node->left) build_from_tree(node->left, paren);
        if (node->right) build_from_tree(node->right, paren);
        paren.push_back(false);  // Close
    }
};

// LOUDS (Level-Order Unary Degree Sequence) representation
class louds_tree {
    bit_vector louds;
    size_t n_nodes;

public:
    louds_tree() : n_nodes(0) {}

    // Build from level-order traversal with degrees
    louds_tree(const std::vector<size_t>& degrees) : n_nodes(degrees.size()) {
        std::vector<bool> bits;
        bits.push_back(true);  // Super-root
        bits.push_back(false);

        for (size_t degree : degrees) {
            for (size_t i = 0; i < degree; ++i) {
                bits.push_back(true);
            }
            bits.push_back(false);
        }

        louds = bit_vector(bits.begin(), bits.end());
        louds.build_auxiliary();
    }

    size_t root() const { return 2; }  // After super-root

    size_t parent(size_t v) const {
        if (v == root()) return 0;
        size_t node_idx = louds.rank0(v);
        return louds.select1(node_idx) + 1;
    }

    size_t first_child(size_t v) const {
        size_t i = louds.select0(louds.rank1(v)) + 1;
        if (louds[i]) return i;
        return 0;  // No children
    }

    size_t next_sibling(size_t v) const {
        if (louds[v + 1]) return v + 1;
        return 0;  // No next sibling
    }

    size_t degree(size_t v) const {
        size_t start = louds.select0(louds.rank1(v)) + 1;
        size_t count = 0;
        while (louds[start + count]) count++;
        return count;
    }

    bool is_leaf(size_t v) const {
        return degree(v) == 0;
    }
};

} // namespace stepanov::succinct

#endif // STEPANOV_SUCCINCT_HPP