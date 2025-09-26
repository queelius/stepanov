#ifndef STEPANOV_COMPRESSION_ADAPTIVE_HPP
#define STEPANOV_COMPRESSION_ADAPTIVE_HPP

#include <vector>
#include <array>
#include <deque>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <queue>
#include <span>
#include <concepts>

namespace stepanov::compression::adaptive {

// ============================================================================
// Adaptive and Online Compression Algorithms
// ============================================================================
// These algorithms adapt their models during compression, learning patterns
// as they process data without requiring a separate modeling pass.

// Dynamic Huffman Coding - Updates tree structure as symbols are processed
template<typename T>
class dynamic_huffman {
    struct node {
        T symbol;
        size_t weight;
        size_t number;  // Node number for sibling property
        node* parent;
        node* left;
        node* right;

        bool is_leaf() const { return left == nullptr && right == nullptr; }

        node(T sym = T{}, size_t w = 0, size_t num = 0)
            : symbol(sym), weight(w), number(num),
              parent(nullptr), left(nullptr), right(nullptr) {}
    };

    node* root;
    node* nyt;  // Not Yet Transmitted node
    std::unordered_map<T, node*> symbol_nodes;
    std::vector<node*> node_list;  // Ordered by node numbers
    size_t next_node_number;

    // Vitter's algorithm for maintaining sibling property
    void update_tree(node* current) {
        while (current != nullptr) {
            // Find node to swap with (same weight, higher number)
            node* swap_node = find_swap_node(current);

            if (swap_node != nullptr && swap_node != current->parent) {
                swap_nodes(current, swap_node);
            }

            // Increment weight
            current->weight++;
            current = current->parent;
        }
    }

    node* find_swap_node(node* n) {
        // Find the last node with same weight in the ordering
        for (auto it = node_list.rbegin(); it != node_list.rend(); ++it) {
            if ((*it)->weight == n->weight && (*it) != n) {
                return *it;
            }
            if ((*it)->weight > n->weight) break;
        }
        return nullptr;
    }

    void swap_nodes(node* a, node* b) {
        // Swap positions in tree
        if (a->parent != nullptr) {
            if (a->parent->left == a) a->parent->left = b;
            else a->parent->right = b;
        }
        if (b->parent != nullptr) {
            if (b->parent->left == b) b->parent->left = a;
            else b->parent->right = a;
        }

        std::swap(a->parent, b->parent);
        std::swap(a->number, b->number);

        // Update node list
        std::swap(node_list[a->number], node_list[b->number]);
    }

    void add_new_symbol(T symbol) {
        // Create new internal node and leaf
        auto* new_internal = new node(T{}, 1, next_node_number++);
        auto* new_leaf = new node(symbol, 1, next_node_number++);

        // Replace NYT with new internal node
        if (nyt->parent != nullptr) {
            if (nyt->parent->left == nyt) nyt->parent->left = new_internal;
            else nyt->parent->right = new_internal;
        } else {
            root = new_internal;
        }

        new_internal->parent = nyt->parent;
        new_internal->left = nyt;
        new_internal->right = new_leaf;

        nyt->parent = new_internal;
        new_leaf->parent = new_internal;

        // Update NYT number
        nyt->number = next_node_number++;

        // Add to structures
        symbol_nodes[symbol] = new_leaf;
        node_list.push_back(new_internal);
        node_list.push_back(new_leaf);

        // Update weights up the tree
        update_tree(new_internal->parent);
    }

public:
    dynamic_huffman() : next_node_number(0) {
        nyt = new node(T{}, 0, next_node_number++);
        root = nyt;
        node_list.push_back(nyt);
    }

    ~dynamic_huffman() {
        // Clean up tree
        delete_tree(root);
    }

    // Encode symbol and update tree
    std::vector<bool> encode_symbol(T symbol) {
        std::vector<bool> output;

        auto it = symbol_nodes.find(symbol);
        if (it != symbol_nodes.end()) {
            // Symbol exists - output its code
            node* n = it->second;
            std::vector<bool> code;

            while (n->parent != nullptr) {
                code.push_back(n->parent->right == n);
                n = n->parent;
            }

            output.insert(output.end(), code.rbegin(), code.rend());

            // Update tree
            update_tree(it->second);
        } else {
            // New symbol - output NYT code + fixed code for symbol
            node* n = nyt;
            std::vector<bool> nyt_code;

            while (n->parent != nullptr) {
                nyt_code.push_back(n->parent->right == n);
                n = n->parent;
            }

            output.insert(output.end(), nyt_code.rbegin(), nyt_code.rend());

            // Add fixed-length code for new symbol (simplified to 8 bits)
            for (int i = 7; i >= 0; --i) {
                output.push_back((static_cast<uint8_t>(symbol) >> i) & 1);
            }

            // Add symbol to tree
            add_new_symbol(symbol);
        }

        return output;
    }

    // Decode bits and update tree
    T decode_symbol(std::queue<bool>& bits) {
        node* current = root;

        // Navigate to leaf
        while (!current->is_leaf()) {
            if (bits.empty()) throw std::runtime_error("Insufficient bits");

            if (bits.front()) {
                current = current->right;
            } else {
                current = current->left;
            }
            bits.pop();
        }

        if (current == nyt) {
            // Read fixed-length code for new symbol
            T symbol = 0;
            for (int i = 0; i < 8; ++i) {
                if (bits.empty()) throw std::runtime_error("Insufficient bits");
                symbol = (symbol << 1) | bits.front();
                bits.pop();
            }

            add_new_symbol(symbol);
            return symbol;
        } else {
            // Existing symbol
            update_tree(current);
            return current->symbol;
        }
    }

private:
    void delete_tree(node* n) {
        if (n == nullptr) return;
        delete_tree(n->left);
        delete_tree(n->right);
        delete n;
    }
};

// Adaptive Arithmetic Coding with fast updates
class adaptive_arithmetic {
    static constexpr uint32_t MAX_RANGE = 0xFFFFFFFF;
    static constexpr uint32_t QUARTER = 0x40000000;
    static constexpr uint32_t HALF = 0x80000000;
    static constexpr uint32_t THREE_QUARTERS = 0xC0000000;

    struct symbol_stats {
        uint32_t count;
        uint32_t cumulative;
    };

    std::array<symbol_stats, 256> model;
    uint32_t total_count;

    // Fenwick tree for fast cumulative frequency updates
    class fenwick_tree {
        std::vector<uint32_t> tree;

    public:
        fenwick_tree(size_t n) : tree(n + 1, 0) {}

        void update(size_t idx, int32_t delta) {
            idx++;  // 1-indexed
            while (idx < tree.size()) {
                tree[idx] += delta;
                idx += idx & -idx;
            }
        }

        uint32_t query(size_t idx) const {
            idx++;
            uint32_t sum = 0;
            while (idx > 0) {
                sum += tree[idx];
                idx -= idx & -idx;
            }
            return sum;
        }

        uint32_t range_query(size_t left, size_t right) const {
            return query(right) - (left > 0 ? query(left - 1) : 0);
        }
    };

    fenwick_tree frequency_tree;

public:
    adaptive_arithmetic() : frequency_tree(256), total_count(256) {
        // Initialize with uniform distribution
        for (size_t i = 0; i < 256; ++i) {
            model[i].count = 1;
            model[i].cumulative = i;
            frequency_tree.update(i, 1);
        }
    }

    // Encode symbol with adaptive model
    void encode_symbol(uint8_t symbol, uint32_t& low, uint32_t& high,
                       std::vector<uint8_t>& output) {
        // Get symbol range
        uint32_t symbol_low = frequency_tree.query(symbol > 0 ? symbol - 1 : 0);
        uint32_t symbol_high = frequency_tree.query(symbol);

        // Update coding interval
        uint64_t range = static_cast<uint64_t>(high) - low + 1;
        high = low + (range * symbol_high / total_count) - 1;
        low = low + (range * symbol_low / total_count);

        // Output bits and rescale
        while (true) {
            if (high < HALF) {
                output.push_back(0);
                low = 2 * low;
                high = 2 * high + 1;
            } else if (low >= HALF) {
                output.push_back(1);
                low = 2 * (low - HALF);
                high = 2 * (high - HALF) + 1;
            } else if (low >= QUARTER && high < THREE_QUARTERS) {
                // Underflow prevention
                low = 2 * (low - QUARTER);
                high = 2 * (high - QUARTER) + 1;
            } else {
                break;
            }
        }

        // Update model
        update_model(symbol);
    }

    // Update frequency model
    void update_model(uint8_t symbol) {
        frequency_tree.update(symbol, 1);
        model[symbol].count++;
        total_count++;

        // Rescale if total gets too large
        if (total_count > 65536) {
            rescale_model();
        }
    }

    void rescale_model() {
        total_count = 0;
        for (size_t i = 0; i < 256; ++i) {
            model[i].count = (model[i].count + 1) / 2;  // Keep at least 1
            frequency_tree.update(i, model[i].count - frequency_tree.range_query(i, i));
            total_count += model[i].count;
        }
    }
};

// Move-to-Front with Local Adaptivity
template<typename T>
class move_to_front_local {
    std::vector<T> list;
    size_t window_size;
    std::deque<T> recent_symbols;

    // Local context for better prediction
    struct context_model {
        std::unordered_map<T, size_t> frequency;
        std::vector<T> mtf_list;

        void update(T symbol) {
            frequency[symbol]++;

            // Move to front in local list
            auto it = std::find(mtf_list.begin(), mtf_list.end(), symbol);
            if (it != mtf_list.end()) {
                mtf_list.erase(it);
            }
            mtf_list.insert(mtf_list.begin(), symbol);

            // Limit list size
            if (mtf_list.size() > 32) {
                mtf_list.resize(32);
            }
        }

        size_t get_position(T symbol) const {
            auto it = std::find(mtf_list.begin(), mtf_list.end(), symbol);
            return it != mtf_list.end() ? std::distance(mtf_list.begin(), it) : mtf_list.size();
        }
    };

    std::unordered_map<uint32_t, context_model> contexts;

public:
    move_to_front_local(size_t alphabet_size = 256, size_t win_size = 16)
        : window_size(win_size) {
        // Initialize with identity permutation
        list.reserve(alphabet_size);
        for (size_t i = 0; i < alphabet_size; ++i) {
            list.push_back(static_cast<T>(i));
        }
    }

    // Encode with local context
    size_t encode(T symbol) {
        // Compute context hash from recent symbols
        uint32_t context = compute_context_hash();

        // Use context-specific model if available
        auto& ctx_model = contexts[context];
        size_t local_pos = ctx_model.get_position(symbol);

        // Update global list
        auto it = std::find(list.begin(), list.end(), symbol);
        size_t position = std::distance(list.begin(), it);

        if (it != list.end()) {
            list.erase(it);
            list.insert(list.begin(), symbol);
        }

        // Update context model
        ctx_model.update(symbol);

        // Update recent symbols
        recent_symbols.push_back(symbol);
        if (recent_symbols.size() > window_size) {
            recent_symbols.pop_front();
        }

        // Return weighted position (prefer local prediction)
        return local_pos < position ? local_pos : position;
    }

    T decode(size_t position) {
        if (position >= list.size()) {
            throw std::out_of_range("Invalid MTF position");
        }

        T symbol = list[position];

        // Move to front
        list.erase(list.begin() + position);
        list.insert(list.begin(), symbol);

        // Update recent symbols
        recent_symbols.push_back(symbol);
        if (recent_symbols.size() > window_size) {
            recent_symbols.pop_front();
        }

        return symbol;
    }

private:
    uint32_t compute_context_hash() const {
        uint32_t hash = 0;
        for (const T& sym : recent_symbols) {
            hash = hash * 31 + std::hash<T>{}(sym);
        }
        return hash;
    }
};

// Recency Rank Encoding - Exploits temporal locality
template<typename T>
class recency_rank_encoder {
    struct cache_entry {
        T value;
        size_t last_access;
        size_t frequency;
    };

    std::vector<cache_entry> cache;
    size_t max_cache_size;
    size_t current_time;

    // Aging function for cache entries
    double compute_score(const cache_entry& entry) const {
        double recency = 1.0 / (current_time - entry.last_access + 1);
        double frequency = std::log(entry.frequency + 1);
        return recency * 0.7 + frequency * 0.3;  // Weighted combination
    }

public:
    recency_rank_encoder(size_t cache_size = 256)
        : max_cache_size(cache_size), current_time(0) {
        cache.reserve(cache_size);
    }

    // Encode symbol using recency rank
    struct encoded_symbol {
        bool in_cache;
        size_t rank;  // Position in cache if in_cache
        T value;      // Raw value if not in cache

        size_t encoded_size() const {
            return in_cache ? 1 + std::bit_width(rank) : 1 + sizeof(T) * 8;
        }
    };

    encoded_symbol encode(T symbol) {
        current_time++;

        // Search in cache
        auto it = std::find_if(cache.begin(), cache.end(),
            [symbol](const cache_entry& e) { return e.value == symbol; });

        if (it != cache.end()) {
            // Found in cache
            size_t rank = std::distance(cache.begin(), it);

            // Update entry
            it->last_access = current_time;
            it->frequency++;

            // Reorder cache by score
            std::sort(cache.begin(), cache.end(),
                [this](const cache_entry& a, const cache_entry& b) {
                    return compute_score(a) > compute_score(b);
                });

            return {true, rank, T{}};
        } else {
            // Not in cache - add it
            cache_entry new_entry{symbol, current_time, 1};

            if (cache.size() >= max_cache_size) {
                // Evict least useful entry
                auto min_it = std::min_element(cache.begin(), cache.end(),
                    [this](const cache_entry& a, const cache_entry& b) {
                        return compute_score(a) > compute_score(b);
                    });
                *min_it = new_entry;
            } else {
                cache.push_back(new_entry);
            }

            // Reorder cache
            std::sort(cache.begin(), cache.end(),
                [this](const cache_entry& a, const cache_entry& b) {
                    return compute_score(a) > compute_score(b);
                });

            return {false, 0, symbol};
        }
    }

    T decode(const encoded_symbol& encoded) {
        current_time++;

        if (encoded.in_cache) {
            if (encoded.rank >= cache.size()) {
                throw std::runtime_error("Invalid cache rank");
            }

            T symbol = cache[encoded.rank].value;

            // Update cache entry
            cache[encoded.rank].last_access = current_time;
            cache[encoded.rank].frequency++;

            // Reorder cache
            std::sort(cache.begin(), cache.end(),
                [this](const cache_entry& a, const cache_entry& b) {
                    return compute_score(a) > compute_score(b);
                });

            return symbol;
        } else {
            // Add to cache
            cache_entry new_entry{encoded.value, current_time, 1};

            if (cache.size() >= max_cache_size) {
                auto min_it = std::min_element(cache.begin(), cache.end(),
                    [this](const cache_entry& a, const cache_entry& b) {
                        return compute_score(a) > compute_score(b);
                    });
                *min_it = new_entry;
            } else {
                cache.push_back(new_entry);
            }

            return encoded.value;
        }
    }
};

// Prediction by Partial Matching (PPM) with adaptive order
class ppm_adaptive {
    static constexpr size_t MAX_ORDER = 5;

    struct context_node {
        std::unordered_map<uint8_t, uint32_t> counts;
        std::unordered_map<uint8_t, std::unique_ptr<context_node>> children;
        uint32_t total_count;
        uint32_t escape_count;

        context_node() : total_count(0), escape_count(1) {}

        double get_probability(uint8_t symbol) const {
            auto it = counts.find(symbol);
            if (it != counts.end()) {
                return static_cast<double>(it->second) / (total_count + escape_count);
            }
            return static_cast<double>(escape_count) / (total_count + escape_count);
        }

        void update(uint8_t symbol) {
            counts[symbol]++;
            total_count++;

            // Adapt escape probability
            if (counts.size() > 10) {
                escape_count = std::max(1u, escape_count - 1);
            }
        }
    };

    context_node root;
    std::deque<uint8_t> context_buffer;
    size_t current_order;

public:
    ppm_adaptive() : current_order(0) {}

    // Encode with adaptive order selection
    double encode_symbol(uint8_t symbol) {
        double probability = 1.0;
        bool found = false;

        // Try contexts from highest to lowest order
        for (int order = std::min(current_order, context_buffer.size()); order >= 0; --order) {
            context_node* node = &root;

            // Navigate to context
            for (size_t i = context_buffer.size() - order; i < context_buffer.size(); ++i) {
                auto it = node->children.find(context_buffer[i]);
                if (it == node->children.end()) {
                    break;
                }
                node = it->second.get();
            }

            double p = node->get_probability(symbol);
            probability *= p;

            if (node->counts.find(symbol) != node->counts.end()) {
                found = true;
                break;
            }
        }

        // Update all contexts
        update_contexts(symbol);

        // Adapt order based on prediction success
        if (found && current_order < MAX_ORDER) {
            current_order++;
        } else if (!found && current_order > 0) {
            current_order--;
        }

        return probability;
    }

private:
    void update_contexts(uint8_t symbol) {
        // Update all context levels
        context_node* node = &root;
        node->update(symbol);

        for (uint8_t ctx_symbol : context_buffer) {
            if (node->children.find(ctx_symbol) == node->children.end()) {
                node->children[ctx_symbol] = std::make_unique<context_node>();
            }
            node = node->children[ctx_symbol].get();
            node->update(symbol);
        }

        // Update context buffer
        context_buffer.push_back(symbol);
        if (context_buffer.size() > MAX_ORDER) {
            context_buffer.pop_front();
        }
    }
};

} // namespace stepanov::compression::adaptive

#endif // STEPANOV_COMPRESSION_ADAPTIVE_HPP