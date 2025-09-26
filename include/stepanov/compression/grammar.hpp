#ifndef STEPANOV_COMPRESSION_GRAMMAR_HPP
#define STEPANOV_COMPRESSION_GRAMMAR_HPP

#include <vector>
#include <unordered_map>
#include <memory>
#include <stack>
#include <queue>
#include <algorithm>
#include <string>
#include <span>
#include <concepts>

namespace stepanov::compression::grammar {

// ============================================================================
// Grammar-Based Compression
// ============================================================================
// These algorithms represent data as context-free grammars,
// achieving compression by finding repeated patterns and encoding them as rules.

// Symbol type for grammar rules
template<typename T>
struct grammar_symbol {
    bool is_terminal;
    union {
        T terminal_value;
        size_t nonterminal_id;
    };

    grammar_symbol(T val) : is_terminal(true), terminal_value(val) {}
    grammar_symbol(size_t id, bool) : is_terminal(false), nonterminal_id(id) {}

    bool operator==(const grammar_symbol& other) const {
        if (is_terminal != other.is_terminal) return false;
        return is_terminal ? terminal_value == other.terminal_value
                           : nonterminal_id == other.nonterminal_id;
    }
};

// Context-Free Grammar representation
template<typename T>
class context_free_grammar {
public:
    using symbol = grammar_symbol<T>;
    using production = std::vector<symbol>;

private:
    std::vector<production> rules;      // Grammar rules (rule[0] is start symbol)
    size_t next_nonterminal_id;

public:
    context_free_grammar() : next_nonterminal_id(0) {}

    // Add a new production rule
    size_t add_rule(const production& rule) {
        size_t id = next_nonterminal_id++;
        rules.push_back(rule);
        return id;
    }

    // Generate the original sequence from grammar
    std::vector<T> expand() const {
        if (rules.empty()) return {};

        std::vector<T> result;
        expand_rule(0, result);
        return result;
    }

    // Get compression ratio
    double compression_ratio(size_t original_size) const {
        size_t grammar_size = 0;
        for (const auto& rule : rules) {
            grammar_size += rule.size() + 1;  // +1 for rule separator
        }
        return static_cast<double>(grammar_size) / original_size;
    }

    const std::vector<production>& get_rules() const { return rules; }

private:
    void expand_rule(size_t rule_id, std::vector<T>& output) const {
        if (rule_id >= rules.size()) return;

        for (const auto& sym : rules[rule_id]) {
            if (sym.is_terminal) {
                output.push_back(sym.terminal_value);
            } else {
                expand_rule(sym.nonterminal_id, output);
            }
        }
    }
};

// Sequitur Algorithm - Builds a hierarchical grammar with no repeated digrams
template<typename T>
class sequitur_compressor {
    using symbol = grammar_symbol<T>;
    using production = std::vector<symbol>;

    struct digram {
        symbol first, second;

        bool operator==(const digram& other) const {
            return first == other.first && second == other.second;
        }
    };

    struct digram_hash {
        size_t operator()(const digram& d) const {
            size_t h1 = d.first.is_terminal ?
                       std::hash<T>{}(d.first.terminal_value) :
                       std::hash<size_t>{}(d.first.nonterminal_id);
            size_t h2 = d.second.is_terminal ?
                       std::hash<T>{}(d.second.terminal_value) :
                       std::hash<size_t>{}(d.second.nonterminal_id);
            return h1 ^ (h2 << 1);
        }
    };

    std::unordered_map<digram, size_t, digram_hash> digram_index;
    context_free_grammar<T> grammar;
    production main_rule;
    size_t next_rule_id;

public:
    sequitur_compressor() : next_rule_id(1) {}  // Rule 0 is main rule

    context_free_grammar<T> compress(const std::vector<T>& input) {
        if (input.empty()) return grammar;

        // Initialize main rule with input
        main_rule.clear();
        for (const T& val : input) {
            main_rule.push_back(symbol(val));
        }

        // Apply Sequitur rules
        bool changed = true;
        while (changed) {
            changed = false;

            // Find repeated digrams
            std::unordered_map<digram, std::vector<size_t>, digram_hash> digram_positions;

            for (size_t i = 0; i < main_rule.size() - 1; ++i) {
                digram d{main_rule[i], main_rule[i + 1]};
                digram_positions[d].push_back(i);
            }

            // Replace repeated digrams with new rules
            for (const auto& [dig, positions] : digram_positions) {
                if (positions.size() > 1) {
                    // Create new rule for this digram
                    production new_rule{dig.first, dig.second};
                    size_t rule_id = next_rule_id++;

                    // Replace all occurrences
                    for (int i = positions.size() - 1; i >= 0; --i) {
                        size_t pos = positions[i];
                        main_rule.erase(main_rule.begin() + pos, main_rule.begin() + pos + 2);
                        main_rule.insert(main_rule.begin() + pos, symbol(rule_id, false));
                    }

                    grammar.add_rule(new_rule);
                    changed = true;
                    break;  // Restart after modification
                }
            }
        }

        // Add main rule as rule 0
        grammar = context_free_grammar<T>();
        grammar.add_rule(main_rule);

        return grammar;
    }
};

// Re-Pair Algorithm - Recursive pairing for optimal grammar extraction
template<typename T>
class repair_compressor {
    using symbol = grammar_symbol<T>;
    using production = std::vector<symbol>;

    struct pair {
        symbol left, right;
        size_t count;

        bool operator<(const pair& other) const {
            return count > other.count;  // Max heap
        }
    };

public:
    context_free_grammar<T> compress(const std::vector<T>& input) {
        if (input.empty()) return {};

        std::vector<symbol> sequence;
        for (const T& val : input) {
            sequence.push_back(symbol(val));
        }

        context_free_grammar<T> grammar;
        size_t next_nonterminal = 0;

        // Custom hash for pair
        struct pair_hash {
            size_t operator()(const std::pair<size_t, size_t>& p) const {
                return p.first ^ (p.second << 1);
            }
        };

        while (sequence.size() > 1) {
            // Count all pairs
            std::unordered_map<std::pair<size_t, size_t>, size_t, pair_hash> pair_counts;

            for (size_t i = 0; i < sequence.size() - 1; ++i) {
                size_t left_id = sequence[i].is_terminal ?
                    std::hash<T>{}(sequence[i].terminal_value) :
                    sequence[i].nonterminal_id + 1000000;
                size_t right_id = sequence[i + 1].is_terminal ?
                    std::hash<T>{}(sequence[i + 1].terminal_value) :
                    sequence[i + 1].nonterminal_id + 1000000;

                pair_counts[{left_id, right_id}]++;
            }

            // Find most frequent pair
            if (pair_counts.empty()) break;

            auto max_it = std::max_element(pair_counts.begin(), pair_counts.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

            if (max_it->second <= 1) break;  // No repeated pairs

            // Create new rule for most frequent pair
            size_t i = 0;
            while (i < sequence.size() - 1) {
                size_t left_id = sequence[i].is_terminal ?
                    std::hash<T>{}(sequence[i].terminal_value) :
                    sequence[i].nonterminal_id + 1000000;
                size_t right_id = sequence[i + 1].is_terminal ?
                    std::hash<T>{}(sequence[i + 1].terminal_value) :
                    sequence[i + 1].nonterminal_id + 1000000;

                if (std::make_pair(left_id, right_id) == max_it->first) {
                    // Replace pair with new nonterminal
                    production new_rule{sequence[i], sequence[i + 1]};
                    size_t rule_id = grammar.add_rule(new_rule);

                    sequence[i] = symbol(rule_id, false);
                    sequence.erase(sequence.begin() + i + 1);
                } else {
                    ++i;
                }
            }
        }

        // Add main rule
        grammar.add_rule(sequence);
        return grammar;
    }
};

// Straight-Line Program (SLP) - Grammar where each nonterminal appears exactly once
template<typename T>
class slp_compressor {
    using symbol = grammar_symbol<T>;

    struct slp_node {
        bool is_leaf;
        union {
            T value;
            struct {
                size_t left;
                size_t right;
            } children;
        };
        size_t length;  // Length of expanded string

        slp_node(T val) : is_leaf(true), value(val), length(1) {}
        slp_node(size_t l, size_t r, size_t len)
            : is_leaf(false), length(len) {
            children.left = l;
            children.right = r;
        }
    };

    std::vector<slp_node> nodes;

public:
    // Build SLP using balanced binary grammar
    context_free_grammar<T> compress(const std::vector<T>& input) {
        if (input.empty()) return {};

        nodes.clear();

        // Create leaf nodes
        for (const T& val : input) {
            nodes.push_back(slp_node(val));
        }

        // Build balanced binary tree
        std::queue<size_t> queue;
        for (size_t i = 0; i < input.size(); ++i) {
            queue.push(i);
        }

        while (queue.size() > 1) {
            size_t left = queue.front(); queue.pop();
            size_t right = queue.front(); queue.pop();

            size_t new_length = nodes[left].length + nodes[right].length;
            nodes.push_back(slp_node(left, right, new_length));
            queue.push(nodes.size() - 1);
        }

        // Convert to context-free grammar
        return slp_to_grammar();
    }

    // Random access in compressed string - O(log n) time
    T access(size_t index) const {
        if (nodes.empty() || index >= nodes.back().length) {
            throw std::out_of_range("Index out of bounds");
        }

        return access_recursive(nodes.size() - 1, index);
    }

private:
    T access_recursive(size_t node_id, size_t index) const {
        const auto& node = nodes[node_id];

        if (node.is_leaf) {
            return node.value;
        }

        size_t left_length = nodes[node.children.left].length;
        if (index < left_length) {
            return access_recursive(node.children.left, index);
        } else {
            return access_recursive(node.children.right, index - left_length);
        }
    }

    context_free_grammar<T> slp_to_grammar() const {
        context_free_grammar<T> grammar;

        // Convert each node to a grammar rule
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& node = nodes[i];
            typename context_free_grammar<T>::production rule;

            if (node.is_leaf) {
                rule.push_back(symbol(node.value));
            } else {
                rule.push_back(symbol(node.children.left, false));
                rule.push_back(symbol(node.children.right, false));
            }

            grammar.add_rule(rule);
        }

        return grammar;
    }
};

// Grammar Transform - LZ77-style compression using grammar rules
template<typename T>
class grammar_transform {
    using symbol = grammar_symbol<T>;

    struct rule_reference {
        size_t rule_id;
        size_t start;
        size_t length;
    };

public:
    // Compress using grammar-based LZ77 variant
    context_free_grammar<T> compress(const std::vector<T>& input, size_t window_size = 4096) {
        context_free_grammar<T> grammar;
        std::vector<rule_reference> references;

        // Dictionary of seen patterns
        std::unordered_map<std::vector<T>, size_t, vector_hash<T>> pattern_to_rule;

        size_t pos = 0;
        while (pos < input.size()) {
            // Find longest match in dictionary
            size_t best_length = 0;
            size_t best_rule = 0;

            for (size_t len = std::min(window_size, input.size() - pos); len > 0; --len) {
                std::vector<T> pattern(input.begin() + pos, input.begin() + pos + len);

                auto it = pattern_to_rule.find(pattern);
                if (it != pattern_to_rule.end()) {
                    best_length = len;
                    best_rule = it->second;
                    break;
                }
            }

            if (best_length > 3) {  // Threshold for using reference
                // Use existing rule
                references.push_back({best_rule, pos, best_length});
                pos += best_length;
            } else {
                // Create new rule for this position
                size_t end = std::min(pos + window_size, input.size());
                for (size_t len = 1; len <= end - pos; ++len) {
                    std::vector<T> pattern(input.begin() + pos, input.begin() + pos + len);

                    if (pattern_to_rule.find(pattern) == pattern_to_rule.end()) {
                        typename context_free_grammar<T>::production rule;
                        for (const T& val : pattern) {
                            rule.push_back(symbol(val));
                        }

                        size_t rule_id = grammar.add_rule(rule);
                        pattern_to_rule[pattern] = rule_id;

                        if (len >= 4) break;  // Limit rule size
                    }
                }
                ++pos;
            }
        }

        return grammar;
    }

private:
    template<typename U>
    struct vector_hash {
        size_t operator()(const std::vector<U>& v) const {
            size_t hash = 0;
            for (const U& val : v) {
                hash ^= std::hash<U>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };
};

// Smallest Grammar Problem solver using greedy approximation
template<typename T>
class smallest_grammar {
public:
    // Find approximately smallest grammar using greedy heuristics
    static context_free_grammar<T> find_smallest(const std::vector<T>& input) {
        // Try multiple algorithms and return smallest
        sequitur_compressor<T> sequitur;
        repair_compressor<T> repair;
        slp_compressor<T> slp;

        auto g1 = sequitur.compress(input);
        auto g2 = repair.compress(input);
        auto g3 = slp.compress(input);

        // Compare grammar sizes
        size_t s1 = grammar_size(g1);
        size_t s2 = grammar_size(g2);
        size_t s3 = grammar_size(g3);

        if (s1 <= s2 && s1 <= s3) return g1;
        if (s2 <= s3) return g2;
        return g3;
    }

private:
    static size_t grammar_size(const context_free_grammar<T>& g) {
        size_t size = 0;
        for (const auto& rule : g.get_rules()) {
            size += rule.size();
        }
        return size;
    }
};

} // namespace stepanov::compression::grammar

#endif // STEPANOV_COMPRESSION_GRAMMAR_HPP