#pragma once

#include <algorithm>
#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <deque>
#include <optional>
#include <memory>
#include <numeric>
#include <limits>
#include <cassert>
#include "concepts.hpp"
#include "iterators.hpp"
#include "ranges.hpp"

namespace stepanov {

// Generic string matching algorithms

// Knuth-Morris-Pratt algorithm
template<typename Iterator, typename PatIterator>
    requires forward_iterator<Iterator> && forward_iterator<PatIterator>
std::optional<Iterator> kmp_search(Iterator first, Iterator last,
                                   PatIterator pat_first, PatIterator pat_last) {
    using value_type = std::iter_value_t<PatIterator>;

    auto pattern_size = std::distance(pat_first, pat_last);
    if (pattern_size == 0) return first;

    // Build failure function
    std::vector<std::ptrdiff_t> failure(pattern_size, 0);
    std::ptrdiff_t j = 0;

    auto pat_it = pat_first;
    ++pat_it;

    for (std::ptrdiff_t i = 1; i < pattern_size; ++i) {
        auto pat_i = pat_first;
        std::advance(pat_i, i);

        while (j > 0 && *pat_i != *std::next(pat_first, j)) {
            j = failure[j - 1];
        }

        if (*pat_i == *std::next(pat_first, j)) {
            ++j;
        }

        failure[i] = j;
    }

    // Search
    j = 0;
    for (auto it = first; it != last; ++it) {
        while (j > 0 && *it != *std::next(pat_first, j)) {
            j = failure[j - 1];
        }

        if (*it == *std::next(pat_first, j)) {
            ++j;
        }

        if (j == pattern_size) {
            auto result = it;
            std::advance(result, 1 - pattern_size);
            return result;
        }
    }

    return std::nullopt;
}

// Boyer-Moore algorithm
template<typename Iterator, typename PatIterator>
    requires random_access_iterator<Iterator> && random_access_iterator<PatIterator>
class boyer_moore {
private:
    using value_type = std::iter_value_t<PatIterator>;
    std::vector<value_type> pattern;
    std::unordered_map<value_type, std::ptrdiff_t> bad_char_skip;
    std::vector<std::ptrdiff_t> good_suffix_skip;

    void compute_bad_char_skip() {
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(pattern.size()); ++i) {
            bad_char_skip[pattern[i]] = i;
        }
    }

    void compute_good_suffix_skip() {
        std::ptrdiff_t m = pattern.size();
        good_suffix_skip.resize(m, m);

        std::vector<std::ptrdiff_t> suffix(m);
        suffix[m - 1] = m;

        std::ptrdiff_t g = m - 1;
        std::ptrdiff_t f = 0;

        for (std::ptrdiff_t i = m - 2; i >= 0; --i) {
            if (i > g && suffix[i + m - 1 - f] < i - g) {
                suffix[i] = suffix[i + m - 1 - f];
            } else {
                if (i < g) g = i;
                f = i;

                while (g >= 0 && pattern[g] == pattern[g + m - 1 - f]) {
                    --g;
                }

                suffix[i] = f - g;
            }
        }

        for (std::ptrdiff_t i = 0; i < m - 1; ++i) {
            good_suffix_skip[m - 1 - suffix[i]] = m - 1 - i;
        }

        for (std::ptrdiff_t i = 0; i < m - 1; ++i) {
            good_suffix_skip[m - 1 - suffix[i]] = m - 1 - i;
        }
    }

public:
    boyer_moore(PatIterator first, PatIterator last)
        : pattern(first, last) {
        if (!pattern.empty()) {
            compute_bad_char_skip();
            compute_good_suffix_skip();
        }
    }

    std::optional<Iterator> search(Iterator first, Iterator last) const {
        if (pattern.empty()) return first;

        auto text_size = std::distance(first, last);
        auto pat_size = static_cast<std::ptrdiff_t>(pattern.size());

        if (text_size < pat_size) return std::nullopt;

        std::ptrdiff_t i = 0;

        while (i <= text_size - pat_size) {
            std::ptrdiff_t j = pat_size - 1;

            while (j >= 0 && pattern[j] == *(first + i + j)) {
                --j;
            }

            if (j < 0) {
                return first + i;
            }

            auto bad_char_it = bad_char_skip.find(*(first + i + j));
            std::ptrdiff_t bad_char = (bad_char_it != bad_char_skip.end())
                                      ? j - bad_char_it->second
                                      : j + 1;

            i += std::max(good_suffix_skip[j], bad_char);
        }

        return std::nullopt;
    }
};

// Rabin-Karp algorithm with rolling hash
template<typename Iterator>
    requires forward_iterator<Iterator>
class rabin_karp {
private:
    static constexpr std::size_t prime = 31;
    static constexpr std::size_t mod = 1000000009;

    std::size_t compute_hash(Iterator first, Iterator last) const {
        std::size_t hash = 0;
        std::size_t pow = 1;

        for (auto it = first; it != last; ++it) {
            hash = (hash + static_cast<std::size_t>(*it) * pow) % mod;
            pow = (pow * prime) % mod;
        }

        return hash;
    }

public:
    std::optional<Iterator> search(Iterator text_first, Iterator text_last,
                                   Iterator pat_first, Iterator pat_last) const {
        auto pat_size = std::distance(pat_first, pat_last);
        auto text_size = std::distance(text_first, text_last);

        if (pat_size == 0) return text_first;
        if (text_size < pat_size) return std::nullopt;

        std::size_t pat_hash = compute_hash(pat_first, pat_last);
        std::size_t text_hash = 0;
        std::size_t pow = 1;

        // Compute pow^(pat_size-1)
        for (std::ptrdiff_t i = 0; i < pat_size - 1; ++i) {
            pow = (pow * prime) % mod;
        }

        // Initial hash of text
        auto text_end = text_first;
        std::advance(text_end, pat_size);
        text_hash = compute_hash(text_first, text_end);

        // Sliding window
        auto window_start = text_first;

        for (std::ptrdiff_t i = 0; i <= text_size - pat_size; ++i) {
            if (text_hash == pat_hash) {
                // Double-check with actual comparison
                if (std::equal(pat_first, pat_last, window_start)) {
                    return window_start;
                }
            }

            if (i < text_size - pat_size) {
                // Roll the hash
                auto old_char = *window_start;
                ++window_start;
                auto new_char = *std::next(window_start, pat_size - 1);

                text_hash = (text_hash + mod - (static_cast<std::size_t>(old_char) * pow) % mod) % mod;
                text_hash = (text_hash * prime + static_cast<std::size_t>(new_char)) % mod;
            }
        }

        return std::nullopt;
    }
};

// Aho-Corasick algorithm for multiple pattern matching
template<typename CharT>
class aho_corasick {
private:
    struct node {
        std::unordered_map<CharT, std::unique_ptr<node>> children;
        node* failure = nullptr;
        std::vector<std::size_t> output;  // Pattern IDs that end at this node
    };

    std::unique_ptr<node> root;
    std::vector<std::basic_string<CharT>> patterns;

    void build_failure_links() {
        std::queue<node*> q;

        // Initialize failure links for root's children
        for (auto& [ch, child] : root->children) {
            child->failure = root.get();
            q.push(child.get());
        }

        while (!q.empty()) {
            node* curr = q.front();
            q.pop();

            for (auto& [ch, child] : curr->children) {
                q.push(child.get());

                node* f = curr->failure;
                while (f && f->children.find(ch) == f->children.end()) {
                    f = f->failure;
                }

                child->failure = f ? f->children[ch].get() : root.get();

                // Merge output patterns
                if (child->failure) {
                    child->output.insert(child->output.end(),
                                        child->failure->output.begin(),
                                        child->failure->output.end());
                }
            }
        }
    }

public:
    aho_corasick() : root(std::make_unique<node>()) {}

    void add_pattern(const std::basic_string<CharT>& pattern) {
        std::size_t id = patterns.size();
        patterns.push_back(pattern);

        node* curr = root.get();
        for (CharT ch : pattern) {
            if (curr->children.find(ch) == curr->children.end()) {
                curr->children[ch] = std::make_unique<node>();
            }
            curr = curr->children[ch].get();
        }

        curr->output.push_back(id);
    }

    void build() {
        build_failure_links();
    }

    std::vector<std::pair<std::size_t, std::size_t>> search(const std::basic_string<CharT>& text) {
        std::vector<std::pair<std::size_t, std::size_t>> matches;

        node* curr = root.get();

        for (std::size_t i = 0; i < text.size(); ++i) {
            CharT ch = text[i];

            while (curr != root.get() && curr->children.find(ch) == curr->children.end()) {
                curr = curr->failure;
            }

            if (curr->children.find(ch) != curr->children.end()) {
                curr = curr->children[ch].get();
            }

            for (std::size_t pattern_id : curr->output) {
                std::size_t pos = i + 1 - patterns[pattern_id].size();
                matches.push_back({pos, pattern_id});
            }
        }

        return matches;
    }
};

// Suffix array construction using SA-IS algorithm (simplified)
template<typename Iterator>
    requires random_access_iterator<Iterator>
std::vector<std::ptrdiff_t> build_suffix_array(Iterator first, Iterator last) {
    auto n = std::distance(first, last);
    std::vector<std::ptrdiff_t> sa(n);

    if (n == 0) return sa;
    if (n == 1) {
        sa[0] = 0;
        return sa;
    }

    // Simple O(n log n) implementation using sorting
    std::iota(sa.begin(), sa.end(), 0);

    std::sort(sa.begin(), sa.end(), [first, n](std::ptrdiff_t i, std::ptrdiff_t j) {
        auto it1 = first + i;
        auto it2 = first + j;
        auto end = first + n;

        while (it1 != end && it2 != end) {
            if (*it1 != *it2) return *it1 < *it2;
            ++it1;
            ++it2;
        }

        return it2 != end;
    });

    return sa;
}

// LCP array construction
template<typename Iterator>
    requires random_access_iterator<Iterator>
std::vector<std::ptrdiff_t> build_lcp_array(Iterator first, Iterator last,
                                             const std::vector<std::ptrdiff_t>& sa) {
    auto n = std::distance(first, last);
    std::vector<std::ptrdiff_t> lcp(n, 0);
    std::vector<std::ptrdiff_t> rank(n);

    for (std::ptrdiff_t i = 0; i < n; ++i) {
        rank[sa[i]] = i;
    }

    std::ptrdiff_t k = 0;

    for (std::ptrdiff_t i = 0; i < n; ++i) {
        if (rank[i] == n - 1) {
            k = 0;
            continue;
        }

        std::ptrdiff_t j = sa[rank[i] + 1];

        while (i + k < n && j + k < n && *(first + i + k) == *(first + j + k)) {
            ++k;
        }

        lcp[rank[i]] = k;

        if (k > 0) --k;
    }

    return lcp;
}

// Longest common subsequence
template<typename Iterator1, typename Iterator2>
    requires forward_iterator<Iterator1> && forward_iterator<Iterator2>
std::vector<std::iter_value_t<Iterator1>> longest_common_subsequence(
    Iterator1 first1, Iterator1 last1,
    Iterator2 first2, Iterator2 last2) {

    auto m = std::distance(first1, last1);
    auto n = std::distance(first2, last2);

    std::vector<std::vector<std::ptrdiff_t>> dp(m + 1, std::vector<std::ptrdiff_t>(n + 1, 0));

    auto it1 = first1;
    for (std::ptrdiff_t i = 1; i <= m; ++i, ++it1) {
        auto it2 = first2;
        for (std::ptrdiff_t j = 1; j <= n; ++j, ++it2) {
            if (*it1 == *it2) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    // Reconstruct LCS
    std::vector<std::iter_value_t<Iterator1>> lcs;
    std::ptrdiff_t i = m, j = n;

    while (i > 0 && j > 0) {
        auto val1 = *std::next(first1, i - 1);
        auto val2 = *std::next(first2, j - 1);

        if (val1 == val2) {
            lcs.push_back(val1);
            --i;
            --j;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            --i;
        } else {
            --j;
        }
    }

    std::reverse(lcs.begin(), lcs.end());
    return lcs;
}

// Edit distance (Levenshtein)
template<typename Iterator1, typename Iterator2>
    requires forward_iterator<Iterator1> && forward_iterator<Iterator2>
std::size_t edit_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2) {
    auto m = std::distance(first1, last1);
    auto n = std::distance(first2, last2);

    if (m == 0) return n;
    if (n == 0) return m;

    std::vector<std::size_t> prev(n + 1);
    std::vector<std::size_t> curr(n + 1);

    std::iota(prev.begin(), prev.end(), 0);

    auto it1 = first1;
    for (std::size_t i = 1; i <= static_cast<std::size_t>(m); ++i, ++it1) {
        curr[0] = i;

        auto it2 = first2;
        for (std::size_t j = 1; j <= static_cast<std::size_t>(n); ++j, ++it2) {
            std::size_t cost = (*it1 == *it2) ? 0 : 1;

            curr[j] = std::min({
                prev[j] + 1,      // deletion
                curr[j-1] + 1,    // insertion
                prev[j-1] + cost  // substitution
            });
        }

        std::swap(prev, curr);
    }

    return prev[n];
}

// Damerau-Levenshtein distance (with transpositions)
template<typename Iterator1, typename Iterator2>
    requires random_access_iterator<Iterator1> && random_access_iterator<Iterator2>
std::size_t damerau_levenshtein_distance(Iterator1 first1, Iterator1 last1,
                                         Iterator2 first2, Iterator2 last2) {
    auto m = std::distance(first1, last1);
    auto n = std::distance(first2, last2);

    if (m == 0) return n;
    if (n == 0) return m;

    std::vector<std::vector<std::size_t>> dp(m + 1, std::vector<std::size_t>(n + 1));

    for (std::size_t i = 0; i <= static_cast<std::size_t>(m); ++i) dp[i][0] = i;
    for (std::size_t j = 0; j <= static_cast<std::size_t>(n); ++j) dp[0][j] = j;

    for (std::size_t i = 1; i <= static_cast<std::size_t>(m); ++i) {
        for (std::size_t j = 1; j <= static_cast<std::size_t>(n); ++j) {
            std::size_t cost = (*(first1 + i - 1) == *(first2 + j - 1)) ? 0 : 1;

            dp[i][j] = std::min({
                dp[i-1][j] + 1,      // deletion
                dp[i][j-1] + 1,      // insertion
                dp[i-1][j-1] + cost  // substitution
            });

            // Transposition
            if (i > 1 && j > 1 &&
                *(first1 + i - 1) == *(first2 + j - 2) &&
                *(first1 + i - 2) == *(first2 + j - 1)) {
                dp[i][j] = std::min(dp[i][j], dp[i-2][j-2] + cost);
            }
        }
    }

    return dp[m][n];
}

// Rope data structure for efficient string operations
template<typename CharT>
class rope {
private:
    struct node {
        std::basic_string<CharT> data;
        std::unique_ptr<node> left;
        std::unique_ptr<node> right;
        std::size_t weight;  // Total size of left subtree

        node(const std::basic_string<CharT>& s)
            : data(s), weight(s.size()) {}

        node(std::unique_ptr<node> l, std::unique_ptr<node> r)
            : left(std::move(l)), right(std::move(r)) {
            weight = left ? get_size(left.get()) : 0;
        }

        static std::size_t get_size(const node* n) {
            if (!n) return 0;
            if (!n->left && !n->right) return n->data.size();
            return get_size(n->left.get()) + get_size(n->right.get());
        }
    };

    std::unique_ptr<node> root;
    static constexpr std::size_t leaf_size_threshold = 1024;

    CharT index_impl(const node* n, std::size_t i) const {
        if (!n->left && !n->right) {
            return n->data[i];
        }

        if (i < n->weight) {
            return index_impl(n->left.get(), i);
        } else {
            return index_impl(n->right.get(), i - n->weight);
        }
    }

    std::unique_ptr<node> concat_impl(std::unique_ptr<node> left,
                                      std::unique_ptr<node> right) {
        if (!left) return right;
        if (!right) return left;

        return std::make_unique<node>(std::move(left), std::move(right));
    }

public:
    rope() = default;

    explicit rope(const std::basic_string<CharT>& s) {
        if (s.size() <= leaf_size_threshold) {
            root = std::make_unique<node>(s);
        } else {
            std::size_t mid = s.size() / 2;
            auto left = std::make_unique<node>(s.substr(0, mid));
            auto right = std::make_unique<node>(s.substr(mid));
            root = std::make_unique<node>(std::move(left), std::move(right));
        }
    }

    CharT operator[](std::size_t i) const {
        return index_impl(root.get(), i);
    }

    std::size_t size() const {
        return node::get_size(root.get());
    }

    void concat(rope&& other) {
        root = concat_impl(std::move(root), std::move(other.root));
    }

    rope split(std::size_t pos) {
        // Simplified: would need to implement tree splitting
        rope result;
        return result;
    }
};

// Z-algorithm for pattern matching
template<typename Iterator>
    requires random_access_iterator<Iterator>
std::vector<std::ptrdiff_t> z_algorithm(Iterator first, Iterator last) {
    auto n = std::distance(first, last);
    std::vector<std::ptrdiff_t> z(n, 0);

    if (n == 0) return z;

    z[0] = n;
    std::ptrdiff_t l = 0, r = 0;

    for (std::ptrdiff_t i = 1; i < n; ++i) {
        if (i <= r) {
            z[i] = std::min(r - i + 1, z[i - l]);
        }

        while (i + z[i] < n && *(first + z[i]) == *(first + i + z[i])) {
            ++z[i];
        }

        if (i + z[i] - 1 > r) {
            l = i;
            r = i + z[i] - 1;
        }
    }

    return z;
}

// Manacher's algorithm for finding palindromes
template<typename Iterator>
    requires random_access_iterator<Iterator>
std::vector<std::ptrdiff_t> manacher_odd(Iterator first, Iterator last) {
    auto n = std::distance(first, last);
    std::vector<std::ptrdiff_t> p(n, 0);

    std::ptrdiff_t l = 0, r = -1;

    for (std::ptrdiff_t i = 0; i < n; ++i) {
        std::ptrdiff_t k = (i > r) ? 1 : std::min(p[l + r - i], r - i + 1);

        while (i - k >= 0 && i + k < n && *(first + i - k) == *(first + i + k)) {
            ++k;
        }

        p[i] = k--;

        if (i + k > r) {
            l = i - k;
            r = i + k;
        }
    }

    return p;
}

template<typename Iterator>
    requires random_access_iterator<Iterator>
std::vector<std::ptrdiff_t> manacher_even(Iterator first, Iterator last) {
    auto n = std::distance(first, last);
    std::vector<std::ptrdiff_t> p(n, 0);

    std::ptrdiff_t l = 0, r = -1;

    for (std::ptrdiff_t i = 0; i < n; ++i) {
        std::ptrdiff_t k = (i > r) ? 0 : std::min(p[l + r - i + 1], r - i + 1);

        while (i - k - 1 >= 0 && i + k < n && *(first + i - k - 1) == *(first + i + k)) {
            ++k;
        }

        p[i] = k--;

        if (i + k > r) {
            l = i - k - 1;
            r = i + k;
        }
    }

    return p;
}

// Longest palindromic substring
template<typename Iterator>
    requires random_access_iterator<Iterator>
std::pair<Iterator, Iterator> longest_palindrome(Iterator first, Iterator last) {
    auto n = std::distance(first, last);
    if (n == 0) return {first, first};

    auto odd = manacher_odd(first, last);
    auto even = manacher_even(first, last);

    std::ptrdiff_t max_len = 0;
    Iterator result_first = first, result_last = first;

    // Check odd-length palindromes
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        std::ptrdiff_t len = 2 * odd[i] - 1;
        if (len > max_len) {
            max_len = len;
            result_first = first + i - odd[i] + 1;
            result_last = first + i + odd[i];
        }
    }

    // Check even-length palindromes
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        std::ptrdiff_t len = 2 * even[i];
        if (len > max_len) {
            max_len = len;
            result_first = first + i - even[i];
            result_last = first + i + even[i];
        }
    }

    return {result_first, result_last};
}

} // namespace stepanov