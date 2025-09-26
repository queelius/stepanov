#pragma once

/**
 * stepanov::text - Text processing with algebraic operations
 *
 * Integrates concepts from Alga library following Stepanov principles:
 * - Monoid-based text operations
 * - Functor transformations
 * - Composable text algorithms
 * - Type erasure for runtime polymorphism
 */

#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <functional>
#include <algorithm>
#include <ranges>
#include <cctype>
#include <sstream>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <random>
#include "type_erasure.hpp"
#include "concepts.hpp"

namespace stepanov::text {

// =============================================================================
// Core text concepts
// =============================================================================

template<typename T>
concept text_like = requires(const T& t) {
    { t.size() } -> std::convertible_to<std::size_t>;
    { t.empty() } -> std::convertible_to<bool>;
    { t[0] } -> std::convertible_to<char>;
};

template<typename F, typename In, typename Out>
concept text_transformer = requires(const F& f, const In& input) {
    { f(input) } -> std::convertible_to<Out>;
};

// =============================================================================
// Monoid operations for text
// =============================================================================

/**
 * Text concatenation monoid
 * Identity: empty string
 * Operation: concatenation
 */
struct concatenation_monoid {
    static std::string identity() { return ""; }

    static std::string operation(const std::string& a, const std::string& b) {
        return a + b;
    }

    template<typename Range>
        requires std::ranges::range<Range> &&
                 std::convertible_to<std::ranges::range_value_t<Range>, std::string>
    static std::string fold(Range&& range) {
        return std::accumulate(std::begin(range), std::end(range),
                              identity(), operation);
    }
};

/**
 * Text joining monoid with separator
 */
template<char Separator = ' '>
struct join_monoid {
    static std::string identity() { return ""; }

    static std::string operation(const std::string& a, const std::string& b) {
        if (a.empty()) return b;
        if (b.empty()) return a;
        return a + Separator + b;
    }

    template<typename Range>
        requires std::ranges::range<Range> &&
                 std::convertible_to<std::ranges::range_value_t<Range>, std::string>
    static std::string fold(Range&& range) {
        return std::accumulate(std::begin(range), std::end(range),
                              identity(), operation);
    }
};

/**
 * Line concatenation monoid
 */
struct line_monoid {
    static std::string identity() { return ""; }

    static std::string operation(const std::string& a, const std::string& b) {
        if (a.empty()) return b;
        if (b.empty()) return a;
        return a + '\n' + b;
    }

    template<typename Range>
    static std::string fold(Range&& range) {
        return std::accumulate(std::begin(range), std::end(range),
                              identity(), operation);
    }
};

// =============================================================================
// Text transformations (Functors)
// =============================================================================

/**
 * Case transformation functors
 */
struct to_lower {
    std::string operator()(std::string_view input) const {
        std::string result;
        result.reserve(input.size());
        std::transform(input.begin(), input.end(),
                      std::back_inserter(result),
                      [](char c) { return std::tolower(c); });
        return result;
    }

    constexpr const char* description() const { return "to_lower"; }
};

struct to_upper {
    std::string operator()(std::string_view input) const {
        std::string result;
        result.reserve(input.size());
        std::transform(input.begin(), input.end(),
                      std::back_inserter(result),
                      [](char c) { return std::toupper(c); });
        return result;
    }

    constexpr const char* description() const { return "to_upper"; }
};

/**
 * Trimming functors
 */
struct trim_left {
    std::string operator()(std::string_view input) const {
        auto it = std::find_if(input.begin(), input.end(),
                               [](char c) { return !std::isspace(c); });
        return std::string(it, input.end());
    }

    constexpr const char* description() const { return "trim_left"; }
};

struct trim_right {
    std::string operator()(std::string_view input) const {
        auto it = std::find_if(input.rbegin(), input.rend(),
                               [](char c) { return !std::isspace(c); });
        return std::string(input.begin(), it.base());
    }

    constexpr const char* description() const { return "trim_right"; }
};

struct trim {
    std::string operator()(std::string_view input) const {
        return trim_left{}(trim_right{}(input));
    }

    constexpr const char* description() const { return "trim"; }
};

/**
 * Text replacement functor
 */
class replace {
private:
    std::string from_;
    std::string to_;

public:
    replace(std::string from, std::string to)
        : from_(std::move(from)), to_(std::move(to)) {}

    std::string operator()(std::string_view input) const {
        if (from_.empty()) return std::string(input);

        std::string result;
        std::size_t pos = 0;
        std::size_t found = input.find(from_);

        while (found != std::string_view::npos) {
            result.append(input, pos, found - pos);
            result.append(to_);
            pos = found + from_.size();
            found = input.find(from_, pos);
        }
        result.append(input, pos);

        return result;
    }

    std::string description() const {
        return "replace('" + from_ + "' -> '" + to_ + "')";
    }
};

/**
 * Regular expression replacement
 */
class regex_replace {
private:
    std::regex pattern_;
    std::string replacement_;

public:
    regex_replace(const std::string& pattern, std::string replacement)
        : pattern_(pattern), replacement_(std::move(replacement)) {}

    std::string operator()(std::string_view input) const {
        return std::regex_replace(std::string(input), pattern_, replacement_);
    }

    std::string description() const {
        return "regex_replace";
    }
};

// =============================================================================
// Text splitting and tokenization
// =============================================================================

/**
 * Split text into tokens
 */
class tokenizer {
private:
    char delimiter_;

public:
    explicit tokenizer(char delimiter = ' ') : delimiter_(delimiter) {}

    std::vector<std::string> operator()(std::string_view input) const {
        std::vector<std::string> tokens;
        std::stringstream ss{std::string(input)};
        std::string token;

        while (std::getline(ss, token, delimiter_)) {
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }

        return tokens;
    }

    std::string description() const {
        return std::string("tokenize('") + delimiter_ + "')";
    }
};

/**
 * Split by whitespace
 */
struct split_words {
    std::vector<std::string> operator()(std::string_view input) const {
        std::vector<std::string> words;
        std::istringstream iss{std::string(input)};
        std::string word;

        while (iss >> word) {
            words.push_back(word);
        }

        return words;
    }

    constexpr const char* description() const { return "split_words"; }
};

/**
 * Split into lines
 */
struct split_lines {
    std::vector<std::string> operator()(std::string_view input) const {
        std::vector<std::string> lines;
        std::stringstream ss{std::string(input)};
        std::string line;

        while (std::getline(ss, line)) {
            lines.push_back(line);
        }

        return lines;
    }

    constexpr const char* description() const { return "split_lines"; }
};

// =============================================================================
// Text analysis
// =============================================================================

/**
 * Character frequency analysis
 */
struct char_frequency {
    std::unordered_map<char, std::size_t> operator()(std::string_view input) const {
        std::unordered_map<char, std::size_t> freq;
        for (char c : input) {
            ++freq[c];
        }
        return freq;
    }

    constexpr const char* description() const { return "char_frequency"; }
};

/**
 * Word frequency analysis
 */
struct word_frequency {
    std::unordered_map<std::string, std::size_t> operator()(std::string_view input) const {
        std::unordered_map<std::string, std::size_t> freq;
        auto words = split_words{}(input);

        for (const auto& word : words) {
            ++freq[to_lower{}(word)];
        }

        return freq;
    }

    constexpr const char* description() const { return "word_frequency"; }
};

/**
 * N-gram extraction
 */
template<std::size_t N>
class ngram_extractor {
public:
    std::vector<std::string> operator()(std::string_view input) const {
        std::vector<std::string> ngrams;
        if (input.size() < N) return ngrams;

        for (std::size_t i = 0; i <= input.size() - N; ++i) {
            ngrams.emplace_back(input.substr(i, N));
        }

        return ngrams;
    }

    std::string description() const {
        return "ngram<" + std::to_string(N) + ">";
    }
};

// =============================================================================
// Monadic composition for text processing
// =============================================================================

/**
 * Optional text transformation with error handling
 */
template<typename F>
class safe_transform {
private:
    F transform_;

public:
    explicit safe_transform(F f) : transform_(std::move(f)) {}

    std::optional<std::string> operator()(std::string_view input) const {
        try {
            return transform_(input);
        } catch (...) {
            return std::nullopt;
        }
    }
};

/**
 * Chain text transformations
 */
template<typename F, typename G>
class composed_transform {
private:
    F first_;
    G second_;

public:
    composed_transform(F f, G g) : first_(std::move(f)), second_(std::move(g)) {}

    auto operator()(std::string_view input) const {
        return second_(first_(input));
    }

    std::string description() const {
        std::string desc = "compose(";
        if constexpr (requires { first_.description(); }) {
            desc += first_.description();
        } else {
            desc += "transform1";
        }
        desc += ", ";
        if constexpr (requires { second_.description(); }) {
            desc += second_.description();
        } else {
            desc += "transform2";
        }
        desc += ")";
        return desc;
    }
};

// Operator for composing transformations
template<typename F, typename G>
composed_transform<F, G> operator|(F f, G g) {
    return composed_transform<F, G>(std::move(f), std::move(g));
}

// =============================================================================
// Pattern matching
// =============================================================================

/**
 * Simple pattern matcher
 */
class pattern_matcher {
private:
    std::string pattern_;

public:
    explicit pattern_matcher(std::string pattern)
        : pattern_(std::move(pattern)) {}

    bool operator()(std::string_view input) const {
        return input.find(pattern_) != std::string_view::npos;
    }

    std::vector<std::size_t> find_all(std::string_view input) const {
        std::vector<std::size_t> positions;
        std::size_t pos = input.find(pattern_);

        while (pos != std::string_view::npos) {
            positions.push_back(pos);
            pos = input.find(pattern_, pos + 1);
        }

        return positions;
    }

    std::string description() const {
        return "pattern('" + pattern_ + "')";
    }
};

/**
 * Regular expression matcher
 */
class regex_matcher {
private:
    std::regex pattern_;
    std::string pattern_str_;

public:
    explicit regex_matcher(const std::string& pattern)
        : pattern_(pattern), pattern_str_(pattern) {}

    bool operator()(std::string_view input) const {
        return std::regex_search(std::string(input), pattern_);
    }

    std::vector<std::string> extract_all(std::string_view input) const {
        std::vector<std::string> matches;
        std::string str(input);
        std::smatch match;

        while (std::regex_search(str, match, pattern_)) {
            matches.push_back(match.str());
            str = match.suffix();
        }

        return matches;
    }

    std::string description() const {
        return "regex('" + pattern_str_ + "')";
    }
};

// =============================================================================
// Text metrics
// =============================================================================

/**
 * Levenshtein distance
 */
struct levenshtein_distance {
    std::size_t operator()(std::string_view a, std::string_view b) const {
        const std::size_t m = a.size();
        const std::size_t n = b.size();

        if (m == 0) return n;
        if (n == 0) return m;

        std::vector<std::vector<std::size_t>> dp(m + 1,
                                                  std::vector<std::size_t>(n + 1));

        for (std::size_t i = 0; i <= m; ++i) dp[i][0] = i;
        for (std::size_t j = 0; j <= n; ++j) dp[0][j] = j;

        for (std::size_t i = 1; i <= m; ++i) {
            for (std::size_t j = 1; j <= n; ++j) {
                const std::size_t cost = (a[i-1] == b[j-1]) ? 0 : 1;
                dp[i][j] = std::min({
                    dp[i-1][j] + 1,      // deletion
                    dp[i][j-1] + 1,      // insertion
                    dp[i-1][j-1] + cost  // substitution
                });
            }
        }

        return dp[m][n];
    }

    constexpr const char* description() const { return "levenshtein_distance"; }
};

/**
 * Hamming distance (for equal-length strings)
 */
struct hamming_distance {
    std::optional<std::size_t> operator()(std::string_view a, std::string_view b) const {
        if (a.size() != b.size()) return std::nullopt;

        std::size_t distance = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) ++distance;
        }

        return distance;
    }

    constexpr const char* description() const { return "hamming_distance"; }
};

// =============================================================================
// Advanced text algorithms
// =============================================================================

/**
 * Porter stemmer (simplified version)
 */
class porter_stemmer {
private:
    bool is_consonant(std::string_view str, std::size_t i) const {
        static const std::string vowels = "aeiouAEIOU";
        return vowels.find(str[i]) == std::string::npos;
    }

    std::size_t measure(std::string_view str) const {
        std::size_t m = 0;
        std::size_t i = 0;
        const std::size_t n = str.size();

        // Skip initial consonants
        while (i < n && is_consonant(str, i)) ++i;

        // Count VC pairs
        while (i < n) {
            // Skip vowels
            while (i < n && !is_consonant(str, i)) ++i;
            if (i >= n) break;

            // Skip consonants
            while (i < n && is_consonant(str, i)) ++i;
            ++m;
        }

        return m;
    }

public:
    std::string operator()(std::string_view word) const {
        std::string stem(word);

        // Step 1a: plurals
        if (stem.ends_with("sses")) {
            stem.erase(stem.size() - 2);
        } else if (stem.ends_with("ies")) {
            stem.erase(stem.size() - 2);
        } else if (stem.ends_with("s") && !stem.ends_with("ss")) {
            stem.pop_back();
        }

        // Step 1b: past tense
        bool flag = false;
        if (stem.ends_with("eed")) {
            if (measure(stem.substr(0, stem.size() - 3)) > 0) {
                stem.erase(stem.size() - 1);
            }
        } else if (stem.ends_with("ed")) {
            stem.erase(stem.size() - 2);
            flag = true;
        } else if (stem.ends_with("ing")) {
            stem.erase(stem.size() - 3);
            flag = true;
        }

        // Additional rules could be added here

        return stem;
    }

    constexpr const char* description() const { return "porter_stemmer"; }
};

/**
 * Text similarity using Jaccard coefficient
 */
struct jaccard_similarity {
    double operator()(std::string_view text1, std::string_view text2) const {
        auto words1 = split_words{}(text1);
        auto words2 = split_words{}(text2);

        std::unordered_set<std::string> set1(words1.begin(), words1.end());
        std::unordered_set<std::string> set2(words2.begin(), words2.end());

        std::size_t intersection = 0;
        for (const auto& word : set1) {
            if (set2.count(word) > 0) {
                ++intersection;
            }
        }

        std::size_t union_size = set1.size() + set2.size() - intersection;

        if (union_size == 0) return 1.0;
        return static_cast<double>(intersection) / union_size;
    }

    constexpr const char* description() const { return "jaccard_similarity"; }
};

// =============================================================================
// Text pipeline builder
// =============================================================================

template<typename... Transforms>
class text_pipeline {
private:
    std::tuple<Transforms...> transforms_;

public:
    explicit text_pipeline(Transforms... transforms)
        : transforms_(std::move(transforms)...) {}

    std::string process(std::string_view input) const {
        return process_impl(input, std::index_sequence_for<Transforms...>{});
    }

    std::string operator()(std::string_view input) const {
        return process(input);
    }

private:
    template<std::size_t... Is>
    std::string process_impl(std::string_view input,
                            std::index_sequence<Is...>) const {
        std::string result(input);
        ((result = std::get<Is>(transforms_)(result)), ...);
        return result;
    }
};

// Factory function for creating pipelines
template<typename... Transforms>
auto make_pipeline(Transforms... transforms) {
    return text_pipeline<Transforms...>(std::move(transforms)...);
}

// =============================================================================
// Type-erased text processor (uses any_text_processor from type_erasure.hpp)
// =============================================================================

using any_string_processor = any_text_processor<std::string_view, std::string>;
using any_tokenizer = any_text_processor<std::string_view, std::vector<std::string>>;
using any_analyzer = any_text_processor<std::string_view, std::unordered_map<std::string, std::size_t>>;

// Factory functions for type-erased processors
inline any_string_processor make_any_processor(auto processor) {
    return any_string_processor(std::move(processor));
}

inline any_tokenizer make_any_tokenizer(auto tokenizer) {
    return any_tokenizer(std::move(tokenizer));
}

// =============================================================================
// Utility functions
// =============================================================================

// Check if string is palindrome
inline bool is_palindrome(std::string_view str) {
    auto cleaned = to_lower{}(trim{}(str));
    std::string reversed(cleaned.rbegin(), cleaned.rend());
    return cleaned == reversed;
}

// Count occurrences of substring
inline std::size_t count_occurrences(std::string_view text, std::string_view pattern) {
    std::size_t count = 0;
    std::size_t pos = text.find(pattern);

    while (pos != std::string_view::npos) {
        ++count;
        pos = text.find(pattern, pos + 1);
    }

    return count;
}

// Generate random text (for testing)
inline std::string generate_random_text(std::size_t length,
                                        const std::string& charset =
                                        "abcdefghijklmnopqrstuvwxyz ") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, charset.size() - 1);

    std::string result;
    result.reserve(length);

    for (std::size_t i = 0; i < length; ++i) {
        result.push_back(charset[dis(gen)]);
    }

    return result;
}

} // namespace stepanov::text