// similarity.hpp - String distance and similarity metrics
//
// These are useful, self-contained string algorithms that complement
// the parser combinator core. They measure how "close" two strings are.
//
//   levenshtein(a, b) — edit distance (insertions, deletions, substitutions)
//   jaro_winkler(a, b) — similarity score in [0, 1]

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <vector>

namespace alga {

// =============================================================================
// Levenshtein Distance
// =============================================================================

/// Classic edit distance: minimum number of single-character insertions,
/// deletions, or substitutions to transform one string into another.
///
/// Complexity: O(n * m) time and O(min(n, m)) space (single-row optimization).
inline auto levenshtein(std::string_view a, std::string_view b) -> std::size_t {
    // Ensure a is the shorter string for space optimization
    if (a.size() > b.size()) std::swap(a, b);

    const auto m = a.size();
    const auto n = b.size();

    // Single row of the DP matrix
    std::vector<std::size_t> row(m + 1);
    for (std::size_t j = 0; j <= m; ++j) row[j] = j;

    for (std::size_t i = 1; i <= n; ++i) {
        std::size_t prev = row[0];
        row[0] = i;

        for (std::size_t j = 1; j <= m; ++j) {
            std::size_t cost = (a[j - 1] == b[i - 1]) ? 0 : 1;
            std::size_t temp = row[j];
            row[j] = std::min({
                row[j] + 1,       // deletion
                row[j - 1] + 1,   // insertion
                prev + cost        // substitution
            });
            prev = temp;
        }
    }

    return row[m];
}

// =============================================================================
// Jaro-Winkler Similarity
// =============================================================================

/// Jaro-Winkler similarity score in [0, 1].
/// Higher values indicate more similar strings.
/// The Winkler modification boosts scores for strings sharing a common prefix.
///
/// Returns 1.0 for identical strings, 0.0 for completely different strings.
inline auto jaro_winkler(std::string_view s1, std::string_view s2,
                         double winkler_boost = 0.1) -> double {
    if (s1.empty() && s2.empty()) return 1.0;
    if (s1.empty() || s2.empty()) return 0.0;

    const auto len1 = s1.size();
    const auto len2 = s2.size();

    // Match window: characters within floor(max(|s1|,|s2|)/2) - 1
    const auto match_distance =
        static_cast<std::size_t>(std::max(len1, len2) / 2) > 0
            ? static_cast<std::size_t>(std::max(len1, len2) / 2) - 1
            : 0;

    std::vector<bool> s1_matched(len1, false);
    std::vector<bool> s2_matched(len2, false);

    std::size_t matches = 0;
    std::size_t transpositions = 0;

    // Find matching characters
    for (std::size_t i = 0; i < len1; ++i) {
        auto start = (i > match_distance) ? i - match_distance : 0;
        auto end = std::min(i + match_distance + 1, len2);

        for (auto j = start; j < end; ++j) {
            if (s2_matched[j] || s1[i] != s2[j]) continue;
            s1_matched[i] = true;
            s2_matched[j] = true;
            ++matches;
            break;
        }
    }

    if (matches == 0) return 0.0;

    // Count transpositions
    std::size_t k = 0;
    for (std::size_t i = 0; i < len1; ++i) {
        if (!s1_matched[i]) continue;
        while (!s2_matched[k]) ++k;
        if (s1[i] != s2[k]) ++transpositions;
        ++k;
    }

    double m = static_cast<double>(matches);
    double jaro = (m / static_cast<double>(len1) +
                   m / static_cast<double>(len2) +
                   (m - static_cast<double>(transpositions) / 2.0) / m) / 3.0;

    // Winkler boost for common prefix (up to 4 characters)
    std::size_t prefix_len = 0;
    auto max_prefix = std::min({len1, len2, std::size_t{4}});
    for (std::size_t i = 0; i < max_prefix; ++i) {
        if (s1[i] != s2[i]) break;
        ++prefix_len;
    }

    return jaro + static_cast<double>(prefix_len) * winkler_boost * (1.0 - jaro);
}

} // namespace alga
