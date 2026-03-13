// utf8_alpha.hpp - UTF-8 alphabetic strings preserving monoid structure
//
// Extends lc_alpha to Unicode: strings of alphabetic code points
// that form a monoid under concatenation. Handles multi-byte UTF-8.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace alga {

/// UTF-8 alphabetic string — monoid under concatenation.
///
/// Like lc_alpha but accepts any Unicode alphabetic character.
/// The monoid structure is identical: concatenation with empty identity.
class utf8_alpha {
    std::string data_;
    std::size_t char_count_;  // number of Unicode code points

    utf8_alpha(std::string s, std::size_t count)
        : data_(std::move(s)), char_count_(count) {}

    friend auto make_utf8_alpha(std::string_view) -> std::optional<utf8_alpha>;

public:
    utf8_alpha() : char_count_(0) {}

    [[nodiscard]] auto operator*(const utf8_alpha& other) const -> utf8_alpha {
        return utf8_alpha(data_ + other.data_, char_count_ + other.char_count_);
    }

    [[nodiscard]] auto str() const -> const std::string& { return data_; }
    [[nodiscard]] auto char_count() const -> std::size_t { return char_count_; }
    [[nodiscard]] auto byte_size() const -> std::size_t { return data_.size(); }
    [[nodiscard]] auto empty() const -> bool { return data_.empty(); }

    /// Extract Unicode code points
    [[nodiscard]] auto codepoints() const -> std::vector<uint32_t> {
        std::vector<uint32_t> result;
        auto it = data_.begin();
        while (it != data_.end()) {
            uint32_t cp = 0;
            auto byte = static_cast<uint8_t>(*it);

            if (byte < 0x80) {
                cp = byte;
                ++it;
            } else if ((byte & 0xE0) == 0xC0) {
                cp = byte & 0x1F;
                ++it; cp = (cp << 6) | (static_cast<uint8_t>(*it) & 0x3F); ++it;
            } else if ((byte & 0xF0) == 0xE0) {
                cp = byte & 0x0F;
                ++it; cp = (cp << 6) | (static_cast<uint8_t>(*it) & 0x3F);
                ++it; cp = (cp << 6) | (static_cast<uint8_t>(*it) & 0x3F); ++it;
            } else if ((byte & 0xF8) == 0xF0) {
                cp = byte & 0x07;
                ++it; cp = (cp << 6) | (static_cast<uint8_t>(*it) & 0x3F);
                ++it; cp = (cp << 6) | (static_cast<uint8_t>(*it) & 0x3F);
                ++it; cp = (cp << 6) | (static_cast<uint8_t>(*it) & 0x3F); ++it;
            }
            result.push_back(cp);
        }
        return result;
    }

    auto operator<=>(const utf8_alpha&) const = default;
    bool operator==(const utf8_alpha&) const = default;
};

namespace detail {

/// Check if a Unicode code point is alphabetic (simplified).
/// Covers Latin, Greek, Cyrillic, and basic CJK ranges.
inline bool is_unicode_alpha(uint32_t cp) {
    // ASCII lowercase/uppercase
    if (cp >= 0x41 && cp <= 0x5A) return true;   // A-Z
    if (cp >= 0x61 && cp <= 0x7A) return true;   // a-z
    // Latin Extended
    if (cp >= 0xC0 && cp <= 0x024F) return true;
    // Greek
    if (cp >= 0x0370 && cp <= 0x03FF) return true;
    // Cyrillic
    if (cp >= 0x0400 && cp <= 0x04FF) return true;
    // CJK Unified Ideographs (subset)
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    // Hiragana
    if (cp >= 0x3040 && cp <= 0x309F) return true;
    // Katakana
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;

    return false;
}

} // namespace detail

/// Factory: validates that all code points are alphabetic.
inline auto make_utf8_alpha(std::string_view sv) -> std::optional<utf8_alpha> {
    std::size_t count = 0;
    auto it = sv.begin();

    while (it != sv.end()) {
        uint32_t cp = 0;
        auto byte = static_cast<uint8_t>(*it);
        int len = 0;

        if (byte < 0x80) {
            cp = byte; len = 1;
        } else if ((byte & 0xE0) == 0xC0) {
            cp = byte & 0x1F; len = 2;
        } else if ((byte & 0xF0) == 0xE0) {
            cp = byte & 0x0F; len = 3;
        } else if ((byte & 0xF8) == 0xF0) {
            cp = byte & 0x07; len = 4;
        } else {
            return std::nullopt;  // Invalid UTF-8
        }

        if (std::distance(it, sv.end()) < len) return std::nullopt;

        ++it;
        for (int i = 1; i < len; ++i) {
            if ((static_cast<uint8_t>(*it) & 0xC0) != 0x80) return std::nullopt;
            cp = (cp << 6) | (static_cast<uint8_t>(*it) & 0x3F);
            ++it;
        }

        if (!detail::is_unicode_alpha(cp)) return std::nullopt;
        ++count;
    }

    return utf8_alpha(std::string(sv), count);
}

} // namespace alga
