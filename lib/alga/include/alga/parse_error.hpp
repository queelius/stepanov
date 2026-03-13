// parse_error.hpp - Error handling with position tracking
//
// Parse errors are values, not exceptions. This follows the algebraic
// approach: failure is a valid outcome, not an exceptional condition.

#pragma once

#include <cstddef>
#include <string>

namespace alga {

// =============================================================================
// Position — Location in source text
// =============================================================================

/// Tracks line/column/byte position in source text.
/// Lines and columns are 1-indexed (human convention).
struct position {
    std::size_t line = 1;
    std::size_t column = 1;
    std::size_t byte_offset = 0;
};

// =============================================================================
// Parse Error
// =============================================================================

/// Rich error information for parse failures.
struct parse_error {
    position pos;
    std::string expected;
    std::string got;
    std::string message;
};

// =============================================================================
// Position Tracker
// =============================================================================

/// Tracks position as we advance through input.
/// Counts newlines to maintain line/column information.
template <typename Iterator>
class position_tracker {
    Iterator start_;
    position pos_;

public:
    explicit position_tracker(Iterator start)
        : start_(start), pos_{} {}

    /// Advance to a new position, updating line/column counts
    void advance(Iterator new_pos) {
        auto it = start_;
        // Reset and recount from the beginning
        pos_ = position{};
        while (it != new_pos) {
            if (*it == '\n') {
                ++pos_.line;
                pos_.column = 1;
            } else {
                ++pos_.column;
            }
            ++pos_.byte_offset;
            ++it;
        }
    }

    /// Get current position
    [[nodiscard]] auto current() const -> position { return pos_; }
};

} // namespace alga
