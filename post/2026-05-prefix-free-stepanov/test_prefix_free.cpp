#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <span>
#include <vector>
#include "prefix_free.hpp"

using namespace prefix_free;

TEST(PrefixFreeTest, BitIORoundTrip) {
    std::array<std::uint8_t, 2> buf{};
    std::span<std::uint8_t> buf_span(buf);
    BitWriter w(buf_span);
    const std::vector<bool> pattern{true, false, true, true, false, false, true};
    for (bool b : pattern) w.write(b);
    w.align();

    std::span<const std::uint8_t> read_span(buf.data(), w.bytes_written());
    BitReader r(read_span);
    for (bool b : pattern) {
        EXPECT_EQ(r.read(), b);
    }
}

TEST(PrefixFreeTest, ConceptsAreSatisfied) {
    static_assert(BitSink<BitWriter>);
    static_assert(BitSource<BitReader>);
}
