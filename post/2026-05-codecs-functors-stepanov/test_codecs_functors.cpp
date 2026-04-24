#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <span>
#include <vector>
#include "codecs_functors.hpp"

using namespace codecs_functors;

TEST(CodecsFunctorsTest, BitWriterWritesIndividualBits) {
    std::array<std::uint8_t, 2> buf{};
    std::span<std::uint8_t> buf_span(buf);
    BitWriter w(buf_span);
    // Pattern written in order: 1 0 1 1 0 0 0 1.
    // LSB-first packing: bit0=1, bit1=0, bit2=1, bit3=1, bit4=0, bit5=0, bit6=0, bit7=1.
    // byte = 1 + 4 + 8 + 128 = 141 = 0b10001101 = 0x8D.
    w.write(true);
    w.write(false);
    w.write(true);
    w.write(true);
    w.write(false);
    w.write(false);
    w.write(false);
    w.write(true);
    EXPECT_EQ(w.bytes_written(), 1u);
    EXPECT_EQ(buf[0], 0x8Du);
}

TEST(CodecsFunctorsTest, BitWriterAlignsPartialByte) {
    std::array<std::uint8_t, 2> buf{};
    std::span<std::uint8_t> buf_span(buf);
    BitWriter w(buf_span);
    w.write(true);
    w.write(true);
    w.write(true);
    w.align();
    EXPECT_EQ(w.bytes_written(), 1u);
    EXPECT_EQ(buf[0], 0x07u);
}

TEST(CodecsFunctorsTest, BitReaderReadsBackBitWriterOutput) {
    std::array<std::uint8_t, 2> buf{};
    std::span<std::uint8_t> buf_span(buf);
    BitWriter w(buf_span);
    const std::vector<bool> pattern{true, false, true, true, false, false, false, true,
                                    false, true, false, true};
    for (bool b : pattern) w.write(b);
    w.align();
    EXPECT_EQ(w.bytes_written(), 2u);

    std::span<const std::uint8_t> read_span(buf.data(), w.bytes_written());
    BitReader r(read_span);
    for (bool b : pattern) {
        EXPECT_EQ(r.read(), b);
    }
}

TEST(CodecsFunctorsTest, ConceptsAreSatisfied) {
    static_assert(BitSink<BitWriter>);
    static_assert(BitSource<BitReader>);
}
