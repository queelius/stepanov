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

namespace {

// Round-trip helper used by codec tests in this file.
template<typename Codec, typename T>
T round_trip(const T& v) {
    std::array<std::uint8_t, 1024> buf{};
    std::span<std::uint8_t> buf_span(buf);
    BitWriter w(buf_span);
    Codec::encode(v, w);
    w.align();
    std::span<const std::uint8_t> read_span(buf.data(), w.bytes_written());
    BitReader r(read_span);
    return Codec::decode(r);
}

}  // namespace

TEST(CodecsFunctorsTest, GammaRoundTrip) {
    for (std::uint64_t n : {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3},
                            std::uint64_t{100}, std::uint64_t{1024},
                            std::uint64_t{1} << 30}) {
        EXPECT_EQ(round_trip<Gamma>(n), n) << "n = " << n;
    }
}

TEST(CodecsFunctorsTest, OptRoundTripPresent) {
    using Codec = Opt<Gamma>;
    Codec::value_type v = std::uint64_t{42};
    EXPECT_EQ(round_trip<Codec>(v), v);
}

TEST(CodecsFunctorsTest, OptRoundTripAbsent) {
    using Codec = Opt<Gamma>;
    Codec::value_type v = std::nullopt;
    EXPECT_EQ(round_trip<Codec>(v), v);
}

TEST(CodecsFunctorsTest, EitherRoundTripLeft) {
    using Codec = Either<Gamma, Gamma>;
    Codec::value_type v{std::in_place_index<0>, std::uint64_t{7}};
    auto out = round_trip<Codec>(v);
    EXPECT_EQ(out.index(), 0u);
    EXPECT_EQ(std::get<0>(out), 7u);
}

TEST(CodecsFunctorsTest, EitherRoundTripRight) {
    using Codec = Either<Gamma, Gamma>;
    Codec::value_type v{std::in_place_index<1>, std::uint64_t{13}};
    auto out = round_trip<Codec>(v);
    EXPECT_EQ(out.index(), 1u);
    EXPECT_EQ(std::get<1>(out), 13u);
}

TEST(CodecsFunctorsTest, Either3RoundTripAllBranches) {
    using Codec = Either3<Gamma, Gamma, Gamma>;
    for (std::size_t branch = 0; branch < 3; ++branch) {
        Codec::value_type v;
        if (branch == 0) v.template emplace<0>(std::uint64_t{1});
        if (branch == 1) v.template emplace<1>(std::uint64_t{2});
        if (branch == 2) v.template emplace<2>(std::uint64_t{3});
        auto out = round_trip<Codec>(v);
        EXPECT_EQ(out.index(), branch);
        if (branch == 0) EXPECT_EQ(std::get<0>(out), 1u);
        if (branch == 1) EXPECT_EQ(std::get<1>(out), 2u);
        if (branch == 2) EXPECT_EQ(std::get<2>(out), 3u);
    }
}

TEST(CodecsFunctorsTest, PairRoundTrip) {
    using Codec = Pair<Gamma, Gamma>;
    Codec::value_type v{std::uint64_t{17}, std::uint64_t{42}};
    auto out = round_trip<Codec>(v);
    EXPECT_EQ(out.first, 17u);
    EXPECT_EQ(out.second, 42u);
}

TEST(CodecsFunctorsTest, PairOfDifferentCodecsRoundTrip) {
    using Codec = Pair<Gamma, Opt<Gamma>>;
    Codec::value_type v{std::uint64_t{5}, std::optional<std::uint64_t>{std::uint64_t{99}}};
    auto out = round_trip<Codec>(v);
    EXPECT_EQ(out.first, 5u);
    ASSERT_TRUE(out.second.has_value());
    EXPECT_EQ(*out.second, 99u);
}

TEST(CodecsFunctorsTest, VecEmpty) {
    using Codec = Vec<Gamma>;
    Codec::value_type v;
    auto out = round_trip<Codec>(v);
    EXPECT_TRUE(out.empty());
}

TEST(CodecsFunctorsTest, VecSingleton) {
    using Codec = Vec<Gamma>;
    Codec::value_type v{std::uint64_t{7}};
    auto out = round_trip<Codec>(v);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], 7u);
}

TEST(CodecsFunctorsTest, VecManyElements) {
    using Codec = Vec<Gamma>;
    Codec::value_type v{1, 2, 3, 5, 8, 13, 21, 34, 55};
    auto out = round_trip<Codec>(v);
    EXPECT_EQ(out, v);
}
