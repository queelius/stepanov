# Stepanov Compression Framework

## Design Philosophy

This compression module embodies the principle of "less but better":

- **Simplicity over complexity**: Each algorithm does one thing well
- **Speed over maximum compression**: Fast defaults that are "good enough"
- **Composability over monoliths**: Small pieces that combine elegantly
- **Generic over specific**: Works with any probability model
- **Educational over obfuscated**: Code that teaches compression fundamentals

## Core Components

### 1. Classic Algorithms (Just the Best Ones)

```cpp
// LZ77 - Simple, effective, universal
lz77_compressor lz77;
auto compressed = lz77.compress(data);

// Fast LZ - When speed matters most
fast_lz_compressor fast;
auto compressed = fast.compress(data);  // Optimized for speed

// Huffman - Classic, elegant, educational
huffman_compressor huffman;
auto compressed = huffman.compress(data);
```

### 2. Simple Transforms

```cpp
// Run-length encoding for repetitive data
run_length_transform rle;
auto transformed = rle.forward(data);

// Move-to-front for improving locality
move_to_front_transform mtf;
auto transformed = mtf.forward(data);
```

### 3. Composable Pipelines

The real power comes from composition:

```cpp
// Elegant pipeline composition
auto compressor = make_pipeline(
    move_to_front_transform(),     // Improve locality
    run_length_transform(),         // Handle runs
    fast_lz_compressor()           // Final compression
);

auto compressed = compressor.compress(data);
```

### 4. Generic Compression Concepts

The framework defines clean concepts for extensibility:

```cpp
template<typename C>
concept Compressor = requires(C c, byte_span input) {
    { c.compress(input) } -> compressed_data;
    { c.decompress(compressed) } -> byte_vector;
    { c.worst_case_expansion() } -> float;
};

template<typename M>
concept ProbabilityModel = requires(M m, symbol s) {
    { m.probability(s) } -> float;
    { m.cumulative_probability(s) } -> float;
    { m.update(s) };
};
```

### 5. Generic Arithmetic Coding

Works with ANY probability model:

```cpp
// Bring your own model
template<ProbabilityModel Model>
class arithmetic_coder {
    Model model;
    // Generic arithmetic coding using model's probabilities
};

// Use with simple models
arithmetic_coder<uniform_model> simple_coder;
arithmetic_coder<adaptive_model> adaptive_coder;

// Or plug in neural networks, Markov models, etc.
arithmetic_coder<YourCustomModel> custom_coder;
```

### 6. Practical Utilities

```cpp
// Is this data already compressed?
if (is_compressed(data)) {
    // Skip compression
}

// What type of data is this?
auto type = analyze_data(data);  // Returns: text, binary, sparse, repetitive

// Benchmark any compressor
auto result = benchmark(compressor, data);
std::cout << "Ratio: " << result.compression_ratio << "\n";
std::cout << "Speed: " << result.compress_time_ms << " ms\n";
```

## What This Is NOT

- **NOT** a kitchen-sink library with 50 algorithms
- **NOT** optimized to the point of unreadability
- **NOT** a neural compression implementation (bring your own models)
- **NOT** trying to beat specialized libraries in their domains

## What This IS

- **Beautiful code** that teaches compression fundamentals
- **Fast enough** for real-world use
- **Simple enough** to understand completely
- **Generic enough** to extend with your own ideas
- **Composable enough** to build custom solutions

## Examples

### Simple Compression
```cpp
fast_lz_compressor compressor;
auto compressed = compressor.compress(data);
auto decompressed = compressor.decompress(compressed);
```

### Smart Compression
```cpp
// Analyze and choose the best algorithm
auto type = analyze_data(data);
if (type == data_type::text) {
    huffman_compressor comp;
    return comp.compress(data);
} else if (type == data_type::repetitive) {
    auto pipeline = make_pipeline(
        run_length_transform(),
        fast_lz_compressor()
    );
    return pipeline.compress(data);
}
```

### Custom Models
```cpp
// Your own probability model
class markov_model {
    float probability(uint16_t symbol) { /* your code */ }
    float cumulative_probability(uint16_t s) { /* your code */ }
    void update(uint16_t symbol) { /* your code */ }
    size_t symbol_count() { return 256; }
};

// Use it with arithmetic coding
arithmetic_coder<markov_model> coder;
auto compressed = coder.compress(data);
```

## Performance

The framework prioritizes:

1. **Decompression speed** (most important)
2. **Compression speed** (important)
3. **Compression ratio** (good enough)
4. **Memory usage** (reasonable)

Typical performance:
- Fast LZ: ~500 MB/s compression, ~2 GB/s decompression
- LZ77: ~100 MB/s compression, ~300 MB/s decompression
- Arithmetic: ~50 MB/s with adaptive model

## The 80/20 Rule

This library provides 80% of compression functionality with 20% of the complexity:

- ✅ Fast compression for common cases
- ✅ Good enough compression ratios
- ✅ Simple, understandable code
- ✅ Easy composition and extension
- ❌ State-of-the-art compression ratios
- ❌ Highly optimized assembly code
- ❌ Complex neural architectures

## Conclusion

The Stepanov compression framework demonstrates that elegant, simple code can achieve practical compression without over-engineering. It's a toolkit for builders, not a black box for users.

Remember: **More is less. The best code is not when there's nothing left to add, but when there's nothing left to remove.**