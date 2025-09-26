# Compositional Allocators System

## Overview

The Stepanov library provides a sophisticated compositional allocator system following generic programming principles. Allocators can be composed together like LEGO blocks to create complex memory management strategies with zero runtime overhead.

## Design Philosophy

1. **Composability**: Allocators combine through template composition
2. **Zero-overhead abstraction**: Compile-time composition with no runtime cost
3. **Policy-based design**: Flexible configuration through template parameters
4. **Generic programming**: Works with any type satisfying required concepts
5. **STL compatibility**: Can be used with standard containers

## Basic Allocators

### heap_allocator
Standard malloc/free wrapper:
```cpp
heap_allocator alloc;
void* p = alloc.allocate(100);
alloc.deallocate(p, 100);
```

### fixed_pool_allocator<BlockSize, NumBlocks>
Fixed-size memory pool:
```cpp
fixed_pool_allocator<64, 100> pool;  // 100 blocks of 64 bytes
void* p = pool.allocate(64);
pool.deallocate(p, 64);
```

### region_allocator
Bulk allocation with single deallocation:
```cpp
region_allocator<> region(65536);  // 64KB regions
void* p = region.allocate(100);
// Individual deallocations do nothing
region.deallocate_all();  // Free everything at once
```

### stack_allocator_with_markers<Size>
Stack-based allocation with save/restore points:
```cpp
stack_allocator_with_markers<4096> stack;
auto marker = stack.save_state();
void* p = stack.allocate(100);
stack.restore_state(marker);  // Rewind allocations
```

### freelist_allocator<Allocator>
Maintains a freelist for recycling:
```cpp
freelist_allocator<heap_allocator> freelist;
void* p = freelist.allocate(100);
freelist.deallocate(p, 100);  // Added to freelist
void* p2 = freelist.allocate(100);  // Reuses from freelist
```

### slab_allocator<T, ObjectsPerSlab>
Efficient for many same-sized objects:
```cpp
slab_allocator<MyClass, 64> slab;
MyClass* obj = slab.allocate();
slab.deallocate(obj);
```

### bitmapped_block_allocator<BlockSize, NumBlocks>
Uses bitmap for tracking allocations:
```cpp
bitmapped_block_allocator<128, 256> bitmap;
void* p = bitmap.allocate(128);
bitmap.deallocate(p, 128);
```

## Allocator Combinators

### fallback_allocator<Primary, Fallback>
Try primary, fallback to secondary:
```cpp
using my_alloc = fallback_allocator<
    fixed_pool_allocator<64, 100>,
    heap_allocator
>;
```

### segregator_allocator<Threshold, Small, Large>
Route allocations by size:
```cpp
using my_alloc = segregator_allocator<
    256,                              // Threshold
    fixed_pool_allocator<256, 100>,  // For small allocations
    heap_allocator                   // For large allocations
>;
```

### affixed_allocator<Allocator, Prefix, Suffix>
Add metadata to allocations:
```cpp
struct debug_info {
    std::size_t size;
    std::uint32_t magic;
};

using debug_alloc = affixed_allocator<
    heap_allocator,
    debug_info,  // Prefix
    void         // No suffix
>;
```

### stats_allocator<Allocator>
Track allocation statistics:
```cpp
stats_allocator<heap_allocator> alloc;
void* p = alloc.allocate(100);
std::cout << "Allocations: " << alloc.stats().allocations << "\n";
std::cout << "Peak bytes: " << alloc.stats().peak_bytes << "\n";
```

### synchronized_allocator<Allocator>
Thread-safe wrapper:
```cpp
using safe_alloc = synchronized_allocator<heap_allocator>;
```

### thread_local_allocator<Allocator>
Per-thread allocator instances:
```cpp
using tls_alloc = thread_local_allocator<heap_allocator>;
```

## Complex Compositions

### Example 1: Embedded System Allocator
```cpp
using embedded_alloc = fallback_allocator<
    stack_allocator<4096>,           // Use stack first
    fixed_pool_allocator<256, 32>    // Fallback to fixed pool
>;
```

### Example 2: Game Engine Allocator
```cpp
using game_alloc = stats_allocator<
    segregator_allocator<
        64,                               // Small threshold
        fixed_pool_allocator<64, 1024>,  // Pool for particles
        segregator_allocator<
            1024,                         // Medium threshold
            freelist_allocator<heap_allocator>,  // Recycled heap
            heap_allocator                // Direct heap for large
        >
    >
>;
```

### Example 3: Production System Allocator
```cpp
using production_alloc = synchronized_allocator<
    affixed_allocator<
        stats_allocator<
            segregator_allocator<
                256,
                bitmapped_block_allocator<256, 128>,
                freelist_allocator<heap_allocator>
            >
        >,
        allocation_header,
        void
    >
>;
```

## STL Integration

Use custom allocators with STL containers:

```cpp
// Create allocator
using my_alloc = fallback_allocator<
    fixed_pool_allocator<32, 1024>,
    heap_allocator
>;
my_alloc allocator;

// Adapt for STL
using stl_alloc = stl_allocator_adaptor<int, my_alloc>;
std::vector<int, stl_alloc> vec{stl_alloc(&allocator)};

vec.push_back(42);  // Uses custom allocator
```

## Performance Characteristics

| Allocator | Allocation | Deallocation | Memory Overhead | Fragmentation |
|-----------|------------|--------------|-----------------|---------------|
| heap_allocator | O(log n) | O(log n) | Low | High |
| fixed_pool | O(1) | O(1) | Fixed | None |
| region | O(1) | O(1) batch | Very low | None |
| stack | O(1) | O(1) batch | None | None |
| freelist | O(1) typical | O(1) | Low | Medium |
| slab | O(1) | O(1) | Medium | Low |
| bitmapped | O(n) worst | O(1) | Fixed bitmap | None |

## Usage Guidelines

1. **For embedded systems**: Use stack or fixed pool allocators
2. **For game engines**: Use segregator with pools for common sizes
3. **For servers**: Use region allocators per request
4. **For general apps**: Use freelist with heap fallback
5. **For debugging**: Wrap with affixed and stats allocators
6. **For multithreading**: Wrap with synchronized or thread_local

## Advanced Features

### Custom Allocator Traits
Allocators can optionally provide:
- `owns(p)`: Check if pointer belongs to allocator
- `expand(p, old_size, delta)`: Try to expand in-place
- `reallocate(p, old_size, new_size)`: Optimized reallocation
- `deallocate_all()`: Bulk deallocation

### Alignment Support
All allocators support custom alignment:
```cpp
void* p = alloc.allocate(size, alignment);
```

### Zero-Cost Composition
Template-based composition ensures no runtime overhead:
```cpp
// This complex composition has zero runtime cost
using complex = fallback_allocator<
    segregator_allocator<256,
        fixed_pool_allocator<256, 100>,
        heap_allocator
    >,
    heap_allocator
>;
```

## Implementation Notes

1. All allocators follow the same interface for composability
2. Template metaprogramming enables compile-time optimization
3. SFINAE and concepts ensure type safety
4. Header-only implementation for maximum optimization
5. No virtual functions or runtime polymorphism

## See Also

- `example_allocators.cpp`: Comprehensive examples
- `test_allocators.cpp`: Unit tests
- `include/stepanov/allocators.hpp`: Full implementation