#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <limits>
#include <atomic>
#include <cstring>
#include <array>
#include <vector>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <chrono>
#include "memory.hpp"
#include "concepts.hpp"

namespace stepanov {

// Forward declarations
template<typename Primary, typename Fallback> class fallback_allocator;
template<std::size_t Threshold, typename Small, typename Large> class segregator_allocator;
template<typename Allocator, typename Prefix, typename Suffix> class affixed_allocator;
template<typename Allocator> class stats_allocator;
template<typename Allocator> class thread_local_allocator;
template<typename Allocator> class synchronized_allocator;

// Allocator traits and concepts
template<typename A>
concept has_owns = requires(A a, const void* p) {
    { a.owns(p) } -> std::convertible_to<bool>;
};

template<typename A>
concept has_expand = requires(A a, void* p, std::size_t old_size, std::size_t delta) {
    { a.expand(p, old_size, delta) } -> std::convertible_to<bool>;
};

template<typename A>
concept has_reallocate = requires(A a, void* p, std::size_t old_size, std::size_t new_size) {
    { a.reallocate(p, old_size, new_size) } -> std::same_as<void*>;
};

template<typename A>
concept has_deallocate_all = requires(A a) {
    { a.deallocate_all() } -> std::same_as<void>;
};

// Helper to get allocation result
struct allocation_result {
    void* ptr;
    std::size_t allocated_size;

    explicit operator bool() const noexcept { return ptr != nullptr; }
};

// Base allocator interface traits
template<typename Allocator>
struct allocator_traits {
    using allocator_type = Allocator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    static constexpr std::size_t alignment = alignof(std::max_align_t);

    static void* allocate(Allocator& a, std::size_t size, std::size_t align = alignment) {
        if constexpr (requires { a.allocate(size, align); }) {
            return a.allocate(size, align);
        } else if constexpr (requires { a.allocate(size); }) {
            return a.allocate(size);
        } else {
            static_assert(sizeof(Allocator) == 0, "Allocator must have allocate method");
        }
    }

    static void deallocate(Allocator& a, void* p, std::size_t size) noexcept {
        a.deallocate(p, size);
    }

    static bool owns(const Allocator& a, const void* p) {
        if constexpr (has_owns<Allocator>) {
            return a.owns(p);
        } else {
            return false;
        }
    }

    static bool expand(Allocator& a, void* p, std::size_t old_size, std::size_t delta) {
        if constexpr (has_expand<Allocator>) {
            return a.expand(p, old_size, delta);
        } else {
            return false;
        }
    }

    static void* reallocate(Allocator& a, void* p, std::size_t old_size, std::size_t new_size) {
        if constexpr (has_reallocate<Allocator>) {
            return a.reallocate(p, old_size, new_size);
        } else {
            void* new_p = allocate(a, new_size);
            if (new_p && p) {
                std::memcpy(new_p, p, std::min(old_size, new_size));
                deallocate(a, p, old_size);
            }
            return new_p;
        }
    }

    static void deallocate_all(Allocator& a) {
        if constexpr (has_deallocate_all<Allocator>) {
            a.deallocate_all();
        }
    }
};

// Heap allocator - wrapper around malloc/free
class heap_allocator {
public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        if (alignment <= alignof(std::max_align_t)) {
            return std::malloc(size);
        } else {
            return std::aligned_alloc(alignment, size);
        }
    }

    void deallocate(void* p, std::size_t) noexcept {
        std::free(p);
    }

    void* reallocate(void* p, std::size_t, std::size_t new_size) {
        return std::realloc(p, new_size);
    }
};

// Null allocator - always fails
class null_allocator {
public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    void* allocate(std::size_t, std::size_t = alignof(std::max_align_t)) noexcept {
        return nullptr;
    }

    void deallocate(void*, std::size_t) noexcept {}

    bool owns(const void*) const noexcept { return false; }
};

// Fixed pool allocator - allocates from a fixed-size buffer
template<std::size_t BlockSize, std::size_t NumBlocks>
class fixed_pool_allocator {
    static_assert(BlockSize >= sizeof(void*), "Block size too small");

    struct free_node {
        free_node* next;
    };

    alignas(std::max_align_t) char buffer_[BlockSize * NumBlocks];
    free_node* free_list_;
    std::size_t allocated_blocks_;

    void initialize() {
        free_list_ = nullptr;
        for (std::size_t i = 0; i < NumBlocks; ++i) {
            auto* node = reinterpret_cast<free_node*>(buffer_ + i * BlockSize);
            node->next = free_list_;
            free_list_ = node;
        }
        allocated_blocks_ = 0;
    }

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    fixed_pool_allocator() { initialize(); }

    void* allocate(std::size_t size, std::size_t = alignof(std::max_align_t)) noexcept {
        if (size > BlockSize || !free_list_) {
            return nullptr;
        }

        auto* node = free_list_;
        free_list_ = free_list_->next;
        ++allocated_blocks_;
        return node;
    }

    void deallocate(void* p, std::size_t) noexcept {
        if (owns(p)) {
            auto* node = static_cast<free_node*>(p);
            node->next = free_list_;
            free_list_ = node;
            --allocated_blocks_;
        }
    }

    bool owns(const void* p) const noexcept {
        const char* cp = static_cast<const char*>(p);
        return cp >= buffer_ && cp < buffer_ + sizeof(buffer_) &&
               (cp - buffer_) % BlockSize == 0;
    }

    void deallocate_all() noexcept {
        initialize();
    }

    std::size_t allocated() const noexcept { return allocated_blocks_; }
    std::size_t available() const noexcept { return NumBlocks - allocated_blocks_; }
    static constexpr std::size_t block_size() noexcept { return BlockSize; }
    static constexpr std::size_t capacity() noexcept { return NumBlocks; }
};

// Freelist allocator - maintains a freelist for recycling
template<typename Allocator>
class freelist_allocator {
    struct free_node {
        free_node* next;
        std::size_t size;
    };

    Allocator allocator_;
    free_node* free_list_;
    std::size_t min_block_size_;
    std::size_t max_free_blocks_;
    std::size_t current_free_blocks_;

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    explicit freelist_allocator(
        Allocator allocator = Allocator(),
        std::size_t min_block_size = 16,
        std::size_t max_free_blocks = 128)
        : allocator_(std::move(allocator))
        , free_list_(nullptr)
        , min_block_size_(min_block_size)
        , max_free_blocks_(max_free_blocks)
        , current_free_blocks_(0) {}

    ~freelist_allocator() {
        clear_freelist();
    }

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        // Try to find a suitable block in the freelist
        free_node** prev = &free_list_;
        free_node* curr = free_list_;

        while (curr) {
            if (curr->size >= size) {
                *prev = curr->next;
                --current_free_blocks_;
                return curr;
            }
            prev = &curr->next;
            curr = curr->next;
        }

        // Allocate from underlying allocator
        std::size_t actual_size = std::max(size, sizeof(free_node));
        return allocator_traits<Allocator>::allocate(allocator_, actual_size, alignment);
    }

    void deallocate(void* p, std::size_t size) noexcept {
        if (!p) return;

        // Add to freelist if not full and size is reasonable
        if (current_free_blocks_ < max_free_blocks_ && size >= min_block_size_) {
            auto* node = static_cast<free_node*>(p);
            node->size = size;
            node->next = free_list_;
            free_list_ = node;
            ++current_free_blocks_;
        } else {
            allocator_traits<Allocator>::deallocate(allocator_, p, size);
        }
    }

    void clear_freelist() noexcept {
        while (free_list_) {
            auto* next = free_list_->next;
            allocator_traits<Allocator>::deallocate(
                allocator_, free_list_, free_list_->size);
            free_list_ = next;
        }
        current_free_blocks_ = 0;
    }

    std::size_t freelist_size() const noexcept { return current_free_blocks_; }
};

// Region allocator - bulk allocation with single deallocation
template<typename Allocator = heap_allocator>
class region_allocator {
    struct region {
        region* next;
        std::size_t size;
        std::size_t used;
        alignas(std::max_align_t) char data[1];  // Flexible array member pattern
    };

    Allocator allocator_;
    region* head_;
    std::size_t region_size_;
    std::size_t total_allocated_;

    void grow(std::size_t min_size) {
        std::size_t size = std::max(min_size, region_size_);
        std::size_t total_size = sizeof(region) - 1 + size;

        void* mem = allocator_traits<Allocator>::allocate(allocator_, total_size);
        if (!mem) throw std::bad_alloc();

        auto* r = static_cast<region*>(mem);
        r->next = head_;
        r->size = size;
        r->used = 0;
        head_ = r;
    }

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    explicit region_allocator(
        std::size_t region_size = 65536,
        Allocator allocator = Allocator())
        : allocator_(std::move(allocator))
        , head_(nullptr)
        , region_size_(region_size)
        , total_allocated_(0) {}

    ~region_allocator() {
        deallocate_all();
    }

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        // Align the allocation
        std::size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

        // Check current region
        if (head_ && head_->used + aligned_size <= head_->size) {
            void* p = head_->data + head_->used;
            head_->used += aligned_size;
            total_allocated_ += aligned_size;
            return p;
        }

        // Need new region
        grow(aligned_size);
        void* p = head_->data;
        head_->used = aligned_size;
        total_allocated_ += aligned_size;
        return p;
    }

    void deallocate(void*, std::size_t) noexcept {
        // Region allocator doesn't support individual deallocations
    }

    void deallocate_all() noexcept {
        while (head_) {
            region* next = head_->next;
            std::size_t total_size = sizeof(region) - 1 + head_->size;
            allocator_traits<Allocator>::deallocate(allocator_, head_, total_size);
            head_ = next;
        }
        total_allocated_ = 0;
    }

    std::size_t bytes_allocated() const noexcept { return total_allocated_; }
};

// Stack allocator with markers
template<typename Allocator = heap_allocator>
class stack_allocator_with_markers {
public:
    struct marker {
        std::size_t position;
        explicit marker(std::size_t pos) : position(pos) {}
    };

private:
    Allocator allocator_;
    char* buffer_;
    std::size_t capacity_;
    std::size_t position_;

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    explicit stack_allocator_with_markers(
        std::size_t capacity,
        Allocator allocator = Allocator())
        : allocator_(std::move(allocator))
        , buffer_(nullptr)
        , capacity_(capacity)
        , position_(0) {
        buffer_ = static_cast<char*>(
            allocator_traits<Allocator>::allocate(allocator_, capacity_));
        if (!buffer_) throw std::bad_alloc();
    }

    ~stack_allocator_with_markers() {
        if (buffer_) {
            allocator_traits<Allocator>::deallocate(allocator_, buffer_, capacity_);
        }
    }

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        // Align the position
        std::size_t aligned_pos = (position_ + alignment - 1) & ~(alignment - 1);

        if (aligned_pos + size > capacity_) {
            return nullptr;
        }

        void* p = buffer_ + aligned_pos;
        position_ = aligned_pos + size;
        return p;
    }

    void deallocate(void*, std::size_t) noexcept {
        // Stack allocator doesn't support individual deallocations
    }

    marker save_state() const noexcept {
        return marker(position_);
    }

    void restore_state(const marker& m) noexcept {
        if (m.position <= position_) {
            position_ = m.position;
        }
    }

    void reset() noexcept {
        position_ = 0;
    }

    std::size_t used() const noexcept { return position_; }
    std::size_t available() const noexcept { return capacity_ - position_; }
    std::size_t capacity() const noexcept { return capacity_; }
};

// Ring buffer allocator
template<std::size_t Size>
class ring_buffer_allocator {
    alignas(std::max_align_t) char buffer_[Size];
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> tail_{0};

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        std::size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

        std::size_t current_head = head_.load(std::memory_order_acquire);
        std::size_t new_head = (current_head + aligned_size) % Size;

        // Check for wrap-around collision
        std::size_t current_tail = tail_.load(std::memory_order_acquire);
        if (current_head < current_tail && new_head >= current_tail) {
            return nullptr;  // Would overwrite unreleased memory
        }

        // Try to update head
        if (head_.compare_exchange_strong(
                current_head, new_head, std::memory_order_release)) {
            return buffer_ + current_head;
        }

        return nullptr;  // Another thread beat us
    }

    void deallocate(void* p, std::size_t size) noexcept {
        if (!owns(p)) return;

        std::size_t aligned_size = (size + alignof(std::max_align_t) - 1) &
                                   ~(alignof(std::max_align_t) - 1);
        std::size_t offset = static_cast<char*>(p) - buffer_;

        // Update tail if this was the oldest allocation
        std::size_t expected = offset;
        tail_.compare_exchange_strong(
            expected, (offset + aligned_size) % Size, std::memory_order_release);
    }

    bool owns(const void* p) const noexcept {
        const char* cp = static_cast<const char*>(p);
        return cp >= buffer_ && cp < buffer_ + Size;
    }

    void reset() noexcept {
        head_.store(0, std::memory_order_release);
        tail_.store(0, std::memory_order_release);
    }
};

// Buddy allocator
template<std::size_t MinBlockSize = 32, std::size_t MaxBlockSize = 1048576>
class buddy_allocator {
    static_assert((MinBlockSize & (MinBlockSize - 1)) == 0, "Min block size must be power of 2");
    static_assert((MaxBlockSize & (MaxBlockSize - 1)) == 0, "Max block size must be power of 2");
    static_assert(MaxBlockSize >= MinBlockSize, "Max must be >= min");

    static constexpr std::size_t num_levels =
        __builtin_ctz(MaxBlockSize) - __builtin_ctz(MinBlockSize) + 1;

    struct block_header {
        block_header* next;
        block_header* prev;
        std::size_t size;
        bool free;
    };

    alignas(std::max_align_t) char memory_[MaxBlockSize];
    std::array<block_header*, num_levels> free_lists_;

    static std::size_t size_to_level(std::size_t size) {
        if (size <= MinBlockSize) return 0;

        // Find the smallest power of 2 >= size
        std::size_t required_size = MinBlockSize;
        std::size_t level = 0;
        while (required_size < size && level < num_levels - 1) {
            required_size <<= 1;
            ++level;
        }
        return level;
    }

    static std::size_t level_to_size(std::size_t level) {
        return MinBlockSize << level;
    }

    void* split_block(std::size_t level, std::size_t target_level) {
        // Check if we have a block at the desired level
        if (free_lists_[level]) {
            block_header* block = free_lists_[level];
            remove_from_free_list(level, block);

            if (level == target_level) {
                block->free = false;
                return block + 1;
            }

            // Split the block for smaller allocation
            std::size_t half_size = level_to_size(level - 1);
            block_header* buddy = reinterpret_cast<block_header*>(
                reinterpret_cast<char*>(block) + half_size);

            // Set up buddy block
            buddy->size = half_size;
            buddy->free = true;
            add_to_free_list(level - 1, buddy);

            // Set up original block
            block->size = half_size;
            block->free = true;
            add_to_free_list(level - 1, block);

            // Recursively allocate from the smaller level
            return split_block(level - 1, target_level);
        }

        // Try to get a block from a larger level
        if (level + 1 < num_levels) {
            return split_block(level + 1, target_level);
        }

        return nullptr;
    }

    void add_to_free_list(std::size_t level, block_header* block) {
        block->next = free_lists_[level];
        block->prev = nullptr;
        if (free_lists_[level]) {
            free_lists_[level]->prev = block;
        }
        free_lists_[level] = block;
    }

    void remove_from_free_list(std::size_t level, block_header* block) {
        if (block->prev) {
            block->prev->next = block->next;
        } else {
            free_lists_[level] = block->next;
        }
        if (block->next) {
            block->next->prev = block->prev;
        }
    }

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    buddy_allocator() {
        std::fill(free_lists_.begin(), free_lists_.end(), nullptr);

        // Initialize with one maximum-size block
        block_header* initial = reinterpret_cast<block_header*>(memory_);
        initial->size = MaxBlockSize;
        initial->free = true;
        initial->next = nullptr;
        initial->prev = nullptr;
        free_lists_[num_levels - 1] = initial;
    }

    void* allocate(std::size_t size, std::size_t = alignof(std::max_align_t)) {
        size += sizeof(block_header);  // Account for header
        std::size_t level = size_to_level(size);

        // Check if level is valid
        if (level >= num_levels) return nullptr;

        // Find a free block at this level or split a larger one
        for (std::size_t l = level; l < num_levels; ++l) {
            if (free_lists_[l]) {
                return split_block(l, level);
            }
        }

        return nullptr;
    }

    void deallocate(void* p, std::size_t) noexcept {
        if (!p || !owns(p)) return;

        block_header* block = static_cast<block_header*>(p) - 1;
        std::size_t level = size_to_level(block->size);

        // Try to coalesce with buddy
        while (level < num_levels - 1) {
            std::size_t buddy_offset = (reinterpret_cast<char*>(block) - memory_) ^ block->size;
            if (buddy_offset >= MaxBlockSize) break;

            block_header* buddy = reinterpret_cast<block_header*>(memory_ + buddy_offset);
            if (!buddy->free || buddy->size != block->size) break;

            remove_from_free_list(level, buddy);

            // Merge with buddy
            if (buddy < block) {
                block = buddy;
            }
            block->size *= 2;
            ++level;
        }

        block->free = true;
        add_to_free_list(level, block);
    }

    bool owns(const void* p) const noexcept {
        const char* cp = static_cast<const char*>(p);
        return cp >= memory_ && cp < memory_ + MaxBlockSize;
    }
};

// Slab allocator
template<typename T, std::size_t ObjectsPerSlab = 64>
class slab_allocator {
    struct slab {
        slab* next;
        std::size_t free_count;
        std::uint64_t free_bitmap[(ObjectsPerSlab + 63) / 64];
        alignas(T) char objects[sizeof(T) * ObjectsPerSlab];

        slab() : next(nullptr), free_count(ObjectsPerSlab) {
            std::fill(std::begin(free_bitmap), std::end(free_bitmap), ~std::uint64_t(0));
        }

        void* allocate() {
            if (free_count == 0) return nullptr;

            for (std::size_t i = 0; i < ObjectsPerSlab; ++i) {
                std::size_t word = i / 64;
                std::size_t bit = i % 64;

                if (free_bitmap[word] & (std::uint64_t(1) << bit)) {
                    free_bitmap[word] &= ~(std::uint64_t(1) << bit);
                    --free_count;
                    return objects + i * sizeof(T);
                }
            }
            return nullptr;
        }

        bool deallocate(void* p) {
            char* cp = static_cast<char*>(p);
            if (cp < objects || cp >= objects + sizeof(objects)) {
                return false;
            }

            std::size_t index = (cp - objects) / sizeof(T);
            std::size_t word = index / 64;
            std::size_t bit = index % 64;

            if (!(free_bitmap[word] & (std::uint64_t(1) << bit))) {
                free_bitmap[word] |= (std::uint64_t(1) << bit);
                ++free_count;
                return true;
            }
            return false;
        }

        bool is_empty() const { return free_count == ObjectsPerSlab; }
        bool is_full() const { return free_count == 0; }
    };

    slab* partial_slabs_;
    slab* full_slabs_;
    slab* empty_slabs_;
    std::size_t slab_count_;
    std::size_t max_empty_slabs_;

    void move_slab(slab*& from_list, slab*& to_list, slab* s) {
        // Remove from current list
        if (from_list == s) {
            from_list = s->next;
        } else {
            slab* prev = from_list;
            while (prev && prev->next != s) {
                prev = prev->next;
            }
            if (prev) {
                prev->next = s->next;
            }
        }

        // Add to new list
        s->next = to_list;
        to_list = s;
    }

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    explicit slab_allocator(std::size_t max_empty_slabs = 2)
        : partial_slabs_(nullptr)
        , full_slabs_(nullptr)
        , empty_slabs_(nullptr)
        , slab_count_(0)
        , max_empty_slabs_(max_empty_slabs) {}

    ~slab_allocator() {
        auto free_list = [](slab* list) {
            while (list) {
                slab* next = list->next;
                delete list;
                list = next;
            }
        };

        free_list(partial_slabs_);
        free_list(full_slabs_);
        free_list(empty_slabs_);
    }

    void* allocate_raw(std::size_t size, std::size_t = alignof(std::max_align_t)) {
        if (size > sizeof(T)) return nullptr;  // Can't allocate larger than object size

        // Try partial slabs first
        if (partial_slabs_) {
            void* p = partial_slabs_->allocate();
            if (partial_slabs_->is_full()) {
                move_slab(partial_slabs_, full_slabs_, partial_slabs_);
            }
            return p;
        }

        // Try empty slabs
        if (empty_slabs_) {
            slab* s = empty_slabs_;
            move_slab(empty_slabs_, partial_slabs_, s);
            return s->allocate();
        }

        // Allocate new slab
        slab* new_slab = new slab();
        ++slab_count_;
        new_slab->next = partial_slabs_;
        partial_slabs_ = new_slab;
        return new_slab->allocate();
    }

    T* allocate(std::size_t n = 1) {
        if (n != 1) return nullptr;
        return static_cast<T*>(allocate_raw(sizeof(T)));
    }

    void deallocate(void* p, std::size_t) noexcept {
        if (!p) return;

        // Find the slab containing this object
        auto try_deallocate = [&](slab*& list) -> bool {
            slab* s = list;
            while (s) {
                if (s->deallocate(p)) {
                    if (s->is_empty() && list == partial_slabs_) {
                        // Move to empty list
                        move_slab(partial_slabs_, empty_slabs_, s);

                        // Trim excess empty slabs
                        std::size_t empty_count = 0;
                        slab* e = empty_slabs_;
                        slab* prev = nullptr;
                        while (e) {
                            ++empty_count;
                            if (empty_count > max_empty_slabs_) {
                                if (prev) {
                                    prev->next = e->next;
                                } else {
                                    empty_slabs_ = e->next;
                                }
                                slab* to_delete = e;
                                e = e->next;
                                delete to_delete;
                                --slab_count_;
                            } else {
                                prev = e;
                                e = e->next;
                            }
                        }
                    } else if (!s->is_full() && list == full_slabs_) {
                        // Move to partial list
                        move_slab(full_slabs_, partial_slabs_, s);
                    }
                    return true;
                }
                s = s->next;
            }
            return false;
        };

        try_deallocate(partial_slabs_) ||
        try_deallocate(full_slabs_) ||
        try_deallocate(empty_slabs_);
    }

    void deallocate(T* p, std::size_t n = 1) noexcept {
        deallocate(static_cast<void*>(p), n * sizeof(T));
    }

    std::size_t slab_count() const noexcept { return slab_count_; }
};

// Bitmapped block allocator
template<std::size_t BlockSize, std::size_t NumBlocks>
class bitmapped_block_allocator {
    static constexpr std::size_t words_needed = (NumBlocks + 63) / 64;

    alignas(std::max_align_t) char blocks_[BlockSize * NumBlocks];
    std::uint64_t bitmap_[words_needed];
    std::size_t hint_word_;  // Hint for next free block

    std::pair<std::size_t, std::size_t> find_free_bit() const {
        // Start from hint
        for (std::size_t w = hint_word_; w < words_needed; ++w) {
            if (bitmap_[w] != 0) {
                std::size_t bit = __builtin_ctzll(bitmap_[w]);
                return {w, bit};
            }
        }
        // Wrap around
        for (std::size_t w = 0; w < hint_word_; ++w) {
            if (bitmap_[w] != 0) {
                std::size_t bit = __builtin_ctzll(bitmap_[w]);
                return {w, bit};
            }
        }
        return {words_needed, 0};  // No free blocks
    }

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    bitmapped_block_allocator() : hint_word_(0) {
        std::fill(std::begin(bitmap_), std::end(bitmap_), ~std::uint64_t(0));
        // Clear unused bits in last word
        if constexpr (NumBlocks % 64 != 0) {
            bitmap_[words_needed - 1] &= (std::uint64_t(1) << (NumBlocks % 64)) - 1;
        }
    }

    void* allocate(std::size_t size, std::size_t = alignof(std::max_align_t)) {
        if (size > BlockSize) return nullptr;

        auto [word, bit] = find_free_bit();
        if (word >= words_needed) return nullptr;

        bitmap_[word] &= ~(std::uint64_t(1) << bit);
        hint_word_ = word;

        std::size_t block_index = word * 64 + bit;
        return blocks_ + block_index * BlockSize;
    }

    void deallocate(void* p, std::size_t) noexcept {
        if (!owns(p)) return;

        std::size_t offset = static_cast<char*>(p) - blocks_;
        std::size_t block_index = offset / BlockSize;
        std::size_t word = block_index / 64;
        std::size_t bit = block_index % 64;

        bitmap_[word] |= (std::uint64_t(1) << bit);
        hint_word_ = std::min(hint_word_, word);
    }

    bool owns(const void* p) const noexcept {
        const char* cp = static_cast<const char*>(p);
        return cp >= blocks_ && cp < blocks_ + sizeof(blocks_) &&
               (cp - blocks_) % BlockSize == 0;
    }

    std::size_t allocated_blocks() const noexcept {
        std::size_t count = 0;
        for (std::size_t w = 0; w < words_needed; ++w) {
            count += __builtin_popcountll(~bitmap_[w]);
        }
        return count;
    }

    std::size_t free_blocks() const noexcept {
        return NumBlocks - allocated_blocks();
    }

    static constexpr std::size_t block_size() noexcept { return BlockSize; }
    static constexpr std::size_t capacity() noexcept { return NumBlocks; }
};

// Fallback allocator combinator
template<typename Primary, typename Fallback>
class fallback_allocator {
    Primary primary_;
    Fallback fallback_;

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    fallback_allocator() = default;

    explicit fallback_allocator(Primary primary, Fallback fallback)
        : primary_(std::move(primary)), fallback_(std::move(fallback)) {}

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        void* p = allocator_traits<Primary>::allocate(primary_, size, alignment);
        if (!p) {
            p = allocator_traits<Fallback>::allocate(fallback_, size, alignment);
        }
        return p;
    }

    void deallocate(void* p, std::size_t size) noexcept {
        if (allocator_traits<Primary>::owns(primary_, p)) {
            allocator_traits<Primary>::deallocate(primary_, p, size);
        } else {
            allocator_traits<Fallback>::deallocate(fallback_, p, size);
        }
    }

    bool owns(const void* p) const {
        return allocator_traits<Primary>::owns(primary_, p) ||
               allocator_traits<Fallback>::owns(fallback_, p);
    }

    bool expand(void* p, std::size_t old_size, std::size_t delta) {
        if (allocator_traits<Primary>::owns(primary_, p)) {
            return allocator_traits<Primary>::expand(primary_, p, old_size, delta);
        } else {
            return allocator_traits<Fallback>::expand(fallback_, p, old_size, delta);
        }
    }

    void* reallocate(void* p, std::size_t old_size, std::size_t new_size) {
        if (!p) {
            return allocate(new_size);
        }

        if (allocator_traits<Primary>::owns(primary_, p)) {
            return allocator_traits<Primary>::reallocate(primary_, p, old_size, new_size);
        } else {
            return allocator_traits<Fallback>::reallocate(fallback_, p, old_size, new_size);
        }
    }

    Primary& primary() { return primary_; }
    const Primary& primary() const { return primary_; }
    Fallback& fallback() { return fallback_; }
    const Fallback& fallback() const { return fallback_; }
};

// Segregator allocator combinator
template<std::size_t Threshold, typename Small, typename Large>
class segregator_allocator {
    Small small_;
    Large large_;

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    segregator_allocator() = default;

    explicit segregator_allocator(Small small, Large large)
        : small_(std::move(small)), large_(std::move(large)) {}

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        if (size <= Threshold) {
            return allocator_traits<Small>::allocate(small_, size, alignment);
        } else {
            return allocator_traits<Large>::allocate(large_, size, alignment);
        }
    }

    void deallocate(void* p, std::size_t size) noexcept {
        if (!p) return;

        // First try based on size
        if (size <= Threshold) {
            // If small allocator owns it, deallocate there
            if (allocator_traits<Small>::owns(small_, p)) {
                allocator_traits<Small>::deallocate(small_, p, size);
                return;
            }
        }

        // Otherwise deallocate from large
        allocator_traits<Large>::deallocate(large_, p, size);
    }

    bool owns(const void* p) const {
        return allocator_traits<Small>::owns(small_, p) ||
               allocator_traits<Large>::owns(large_, p);
    }

    static constexpr std::size_t threshold() noexcept { return Threshold; }
    Small& small() { return small_; }
    const Small& small() const { return small_; }
    Large& large() { return large_; }
    const Large& large() const { return large_; }
};

// Affixed allocator combinator
template<typename Allocator, typename Prefix, typename Suffix = void>
class affixed_allocator {
    static constexpr bool has_prefix = !std::is_void_v<Prefix>;
    static constexpr bool has_suffix = !std::is_void_v<Suffix>;

    static constexpr std::size_t prefix_size = []() {
        if constexpr (has_prefix) return sizeof(Prefix);
        else return std::size_t(0);
    }();

    static constexpr std::size_t suffix_size = []() {
        if constexpr (has_suffix) return sizeof(Suffix);
        else return std::size_t(0);
    }();

    Allocator allocator_;

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    affixed_allocator() = default;

    explicit affixed_allocator(Allocator allocator)
        : allocator_(std::move(allocator)) {}

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        std::size_t total_size = prefix_size + size + suffix_size;
        void* p = allocator_traits<Allocator>::allocate(allocator_, total_size, alignment);

        if (p) {
            if constexpr (has_prefix) {
                new(p) Prefix{};
            }
            if constexpr (has_suffix) {
                new(static_cast<char*>(p) + prefix_size + size) Suffix{};
            }
            return static_cast<char*>(p) + prefix_size;
        }
        return nullptr;
    }

    void deallocate(void* p, std::size_t size) noexcept {
        if (!p) return;

        void* actual_p = static_cast<char*>(p) - prefix_size;

        if constexpr (has_prefix) {
            static_cast<Prefix*>(actual_p)->~Prefix();
        }
        if constexpr (has_suffix) {
            auto* suffix_p = static_cast<Suffix*>(
                static_cast<void*>(static_cast<char*>(p) + size));
            suffix_p->~Suffix();
        }

        std::size_t total_size = prefix_size + size + suffix_size;
        allocator_traits<Allocator>::deallocate(allocator_, actual_p, total_size);
    }

    template<typename T = Prefix>
    T* prefix(void* p) {
        static_assert(has_prefix, "No prefix type specified");
        return static_cast<T*>(static_cast<void*>(static_cast<char*>(p) - prefix_size));
    }

    template<typename T = Suffix>
    T* suffix(void* p, std::size_t size) {
        static_assert(has_suffix, "No suffix type specified");
        return static_cast<T*>(static_cast<void*>(static_cast<char*>(p) + size));
    }

    Allocator& inner() { return allocator_; }
    const Allocator& inner() const { return allocator_; }
};

// Stats allocator combinator
template<typename Allocator>
class stats_allocator {
public:
    struct statistics {
        std::size_t allocations = 0;
        std::size_t deallocations = 0;
        std::size_t bytes_allocated = 0;
        std::size_t bytes_deallocated = 0;
        std::size_t peak_bytes = 0;
        std::size_t current_bytes = 0;
        std::size_t failed_allocations = 0;

        void record_allocation(std::size_t size) {
            ++allocations;
            bytes_allocated += size;
            current_bytes += size;
            peak_bytes = std::max(peak_bytes, current_bytes);
        }

        void record_deallocation(std::size_t size) {
            ++deallocations;
            bytes_deallocated += size;
            current_bytes -= size;
        }

        void record_failure() {
            ++failed_allocations;
        }
    };

private:
    Allocator allocator_;
    mutable statistics stats_;

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    stats_allocator() = default;

    explicit stats_allocator(Allocator allocator)
        : allocator_(std::move(allocator)) {}

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        void* p = allocator_traits<Allocator>::allocate(allocator_, size, alignment);
        if (p) {
            stats_.record_allocation(size);
        } else {
            stats_.record_failure();
        }
        return p;
    }

    void deallocate(void* p, std::size_t size) noexcept {
        if (p) {
            stats_.record_deallocation(size);
        }
        allocator_traits<Allocator>::deallocate(allocator_, p, size);
    }

    bool owns(const void* p) const {
        return allocator_traits<Allocator>::owns(allocator_, p);
    }

    const statistics& stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_ = statistics{}; }

    Allocator& inner() { return allocator_; }
    const Allocator& inner() const { return allocator_; }
};

// Thread-local allocator combinator
template<typename Allocator>
class thread_local_allocator {
    struct thread_data {
        Allocator allocator;
        thread_data() = default;
    };

    static thread_local thread_data* tls_data_;
    mutable std::unordered_map<std::thread::id, std::unique_ptr<thread_data>> allocators_;
    mutable std::mutex mutex_;

    thread_data* get_thread_data() const {
        if (!tls_data_) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto id = std::this_thread::get_id();
            auto& ptr = allocators_[id];
            if (!ptr) {
                ptr = std::make_unique<thread_data>();
            }
            tls_data_ = ptr.get();
        }
        return tls_data_;
    }

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        auto* data = get_thread_data();
        return allocator_traits<Allocator>::allocate(data->allocator, size, alignment);
    }

    void deallocate(void* p, std::size_t size) noexcept {
        auto* data = get_thread_data();
        allocator_traits<Allocator>::deallocate(data->allocator, p, size);
    }

    void cleanup_thread() {
        std::lock_guard<std::mutex> lock(mutex_);
        allocators_.erase(std::this_thread::get_id());
        tls_data_ = nullptr;
    }

    std::size_t thread_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocators_.size();
    }
};

template<typename Allocator>
thread_local typename thread_local_allocator<Allocator>::thread_data*
    thread_local_allocator<Allocator>::tls_data_ = nullptr;

// Synchronized allocator combinator
template<typename Allocator>
class synchronized_allocator {
    mutable std::mutex mutex_;
    Allocator allocator_;

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    synchronized_allocator() = default;

    explicit synchronized_allocator(Allocator allocator)
        : allocator_(std::move(allocator)) {}

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocator_traits<Allocator>::allocate(allocator_, size, alignment);
    }

    void deallocate(void* p, std::size_t size) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        allocator_traits<Allocator>::deallocate(allocator_, p, size);
    }

    bool owns(const void* p) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocator_traits<Allocator>::owns(allocator_, p);
    }

    Allocator& inner() { return allocator_; }
    const Allocator& inner() const { return allocator_; }
};

// Bucketizing allocator - different pools for size classes
template<typename Allocator = heap_allocator>
class bucketizing_allocator {
    static constexpr std::size_t num_buckets = 16;
    static constexpr std::size_t bucket_sizes[num_buckets] = {
        8, 16, 24, 32, 48, 64, 96, 128,
        192, 256, 384, 512, 768, 1024, 1536, 2048
    };

    struct bucket {
        freelist_allocator<Allocator> allocator;
        std::size_t size;

        explicit bucket(std::size_t s) : allocator(Allocator{}, s, 32), size(s) {}
    };

    std::array<bucket, num_buckets> buckets_;
    Allocator fallback_;

    static std::size_t size_to_bucket(std::size_t size) {
        for (std::size_t i = 0; i < num_buckets; ++i) {
            if (size <= bucket_sizes[i]) {
                return i;
            }
        }
        return num_buckets;  // Use fallback
    }

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    bucketizing_allocator() : buckets_{
        bucket{8}, bucket{16}, bucket{24}, bucket{32},
        bucket{48}, bucket{64}, bucket{96}, bucket{128},
        bucket{192}, bucket{256}, bucket{384}, bucket{512},
        bucket{768}, bucket{1024}, bucket{1536}, bucket{2048}
    } {}

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        std::size_t bucket_index = size_to_bucket(size);

        if (bucket_index < num_buckets) {
            // Always allocate the bucket size, not the requested size
            return buckets_[bucket_index].allocator.allocate(
                bucket_sizes[bucket_index], alignment);
        } else {
            return allocator_traits<Allocator>::allocate(fallback_, size, alignment);
        }
    }

    void deallocate(void* p, std::size_t size) noexcept {
        if (!p) return;

        std::size_t bucket_index = size_to_bucket(size);

        if (bucket_index < num_buckets) {
            // Always deallocate the bucket size, not the original size
            buckets_[bucket_index].allocator.deallocate(p, bucket_sizes[bucket_index]);
        } else {
            allocator_traits<Allocator>::deallocate(fallback_, p, size);
        }
    }
};

// Helper type aliases for common compositions
template<std::size_t Size>
using stack_with_heap = fallback_allocator<
    stack_allocator<Size>,
    heap_allocator
>;

template<std::size_t SmallSize, std::size_t SmallThreshold>
using small_object_allocator = segregator_allocator<
    SmallThreshold,
    fixed_pool_allocator<SmallSize, 1024>,
    heap_allocator
>;

template<typename Allocator>
using tracking_allocator = stats_allocator<Allocator>;

template<typename Allocator>
using debug_allocator = affixed_allocator<
    Allocator,
    std::size_t,  // Size prefix
    std::uint32_t  // Magic number suffix
>;

// STL allocator adaptor for compositional allocators
template<typename T, typename Allocator>
class stl_allocator_adaptor {
    Allocator* allocator_;

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::false_type;

    template<typename U>
    struct rebind {
        using other = stl_allocator_adaptor<U, Allocator>;
    };

    explicit stl_allocator_adaptor(Allocator* allocator) noexcept
        : allocator_(allocator) {}

    template<typename U>
    stl_allocator_adaptor(const stl_allocator_adaptor<U, Allocator>& other) noexcept
        : allocator_(other.allocator_) {}

    T* allocate(std::size_t n) {
        void* p = allocator_traits<Allocator>::allocate(
            *allocator_, n * sizeof(T), alignof(T));
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t n) noexcept {
        allocator_traits<Allocator>::deallocate(
            *allocator_, p, n * sizeof(T));
    }

    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* p) noexcept {
        p->~U();
    }

    template<typename U, typename A>
    friend class stl_allocator_adaptor;

    Allocator* get_allocator() const noexcept { return allocator_; }
};

template<typename T1, typename T2, typename Allocator>
bool operator==(const stl_allocator_adaptor<T1, Allocator>& a,
                const stl_allocator_adaptor<T2, Allocator>& b) noexcept {
    return a.get_allocator() == b.get_allocator();
}

template<typename T1, typename T2, typename Allocator>
bool operator!=(const stl_allocator_adaptor<T1, Allocator>& a,
                const stl_allocator_adaptor<T2, Allocator>& b) noexcept {
    return !(a == b);
}

} // namespace stepanov