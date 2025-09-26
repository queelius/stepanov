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
#include "concepts.hpp"

namespace stepanov {

// Memory alignment utilities
template<std::size_t Alignment>
constexpr bool is_power_of_two = (Alignment & (Alignment - 1)) == 0;

template<typename T>
constexpr std::size_t alignment_of = alignof(T);

inline void* align(std::size_t alignment, std::size_t size, void*& ptr, std::size_t& space) {
    if (std::size_t(ptr) % alignment) {
        std::size_t adjustment = alignment - (std::size_t(ptr) % alignment);
        if (space < size + adjustment) return nullptr;
        ptr = static_cast<char*>(ptr) + adjustment;
        space -= adjustment;
    }
    if (space < size) return nullptr;
    return ptr;
}

// Generic allocator concept
template<typename A>
concept allocator = requires(A a, typename A::value_type* p, std::size_t n) {
    typename A::value_type;
    { a.allocate(n) } -> std::same_as<typename A::value_type*>;
    { a.deallocate(p, n) } -> std::same_as<void>;
};

// Stack allocator - allocates from a fixed-size buffer on stack
template<std::size_t N, std::size_t Alignment = alignof(std::max_align_t)>
class stack_allocator {
    alignas(Alignment) char buffer_[N];
    char* ptr_;

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    constexpr stack_allocator() noexcept : ptr_(buffer_) {}

    stack_allocator(const stack_allocator&) = delete;
    stack_allocator& operator=(const stack_allocator&) = delete;

    template<typename T>
    struct rebind {
        using other = stack_allocator<N, Alignment>;
    };

    void* allocate(std::size_t size, std::size_t alignment = Alignment) noexcept {
        std::size_t space = buffer_ + N - ptr_;
        void* result = ptr_;

        if (align(alignment, size, result, space)) {
            ptr_ = static_cast<char*>(result) + size;
            return result;
        }
        return nullptr;
    }

    void deallocate(void*, std::size_t) noexcept {
        // Stack allocator doesn't support individual deallocations
    }

    void reset() noexcept {
        ptr_ = buffer_;
    }

    std::size_t used() const noexcept {
        return ptr_ - buffer_;
    }

    std::size_t available() const noexcept {
        return N - used();
    }

    constexpr std::size_t capacity() const noexcept {
        return N;
    }
};

// Arena allocator - bulk allocation with single deallocation
template<typename Allocator = std::allocator<char>>
class arena_allocator {
    struct block {
        block* next;
        std::size_t size;
        char* data;
    };

    block* head_;
    char* ptr_;
    std::size_t remaining_;
    std::size_t block_size_;
    Allocator alloc_;

    void grow(std::size_t min_size) {
        std::size_t size = std::max(min_size, block_size_);
        char* data = alloc_.allocate(size);

        block* b = reinterpret_cast<block*>(data);
        b->next = head_;
        b->size = size;
        b->data = data + sizeof(block);

        head_ = b;
        ptr_ = b->data;
        remaining_ = size - sizeof(block);
    }

public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    explicit arena_allocator(std::size_t block_size = 4096, const Allocator& alloc = Allocator())
        : head_(nullptr), ptr_(nullptr), remaining_(0),
          block_size_(block_size), alloc_(alloc) {}

    ~arena_allocator() {
        reset();
    }

    arena_allocator(const arena_allocator&) = delete;
    arena_allocator& operator=(const arena_allocator&) = delete;

    arena_allocator(arena_allocator&& other) noexcept
        : head_(other.head_), ptr_(other.ptr_), remaining_(other.remaining_),
          block_size_(other.block_size_), alloc_(std::move(other.alloc_)) {
        other.head_ = nullptr;
        other.ptr_ = nullptr;
        other.remaining_ = 0;
    }

    arena_allocator& operator=(arena_allocator&& other) noexcept {
        if (this != &other) {
            reset();
            head_ = other.head_;
            ptr_ = other.ptr_;
            remaining_ = other.remaining_;
            block_size_ = other.block_size_;
            alloc_ = std::move(other.alloc_);
            other.head_ = nullptr;
            other.ptr_ = nullptr;
            other.remaining_ = 0;
        }
        return *this;
    }

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        std::size_t space = remaining_;
        void* result = ptr_;

        if (!align(alignment, size, result, space)) {
            grow(size + alignment);
            space = remaining_;
            result = ptr_;
            align(alignment, size, result, space);
        }

        ptr_ = static_cast<char*>(result) + size;
        remaining_ = space - size;
        return result;
    }

    void deallocate(void*, std::size_t) noexcept {
        // Arena doesn't support individual deallocations
    }

    void reset() noexcept {
        while (head_) {
            block* next = head_->next;
            alloc_.deallocate(reinterpret_cast<char*>(head_), head_->size);
            head_ = next;
        }
        ptr_ = nullptr;
        remaining_ = 0;
    }

    template<typename T, typename... Args>
    T* construct(Args&&... args) {
        void* p = allocate(sizeof(T), alignof(T));
        return ::new(p) T(std::forward<Args>(args)...);
    }

    template<typename T>
    void destroy(T* p) noexcept {
        p->~T();
    }
};

// Memory pool for fixed-size objects
template<typename T, std::size_t BlockSize = 32>
class fixed_memory_pool {
    union node {
        alignas(T) char storage[sizeof(T)];
        node* next;
    };

    struct block {
        block* next;
        node nodes[BlockSize];
    };

    block* blocks_;
    node* free_list_;
    std::size_t allocated_;
    std::size_t deallocated_;

    void grow() {
        block* b = new block;
        b->next = blocks_;
        blocks_ = b;

        for (std::size_t i = 0; i < BlockSize - 1; ++i) {
            b->nodes[i].next = &b->nodes[i + 1];
        }
        b->nodes[BlockSize - 1].next = free_list_;
        free_list_ = &b->nodes[0];
    }

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    fixed_memory_pool() noexcept
        : blocks_(nullptr), free_list_(nullptr), allocated_(0), deallocated_(0) {}

    ~fixed_memory_pool() {
        while (blocks_) {
            block* next = blocks_->next;
            delete blocks_;
            blocks_ = next;
        }
    }

    fixed_memory_pool(const fixed_memory_pool&) = delete;
    fixed_memory_pool& operator=(const fixed_memory_pool&) = delete;

    fixed_memory_pool(fixed_memory_pool&& other) noexcept
        : blocks_(other.blocks_), free_list_(other.free_list_),
          allocated_(other.allocated_), deallocated_(other.deallocated_) {
        other.blocks_ = nullptr;
        other.free_list_ = nullptr;
        other.allocated_ = 0;
        other.deallocated_ = 0;
    }

    fixed_memory_pool& operator=(fixed_memory_pool&& other) noexcept {
        if (this != &other) {
            this->~fixed_memory_pool();
            new(this) fixed_memory_pool(std::move(other));
        }
        return *this;
    }

    T* allocate(std::size_t n = 1) {
        if (n != 1) {
            throw std::bad_alloc();
        }

        if (!free_list_) {
            grow();
        }

        node* p = free_list_;
        free_list_ = free_list_->next;
        ++allocated_;
        return reinterpret_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t n = 1) noexcept {
        if (p && n == 1) {
            node* n = reinterpret_cast<node*>(p);
            n->next = free_list_;
            free_list_ = n;
            ++deallocated_;
        }
    }

    template<typename... Args>
    T* construct(Args&&... args) {
        T* p = allocate();
        return ::new(p) T(std::forward<Args>(args)...);
    }

    void destroy(T* p) noexcept {
        p->~T();
        deallocate(p);
    }

    std::size_t allocated_count() const noexcept { return allocated_; }
    std::size_t deallocated_count() const noexcept { return deallocated_; }
    std::size_t in_use_count() const noexcept { return allocated_ - deallocated_; }
};

// SIMD-aligned allocator
template<typename T, std::size_t Alignment = 32>
class aligned_allocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template<typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    aligned_allocator() noexcept = default;

    template<typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }

        void* p = std::aligned_alloc(Alignment, n * sizeof(T));
        if (!p) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }

    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* p) noexcept {
        p->~U();
    }
};

template<typename T1, std::size_t A1, typename T2, std::size_t A2>
bool operator==(const aligned_allocator<T1, A1>&, const aligned_allocator<T2, A2>&) noexcept {
    return A1 == A2;
}

template<typename T1, std::size_t A1, typename T2, std::size_t A2>
bool operator!=(const aligned_allocator<T1, A1>& a, const aligned_allocator<T2, A2>& b) noexcept {
    return !(a == b);
}

// STL-compatible allocator adaptor for memory pools
template<typename T, typename Pool>
class pool_allocator {
    Pool* pool_;

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template<typename U>
    struct rebind {
        using other = pool_allocator<U, Pool>;
    };

    explicit pool_allocator(Pool* pool) noexcept : pool_(pool) {}

    template<typename U>
    pool_allocator(const pool_allocator<U, Pool>& other) noexcept
        : pool_(other.pool_) {}

    T* allocate(std::size_t n) {
        if (n != 1) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(pool_->allocate(sizeof(T), alignof(T)));
    }

    void deallocate(T* p, std::size_t n) noexcept {
        if (n == 1) {
            pool_->deallocate(p, sizeof(T));
        }
    }

    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* p) noexcept {
        p->~U();
    }

    Pool* get_pool() const noexcept { return pool_; }
};

template<typename T1, typename T2, typename Pool>
bool operator==(const pool_allocator<T1, Pool>& a, const pool_allocator<T2, Pool>& b) noexcept {
    return a.get_pool() == b.get_pool();
}

template<typename T1, typename T2, typename Pool>
bool operator!=(const pool_allocator<T1, Pool>& a, const pool_allocator<T2, Pool>& b) noexcept {
    return !(a == b);
}

// Memory utilities
template<typename T>
void construct_at(T* p) {
    ::new(static_cast<void*>(p)) T;
}

template<typename T, typename... Args>
void construct_at(T* p, Args&&... args) {
    ::new(static_cast<void*>(p)) T(std::forward<Args>(args)...);
}

template<typename T>
void destroy_at(T* p) noexcept {
    p->~T();
}

template<typename ForwardIterator>
void destroy(ForwardIterator first, ForwardIterator last) noexcept {
    for (; first != last; ++first) {
        destroy_at(std::addressof(*first));
    }
}

template<typename ForwardIterator, typename Size>
void destroy_n(ForwardIterator first, Size n) noexcept {
    for (; n > 0; (void)++first, --n) {
        destroy_at(std::addressof(*first));
    }
}

// Uninitialized memory algorithms
template<typename InputIterator, typename ForwardIterator>
ForwardIterator uninitialized_copy(InputIterator first, InputIterator last, ForwardIterator d_first) {
    using T = typename std::iterator_traits<ForwardIterator>::value_type;
    ForwardIterator current = d_first;
    try {
        for (; first != last; ++first, (void)++current) {
            ::new(static_cast<void*>(std::addressof(*current))) T(*first);
        }
        return current;
    } catch (...) {
        destroy(d_first, current);
        throw;
    }
}

template<typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator uninitialized_copy_n(InputIterator first, Size count, ForwardIterator d_first) {
    using T = typename std::iterator_traits<ForwardIterator>::value_type;
    ForwardIterator current = d_first;
    try {
        for (; count > 0; ++first, (void)++current, --count) {
            ::new(static_cast<void*>(std::addressof(*current))) T(*first);
        }
        return current;
    } catch (...) {
        destroy(d_first, current);
        throw;
    }
}

template<typename ForwardIterator, typename T>
void uninitialized_fill(ForwardIterator first, ForwardIterator last, const T& value) {
    using V = typename std::iterator_traits<ForwardIterator>::value_type;
    ForwardIterator current = first;
    try {
        for (; current != last; ++current) {
            ::new(static_cast<void*>(std::addressof(*current))) V(value);
        }
    } catch (...) {
        destroy(first, current);
        throw;
    }
}

template<typename ForwardIterator, typename Size, typename T>
ForwardIterator uninitialized_fill_n(ForwardIterator first, Size count, const T& value) {
    using V = typename std::iterator_traits<ForwardIterator>::value_type;
    ForwardIterator current = first;
    try {
        for (; count > 0; ++current, --count) {
            ::new(static_cast<void*>(std::addressof(*current))) V(value);
        }
        return current;
    } catch (...) {
        destroy(first, current);
        throw;
    }
}

} // namespace stepanov