#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>
#include <vector>
#include <queue>
#include <functional>
#include <algorithm>
#include <numeric>
#include <execution>
#include <immintrin.h>
#include "iterators.hpp"
#include "ranges.hpp"
#include "concepts.hpp"

namespace stepanov {

// Thread pool for parallel execution
class thread_pool {
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};

    void worker() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return stop_.load() || !tasks_.empty(); });

                if (stop_.load() && tasks_.empty()) {
                    return;
                }

                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

public:
    explicit thread_pool(std::size_t num_threads = std::thread::hardware_concurrency())
        : threads_(num_threads) {
        for (auto& thread : threads_) {
            thread = std::thread(&thread_pool::worker, this);
        }
    }

    ~thread_pool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        condition_.notify_all();

        for (auto& thread : threads_) {
            thread.join();
        }
    }

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using return_type = decltype(f(args...));

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> result = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_.load()) {
                throw std::runtime_error("enqueue on stopped thread_pool");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return result;
    }

    std::size_t size() const { return threads_.size(); }
};

// Lock-free queue using atomic operations
template<typename T>
class lock_free_queue {
    struct node {
        std::atomic<T*> data;
        std::atomic<node*> next;

        node() : data(nullptr), next(nullptr) {}
    };

    std::atomic<node*> head_;
    std::atomic<node*> tail_;

public:
    lock_free_queue() {
        node* dummy = new node;
        head_.store(dummy);
        tail_.store(dummy);
    }

    ~lock_free_queue() {
        while (node* old_head = head_.load()) {
            head_.store(old_head->next);
            delete old_head;
        }
    }

    void push(T item) {
        node* new_node = new node;
        T* data = new T(std::move(item));
        new_node->data.store(data);
        new_node->next.store(nullptr);

        node* prev_tail = tail_.exchange(new_node);
        prev_tail->next.store(new_node);
    }

    bool pop(T& result) {
        node* head = head_.load();

        while (true) {
            node* next = head->next.load();
            if (next == nullptr) {
                return false;
            }

            if (head_.compare_exchange_weak(head, next)) {
                T* data = next->data.load();
                if (data != nullptr) {
                    result = std::move(*data);
                    delete data;
                    delete head;
                    return true;
                }
            }
        }
    }

    bool empty() const {
        node* head = head_.load();
        node* tail = tail_.load();
        return head == tail && head->next.load() == nullptr;
    }
};

// Lock-free stack using atomic operations
template<typename T>
class lock_free_stack {
    struct node {
        T data;
        node* next;

        node(T val) : data(std::move(val)), next(nullptr) {}
    };

    std::atomic<node*> head_{nullptr};
    std::atomic<std::size_t> size_{0};

public:
    ~lock_free_stack() {
        while (node* old_head = head_.load()) {
            head_.store(old_head->next);
            delete old_head;
        }
    }

    void push(T item) {
        node* new_node = new node(std::move(item));
        new_node->next = head_.load();

        while (!head_.compare_exchange_weak(new_node->next, new_node)) {
            // Retry
        }
        size_.fetch_add(1);
    }

    bool pop(T& result) {
        node* old_head = head_.load();

        while (old_head && !head_.compare_exchange_weak(old_head, old_head->next)) {
            // Retry
        }

        if (old_head) {
            result = std::move(old_head->data);
            size_.fetch_sub(1);
            delete old_head;
            return true;
        }
        return false;
    }

    bool empty() const {
        return head_.load() == nullptr;
    }

    std::size_t size() const {
        return size_.load();
    }
};

// Parallel algorithms
namespace parallel {

// Parallel reduce using divide-and-conquer
template<random_access_iterator I, typename T, typename BinaryOp>
T reduce(I first, I last, T init, BinaryOp op, std::size_t grain_size = 1000) {
    auto n = std::distance(first, last);

    if (n <= grain_size) {
        return std::accumulate(first, last, init, op);
    }

    auto mid = first + n / 2;

    auto future = std::async(std::launch::async, [=] {
        return reduce(first, mid, T{}, op, grain_size);
    });

    T second_half = reduce(mid, last, T{}, op, grain_size);
    T first_half = future.get();

    return op(op(init, first_half), second_half);
}

// Parallel scan (prefix sum)
template<random_access_iterator I, typename T, typename BinaryOp>
void inclusive_scan(I first, I last, I d_first, T init, BinaryOp op) {
    auto n = std::distance(first, last);
    if (n == 0) return;

    std::size_t num_threads = std::thread::hardware_concurrency();
    std::size_t chunk_size = (n + num_threads - 1) / num_threads;

    std::vector<T> chunk_sums(num_threads);
    std::vector<std::thread> threads;

    // Phase 1: Compute local scans
    for (std::size_t i = 0; i < num_threads; ++i) {
        auto start = std::min(i * chunk_size, static_cast<std::size_t>(n));
        auto end = std::min((i + 1) * chunk_size, static_cast<std::size_t>(n));

        if (start >= end) break;

        threads.emplace_back([=, &chunk_sums, &op] {
            T sum = first[start];
            d_first[start] = sum;

            for (auto j = start + 1; j < end; ++j) {
                sum = op(sum, first[j]);
                d_first[j] = sum;
            }

            chunk_sums[i] = sum;
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Phase 2: Scan chunk sums
    for (std::size_t i = 1; i < chunk_sums.size(); ++i) {
        chunk_sums[i] = op(chunk_sums[i - 1], chunk_sums[i]);
    }

    // Phase 3: Update chunks with offsets
    threads.clear();
    for (std::size_t i = 1; i < num_threads; ++i) {
        auto start = std::min(i * chunk_size, static_cast<std::size_t>(n));
        auto end = std::min((i + 1) * chunk_size, static_cast<std::size_t>(n));

        if (start >= end) break;

        threads.emplace_back([=, &chunk_sums, &op] {
            T offset = chunk_sums[i - 1];
            for (auto j = start; j < end; ++j) {
                d_first[j] = op(offset, d_first[j]);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

// Parallel partition
template<random_access_iterator I, typename Pred>
I partition(I first, I last, Pred pred) {
    auto n = std::distance(first, last);
    if (n <= 10000) {
        return std::partition(first, last, pred);
    }

    std::size_t num_threads = std::thread::hardware_concurrency();
    std::size_t chunk_size = (n + num_threads - 1) / num_threads;

    std::vector<std::vector<std::size_t>> true_indices(num_threads);
    std::vector<std::vector<std::size_t>> false_indices(num_threads);
    std::vector<std::thread> threads;

    // Phase 1: Collect indices
    for (std::size_t i = 0; i < num_threads; ++i) {
        auto start = std::min(i * chunk_size, static_cast<std::size_t>(n));
        auto end = std::min((i + 1) * chunk_size, static_cast<std::size_t>(n));

        if (start >= end) break;

        threads.emplace_back([=, &pred, &true_indices, &false_indices] {
            for (auto j = start; j < end; ++j) {
                if (pred(first[j])) {
                    true_indices[i].push_back(j);
                } else {
                    false_indices[i].push_back(j);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Phase 2: Calculate positions
    std::size_t true_count = 0;
    for (const auto& v : true_indices) {
        true_count += v.size();
    }

    // Phase 3: Move elements
    std::vector<typename std::iterator_traits<I>::value_type> temp(n);
    std::size_t true_pos = 0;
    std::size_t false_pos = true_count;

    for (const auto& indices : true_indices) {
        for (auto idx : indices) {
            temp[true_pos++] = std::move(first[idx]);
        }
    }

    for (const auto& indices : false_indices) {
        for (auto idx : indices) {
            temp[false_pos++] = std::move(first[idx]);
        }
    }

    std::move(temp.begin(), temp.end(), first);

    return first + true_count;
}

// Parallel merge sort
template<random_access_iterator I, typename Compare>
void merge_sort(I first, I last, Compare comp) {
    auto n = std::distance(first, last);

    if (n <= 10000) {
        std::sort(first, last, comp);
        return;
    }

    auto mid = first + n / 2;

    auto future = std::async(std::launch::async, [=, &comp] {
        merge_sort(first, mid, comp);
    });

    merge_sort(mid, last, comp);
    future.wait();

    std::inplace_merge(first, mid, last, comp);
}

// Parallel quick sort
template<random_access_iterator I, typename Compare>
void quick_sort(I first, I last, Compare comp, std::size_t depth_limit = 16) {
    auto n = std::distance(first, last);

    if (n <= 1) return;

    if (n <= 10000 || depth_limit == 0) {
        std::sort(first, last, comp);
        return;
    }

    // Choose pivot
    auto pivot = first[n / 2];

    // Three-way partition
    auto middle1 = std::partition(first, last, [&](const auto& elem) {
        return comp(elem, pivot);
    });

    auto middle2 = std::partition(middle1, last, [&](const auto& elem) {
        return !comp(pivot, elem);
    });

    // Recursive calls
    auto future = std::async(std::launch::async, [=, &comp] {
        quick_sort(first, middle1, comp, depth_limit - 1);
    });

    quick_sort(middle2, last, comp, depth_limit - 1);
    future.wait();
}

// SIMD vector operations wrapper
namespace simd {

// Vector addition using AVX
inline void add_vectors_avx(const float* a, const float* b, float* result, std::size_t n) {
    std::size_t simd_n = n - (n % 8);

    for (std::size_t i = 0; i < simd_n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }

    // Handle remaining elements
    for (std::size_t i = simd_n; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

// Vector multiplication using AVX
inline void mul_vectors_avx(const float* a, const float* b, float* result, std::size_t n) {
    std::size_t simd_n = n - (n % 8);

    for (std::size_t i = 0; i < simd_n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }

    // Handle remaining elements
    for (std::size_t i = simd_n; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
}

// Dot product using AVX
inline float dot_product_avx(const float* a, const float* b, std::size_t n) {
    __m256 sum = _mm256_setzero_ps();
    std::size_t simd_n = n - (n % 8);

    for (std::size_t i = 0; i < simd_n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    sum_low = _mm_add_ps(sum_low, sum_high);

    __m128 shuf = _mm_shuffle_ps(sum_low, sum_low, _MM_SHUFFLE(2, 3, 0, 1));
    sum_low = _mm_add_ps(sum_low, shuf);
    shuf = _mm_movehl_ps(shuf, sum_low);
    sum_low = _mm_add_ss(sum_low, shuf);

    float result = _mm_cvtss_f32(sum_low);

    // Handle remaining elements
    for (std::size_t i = simd_n; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

} // namespace simd

// Parallel matrix multiplication
template<typename T>
void matrix_multiply_parallel(const T* a, const T* b, T* c,
                             std::size_t m, std::size_t n, std::size_t k,
                             std::size_t block_size = 64) {
    std::size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    auto worker = [=](std::size_t start_row, std::size_t end_row) {
        for (std::size_t i = start_row; i < end_row; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                T sum = 0;
                for (std::size_t l = 0; l < k; ++l) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    };

    std::size_t rows_per_thread = (m + num_threads - 1) / num_threads;

    for (std::size_t i = 0; i < num_threads; ++i) {
        std::size_t start_row = i * rows_per_thread;
        std::size_t end_row = std::min((i + 1) * rows_per_thread, m);

        if (start_row < end_row) {
            threads.emplace_back(worker, start_row, end_row);
        }
    }

    for (auto& t : threads) {
        t.join();
    }
}

// Parallel for_each
template<forward_iterator I, typename F>
void for_each(I first, I last, F f, std::size_t grain_size = 1000) {
    auto n = std::distance(first, last);

    if (n <= grain_size) {
        std::for_each(first, last, f);
        return;
    }

    std::size_t num_threads = std::thread::hardware_concurrency();
    std::size_t chunk_size = (n + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (std::size_t i = 0; i < num_threads; ++i) {
        auto start = first;
        std::advance(start, std::min(i * chunk_size, static_cast<std::size_t>(n)));

        auto end = first;
        std::advance(end, std::min((i + 1) * chunk_size, static_cast<std::size_t>(n)));

        if (start != end) {
            threads.emplace_back([=, &f] {
                std::for_each(start, end, f);
            });
        }
    }

    for (auto& t : threads) {
        t.join();
    }
}

// Parallel transform
template<forward_iterator I1, forward_iterator I2, typename F>
void transform(I1 first, I1 last, I2 d_first, F f, std::size_t grain_size = 1000) {
    auto n = std::distance(first, last);

    if (n <= grain_size) {
        std::transform(first, last, d_first, f);
        return;
    }

    std::size_t num_threads = std::thread::hardware_concurrency();
    std::size_t chunk_size = (n + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (std::size_t i = 0; i < num_threads; ++i) {
        auto start1 = first;
        std::advance(start1, std::min(i * chunk_size, static_cast<std::size_t>(n)));

        auto end1 = first;
        std::advance(end1, std::min((i + 1) * chunk_size, static_cast<std::size_t>(n)));

        auto start2 = d_first;
        std::advance(start2, std::min(i * chunk_size, static_cast<std::size_t>(n)));

        if (start1 != end1) {
            threads.emplace_back([=, &f] {
                std::transform(start1, end1, start2, f);
            });
        }
    }

    for (auto& t : threads) {
        t.join();
    }
}

} // namespace parallel

// Execution policies
struct sequential_execution_policy {};
struct parallel_execution_policy {};
struct parallel_unsequenced_execution_policy {};

inline constexpr sequential_execution_policy seq{};
inline constexpr parallel_execution_policy par{};
inline constexpr parallel_unsequenced_execution_policy par_unseq{};

// Generic algorithm dispatcher based on execution policy
template<typename ExecutionPolicy, typename I, typename F>
void for_each(ExecutionPolicy&& policy, I first, I last, F f) {
    if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, parallel_execution_policy>) {
        parallel::for_each(first, last, f);
    } else {
        std::for_each(first, last, f);
    }
}

} // namespace stepanov