#pragma once

#include <omp.h>
#include <thread>
#include <atomic>
#include <future>
#include <vector>
#include <algorithm>
#include <execution>
#include <cstddef>
#include <functional>
#include "simd_operations.hpp"

namespace stepanov::parallel {

// =============================================================================
// Thread Pool and Parallelization Configuration
// =============================================================================

// Get optimal number of threads
inline size_t get_num_threads() {
    static const size_t num_threads = []() {
        size_t n = std::thread::hardware_concurrency();
        if (n == 0) n = 4; // Fallback

        // Check if OpenMP is available and get its thread count
        #ifdef _OPENMP
        n = omp_get_max_threads();
        #endif

        return n;
    }();
    return num_threads;
}

// Threshold for parallel execution (tunable)
struct parallel_threshold {
    static constexpr size_t matrix_multiply = 64;    // 64x64 matrices
    static constexpr size_t matrix_add = 10000;      // 10K elements
    static constexpr size_t vector_ops = 10000;      // 10K elements
    static constexpr size_t transpose = 256;         // 256x256 matrices
    static constexpr size_t decomposition = 128;     // 128x128 for LU/QR
};

// =============================================================================
// Parallel Matrix Addition/Subtraction
// =============================================================================

template<typename T>
void matrix_add_parallel(const T* a, const T* b, T* result, size_t rows, size_t cols) {
    const size_t total_size = rows * cols;

    if (total_size < parallel_threshold::matrix_add) {
        // Use SIMD but not parallel
        simd::matrix_add_simd(a, b, result, total_size);
        return;
    }

    #pragma omp parallel for schedule(static) if(total_size > parallel_threshold::matrix_add)
    for (size_t i = 0; i < rows; ++i) {
        const size_t offset = i * cols;
        simd::matrix_add_simd(&a[offset], &b[offset], &result[offset], cols);
    }
}

template<typename T>
void matrix_sub_parallel(const T* a, const T* b, T* result, size_t rows, size_t cols) {
    const size_t total_size = rows * cols;

    if (total_size < parallel_threshold::matrix_add) {
        simd::matrix_sub_simd(a, b, result, total_size);
        return;
    }

    #pragma omp parallel for schedule(static) if(total_size > parallel_threshold::matrix_add)
    for (size_t i = 0; i < rows; ++i) {
        const size_t offset = i * cols;
        simd::matrix_sub_simd(&a[offset], &b[offset], &result[offset], cols);
    }
}

// =============================================================================
// Cache-Optimized Blocked Matrix Multiplication with OpenMP
// =============================================================================

template<typename T>
class blocked_matrix_multiply {
private:
    // Cache-optimal block sizes (tuned for modern CPUs)
    static constexpr size_t L1_BLOCK = 32;   // Fits in L1 cache
    static constexpr size_t L2_BLOCK = 256;  // Fits in L2 cache
    static constexpr size_t L3_BLOCK = 1024; // Fits in L3 cache

public:
    static void multiply(const T* __restrict__ a, const T* __restrict__ b,
                        T* __restrict__ c, size_t m, size_t n, size_t k,
                        bool clear_c = true) {
        // Clear output matrix if needed
        if (clear_c) {
            #pragma omp parallel for collapse(2) if(m * n > parallel_threshold::matrix_multiply * parallel_threshold::matrix_multiply)
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    c[i * n + j] = T(0);
                }
            }
        }

        // Choose block size based on matrix size
        const size_t block_size = choose_block_size(m, n, k);

        if (m * n * k < 64 * 64 * 64) {
            // Small matrices - use SIMD kernel directly
            simd::matmul_kernel_simd(a, b, c, m, n, k, k, n, n);
            return;
        }

        // Large matrices - use cache-blocked parallel algorithm
        multiply_blocked_parallel(a, b, c, m, n, k, block_size);
    }

private:
    static size_t choose_block_size(size_t m, size_t n, size_t k) {
        size_t max_dim = std::max({m, n, k});

        if (max_dim <= 128) return L1_BLOCK;
        else if (max_dim <= 512) return std::min(size_t(64), max_dim);
        else if (max_dim <= 2048) return std::min(L2_BLOCK, max_dim);
        else return L3_BLOCK;
    }

    static void multiply_blocked_parallel(const T* __restrict__ a,
                                         const T* __restrict__ b,
                                         T* __restrict__ c,
                                         size_t m, size_t n, size_t k,
                                         size_t block_size) {
        // Three-level blocking for cache hierarchy
        const size_t BS1 = std::min(block_size, L1_BLOCK);
        const size_t BS2 = std::min(block_size, L2_BLOCK);

        #pragma omp parallel for collapse(2) schedule(dynamic, 1)
        for (size_t ii = 0; ii < m; ii += BS2) {
            for (size_t jj = 0; jj < n; jj += BS2) {
                // L2 cache blocking
                const size_t i_end = std::min(ii + BS2, m);
                const size_t j_end = std::min(jj + BS2, n);

                for (size_t kk = 0; kk < k; kk += BS2) {
                    const size_t k_end = std::min(kk + BS2, k);

                    // L1 cache blocking
                    for (size_t i = ii; i < i_end; i += BS1) {
                        const size_t i_block = std::min(BS1, i_end - i);

                        for (size_t j = jj; j < j_end; j += BS1) {
                            const size_t j_block = std::min(BS1, j_end - j);

                            for (size_t k_pos = kk; k_pos < k_end; k_pos += BS1) {
                                const size_t k_block = std::min(BS1, k_end - k_pos);

                                // Use SIMD kernel for the innermost block
                                multiply_block_simd(&a[i * k + k_pos],
                                                  &b[k_pos * n + j],
                                                  &c[i * n + j],
                                                  i_block, j_block, k_block,
                                                  k, n, n,
                                                  k_pos == 0);
                            }
                        }
                    }
                }
            }
        }
    }

    static void multiply_block_simd(const T* __restrict__ a,
                                   const T* __restrict__ b,
                                   T* __restrict__ c,
                                   size_t m, size_t n, size_t k,
                                   size_t lda, size_t ldb, size_t ldc,
                                   bool clear) {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            // Initialize C block if this is the first K block
            if (clear) {
                for (size_t i = 0; i < m; ++i) {
                    simd::prefetch(&a[i * lda], 1);
                    for (size_t j = 0; j < n; ++j) {
                        c[i * ldc + j] = T(0);
                    }
                }
            }

            // Use SIMD kernel with prefetching
            constexpr size_t UNROLL = 4;

            for (size_t i = 0; i < m; i += UNROLL) {
                for (size_t j = 0; j < n; j += simd::double_simd::simd_width) {
                    // Prefetch next blocks
                    if (i + UNROLL < m) {
                        simd::prefetch(&a[(i + UNROLL) * lda], 1);
                        simd::prefetch(&c[(i + UNROLL) * ldc + j], 1);
                    }

                    // Compute block using SIMD
                    const size_t i_limit = std::min(i + UNROLL, m);
                    const size_t j_limit = std::min(j + simd::double_simd::simd_width, n);

                    for (size_t ii = i; ii < i_limit; ++ii) {
                        auto sum = simd::double_simd::setzero();

                        for (size_t kk = 0; kk < k; ++kk) {
                            if constexpr (std::is_same_v<T, double>) {
                                auto a_val = simd::double_simd::set1(a[ii * lda + kk]);
                                auto b_vec = simd::double_simd::load_unaligned(&b[kk * ldb + j]);
                                sum = simd::double_simd::fmadd(a_val, b_vec, sum);
                            }
                        }

                        // Add to C
                        if (j + simd::double_simd::simd_width <= n) {
                            auto c_vec = simd::double_simd::load_unaligned(&c[ii * ldc + j]);
                            c_vec = simd::double_simd::add(c_vec, sum);
                            simd::double_simd::store_unaligned(&c[ii * ldc + j], c_vec);
                        } else {
                            // Handle remainder
                            alignas(64) T temp[simd::double_simd::simd_width];
                            simd::double_simd::store_aligned(temp, sum);
                            for (size_t jj = j; jj < j_limit; ++jj) {
                                c[ii * ldc + jj] += temp[jj - j];
                            }
                        }
                    }
                }
            }
        } else {
            // Fallback for non-SIMD types
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    T sum = clear ? T(0) : c[i * ldc + j];
                    for (size_t kk = 0; kk < k; ++kk) {
                        sum += a[i * lda + kk] * b[kk * ldb + j];
                    }
                    c[i * ldc + j] = sum;
                }
            }
        }
    }
};

// =============================================================================
// Parallel Matrix Transpose
// =============================================================================

template<typename T>
void transpose_parallel(const T* src, T* dst, size_t rows, size_t cols) {
    if (rows * cols < parallel_threshold::transpose * parallel_threshold::transpose) {
        // Small matrix - use SIMD transpose without parallelization
        simd::transpose_block_simd(src, dst, rows, cols, cols, rows);
        return;
    }

    // Cache-blocked parallel transpose
    constexpr size_t BLOCK = 64;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < rows; i += BLOCK) {
        for (size_t j = 0; j < cols; j += BLOCK) {
            const size_t i_end = std::min(i + BLOCK, rows);
            const size_t j_end = std::min(j + BLOCK, cols);

            // Transpose block
            for (size_t ii = i; ii < i_end; ++ii) {
                for (size_t jj = j; jj < j_end; ++jj) {
                    dst[jj * rows + ii] = src[ii * cols + jj];
                }
            }
        }
    }
}

// =============================================================================
// Parallel Reduction Operations
// =============================================================================

template<typename T>
T parallel_reduce(const T* data, size_t size, T identity,
                  std::function<T(T, T)> op) {
    if (size < parallel_threshold::vector_ops) {
        // Serial reduction
        T result = identity;
        for (size_t i = 0; i < size; ++i) {
            result = op(result, data[i]);
        }
        return result;
    }

    // Parallel reduction with OpenMP
    T result = identity;

    #pragma omp parallel
    {
        T local_result = identity;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < size; ++i) {
            local_result = op(local_result, data[i]);
        }

        #pragma omp critical
        result = op(result, local_result);
    }

    return result;
}

// Parallel sum
template<typename T>
T parallel_sum(const T* data, size_t size) {
    return parallel_reduce(data, size, T(0), std::plus<T>());
}

// Parallel product
template<typename T>
T parallel_product(const T* data, size_t size) {
    return parallel_reduce(data, size, T(1), std::multiplies<T>());
}

// Parallel min/max
template<typename T>
T parallel_min(const T* data, size_t size) {
    if (size == 0) return T();

    T result = data[0];

    #pragma omp parallel for reduction(min:result)
    for (size_t i = 1; i < size; ++i) {
        if (data[i] < result) result = data[i];
    }

    return result;
}

template<typename T>
T parallel_max(const T* data, size_t size) {
    if (size == 0) return T();

    T result = data[0];

    #pragma omp parallel for reduction(max:result)
    for (size_t i = 1; i < size; ++i) {
        if (data[i] > result) result = data[i];
    }

    return result;
}

// =============================================================================
// Parallel Linear Algebra Operations
// =============================================================================

// Parallel LU decomposition
template<typename T>
void lu_decompose_parallel(T* matrix, size_t n, size_t* pivot) {
    for (size_t k = 0; k < n - 1; ++k) {
        // Find pivot (sequential - hard to parallelize effectively)
        size_t max_row = k;
        T max_val = std::abs(matrix[k * n + k]);

        for (size_t i = k + 1; i < n; ++i) {
            T val = std::abs(matrix[i * n + k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }

        pivot[k] = max_row;

        // Swap rows if needed
        if (max_row != k) {
            #pragma omp parallel for simd
            for (size_t j = 0; j < n; ++j) {
                std::swap(matrix[k * n + j], matrix[max_row * n + j]);
            }
        }

        // Compute multipliers and eliminate column (parallel)
        #pragma omp parallel for if(n - k > 64)
        for (size_t i = k + 1; i < n; ++i) {
            T factor = matrix[i * n + k] / matrix[k * n + k];
            matrix[i * n + k] = factor; // Store multiplier in lower triangle

            // Update row i (vectorized)
            #pragma omp simd
            for (size_t j = k + 1; j < n; ++j) {
                matrix[i * n + j] -= factor * matrix[k * n + j];
            }
        }
    }
}

// Parallel QR decomposition using Householder reflections
template<typename T>
void qr_decompose_parallel(T* matrix, T* q, T* r, size_t m, size_t n) {
    // Initialize Q to identity
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            q[i * m + j] = (i == j) ? T(1) : T(0);
        }
    }

    // Copy matrix to R
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            r[i * n + j] = matrix[i * n + j];
        }
    }

    std::vector<T> v(m);

    for (size_t k = 0; k < std::min(m - 1, n); ++k) {
        // Compute Householder vector
        T norm = T(0);
        for (size_t i = k; i < m; ++i) {
            norm += r[i * n + k] * r[i * n + k];
        }
        norm = std::sqrt(norm);

        if (r[k * n + k] > 0) norm = -norm;

        v[k] = r[k * n + k] - norm;
        for (size_t i = k + 1; i < m; ++i) {
            v[i] = r[i * n + k];
        }

        T vnorm = T(0);
        for (size_t i = k; i < m; ++i) {
            vnorm += v[i] * v[i];
        }

        if (vnorm > T(0)) {
            // Apply Householder transformation to R (parallel)
            #pragma omp parallel for if(n - k > 32)
            for (size_t j = k; j < n; ++j) {
                T dot = T(0);
                for (size_t i = k; i < m; ++i) {
                    dot += v[i] * r[i * n + j];
                }
                dot = 2 * dot / vnorm;

                for (size_t i = k; i < m; ++i) {
                    r[i * n + j] -= dot * v[i];
                }
            }

            // Apply Householder transformation to Q (parallel)
            #pragma omp parallel for if(m > 32)
            for (size_t j = 0; j < m; ++j) {
                T dot = T(0);
                for (size_t i = k; i < m; ++i) {
                    dot += v[i] * q[i * m + j];
                }
                dot = 2 * dot / vnorm;

                for (size_t i = k; i < m; ++i) {
                    q[i * m + j] -= dot * v[i];
                }
            }
        }
    }
}

// =============================================================================
// Work-Stealing Thread Pool for Dynamic Parallelism
// =============================================================================

template<typename T>
class work_stealing_pool {
private:
    struct task {
        std::function<void()> work;
        std::atomic<bool> done{false};
    };

    std::vector<std::thread> workers;
    std::vector<std::deque<std::unique_ptr<task>>> queues;
    std::vector<std::mutex> queue_locks;
    std::atomic<bool> stop{false};
    std::atomic<size_t> active_tasks{0};

public:
    explicit work_stealing_pool(size_t num_threads = 0) {
        if (num_threads == 0) {
            num_threads = get_num_threads();
        }

        queues.resize(num_threads);
        queue_locks.resize(num_threads);

        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this, i] { worker_thread(i); });
        }
    }

    ~work_stealing_pool() {
        stop = true;
        for (auto& worker : workers) {
            worker.join();
        }
    }

    template<typename F>
    void submit(F&& f, size_t hint = -1) {
        auto t = std::make_unique<task>();
        t->work = std::forward<F>(f);

        size_t queue_idx = (hint == size_t(-1)) ?
            std::hash<std::thread::id>{}(std::this_thread::get_id()) % queues.size() :
            hint % queues.size();

        {
            std::lock_guard<std::mutex> lock(queue_locks[queue_idx]);
            queues[queue_idx].push_back(std::move(t));
        }

        active_tasks++;
    }

    void wait_all() {
        while (active_tasks > 0) {
            std::this_thread::yield();
        }
    }

private:
    void worker_thread(size_t id) {
        while (!stop) {
            std::unique_ptr<task> t = steal_task(id);

            if (t) {
                t->work();
                t->done = true;
                active_tasks--;
            } else {
                std::this_thread::yield();
            }
        }
    }

    std::unique_ptr<task> steal_task(size_t id) {
        // Try own queue first
        {
            std::lock_guard<std::mutex> lock(queue_locks[id]);
            if (!queues[id].empty()) {
                auto t = std::move(queues[id].front());
                queues[id].pop_front();
                return t;
            }
        }

        // Try stealing from other queues
        for (size_t i = 0; i < queues.size(); ++i) {
            if (i == id) continue;

            std::lock_guard<std::mutex> lock(queue_locks[i]);
            if (!queues[i].empty()) {
                auto t = std::move(queues[i].back());
                queues[i].pop_back();
                return t;
            }
        }

        return nullptr;
    }
};

} // namespace stepanov::parallel