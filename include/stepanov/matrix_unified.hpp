#pragma once

/**
 * Unified Matrix Library - Elegant, Composable, High-Performance
 *
 * Design Philosophy:
 * - Single, unified interface for all matrix types
 * - Automatic structure detection and optimization
 * - Clean separation of storage, structure, and operations
 * - Zero-cost abstractions via templates and concepts
 * - Composable building blocks following STL patterns
 */

#include <concepts>
#include <memory>
#include <vector>
#include <span>
#include <algorithm>
#include <ranges>
#include <type_traits>
#include <cstring>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace stepanov {

// =============================================================================
// Core Concepts - Mathematical Requirements
// =============================================================================

template<typename T>
concept arithmetic = std::integral<T> || std::floating_point<T>;

template<typename T>
concept field = arithmetic<T> && requires(T a, T b) {
    { a + b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
    { a * b } -> std::same_as<T>;
    { a / b } -> std::same_as<T>;
};

// =============================================================================
// Storage Policies - Separating Memory Management
// =============================================================================

template<typename T>
class storage_policy {
public:
    virtual ~storage_policy() = default;
    virtual T& at(size_t i, size_t j) = 0;
    virtual const T& at(size_t i, size_t j) const = 0;
    virtual T* data() = 0;
    virtual const T* data() const = 0;
    virtual size_t memory_usage() const = 0;
    virtual bool is_contiguous() const = 0;
};

// Dense storage with proper alignment
template<typename T>
class dense_storage : public storage_policy<T> {
private:
    struct alignas(32) aligned_deleter {
        void operator()(T* p) const {
            #ifdef _WIN32
            _aligned_free(p);
            #else
            std::free(p);
            #endif
        }
    };

    std::unique_ptr<T[], aligned_deleter> data_;
    size_t rows_, cols_;

    static T* allocate(size_t count) {
        #ifdef _WIN32
        return static_cast<T*>(_aligned_malloc(count * sizeof(T), 32));
        #else
        void* ptr;
        if (posix_memalign(&ptr, 32, count * sizeof(T)) != 0)
            throw std::bad_alloc();
        return static_cast<T*>(ptr);
        #endif
    }

public:
    dense_storage(size_t r, size_t c, T val = T{})
        : data_(allocate(r * c)), rows_(r), cols_(c) {
        std::fill_n(data_.get(), r * c, val);
    }

    T& at(size_t i, size_t j) override {
        return data_[i * cols_ + j];
    }

    const T& at(size_t i, size_t j) const override {
        return data_[i * cols_ + j];
    }

    T* data() override { return data_.get(); }
    const T* data() const override { return data_.get(); }

    size_t memory_usage() const override {
        return rows_ * cols_ * sizeof(T);
    }

    bool is_contiguous() const override { return true; }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
};

// =============================================================================
// Matrix Structures - Clean Trait-Based Design
// =============================================================================

// Structure tags for compile-time dispatch
struct dense_tag {};
struct diagonal_tag {};
struct sparse_tag {};
struct symmetric_tag {};
struct triangular_tag {};
struct banded_tag {};
struct low_rank_tag {};

// Structure traits
template<typename Tag>
struct structure_traits {
    static constexpr bool is_structured = false;
    static constexpr bool is_sparse = false;
    static constexpr bool is_symmetric = false;
};

template<>
struct structure_traits<diagonal_tag> {
    static constexpr bool is_structured = true;
    static constexpr bool is_sparse = true;
    static constexpr bool is_symmetric = false;
};

template<>
struct structure_traits<sparse_tag> {
    static constexpr bool is_structured = true;
    static constexpr bool is_sparse = true;
    static constexpr bool is_symmetric = false;
};

template<>
struct structure_traits<symmetric_tag> {
    static constexpr bool is_structured = true;
    static constexpr bool is_sparse = false;
    static constexpr bool is_symmetric = true;
};

// =============================================================================
// Operation Strategies - Policy-Based Design
// =============================================================================

enum class exec_policy {
    sequential,
    parallel,
    vectorized,
    auto_select
};

template<typename T>
struct operation_traits {
    static constexpr size_t simd_width =
        std::is_same_v<T, double> ? 4 :
        std::is_same_v<T, float> ? 8 : 1;

    static constexpr size_t cache_line = 64;
    static constexpr size_t l1_size = 32 * 1024;

    static constexpr size_t parallel_threshold = 500;
    static constexpr size_t simd_threshold = 100;

    static exec_policy select_policy(size_t n) {
        if (n < simd_threshold) return exec_policy::sequential;
        if (n < parallel_threshold) return exec_policy::vectorized;
        return exec_policy::parallel;
    }
};

// =============================================================================
// Unified Matrix Interface
// =============================================================================

template<typename T, typename Structure = dense_tag>
    requires field<T>
class matrix {
public:
    using value_type = T;
    using structure_type = Structure;
    using traits = structure_traits<Structure>;

private:
    std::unique_ptr<storage_policy<T>> storage_;
    size_t rows_, cols_;
    exec_policy policy_;

    // Factory method for creating appropriate storage
    static auto create_storage(size_t r, size_t c, Structure = {}) {
        if constexpr (std::is_same_v<Structure, dense_tag>) {
            return std::make_unique<dense_storage<T>>(r, c);
        }
        // Add other storage types as needed
        else {
            return std::make_unique<dense_storage<T>>(r, c);
        }
    }

public:
    // Constructors
    matrix(size_t r, size_t c)
        : storage_(create_storage(r, c)),
          rows_(r), cols_(c),
          policy_(exec_policy::auto_select) {}

    matrix(size_t r, size_t c, T val)
        : storage_(create_storage(r, c)),
          rows_(r), cols_(c),
          policy_(exec_policy::auto_select) {
        fill(val);
    }

    // Size and structure queries
    [[nodiscard]] size_t rows() const noexcept { return rows_; }
    [[nodiscard]] size_t cols() const noexcept { return cols_; }
    [[nodiscard]] size_t size() const noexcept { return rows_ * cols_; }
    [[nodiscard]] bool is_square() const noexcept { return rows_ == cols_; }
    [[nodiscard]] bool is_structured() const noexcept { return traits::is_structured; }

    // Element access
    T& operator()(size_t i, size_t j) { return storage_->at(i, j); }
    const T& operator()(size_t i, size_t j) const { return storage_->at(i, j); }

    T& at(size_t i, size_t j) {
        if (i >= rows_ || j >= cols_) throw std::out_of_range("Matrix index out of range");
        return (*this)(i, j);
    }

    const T& at(size_t i, size_t j) const {
        if (i >= rows_ || j >= cols_) throw std::out_of_range("Matrix index out of range");
        return (*this)(i, j);
    }

    // Raw data access (if contiguous)
    [[nodiscard]] T* data() { return storage_->data(); }
    [[nodiscard]] const T* data() const { return storage_->data(); }

    // Row and column views (C++20 ranges)
    [[nodiscard]] auto row(size_t i) {
        return std::views::iota(size_t{0}, cols_)
             | std::views::transform([this, i](size_t j) -> T& {
                   return (*this)(i, j);
               });
    }

    [[nodiscard]] auto col(size_t j) {
        return std::views::iota(size_t{0}, rows_)
             | std::views::transform([this, j](size_t i) -> T& {
                   return (*this)(i, j);
               });
    }

    // Initialization
    void fill(T val) {
        if (storage_->is_contiguous()) {
            std::fill_n(data(), size(), val);
        } else {
            for (size_t i = 0; i < rows_; ++i)
                for (size_t j = 0; j < cols_; ++j)
                    (*this)(i, j) = val;
        }
    }

    void zero() { fill(T{0}); }
    void identity() {
        if (!is_square()) {
            throw std::runtime_error("Identity matrix must be square");
        }
        zero();
        for (size_t i = 0; i < rows_; ++i)
            (*this)(i, i) = T{1};
    }

    // Policy control
    void set_execution_policy(exec_policy p) { policy_ = p; }
    [[nodiscard]] exec_policy execution_policy() const { return policy_; }

    // Memory information
    [[nodiscard]] size_t memory_usage() const {
        return storage_->memory_usage();
    }

    // Arithmetic operations with automatic optimization
    matrix operator+(const matrix& other) const {
        return binary_op(other, std::plus<T>{});
    }

    matrix operator-(const matrix& other) const {
        return binary_op(other, std::minus<T>{});
    }

    matrix operator*(const matrix& other) const {
        return multiply(other);
    }

    matrix operator*(T scalar) const {
        return scalar_op(scalar, std::multiplies<T>{});
    }

    friend matrix operator*(T scalar, const matrix& m) {
        return m * scalar;
    }

private:
    // Generic binary operation with policy-based execution
    template<typename Op>
    matrix binary_op(const matrix& other, Op op) const {
        matrix result(rows_, cols_);

        auto policy = (policy_ == exec_policy::auto_select)
            ? operation_traits<T>::select_policy(size())
            : policy_;

        if (storage_->is_contiguous() && other.storage_->is_contiguous()) {
            // Fast path for contiguous storage
            const T* a = data();
            const T* b = other.data();
            T* c = result.data();

            #pragma omp parallel for if(policy == exec_policy::parallel)
            for (size_t i = 0; i < size(); ++i) {
                c[i] = op(a[i], b[i]);
            }
        } else {
            // Element-wise operation
            #pragma omp parallel for collapse(2) if(policy == exec_policy::parallel)
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    result(i, j) = op((*this)(i, j), other(i, j));
                }
            }
        }

        return result;
    }

    // Scalar operation
    template<typename Op>
    matrix scalar_op(T scalar, Op op) const {
        matrix result(rows_, cols_);

        auto policy = (policy_ == exec_policy::auto_select)
            ? operation_traits<T>::select_policy(size())
            : policy_;

        if (storage_->is_contiguous()) {
            const T* a = data();
            T* c = result.data();

            #pragma omp parallel for simd if(policy == exec_policy::parallel)
            for (size_t i = 0; i < size(); ++i) {
                c[i] = op(a[i], scalar);
            }
        } else {
            #pragma omp parallel for collapse(2) if(policy == exec_policy::parallel)
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    result(i, j) = op((*this)(i, j), scalar);
                }
            }
        }

        return result;
    }

    // Matrix multiplication with automatic algorithm selection
    matrix multiply(const matrix& other) const {
        matrix result(rows_, other.cols_);

        auto policy = (policy_ == exec_policy::auto_select)
            ? operation_traits<T>::select_policy(std::max(rows_, other.cols()))
            : policy_;

        if constexpr (traits::is_structured) {
            // Use specialized multiplication for structured matrices
            return multiply_structured(other);
        } else {
            // Use cache-blocked multiplication for dense matrices
            multiply_blocked(*this, other, result, policy);
        }

        return result;
    }

    // Cache-blocked multiplication
    static void multiply_blocked(const matrix& A, const matrix& B, matrix& C, exec_policy policy) {
        constexpr size_t block_size = 64;
        const size_t M = A.rows();
        const size_t N = B.cols();
        const size_t K = A.cols();

        C.zero();

        #pragma omp parallel for collapse(3) if(policy == exec_policy::parallel)
        for (size_t ii = 0; ii < M; ii += block_size) {
            for (size_t jj = 0; jj < N; jj += block_size) {
                for (size_t kk = 0; kk < K; kk += block_size) {
                    // Process block
                    for (size_t i = ii; i < std::min(ii + block_size, M); ++i) {
                        for (size_t j = jj; j < std::min(jj + block_size, N); ++j) {
                            T sum = C(i, j);
                            for (size_t k = kk; k < std::min(kk + block_size, K); ++k) {
                                sum += A(i, k) * B(k, j);
                            }
                            C(i, j) = sum;
                        }
                    }
                }
            }
        }
    }

    // Placeholder for structured multiplication
    matrix multiply_structured(const matrix& other) const {
        // This would dispatch to specialized implementations
        matrix result(rows_, other.cols_);
        multiply_blocked(*this, other, result, policy_);
        return result;
    }
};

// =============================================================================
// Specialized Matrix Types as Type Aliases
// =============================================================================

template<typename T>
using dense_matrix = matrix<T, dense_tag>;

template<typename T>
using diagonal_matrix = matrix<T, diagonal_tag>;

template<typename T>
using sparse_matrix = matrix<T, sparse_tag>;

template<typename T>
using symmetric_matrix = matrix<T, symmetric_tag>;

// =============================================================================
// Matrix Factory Functions
// =============================================================================

// Create identity matrix
template<typename T>
[[nodiscard]] auto identity(size_t n) {
    dense_matrix<T> I(n, n);
    I.identity();
    return I;
}

// Create zero matrix
template<typename T>
[[nodiscard]] auto zeros(size_t rows, size_t cols) {
    return dense_matrix<T>(rows, cols, T{0});
}

// Create ones matrix
template<typename T>
[[nodiscard]] auto ones(size_t rows, size_t cols) {
    return dense_matrix<T>(rows, cols, T{1});
}

// =============================================================================
// Structure Detection and Conversion
// =============================================================================

template<typename T>
class structure_analyzer {
public:
    struct analysis_result {
        bool is_diagonal = true;
        bool is_sparse = false;
        bool is_symmetric = true;
        bool is_triangular_upper = true;
        bool is_triangular_lower = true;
        bool is_banded = false;

        double sparsity = 0.0;
        size_t bandwidth = 0;
        size_t nnz = 0;
    };

    static analysis_result analyze(const dense_matrix<T>& M, T tolerance = T{1e-10}) {
        analysis_result result;
        const size_t n = M.rows();
        const size_t m = M.cols();

        result.is_symmetric = (n == m);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                T val = M(i, j);

                if (std::abs(val) > tolerance) {
                    result.nnz++;
                    if (i != j) result.is_diagonal = false;
                    if (i > j) result.is_triangular_upper = false;
                    if (i < j) result.is_triangular_lower = false;

                    // Update bandwidth
                    if (i > j) {
                        result.bandwidth = std::max(result.bandwidth, i - j);
                    } else {
                        result.bandwidth = std::max(result.bandwidth, j - i);
                    }
                }

                // Check symmetry
                if (result.is_symmetric && i < j) {
                    if (std::abs(val - M(j, i)) > tolerance) {
                        result.is_symmetric = false;
                    }
                }
            }
        }

        result.sparsity = 1.0 - static_cast<double>(result.nnz) / (n * m);
        result.is_sparse = (result.sparsity > 0.9);
        result.is_banded = (result.bandwidth < std::min(n, m) / 10);

        return result;
    }

    // Convert to optimal structure based on analysis
    template<typename Structure>
    static auto convert(const dense_matrix<T>& M) {
        if constexpr (std::is_same_v<Structure, diagonal_tag>) {
            // Extract diagonal
            diagonal_matrix<T> D(M.rows(), M.rows());
            for (size_t i = 0; i < M.rows(); ++i) {
                D(i, i) = M(i, i);
            }
            return D;
        } else {
            // Default: return copy
            return M;
        }
    }
};

// =============================================================================
// High-Level Operations
// =============================================================================

// Matrix-vector multiplication
template<typename T, typename S>
[[nodiscard]] std::vector<T> operator*(const matrix<T, S>& M, const std::vector<T>& v) {
    std::vector<T> result(M.rows(), T{0});

    #pragma omp parallel for
    for (size_t i = 0; i < M.rows(); ++i) {
        T sum = T{0};
        for (size_t j = 0; j < M.cols(); ++j) {
            sum += M(i, j) * v[j];
        }
        result[i] = sum;
    }

    return result;
}

// Transpose
template<typename T, typename S>
[[nodiscard]] auto transpose(const matrix<T, S>& M) {
    dense_matrix<T> result(M.cols(), M.rows());

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M.rows(); ++i) {
        for (size_t j = 0; j < M.cols(); ++j) {
            result(j, i) = M(i, j);
        }
    }

    return result;
}

// Trace (sum of diagonal)
template<typename T, typename S>
[[nodiscard]] T trace(const matrix<T, S>& M) {
    if (!M.is_square()) {
        throw std::runtime_error("Trace requires square matrix");
    }
    T sum = T{0};
    for (size_t i = 0; i < M.rows(); ++i) {
        sum += M(i, i);
    }
    return sum;
}

} // namespace stepanov