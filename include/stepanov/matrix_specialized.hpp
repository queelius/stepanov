#pragma once

/**
 * Specialized Matrix Implementations
 *
 * Clean, focused implementations for specific matrix structures.
 * Each specialization exploits its mathematical properties for optimal performance.
 */

#include "matrix_unified.hpp"
#include <unordered_map>
#include <tuple>
#include <functional>

namespace stepanov {

// =============================================================================
// Diagonal Matrix - O(n) storage, O(n²) operations
// =============================================================================

template<typename T>
class diagonal_storage : public storage_policy<T> {
private:
    std::vector<T> diagonal_;
    size_t size_;
    mutable T zero_ = T{0};  // For returning references to zero elements

public:
    explicit diagonal_storage(size_t n) : diagonal_(n), size_(n) {}

    T& at(size_t i, size_t j) override {
        return (i == j) ? diagonal_[i] : (zero_ = T{0});
    }

    const T& at(size_t i, size_t j) const override {
        return (i == j) ? diagonal_[i] : (zero_ = T{0});
    }

    T* data() override { return diagonal_.data(); }
    const T* data() const override { return diagonal_.data(); }

    size_t memory_usage() const override {
        return diagonal_.size() * sizeof(T);
    }

    bool is_contiguous() const override { return false; }

    // Specialized diagonal operations
    std::vector<T>& diagonal() { return diagonal_; }
    const std::vector<T>& diagonal() const { return diagonal_; }
};

// Specialized diagonal matrix operations
template<typename T>
class diagonal_matrix_ops {
public:
    // Diagonal * Vector: O(n) instead of O(n²)
    static std::vector<T> multiply_vector(const diagonal_storage<T>& D,
                                          const std::vector<T>& v) {
        const auto& diag = D.diagonal();
        std::vector<T> result(diag.size());

        #pragma omp parallel for simd
        for (size_t i = 0; i < diag.size(); ++i) {
            result[i] = diag[i] * v[i];
        }

        return result;
    }

    // Diagonal * Matrix: O(n²) instead of O(n³)
    template<typename S>
    static matrix<T, S> multiply_matrix(const diagonal_storage<T>& D,
                                        const matrix<T, S>& M) {
        const auto& diag = D.diagonal();
        matrix<T, S> result(diag.size(), M.cols());

        #pragma omp parallel for
        for (size_t i = 0; i < diag.size(); ++i) {
            T d = diag[i];
            #pragma omp simd
            for (size_t j = 0; j < M.cols(); ++j) {
                result(i, j) = d * M(i, j);
            }
        }

        return result;
    }

    // Diagonal * Diagonal: O(n) operation
    static diagonal_storage<T> multiply_diagonal(const diagonal_storage<T>& D1,
                                                 const diagonal_storage<T>& D2) {
        const auto& diag1 = D1.diagonal();
        const auto& diag2 = D2.diagonal();
        diagonal_storage<T> result(diag1.size());

        #pragma omp parallel for simd
        for (size_t i = 0; i < diag1.size(); ++i) {
            result.diagonal()[i] = diag1[i] * diag2[i];
        }

        return result;
    }
};

// =============================================================================
// Sparse Matrix - CSR Format
// =============================================================================

template<typename T>
class csr_storage : public storage_policy<T> {
private:
    std::vector<T> values_;
    std::vector<size_t> col_indices_;
    std::vector<size_t> row_ptrs_;
    size_t rows_, cols_;
    mutable T zero_ = T{0};

public:
    csr_storage(size_t r, size_t c)
        : rows_(r), cols_(c), row_ptrs_(r + 1, 0) {}

    // Build interface
    void add_entry(size_t i, size_t j, T val) {
        if (std::abs(val) > T{1e-10}) {
            values_.push_back(val);
            col_indices_.push_back(j);
            row_ptrs_[i + 1]++;
        }
    }

    void finalize() {
        // Convert row counts to cumulative pointers
        for (size_t i = 1; i <= rows_; ++i) {
            row_ptrs_[i] += row_ptrs_[i - 1];
        }
    }

    T& at(size_t i, size_t j) override {
        for (size_t k = row_ptrs_[i]; k < row_ptrs_[i + 1]; ++k) {
            if (col_indices_[k] == j) {
                return values_[k];
            }
        }
        return (zero_ = T{0});
    }

    const T& at(size_t i, size_t j) const override {
        for (size_t k = row_ptrs_[i]; k < row_ptrs_[i + 1]; ++k) {
            if (col_indices_[k] == j) {
                return values_[k];
            }
        }
        return (zero_ = T{0});
    }

    T* data() override { return nullptr; }  // Not contiguous
    const T* data() const override { return nullptr; }

    size_t memory_usage() const override {
        return values_.size() * sizeof(T) +
               col_indices_.size() * sizeof(size_t) +
               row_ptrs_.size() * sizeof(size_t);
    }

    bool is_contiguous() const override { return false; }

    // Sparse-specific operations
    size_t nnz() const { return values_.size(); }
    double sparsity() const { return 1.0 - double(nnz()) / (rows_ * cols_); }

    // Optimized sparse matrix-vector multiplication: O(nnz)
    std::vector<T> multiply_vector(const std::vector<T>& x) const {
        std::vector<T> y(rows_, T{0});

        #pragma omp parallel for if(rows_ > 1000)
        for (size_t i = 0; i < rows_; ++i) {
            T sum = T{0};
            for (size_t k = row_ptrs_[i]; k < row_ptrs_[i + 1]; ++k) {
                sum += values_[k] * x[col_indices_[k]];
            }
            y[i] = sum;
        }

        return y;
    }
};

// =============================================================================
// Symmetric Matrix - Store only upper triangle
// =============================================================================

template<typename T>
class symmetric_storage : public storage_policy<T> {
private:
    std::vector<T> upper_;  // Upper triangle in packed format
    size_t n_;

    // Map 2D index to 1D packed storage
    size_t packed_index(size_t i, size_t j) const {
        if (i > j) std::swap(i, j);
        return i * n_ - i * (i - 1) / 2 + j - i;
    }

public:
    explicit symmetric_storage(size_t n)
        : upper_(n * (n + 1) / 2), n_(n) {}

    T& at(size_t i, size_t j) override {
        return upper_[packed_index(i, j)];
    }

    const T& at(size_t i, size_t j) const override {
        return upper_[packed_index(i, j)];
    }

    T* data() override { return upper_.data(); }
    const T* data() const override { return upper_.data(); }

    size_t memory_usage() const override {
        return upper_.size() * sizeof(T);
    }

    bool is_contiguous() const override { return false; }

    // Symmetric matrix-vector multiplication exploiting symmetry
    std::vector<T> multiply_vector(const std::vector<T>& v) const {
        std::vector<T> result(n_, T{0});

        #pragma omp parallel for
        for (size_t i = 0; i < n_; ++i) {
            T sum = T{0};

            // Diagonal element
            sum += at(i, i) * v[i];

            // Off-diagonal elements (exploit symmetry)
            for (size_t j = i + 1; j < n_; ++j) {
                T val = at(i, j);
                sum += val * v[j];

                #pragma omp atomic
                result[j] += val * v[i];  // Symmetric contribution
            }

            #pragma omp atomic
            result[i] += sum;
        }

        return result;
    }
};

// =============================================================================
// Banded Matrix - Store only bands
// =============================================================================

template<typename T>
class banded_storage : public storage_policy<T> {
private:
    std::vector<T> bands_;
    size_t n_;
    size_t lower_bw_;
    size_t upper_bw_;
    mutable T zero_ = T{0};

    size_t band_index(size_t i, size_t j) const {
        size_t band = j - i + lower_bw_;
        size_t total_bands = lower_bw_ + upper_bw_ + 1;
        return i * total_bands + band;
    }

public:
    banded_storage(size_t n, size_t lower, size_t upper)
        : n_(n), lower_bw_(lower), upper_bw_(upper) {
        size_t total = lower_bw_ + upper_bw_ + 1;
        bands_.resize(n * total, T{0});
    }

    T& at(size_t i, size_t j) override {
        if (j < i - lower_bw_ || j > i + upper_bw_) {
            return (zero_ = T{0});
        }
        return bands_[band_index(i, j)];
    }

    const T& at(size_t i, size_t j) const override {
        if (j < i - lower_bw_ || j > i + upper_bw_) {
            return (zero_ = T{0});
        }
        return bands_[band_index(i, j)];
    }

    T* data() override { return bands_.data(); }
    const T* data() const override { return bands_.data(); }

    size_t memory_usage() const override {
        return bands_.size() * sizeof(T);
    }

    bool is_contiguous() const override { return false; }

    // Banded matrix-vector multiplication: O(n * bandwidth)
    std::vector<T> multiply_vector(const std::vector<T>& x) const {
        std::vector<T> y(n_, T{0});

        #pragma omp parallel for if(n_ > 1000)
        for (size_t i = 0; i < n_; ++i) {
            T sum = T{0};
            size_t j_start = (i > lower_bw_) ? i - lower_bw_ : 0;
            size_t j_end = std::min(i + upper_bw_ + 1, n_);

            #pragma omp simd reduction(+:sum)
            for (size_t j = j_start; j < j_end; ++j) {
                sum += at(i, j) * x[j];
            }
            y[i] = sum;
        }

        return y;
    }
};

// =============================================================================
// Low-Rank Matrix: A ≈ U * V^T
// =============================================================================

template<typename T>
class low_rank_storage : public storage_policy<T> {
private:
    matrix<T> U_;  // n × r
    matrix<T> V_;  // m × r
    size_t rank_;
    mutable T temp_ = T{0};

public:
    low_rank_storage(size_t n, size_t m, size_t r)
        : U_(n, r), V_(m, r), rank_(r) {}

    low_rank_storage(const matrix<T>& U, const matrix<T>& V)
        : U_(U), V_(V), rank_(U.cols()) {}

    T& at(size_t i, size_t j) override {
        // Note: This is inefficient for element access
        // Low-rank matrices should be used for operations, not element access
        temp_ = T{0};
        for (size_t k = 0; k < rank_; ++k) {
            temp_ += U_(i, k) * V_(j, k);
        }
        return temp_;
    }

    const T& at(size_t i, size_t j) const override {
        temp_ = T{0};
        for (size_t k = 0; k < rank_; ++k) {
            temp_ += U_(i, k) * V_(j, k);
        }
        return temp_;
    }

    T* data() override { return nullptr; }
    const T* data() const override { return nullptr; }

    size_t memory_usage() const override {
        return (U_.rows() + V_.rows()) * rank_ * sizeof(T);
    }

    bool is_contiguous() const override { return false; }

    // Low-rank matrix-vector: O(nr + mr) instead of O(nm)
    std::vector<T> multiply_vector(const std::vector<T>& x) const {
        // Compute y = U * (V^T * x)

        // Step 1: z = V^T * x (r × 1)
        std::vector<T> z(rank_, T{0});
        #pragma omp parallel for
        for (size_t k = 0; k < rank_; ++k) {
            T sum = T{0};
            #pragma omp simd reduction(+:sum)
            for (size_t j = 0; j < V_.rows(); ++j) {
                sum += V_(j, k) * x[j];
            }
            z[k] = sum;
        }

        // Step 2: y = U * z (n × 1)
        std::vector<T> y(U_.rows(), T{0});
        #pragma omp parallel for
        for (size_t i = 0; i < U_.rows(); ++i) {
            T sum = T{0};
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < rank_; ++k) {
                sum += U_(i, k) * z[k];
            }
            y[i] = sum;
        }

        return y;
    }

    double compression_ratio() const {
        size_t full_size = U_.rows() * V_.rows() * sizeof(T);
        return double(full_size) / memory_usage();
    }
};

// =============================================================================
// Smart Matrix Factory - Automatic Structure Detection
// =============================================================================

template<typename T>
class smart_matrix_factory {
public:
    enum class structure {
        dense,
        diagonal,
        sparse,
        symmetric,
        banded,
        low_rank
    };

    struct detection_result {
        structure type;
        double confidence;
        size_t memory_saved;
        double speedup_estimate;
    };

    // Analyze matrix and recommend optimal structure
    static detection_result analyze(const dense_matrix<T>& M, T tolerance = T{1e-10}) {
        detection_result result;
        const size_t n = M.rows();
        const size_t m = M.cols();

        // Count non-zeros and check patterns
        size_t nnz = 0;
        bool is_diagonal = true;
        bool is_symmetric = (n == m);
        size_t max_bandwidth = 0;

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                if (std::abs(M(i, j)) > tolerance) {
                    nnz++;
                    if (i != j) is_diagonal = false;
                    max_bandwidth = std::max(max_bandwidth,
                                            static_cast<size_t>(std::abs(static_cast<long>(i) - static_cast<long>(j))));
                }

                if (is_symmetric && i < j) {
                    if (std::abs(M(i, j) - M(j, i)) > tolerance) {
                        is_symmetric = false;
                    }
                }
            }
        }

        double sparsity = 1.0 - double(nnz) / (n * m);
        size_t dense_memory = n * m * sizeof(T);

        // Decision logic
        if (is_diagonal) {
            result.type = structure::diagonal;
            result.confidence = 1.0;
            result.memory_saved = dense_memory - n * sizeof(T);
            result.speedup_estimate = double(n * n) / n;  // O(n²) → O(n)
        } else if (sparsity > 0.9) {
            result.type = structure::sparse;
            result.confidence = 0.9 + sparsity * 0.1;
            result.memory_saved = dense_memory - (nnz * sizeof(T) + (n + nnz) * sizeof(size_t));
            result.speedup_estimate = double(n * m) / nnz;
        } else if (is_symmetric) {
            result.type = structure::symmetric;
            result.confidence = 0.95;
            result.memory_saved = dense_memory - n * (n + 1) / 2 * sizeof(T);
            result.speedup_estimate = 2.0;  // Half the operations
        } else if (max_bandwidth < std::min(n, m) / 10) {
            result.type = structure::banded;
            result.confidence = 0.85;
            size_t band_memory = n * (2 * max_bandwidth + 1) * sizeof(T);
            result.memory_saved = dense_memory - band_memory;
            result.speedup_estimate = double(n * m) / (n * max_bandwidth);
        } else {
            result.type = structure::dense;
            result.confidence = 0.5;
            result.memory_saved = 0;
            result.speedup_estimate = 1.0;
        }

        return result;
    }

    // Create optimized matrix from dense matrix
    template<typename Structure = dense_tag>
    static auto create_optimized(const dense_matrix<T>& M) {
        auto analysis = analyze(M);

        switch (analysis.type) {
            case structure::diagonal: {
                diagonal_matrix<T> D(M.rows(), M.rows());
                for (size_t i = 0; i < M.rows(); ++i) {
                    D(i, i) = M(i, i);
                }
                return D;
            }
            // Add other conversions as needed
            default:
                return M;
        }
    }
};

} // namespace stepanov