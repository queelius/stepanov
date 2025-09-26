#pragma once

#include <map>
#include <vector>
#include <algorithm>
#include <utility>
#include <functional>
#include "concepts.hpp"
#include "math.hpp"

namespace stepanov {

/**
 * Sparse Matrix Implementation
 *
 * Optimized for matrices where most entries are the additive identity (zero).
 * Works over any ring, not just numerical types.
 *
 * Design choices:
 * - Coordinate (COO) format for flexibility
 * - Compressed Sparse Row (CSR) conversion for efficient operations
 * - Lazy evaluation for chained operations
 * - Support for ring operations (no division required)
 *
 * Trade-offs vs dense matrix:
 * - Space: O(nnz) vs O(mn)
 * - Addition: O(nnz) vs O(mn)
 * - Multiplication: O(nnz1 * nnz2/n) vs O(mn²)
 * - Element access: O(log nnz) vs O(1)
 */

template<ring T>
class sparse_matrix {
public:
    using index_t = std::pair<size_t, size_t>;
    using entry_t = std::pair<index_t, T>;

private:
    std::map<index_t, T> entries;  // (row, col) -> value
    size_t rows_, cols_;
    T zero_;  // Additive identity

    // Remove zero entries
    void cleanup() {
        auto it = entries.begin();
        while (it != entries.end()) {
            if (it->second == zero_) {
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
    }

public:
    // Constructors
    sparse_matrix(size_t r, size_t c, T zero = T(0))
        : rows_(r), cols_(c), zero_(zero) {}

    // Construct from dense matrix
    template<typename Container>
        requires (!std::is_integral_v<Container>)  // Prevent matching with size_t
    sparse_matrix(const Container& dense, T zero = T(0))
        : rows_(dense.size()), cols_(dense[0].size()), zero_(zero) {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                if (dense[i][j] != zero_) {
                    entries[{i, j}] = dense[i][j];
                }
            }
        }
    }

    // Factory methods for special matrices
    static sparse_matrix identity(size_t n, T one = T(1), T zero = T(0)) {
        sparse_matrix result(n, n, zero);
        for (size_t i = 0; i < n; ++i) {
            result.set(i, i, one);
        }
        return result;
    }

    static sparse_matrix diagonal(const std::vector<T>& diag, T zero = T(0)) {
        size_t n = diag.size();
        sparse_matrix result(n, n, zero);
        for (size_t i = 0; i < n; ++i) {
            if (diag[i] != zero) {
                result.set(i, i, diag[i]);
            }
        }
        return result;
    }

    // Element access
    T get(size_t row, size_t col) const {
        auto it = entries.find({row, col});
        return (it != entries.end()) ? it->second : zero_;
    }

    void set(size_t row, size_t col, const T& value) {
        if (value == zero_) {
            entries.erase({row, col});
        } else {
            entries[{row, col}] = value;
        }
    }

    // Dimensions
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t nnz() const { return entries.size(); }  // Number of non-zeros
    double sparsity() const {
        return 1.0 - static_cast<double>(nnz()) / (rows_ * cols_);
    }

    // Matrix addition
    sparse_matrix operator+(const sparse_matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }

        sparse_matrix result(rows_, cols_, zero_);

        // Add entries from this matrix
        for (const auto& [idx, val] : entries) {
            result.entries[idx] = val;
        }

        // Add entries from other matrix
        for (const auto& [idx, val] : other.entries) {
            auto it = result.entries.find(idx);
            if (it != result.entries.end()) {
                it->second = it->second + val;
            } else {
                result.entries[idx] = val;
            }
        }

        result.cleanup();
        return result;
    }

    // Matrix subtraction
    sparse_matrix operator-(const sparse_matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }

        sparse_matrix result(rows_, cols_, zero_);

        // Add entries from this matrix
        for (const auto& [idx, val] : entries) {
            result.entries[idx] = val;
        }

        // Subtract entries from other matrix
        for (const auto& [idx, val] : other.entries) {
            auto it = result.entries.find(idx);
            if (it != result.entries.end()) {
                it->second = it->second - val;
            } else {
                result.entries[idx] = -val;
            }
        }

        result.cleanup();
        return result;
    }

    // Scalar multiplication
    sparse_matrix operator*(const T& scalar) const {
        sparse_matrix result(rows_, cols_, zero_);

        for (const auto& [idx, val] : entries) {
            T new_val = val * scalar;
            if (new_val != zero_) {
                result.entries[idx] = new_val;
            }
        }

        return result;
    }

    // Matrix multiplication (optimized for sparse matrices)
    sparse_matrix operator*(const sparse_matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        }

        sparse_matrix result(rows_, other.cols_, zero_);

        // Group other matrix entries by row for efficiency
        std::map<size_t, std::vector<std::pair<size_t, T>>> other_by_row;
        for (const auto& [idx, val] : other.entries) {
            other_by_row[idx.first].push_back({idx.second, val});
        }

        // Compute product
        for (const auto& [idx1, val1] : entries) {
            auto [i, k] = idx1;
            auto it = other_by_row.find(k);
            if (it != other_by_row.end()) {
                for (const auto& [j, val2] : it->second) {
                    auto& result_val = result.entries[{i, j}];
                    result_val = result_val + val1 * val2;
                }
            }
        }

        result.cleanup();
        return result;
    }

    // Matrix-vector multiplication
    std::vector<T> operator*(const std::vector<T>& vec) const {
        if (cols_ != vec.size()) {
            throw std::invalid_argument("Matrix-vector dimensions incompatible");
        }

        std::vector<T> result(rows_, zero_);

        for (const auto& [idx, val] : entries) {
            auto [i, j] = idx;
            result[i] = result[i] + val * vec[j];
        }

        return result;
    }

    // Transpose
    sparse_matrix transpose() const {
        sparse_matrix result(cols_, rows_, zero_);

        for (const auto& [idx, val] : entries) {
            auto [i, j] = idx;
            result.entries[{j, i}] = val;
        }

        return result;
    }

    // Power (for square matrices)
    sparse_matrix pow(size_t n) const {
        if (rows_ != cols_) {
            throw std::invalid_argument("Matrix must be square for power operation");
        }

        if (n == 0) {
            return identity(rows_, T(1), zero_);
        }

        sparse_matrix result = *this;
        sparse_matrix base = *this;
        n--;

        while (n > 0) {
            if (n % 2 == 1) {
                result = result * base;
            }
            base = base * base;
            n /= 2;
        }

        return result;
    }

    // Convert to Compressed Sparse Row format for efficient row operations
    struct csr_format {
        std::vector<T> values;
        std::vector<size_t> col_indices;
        std::vector<size_t> row_pointers;
    };

    csr_format to_csr() const {
        csr_format csr;
        csr.row_pointers.resize(rows_ + 1, 0);

        // Count entries per row
        std::vector<size_t> row_counts(rows_, 0);
        for (const auto& [idx, val] : entries) {
            row_counts[idx.first]++;
        }

        // Compute row pointers
        for (size_t i = 0; i < rows_; ++i) {
            csr.row_pointers[i + 1] = csr.row_pointers[i] + row_counts[i];
        }

        // Fill values and column indices
        csr.values.resize(entries.size());
        csr.col_indices.resize(entries.size());

        std::vector<size_t> current_pos = csr.row_pointers;
        for (const auto& [idx, val] : entries) {
            auto [row, col] = idx;
            size_t pos = current_pos[row]++;
            csr.values[pos] = val;
            csr.col_indices[pos] = col;
        }

        return csr;
    }

    // Apply function to non-zero entries
    template<typename Func>
    void apply(Func f) {
        for (auto& [idx, val] : entries) {
            val = f(val);
        }
        cleanup();
    }

    // Element-wise operations
    sparse_matrix hadamard_product(const sparse_matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        sparse_matrix result(rows_, cols_, zero_);

        // Only non-zero in both matrices
        for (const auto& [idx, val1] : entries) {
            auto it = other.entries.find(idx);
            if (it != other.entries.end()) {
                T prod = val1 * it->second;
                if (prod != zero_) {
                    result.entries[idx] = prod;
                }
            }
        }

        return result;
    }

    // Kronecker product
    sparse_matrix kronecker(const sparse_matrix& other) const {
        size_t new_rows = rows_ * other.rows_;
        size_t new_cols = cols_ * other.cols_;
        sparse_matrix result(new_rows, new_cols, zero_);

        for (const auto& [idx1, val1] : entries) {
            auto [i1, j1] = idx1;
            for (const auto& [idx2, val2] : other.entries) {
                auto [i2, j2] = idx2;
                size_t i = i1 * other.rows_ + i2;
                size_t j = j1 * other.cols_ + j2;
                result.entries[{i, j}] = val1 * val2;
            }
        }

        return result;
    }

    // Row and column operations
    std::vector<T> get_row(size_t row) const {
        std::vector<T> result(cols_, zero_);
        for (const auto& [idx, val] : entries) {
            if (idx.first == row) {
                result[idx.second] = val;
            }
        }
        return result;
    }

    std::vector<T> get_col(size_t col) const {
        std::vector<T> result(rows_, zero_);
        for (const auto& [idx, val] : entries) {
            if (idx.second == col) {
                result[idx.first] = val;
            }
        }
        return result;
    }

    void set_row(size_t row, const std::vector<T>& values) {
        if (values.size() != cols_) {
            throw std::invalid_argument("Invalid row size");
        }

        // Remove old row
        auto it = entries.begin();
        while (it != entries.end()) {
            if (it->first.first == row) {
                it = entries.erase(it);
            } else {
                ++it;
            }
        }

        // Set new row
        for (size_t j = 0; j < cols_; ++j) {
            if (values[j] != zero_) {
                entries[{row, j}] = values[j];
            }
        }
    }

    // Iterators for non-zero entries
    auto begin() const { return entries.begin(); }
    auto end() const { return entries.end(); }

    // Submatrix extraction
    sparse_matrix submatrix(size_t row_start, size_t row_end,
                           size_t col_start, size_t col_end) const {
        size_t sub_rows = row_end - row_start;
        size_t sub_cols = col_end - col_start;
        sparse_matrix result(sub_rows, sub_cols, zero_);

        for (const auto& [idx, val] : entries) {
            auto [i, j] = idx;
            if (i >= row_start && i < row_end && j >= col_start && j < col_end) {
                result.entries[{i - row_start, j - col_start}] = val;
            }
        }

        return result;
    }

    // Block matrix construction
    static sparse_matrix block_diagonal(const std::vector<sparse_matrix>& blocks) {
        size_t total_rows = 0, total_cols = 0;
        for (const auto& block : blocks) {
            total_rows += block.rows_;
            total_cols += block.cols_;
        }

        sparse_matrix result(total_rows, total_cols, blocks[0].zero_);
        size_t row_offset = 0, col_offset = 0;

        for (const auto& block : blocks) {
            for (const auto& [idx, val] : block.entries) {
                auto [i, j] = idx;
                result.entries[{i + row_offset, j + col_offset}] = val;
            }
            row_offset += block.rows_;
            col_offset += block.cols_;
        }

        return result;
    }
};

// Type aliases for common sparse matrices
template<typename T>
using sparse_vector = sparse_matrix<T>;  // Column vector

// Utility functions for sparse linear algebra
template<ring T>
T sparse_inner_product(const sparse_matrix<T>& A,
                       const std::vector<T>& x,
                       const std::vector<T>& y) {
    // Compute x^T * A * y efficiently
    T result = T(0);
    for (const auto& [idx, val] : A) {
        auto [i, j] = idx;
        result = result + x[i] * val * y[j];
    }
    return result;
}

} // namespace stepanov