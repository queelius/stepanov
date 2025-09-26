#pragma once

#include <memory>
#include <vector>
#include <functional>
#include "concepts.hpp"
#include "matrix_expressions.hpp"

namespace stepanov::matrix_expr {

/**
 * Type-erased matrix following Sean Parent's runtime polymorphism pattern
 *
 * This allows us to:
 * - Store any matrix type (dense, sparse, diagonal, etc.) in the same container
 * - Switch implementations at runtime
 * - Compose operations without template explosion
 * - Pass matrices through non-template interfaces
 */
template<typename T>
    requires ring<T>
class any_matrix {
private:
    /**
     * Concept (interface) for matrix operations
     */
    struct concept_t {
        virtual ~concept_t() = default;
        virtual std::unique_ptr<concept_t> clone() const = 0;

        // Size queries
        virtual size_t rows() const = 0;
        virtual size_t cols() const = 0;

        // Element access
        virtual T get(size_t i, size_t j) const = 0;
        virtual void set(size_t i, size_t j, const T& val) = 0;

        // Structural information
        virtual bool is_zero(size_t i, size_t j) const = 0;
        virtual bool is_symmetric() const = 0;
        virtual bool is_diagonal() const = 0;
        virtual bool is_triangular() const = 0;
        virtual bool is_sparse() const = 0;

        // Operations
        virtual std::vector<T> multiply_vector(const std::vector<T>& v) const = 0;
        virtual std::unique_ptr<concept_t> multiply_matrix(const concept_t& other) const = 0;
        virtual std::unique_ptr<concept_t> add_matrix(const concept_t& other) const = 0;
        virtual std::unique_ptr<concept_t> transpose() const = 0;
        virtual std::unique_ptr<concept_t> scale(const T& scalar) const = 0;

        // Memory footprint
        virtual size_t memory_usage() const = 0;

        // Materialization
        virtual std::vector<std::vector<T>> to_dense() const = 0;
    };

    /**
     * Model (implementation) for specific matrix types
     */
    template<typename MatrixType>
    struct model_t final : concept_t {
        MatrixType matrix;

        template<typename... Args>
        explicit model_t(Args&&... args) : matrix(std::forward<Args>(args)...) {}

        std::unique_ptr<concept_t> clone() const override {
            return std::make_unique<model_t>(*this);
        }

        size_t rows() const override {
            return matrix.rows();
        }

        size_t cols() const override {
            return matrix.cols();
        }

        T get(size_t i, size_t j) const override {
            return matrix(i, j);
        }

        void set(size_t i, size_t j, const T& val) override {
            // Note: This might not work for all matrix types
            // Some are immutable by design
            if constexpr (requires { matrix(i, j) = val; }) {
                matrix(i, j) = val;
            } else {
                throw std::runtime_error("Matrix type does not support mutation");
            }
        }

        bool is_zero(size_t i, size_t j) const override {
            if constexpr (requires { matrix.is_structural_zero(i, j); }) {
                return matrix.is_structural_zero(i, j);
            } else {
                return matrix(i, j) == T(0);
            }
        }

        bool is_symmetric() const override {
            return std::is_base_of_v<symmetric_tag, typename MatrixType::property_tag>;
        }

        bool is_diagonal() const override {
            return std::is_base_of_v<diagonal_tag, typename MatrixType::property_tag>;
        }

        bool is_triangular() const override {
            return std::is_base_of_v<triangular_lower_tag, typename MatrixType::property_tag> ||
                   std::is_base_of_v<triangular_upper_tag, typename MatrixType::property_tag>;
        }

        bool is_sparse() const override {
            return std::is_base_of_v<sparse_tag, typename MatrixType::property_tag>;
        }

        std::vector<T> multiply_vector(const std::vector<T>& v) const override {
            size_t m = rows();
            size_t n = cols();

            if (v.size() != n) {
                throw std::invalid_argument("Vector dimension mismatch");
            }

            std::vector<T> result(m, T(0));

            // Optimize based on matrix structure
            if (is_diagonal()) {
                for (size_t i = 0; i < m; ++i) {
                    result[i] = matrix(i, i) * v[i];
                }
            } else {
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        if (!is_zero(i, j)) {
                            result[i] += matrix(i, j) * v[j];
                        }
                    }
                }
            }

            return result;
        }

        std::unique_ptr<concept_t> multiply_matrix(const concept_t& other) const override {
            // For now, materialize and multiply
            // Could be optimized with double dispatch
            auto A = to_dense();
            auto B = other.to_dense();

            size_t m = rows();
            size_t n = other.cols();
            size_t k = cols();

            std::vector<std::vector<T>> C(m, std::vector<T>(n, T(0)));

            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    for (size_t p = 0; p < k; ++p) {
                        C[i][j] += A[i][p] * B[p][j];
                    }
                }
            }

            return std::make_unique<model_t<decltype(C)>>(std::move(C));
        }

        std::unique_ptr<concept_t> add_matrix(const concept_t& other) const override {
            if (rows() != other.rows() || cols() != other.cols()) {
                throw std::invalid_argument("Matrix dimension mismatch for addition");
            }

            auto A = to_dense();
            auto B = other.to_dense();

            size_t m = rows();
            size_t n = cols();

            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    A[i][j] += B[i][j];
                }
            }

            return std::make_unique<model_t<decltype(A)>>(std::move(A));
        }

        std::unique_ptr<concept_t> transpose() const override {
            auto A = to_dense();
            size_t m = rows();
            size_t n = cols();

            std::vector<std::vector<T>> AT(n, std::vector<T>(m));

            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    AT[j][i] = A[i][j];
                }
            }

            return std::make_unique<model_t<decltype(AT)>>(std::move(AT));
        }

        std::unique_ptr<concept_t> scale(const T& scalar) const override {
            auto result = clone();
            auto A = result->to_dense();

            for (auto& row : A) {
                for (auto& elem : row) {
                    elem *= scalar;
                }
            }

            return std::make_unique<model_t<decltype(A)>>(std::move(A));
        }

        size_t memory_usage() const override {
            if constexpr (requires { matrix.memory_usage(); }) {
                return matrix.memory_usage();
            } else {
                // Estimate based on dimensions
                return rows() * cols() * sizeof(T);
            }
        }

        std::vector<std::vector<T>> to_dense() const override {
            size_t m = rows();
            size_t n = cols();
            std::vector<std::vector<T>> result(m, std::vector<T>(n, T(0)));

            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    if (!is_zero(i, j)) {
                        result[i][j] = matrix(i, j);
                    }
                }
            }

            return result;
        }
    };

    std::unique_ptr<concept_t> pimpl_;

public:
    // Constructors
    any_matrix() = default;

    template<typename MatrixType>
    any_matrix(MatrixType&& m)
        : pimpl_(std::make_unique<model_t<std::decay_t<MatrixType>>>(
            std::forward<MatrixType>(m))) {}

    // Copy and move
    any_matrix(const any_matrix& other)
        : pimpl_(other.pimpl_ ? other.pimpl_->clone() : nullptr) {}

    any_matrix(any_matrix&&) noexcept = default;

    any_matrix& operator=(const any_matrix& other) {
        if (this != &other) {
            pimpl_ = other.pimpl_ ? other.pimpl_->clone() : nullptr;
        }
        return *this;
    }

    any_matrix& operator=(any_matrix&&) noexcept = default;

    // Factory methods
    static any_matrix identity(size_t n) {
        return any_matrix(diagonal_matrix<T>::identity(n));
    }

    static any_matrix diagonal(const std::vector<T>& diag) {
        return any_matrix(diagonal_matrix<T>(diag));
    }

    static any_matrix from_dense(const std::vector<std::vector<T>>& data) {
        return any_matrix(data);
    }

    // Size queries
    size_t rows() const {
        return pimpl_ ? pimpl_->rows() : 0;
    }

    size_t cols() const {
        return pimpl_ ? pimpl_->cols() : 0;
    }

    // Element access
    T operator()(size_t i, size_t j) const {
        if (!pimpl_) throw std::runtime_error("Empty matrix");
        return pimpl_->get(i, j);
    }

    // Properties
    bool is_symmetric() const {
        return pimpl_ && pimpl_->is_symmetric();
    }

    bool is_diagonal() const {
        return pimpl_ && pimpl_->is_diagonal();
    }

    bool is_triangular() const {
        return pimpl_ && pimpl_->is_triangular();
    }

    bool is_sparse() const {
        return pimpl_ && pimpl_->is_sparse();
    }

    // Operations
    std::vector<T> operator*(const std::vector<T>& v) const {
        if (!pimpl_) throw std::runtime_error("Empty matrix");
        return pimpl_->multiply_vector(v);
    }

    any_matrix operator*(const any_matrix& other) const {
        if (!pimpl_ || !other.pimpl_) {
            throw std::runtime_error("Empty matrix");
        }
        return any_matrix(pimpl_->multiply_matrix(*other.pimpl_));
    }

    any_matrix operator+(const any_matrix& other) const {
        if (!pimpl_ || !other.pimpl_) {
            throw std::runtime_error("Empty matrix");
        }
        return any_matrix(pimpl_->add_matrix(*other.pimpl_));
    }

    any_matrix transpose() const {
        if (!pimpl_) throw std::runtime_error("Empty matrix");
        return any_matrix(pimpl_->transpose());
    }

    any_matrix operator*(const T& scalar) const {
        if (!pimpl_) throw std::runtime_error("Empty matrix");
        return any_matrix(pimpl_->scale(scalar));
    }

    friend any_matrix operator*(const T& scalar, const any_matrix& m) {
        return m * scalar;
    }

    // Memory info
    size_t memory_usage() const {
        return pimpl_ ? pimpl_->memory_usage() : 0;
    }

    // Materialization
    std::vector<std::vector<T>> to_dense() const {
        if (!pimpl_) return {};
        return pimpl_->to_dense();
    }

    // Runtime optimization hints
    any_matrix optimize_for(const std::string& operation) const {
        if (!pimpl_) return *this;

        if (operation == "multiply" && is_diagonal()) {
            // Already optimal
            return *this;
        } else if (operation == "solve" && is_symmetric()) {
            // Could convert to Cholesky factorization
            return *this;
        }
        // Add more optimization strategies

        return *this;
    }

    // Visitor pattern for advanced operations
    template<typename Visitor>
    auto visit(Visitor&& vis) const {
        // This would require more sophisticated double dispatch
        // For now, just use the type-erased interface
        return vis(*this);
    }
};

/**
 * Heterogeneous matrix container
 * Can store matrices of different types and operate on them uniformly
 */
template<typename T>
class matrix_collection {
private:
    std::vector<any_matrix<T>> matrices_;
    std::vector<std::string> names_;

public:
    void add(const std::string& name, any_matrix<T> matrix) {
        names_.push_back(name);
        matrices_.push_back(std::move(matrix));
    }

    template<typename MatrixType>
    void add(const std::string& name, MatrixType&& matrix) {
        names_.push_back(name);
        matrices_.emplace_back(std::forward<MatrixType>(matrix));
    }

    any_matrix<T>& operator[](const std::string& name) {
        auto it = std::find(names_.begin(), names_.end(), name);
        if (it == names_.end()) {
            throw std::out_of_range("Matrix not found: " + name);
        }
        return matrices_[std::distance(names_.begin(), it)];
    }

    // Compute total memory usage
    size_t total_memory() const {
        size_t total = 0;
        for (const auto& m : matrices_) {
            total += m.memory_usage();
        }
        return total;
    }

    // Find most efficient representation for an operation
    any_matrix<T> select_optimal_for(const std::string& operation) const {
        // Heuristic: prefer diagonal for multiplication, symmetric for solving, etc.
        for (const auto& m : matrices_) {
            if (operation == "multiply" && m.is_diagonal()) {
                return m;
            }
            if (operation == "solve" && m.is_triangular()) {
                return m;
            }
        }
        return matrices_.empty() ? any_matrix<T>() : matrices_[0];
    }

    // Batch operations
    std::vector<T> multiply_all_by_vector(const std::vector<T>& v) const {
        std::vector<T> result;
        for (const auto& m : matrices_) {
            auto partial = m * v;
            if (result.empty()) {
                result = partial;
            } else {
                for (size_t i = 0; i < result.size(); ++i) {
                    result[i] += partial[i];
                }
            }
        }
        return result;
    }
};

} // namespace stepanov::matrix_expr