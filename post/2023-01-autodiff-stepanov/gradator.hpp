// gradator.hpp - Pedagogical Automatic Differentiation Library
//
// A teaching-oriented implementation of reverse-mode automatic differentiation.
// Designed to demonstrate autograd fundamentals with clean, readable code.
//
// Key design principles:
// - Functional API: grad(f) returns a callable
// - Explicit context: graph passed explicitly, no global state
// - Value semantics: immutable nodes, functional construction
// - Type distinction: var<T> (differentiable) vs val<T> (constant)
// - Concept-based: any type satisfying Matrix concept works
//
// Author: [Your Name]
// License: MIT

#ifndef GRADATOR_HPP
#define GRADATOR_HPP

#include <elementa.hpp>

#include <any>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace gradator {

// Use elementa's Matrix concept
using elementa::Matrix;
using elementa::matrix;

// =============================================================================
// Forward Declarations
// =============================================================================

class graph;
template <typename T> class var;
template <typename T> class val;

// =============================================================================
// Node ID - Reference into the Graph
// =============================================================================

// A lightweight handle to a node in the computational graph.
// Nodes are stored in the graph's arena and referenced by index.
struct node_id {
    std::size_t index;

    bool operator==(const node_id&) const = default;
    bool operator!=(const node_id&) const = default;
};

// Invalid node ID sentinel
constexpr node_id invalid_node{std::numeric_limits<std::size_t>::max()};

// =============================================================================
// Backward Function Type
// =============================================================================

// Type-erased backward function that computes gradients.
// Takes: the graph, the output gradient (adjoint), and parent node IDs
// Returns: nothing (accumulates into parent gradients)
using backward_fn = std::function<void(graph&, const std::any& out_grad)>;

// =============================================================================
// Node - Single Node in the Computational Graph
// =============================================================================

// A node represents a value in the computational graph.
// Each node stores:
// - Its computed value (forward pass result)
// - Its gradient (accumulated during backward pass)
// - Parent node references (for backprop traversal)
// - A backward function (computes parent gradients)
struct node {
    std::any value;                  // The forward-computed value
    std::any gradient;               // Accumulated gradient (adjoint)
    std::vector<node_id> parents;    // Input nodes for this operation
    backward_fn backward;            // Gradient computation function
    bool requires_grad = true;       // Whether this node needs gradients

    node() = default;

    template <typename T>
    explicit node(T val, bool req_grad = true)
        : value(std::move(val)), requires_grad(req_grad) {}
};

// =============================================================================
// Graph - Computational Graph Context
// =============================================================================

// The computational graph holds all nodes created during forward computation.
// It provides:
// - Node creation (make_var)
// - Backward pass execution
// - Gradient retrieval
//
// Usage pattern:
//   graph g;
//   auto x = g.make_var(matrix<double>{{1, 2}});
//   auto y = sin(x);  // creates nodes in g
//   g.backward(y.id());
//   auto dx = g.gradient(x);
class graph {
private:
    std::vector<node> nodes_;

public:
    graph() = default;

    // Non-copyable (nodes reference into this graph)
    graph(const graph&) = delete;
    graph& operator=(const graph&) = delete;

    // Movable
    graph(graph&&) = default;
    graph& operator=(graph&&) = default;

    // -------------------------------------------------------------------------
    // Node Creation
    // -------------------------------------------------------------------------

    // Create a new variable node (input to the graph)
    template <typename T>
    auto make_var(T value) -> var<T> {
        node_id id{nodes_.size()};
        nodes_.emplace_back(std::move(value), true);
        return var<T>{id, this};
    }

    // Create an intermediate node (result of an operation)
    template <typename T>
    auto make_node(T value, std::vector<node_id> parents, backward_fn bwd) -> var<T> {
        node_id id{nodes_.size()};
        node n;
        n.value = std::move(value);
        n.parents = std::move(parents);
        n.backward = std::move(bwd);
        n.requires_grad = true;
        nodes_.push_back(std::move(n));
        return var<T>{id, this};
    }

    // -------------------------------------------------------------------------
    // Node Access
    // -------------------------------------------------------------------------

    [[nodiscard]] auto get_node(node_id id) -> node& {
        return nodes_.at(id.index);
    }

    [[nodiscard]] auto get_node(node_id id) const -> const node& {
        return nodes_.at(id.index);
    }

    template <typename T>
    [[nodiscard]] auto get_value(node_id id) const -> const T& {
        return std::any_cast<const T&>(nodes_.at(id.index).value);
    }

    template <typename T>
    [[nodiscard]] auto get_gradient(node_id id) const -> const T& {
        return std::any_cast<const T&>(nodes_.at(id.index).gradient);
    }

    [[nodiscard]] auto size() const -> std::size_t {
        return nodes_.size();
    }

    // -------------------------------------------------------------------------
    // Backward Pass (Reverse-Mode AD)
    // -------------------------------------------------------------------------

    // Execute backward pass starting from output node.
    // This implements reverse-mode automatic differentiation:
    // 1. Initialize output gradient to 1 (or identity matrix)
    // 2. Traverse nodes in reverse topological order
    // 3. For each node, call backward() to propagate gradients to parents
    template <typename T>
    void backward(node_id output_id) {
        auto& output_node = nodes_.at(output_id.index);

        // Initialize output gradient
        // For scalars: gradient is 1
        // For matrices: gradient is matrix of ones (∂L/∂L = I for each element)
        if constexpr (Matrix<T>) {
            const auto& val = std::any_cast<const T&>(output_node.value);
            output_node.gradient = elementa::ones<typename T::scalar_type>(val.rows(), val.cols());
        } else {
            output_node.gradient = T{1};
        }

        // Reverse pass: process nodes from output to inputs
        // Since we build the graph in topological order (children after parents),
        // reverse iteration gives us correct backprop order
        for (std::size_t i = output_id.index + 1; i-- > 0;) {
            auto& n = nodes_[i];
            if (n.backward && n.gradient.has_value()) {
                n.backward(*this, n.gradient);
            }
        }
    }

    // Convenience overload for var
    template <typename T>
    void backward(const var<T>& output) {
        backward<T>(output.id());
    }

    // -------------------------------------------------------------------------
    // Gradient Retrieval
    // -------------------------------------------------------------------------

    template <typename T>
    [[nodiscard]] auto gradient(const var<T>& v) const -> T {
        const auto& n = nodes_.at(v.id().index);
        if (!n.gradient.has_value()) {
            // No gradient computed - return zeros
            if constexpr (Matrix<T>) {
                const auto& val = std::any_cast<const T&>(n.value);
                return elementa::zeros<typename T::scalar_type>(val.rows(), val.cols());
            } else {
                return T{0};
            }
        }
        return std::any_cast<T>(n.gradient);
    }

    // -------------------------------------------------------------------------
    // Gradient Accumulation (used by backward functions)
    // -------------------------------------------------------------------------

    template <typename T>
    void accumulate_gradient(node_id id, const T& grad) {
        auto& n = nodes_.at(id.index);
        if (!n.requires_grad) return;

        if (!n.gradient.has_value()) {
            n.gradient = grad;
        } else {
            // Accumulate: existing_grad += grad
            auto& existing = std::any_cast<T&>(n.gradient);
            if constexpr (Matrix<T>) {
                existing = existing + grad;
            } else {
                existing = existing + grad;
            }
        }
    }

    // Clear all gradients (for reuse)
    void zero_grad() {
        for (auto& n : nodes_) {
            n.gradient.reset();
        }
    }
};

// =============================================================================
// var<T> - Differentiable Variable
// =============================================================================

// A var represents a value that participates in gradient computation.
// It holds a reference to a node in the graph.
//
// Key properties:
// - Immutable value (set at construction)
// - No implicit conversion to underlying type
// - Gradient accessible after backward pass
template <typename T>
class var {
private:
    node_id id_;
    graph* graph_;  // Non-owning pointer to parent graph

public:
    using value_type = T;

    var(node_id id, graph* g) : id_(id), graph_(g) {}

    // Access the underlying value
    [[nodiscard]] auto value() const -> const T& {
        return graph_->get_value<T>(id_);
    }

    // Access the node ID
    [[nodiscard]] auto id() const -> node_id {
        return id_;
    }

    // Access the parent graph
    [[nodiscard]] auto parent_graph() const -> graph* {
        return graph_;
    }

    // Get dimensions (for matrices)
    [[nodiscard]] auto rows() const -> std::size_t requires Matrix<T> {
        return value().rows();
    }

    [[nodiscard]] auto cols() const -> std::size_t requires Matrix<T> {
        return value().cols();
    }
};

// =============================================================================
// val<T> - Constant (Non-Differentiable)
// =============================================================================

// A val represents a constant that does not participate in gradient computation.
// Use this for data, hyperparameters, or any value that should not be differentiated.
template <typename T>
class val {
private:
    T value_;

public:
    using value_type = T;

    explicit val(T v) : value_(std::move(v)) {}

    [[nodiscard]] auto value() const -> const T& {
        return value_;
    }

    // Get dimensions (for matrices)
    [[nodiscard]] auto rows() const -> std::size_t requires Matrix<T> {
        return value_.rows();
    }

    [[nodiscard]] auto cols() const -> std::size_t requires Matrix<T> {
        return value_.cols();
    }
};

// =============================================================================
// Type Traits for Differentiable Types
// =============================================================================

template <typename T>
struct is_var : std::false_type {};

template <typename T>
struct is_var<var<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_var_v = is_var<T>::value;

template <typename T>
struct is_val : std::false_type {};

template <typename T>
struct is_val<val<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_val_v = is_val<T>::value;

// Extract underlying value type
template <typename T>
struct value_type { using type = T; };

template <typename T>
struct value_type<var<T>> { using type = T; };

template <typename T>
struct value_type<val<T>> { using type = T; };

template <typename T>
using value_type_t = typename value_type<T>::type;

// =============================================================================
// Differentiable Operations - Scalar/Element-wise
// =============================================================================

// Each operation creates a new node with:
// 1. Forward computation (value)
// 2. Parent references
// 3. Backward function implementing the chain rule

// -----------------------------------------------------------------------------
// Negation: y = -x
// Jacobian: ∂y/∂x = -1
// -----------------------------------------------------------------------------
template <typename T>
auto operator-(const var<T>& x) -> var<T> {
    auto* g = x.parent_graph();
    auto result = -x.value();

    return g->template make_node<T>(
        std::move(result),
        {x.id()},
        [x_id = x.id()](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const T&>(out_grad);
            g.accumulate_gradient(x_id, -grad);
        }
    );
}

// -----------------------------------------------------------------------------
// Addition: z = x + y
// Jacobians: ∂z/∂x = 1, ∂z/∂y = 1
// -----------------------------------------------------------------------------
template <typename T>
auto operator+(const var<T>& x, const var<T>& y) -> var<T> {
    auto* g = x.parent_graph();
    auto result = x.value() + y.value();

    return g->template make_node<T>(
        std::move(result),
        {x.id(), y.id()},
        [x_id = x.id(), y_id = y.id()](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const T&>(out_grad);
            g.accumulate_gradient(x_id, grad);
            g.accumulate_gradient(y_id, grad);
        }
    );
}

// var + val
template <typename T>
auto operator+(const var<T>& x, const val<T>& y) -> var<T> {
    auto* g = x.parent_graph();
    auto result = x.value() + y.value();

    return g->template make_node<T>(
        std::move(result),
        {x.id()},
        [x_id = x.id()](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const T&>(out_grad);
            g.accumulate_gradient(x_id, grad);
        }
    );
}

// val + var
template <typename T>
auto operator+(const val<T>& x, const var<T>& y) -> var<T> {
    return y + x;
}

// -----------------------------------------------------------------------------
// Subtraction: z = x - y
// Jacobians: ∂z/∂x = 1, ∂z/∂y = -1
// -----------------------------------------------------------------------------
template <typename T>
auto operator-(const var<T>& x, const var<T>& y) -> var<T> {
    auto* g = x.parent_graph();
    auto result = x.value() - y.value();

    return g->template make_node<T>(
        std::move(result),
        {x.id(), y.id()},
        [x_id = x.id(), y_id = y.id()](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const T&>(out_grad);
            g.accumulate_gradient(x_id, grad);
            g.accumulate_gradient(y_id, -grad);
        }
    );
}

template <typename T>
auto operator-(const var<T>& x, const val<T>& y) -> var<T> {
    auto* g = x.parent_graph();
    auto result = x.value() - y.value();

    return g->template make_node<T>(
        std::move(result),
        {x.id()},
        [x_id = x.id()](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const T&>(out_grad);
            g.accumulate_gradient(x_id, grad);
        }
    );
}

template <typename T>
auto operator-(const val<T>& x, const var<T>& y) -> var<T> {
    auto* g = y.parent_graph();
    auto result = x.value() - y.value();

    return g->template make_node<T>(
        std::move(result),
        {y.id()},
        [y_id = y.id()](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const T&>(out_grad);
            g.accumulate_gradient(y_id, -grad);
        }
    );
}

// -----------------------------------------------------------------------------
// Element-wise Multiplication (Hadamard): z = x ⊙ y
// Jacobians: ∂z/∂x_ij = y_ij, ∂z/∂y_ij = x_ij
// -----------------------------------------------------------------------------
template <Matrix M>
auto hadamard(const var<M>& x, const var<M>& y) -> var<M> {
    auto* g = x.parent_graph();
    auto result = elementa::hadamard(x.value(), y.value());

    return g->template make_node<M>(
        std::move(result),
        {x.id(), y.id()},
        [x_id = x.id(), y_id = y.id(), xv = x.value(), yv = y.value()]
        (graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            g.accumulate_gradient(x_id, elementa::hadamard(grad, yv));
            g.accumulate_gradient(y_id, elementa::hadamard(grad, xv));
        }
    );
}

// -----------------------------------------------------------------------------
// Scalar Multiplication: y = s * x (where s is a scalar constant)
// Jacobian: ∂y/∂x = s
// -----------------------------------------------------------------------------
template <Matrix M>
auto operator*(typename M::scalar_type s, const var<M>& x) -> var<M> {
    auto* g = x.parent_graph();
    auto result = s * x.value();

    return g->template make_node<M>(
        std::move(result),
        {x.id()},
        [x_id = x.id(), s](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            g.accumulate_gradient(x_id, s * grad);
        }
    );
}

template <Matrix M>
auto operator*(const var<M>& x, typename M::scalar_type s) -> var<M> {
    return s * x;
}

// -----------------------------------------------------------------------------
// Scalar Division: y = x / s
// Jacobian: ∂y/∂x = 1/s
// -----------------------------------------------------------------------------
template <Matrix M>
auto operator/(const var<M>& x, typename M::scalar_type s) -> var<M> {
    return x * (typename M::scalar_type{1} / s);
}

// =============================================================================
// Matrix Operations
// =============================================================================

// -----------------------------------------------------------------------------
// Matrix Multiplication: C = A * B
// Jacobians:
//   ∂L/∂A = ∂L/∂C * B^T
//   ∂L/∂B = A^T * ∂L/∂C
// -----------------------------------------------------------------------------
template <Matrix M>
auto matmul(const var<M>& A, const var<M>& B) -> var<M> {
    auto* g = A.parent_graph();
    auto result = elementa::matmul(A.value(), B.value());

    return g->template make_node<M>(
        std::move(result),
        {A.id(), B.id()},
        [a_id = A.id(), b_id = B.id(), Av = A.value(), Bv = B.value()]
        (graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            // ∂L/∂A = ∂L/∂C * B^T
            g.accumulate_gradient(a_id, elementa::matmul(grad, elementa::transpose(Bv)));
            // ∂L/∂B = A^T * ∂L/∂C
            g.accumulate_gradient(b_id, elementa::matmul(elementa::transpose(Av), grad));
        }
    );
}

template <Matrix M>
auto matmul(const var<M>& A, const val<M>& B) -> var<M> {
    auto* g = A.parent_graph();
    auto result = elementa::matmul(A.value(), B.value());

    return g->template make_node<M>(
        std::move(result),
        {A.id()},
        [a_id = A.id(), Bv = B.value()](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            g.accumulate_gradient(a_id, elementa::matmul(grad, elementa::transpose(Bv)));
        }
    );
}

template <Matrix M>
auto matmul(const val<M>& A, const var<M>& B) -> var<M> {
    auto* g = B.parent_graph();
    auto result = elementa::matmul(A.value(), B.value());

    return g->template make_node<M>(
        std::move(result),
        {B.id()},
        [b_id = B.id(), Av = A.value()](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            g.accumulate_gradient(b_id, elementa::matmul(elementa::transpose(Av), grad));
        }
    );
}

// -----------------------------------------------------------------------------
// Transpose: B = A^T
// Jacobian: ∂L/∂A = (∂L/∂B)^T
// -----------------------------------------------------------------------------
template <Matrix M>
auto transpose(const var<M>& A) -> var<M> {
    auto* g = A.parent_graph();
    auto result = elementa::transpose(A.value());

    return g->template make_node<M>(
        std::move(result),
        {A.id()},
        [a_id = A.id()](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            g.accumulate_gradient(a_id, elementa::transpose(grad));
        }
    );
}

// -----------------------------------------------------------------------------
// Trace: t = tr(A)
// Jacobian: ∂t/∂A = I (identity matrix)
// -----------------------------------------------------------------------------
template <Matrix M>
auto trace(const var<M>& A) -> var<typename M::scalar_type> {
    using S = typename M::scalar_type;
    auto* g = A.parent_graph();
    auto result = elementa::trace(A.value());
    auto n = A.rows();

    return g->template make_node<S>(
        result,
        {A.id()},
        [a_id = A.id(), n](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<S>(out_grad);
            // ∂L/∂A_ij = grad * δ_ij (gradient flows only to diagonal)
            auto grad_A = elementa::eye<S>(n) * grad;
            g.accumulate_gradient(a_id, grad_A);
        }
    );
}

// -----------------------------------------------------------------------------
// Sum: s = Σ_ij A_ij
// Jacobian: ∂s/∂A_ij = 1
// -----------------------------------------------------------------------------
template <Matrix M>
auto sum(const var<M>& A) -> var<typename M::scalar_type> {
    using S = typename M::scalar_type;
    auto* g = A.parent_graph();
    auto result = elementa::sum(A.value());
    auto rows = A.rows();
    auto cols = A.cols();

    return g->template make_node<S>(
        result,
        {A.id()},
        [a_id = A.id(), rows, cols](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<S>(out_grad);
            // Gradient is grad * ones
            g.accumulate_gradient(a_id, elementa::ones<S>(rows, cols) * grad);
        }
    );
}

// =============================================================================
// Transcendental Operations
// =============================================================================

// -----------------------------------------------------------------------------
// Exponential: y = exp(x)
// Jacobian: ∂y/∂x = exp(x) = y
// -----------------------------------------------------------------------------
template <Matrix M>
auto exp(const var<M>& x) -> var<M> {
    auto* g = x.parent_graph();
    auto result = elementa::exp(x.value());

    return g->template make_node<M>(
        result,
        {x.id()},
        [x_id = x.id(), result](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            // ∂L/∂x = ∂L/∂y ⊙ exp(x)
            g.accumulate_gradient(x_id, elementa::hadamard(grad, result));
        }
    );
}

// -----------------------------------------------------------------------------
// Natural Logarithm: y = log(x)
// Jacobian: ∂y/∂x = 1/x
// -----------------------------------------------------------------------------
template <Matrix M>
auto log(const var<M>& x) -> var<M> {
    auto* g = x.parent_graph();
    auto result = elementa::log(x.value());
    auto xv = x.value();

    return g->template make_node<M>(
        std::move(result),
        {x.id()},
        [x_id = x.id(), xv](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            // ∂L/∂x = ∂L/∂y ⊙ (1/x)
            g.accumulate_gradient(x_id, elementa::elem_div(grad, xv));
        }
    );
}

// -----------------------------------------------------------------------------
// Square Root: y = sqrt(x)
// Jacobian: ∂y/∂x = 1/(2*sqrt(x)) = 1/(2y)
// -----------------------------------------------------------------------------
template <Matrix M>
auto sqrt(const var<M>& x) -> var<M> {
    using S = typename M::scalar_type;
    auto* g = x.parent_graph();
    auto result = elementa::sqrt(x.value());

    return g->template make_node<M>(
        result,
        {x.id()},
        [x_id = x.id(), result](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            // ∂L/∂x = ∂L/∂y ⊙ (1/(2*sqrt(x)))
            auto two_sqrt_x = result * S{2};
            g.accumulate_gradient(x_id, elementa::elem_div(grad, two_sqrt_x));
        }
    );
}

// -----------------------------------------------------------------------------
// Power: y = x^p (element-wise)
// Jacobian: ∂y/∂x = p * x^(p-1)
// -----------------------------------------------------------------------------
template <Matrix M>
auto pow(const var<M>& x, typename M::scalar_type p) -> var<M> {
    using S = typename M::scalar_type;
    auto* g = x.parent_graph();
    auto xv = x.value();
    auto result = elementa::pow(xv, p);

    return g->template make_node<M>(
        std::move(result),
        {x.id()},
        [x_id = x.id(), xv, p](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            // ∂L/∂x = ∂L/∂y ⊙ p * x^(p-1)
            auto deriv = elementa::pow(xv, p - S{1}) * p;
            g.accumulate_gradient(x_id, elementa::hadamard(grad, deriv));
        }
    );
}

// =============================================================================
// Statistical/Linear Algebra Operations
// =============================================================================

// -----------------------------------------------------------------------------
// Determinant: d = det(A)
// Jacobian: ∂d/∂A = det(A) * (A^-1)^T = d * A^(-T)
// This is the classic matrix calculus result
// -----------------------------------------------------------------------------
template <Matrix M>
auto det(const var<M>& A) -> var<typename M::scalar_type> {
    using S = typename M::scalar_type;
    auto* g = A.parent_graph();
    auto Av = A.value();
    auto d = elementa::det(Av);

    return g->template make_node<S>(
        d,
        {A.id()},
        [a_id = A.id(), Av, d](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<S>(out_grad);
            // ∂L/∂A = grad * det(A) * A^(-T)
            auto Ainv_T = elementa::transpose(elementa::inverse(Av));
            g.accumulate_gradient(a_id, Ainv_T * (grad * d));
        }
    );
}

// -----------------------------------------------------------------------------
// Log Determinant: ld = log(|det(A)|)
// Jacobian: ∂ld/∂A = (A^-1)^T = A^(-T)
// More numerically stable than det for positive definite matrices
// -----------------------------------------------------------------------------
template <Matrix M>
auto logdet(const var<M>& A) -> var<typename M::scalar_type> {
    using S = typename M::scalar_type;
    auto* g = A.parent_graph();
    auto Av = A.value();
    auto [sign, ld] = elementa::logdet(Av);

    return g->template make_node<S>(
        ld,
        {A.id()},
        [a_id = A.id(), Av](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<S>(out_grad);
            // ∂L/∂A = grad * A^(-T)
            auto Ainv_T = elementa::transpose(elementa::inverse(Av));
            g.accumulate_gradient(a_id, Ainv_T * grad);
        }
    );
}

// -----------------------------------------------------------------------------
// Matrix Inverse: B = A^(-1)
// Jacobian: ∂L/∂A = -A^(-T) * ∂L/∂B * A^(-T)
// Derived from differentiating A * A^(-1) = I
// -----------------------------------------------------------------------------
template <Matrix M>
auto inverse(const var<M>& A) -> var<M> {
    auto* g = A.parent_graph();
    auto Ainv = elementa::inverse(A.value());

    return g->template make_node<M>(
        Ainv,
        {A.id()},
        [a_id = A.id(), Ainv](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            // ∂L/∂A = -A^(-T) * ∂L/∂B * A^(-T)
            auto Ainv_T = elementa::transpose(Ainv);
            auto result = elementa::matmul(elementa::matmul(-Ainv_T, grad), Ainv_T);
            g.accumulate_gradient(a_id, result);
        }
    );
}

// -----------------------------------------------------------------------------
// Linear Solve: X = solve(A, B) means A*X = B, so X = A^(-1)*B
// Jacobians:
//   ∂L/∂A = -A^(-T) * ∂L/∂X * X^T
//   ∂L/∂B = A^(-T) * ∂L/∂X
// -----------------------------------------------------------------------------
template <Matrix M>
auto solve(const var<M>& A, const var<M>& B) -> var<M> {
    auto* g = A.parent_graph();
    auto Av = A.value();
    auto X = elementa::solve(Av, B.value());

    return g->template make_node<M>(
        X,
        {A.id(), B.id()},
        [a_id = A.id(), b_id = B.id(), Av, X](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            // Compute A^(-T) * grad once (used for both)
            auto Ainv_T_grad = elementa::solve(elementa::transpose(Av), grad);

            // ∂L/∂A = -(A^(-T) * grad) * X^T
            g.accumulate_gradient(a_id, -elementa::matmul(Ainv_T_grad, elementa::transpose(X)));
            // ∂L/∂B = A^(-T) * grad
            g.accumulate_gradient(b_id, Ainv_T_grad);
        }
    );
}

template <Matrix M>
auto solve(const var<M>& A, const val<M>& B) -> var<M> {
    auto* g = A.parent_graph();
    auto Av = A.value();
    auto X = elementa::solve(Av, B.value());

    return g->template make_node<M>(
        X,
        {A.id()},
        [a_id = A.id(), Av, X](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            auto Ainv_T_grad = elementa::solve(elementa::transpose(Av), grad);
            g.accumulate_gradient(a_id, -elementa::matmul(Ainv_T_grad, elementa::transpose(X)));
        }
    );
}

template <Matrix M>
auto solve(const val<M>& A, const var<M>& B) -> var<M> {
    auto* g = B.parent_graph();
    auto Av = A.value();
    auto X = elementa::solve(Av, B.value());

    return g->template make_node<M>(
        std::move(X),
        {B.id()},
        [b_id = B.id(), Av](graph& g, const std::any& out_grad) {
            auto grad = std::any_cast<const M&>(out_grad);
            g.accumulate_gradient(b_id, elementa::solve(elementa::transpose(Av), grad));
        }
    );
}

// =============================================================================
// grad() - The Core Differentiation API
// =============================================================================

// grad(f) returns a new function that computes the gradient of f.
// For a function f: T -> S, grad(f) returns a function T -> T
// that gives ∂f/∂input for each element of the input.
//
// The returned function:
// 1. Creates a fresh graph
// 2. Converts input to var
// 3. Evaluates f to build the computational graph
// 4. Runs backward pass
// 5. Returns the gradient
//
// Usage:
//   auto f = [](auto x) { return sum(x * x); };
//   auto df = grad(f);
//   matrix<double> gradient = df(input);  // = 2 * input

template <typename F>
auto grad(F&& f) {
    return [f = std::forward<F>(f)]<typename T>(const T& input) {
        graph g;
        auto x = g.make_var(input);
        auto y = f(x);

        using output_type = typename decltype(y)::value_type;
        g.backward<output_type>(y);

        return g.gradient(x);
    };
}

// grad with explicit graph (for when you need access to intermediates)
template <typename F>
auto grad_with_graph(F&& f) {
    return [f = std::forward<F>(f)]<typename T>(graph& g, const var<T>& x) {
        auto y = f(x);

        using output_type = typename decltype(y)::value_type;
        g.backward<output_type>(y);

        return g.gradient(x);
    };
}

// =============================================================================
// hessian() - Second-Order Derivatives
// =============================================================================

// hessian(f) computes the Hessian matrix (matrix of second derivatives).
// For f: R^n -> R, the Hessian H_ij = ∂²f/∂x_i∂x_j
//
// Implementation: Compute grad(grad(f)) by running AD twice.
// This works because our implementation supports higher-order derivatives.
//
// Note: For pedagogical clarity, we compute Hessian column-by-column
// using the gradient of the gradient. This is O(n) backward passes.

template <typename F>
auto hessian(F&& f) {
    return [f = std::forward<F>(f)]<Matrix M>(const M& input) {
        using T = typename M::scalar_type;
        auto n = input.rows() * input.cols();

        // Result: n x n Hessian matrix
        matrix<T> H(n, n);

        // Compute gradient function
        auto grad_f = grad(f);

        // For each input element, compute how the gradient changes
        // This gives us one column of the Hessian at a time
        T eps = std::sqrt(std::numeric_limits<T>::epsilon());

        for (std::size_t j = 0; j < n; ++j) {
            // Perturb input in direction j
            M input_plus = input;
            M input_minus = input;
            input_plus.data()[j] += eps;
            input_minus.data()[j] -= eps;

            // Compute gradients at perturbed points
            auto grad_plus = grad_f(input_plus);
            auto grad_minus = grad_f(input_minus);

            // Finite difference approximation of Hessian column j
            for (std::size_t i = 0; i < n; ++i) {
                H(i, j) = (grad_plus.data()[i] - grad_minus.data()[i]) / (T{2} * eps);
            }
        }

        return H;
    };
}

// =============================================================================
// Utility Functions
// =============================================================================

// Finite difference gradient (for testing)
template <typename F, Matrix M>
auto finite_diff_gradient(F&& f, const M& x, typename M::scalar_type eps = 1e-7) -> M {
    using T = typename M::scalar_type;
    M grad(x.rows(), x.cols());

    for (std::size_t i = 0; i < x.rows(); ++i) {
        for (std::size_t j = 0; j < x.cols(); ++j) {
            M x_plus = x;
            M x_minus = x;
            x_plus(i, j) += eps;
            x_minus(i, j) -= eps;

            auto f_plus = f(x_plus);
            auto f_minus = f(x_minus);
            grad(i, j) = (f_plus - f_minus) / (T{2} * eps);
        }
    }

    return grad;
}

// Numerical Hessian (for testing)
template <typename F, Matrix M>
auto finite_diff_hessian(F&& f, const M& x, typename M::scalar_type eps = 1e-5) -> matrix<typename M::scalar_type> {
    using T = typename M::scalar_type;
    auto n = x.rows() * x.cols();
    matrix<T> H(n, n);

    auto grad_f = grad(f);

    for (std::size_t j = 0; j < n; ++j) {
        M x_plus = x;
        M x_minus = x;
        x_plus.data()[j] += eps;
        x_minus.data()[j] -= eps;

        auto g_plus = grad_f(x_plus);
        auto g_minus = grad_f(x_minus);

        for (std::size_t i = 0; i < n; ++i) {
            H(i, j) = (g_plus.data()[i] - g_minus.data()[i]) / (T{2} * eps);
        }
    }

    return H;
}

}  // namespace gradator

#endif  // GRADATOR_HPP
