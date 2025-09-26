// quantum_gates.hpp
// Quantum gates and circuit builder for quantum computing simulation
// Implements common gates and quantum algorithms

#pragma once

#include "quantum_state.hpp"
#include <numbers>
#include <functional>
#include <unordered_map>
#include <string>

namespace stepanov::quantum {

// Quantum gate base class
template<typename T = double>
class quantum_gate {
protected:
    size_t num_qubits_;
    std::vector<size_t> target_qubits_;
    std::vector<size_t> control_qubits_;

public:
    quantum_gate(size_t num_qubits,
                std::vector<size_t> targets,
                std::vector<size_t> controls = {})
        : num_qubits_(num_qubits),
          target_qubits_(std::move(targets)),
          control_qubits_(std::move(controls)) {}

    virtual ~quantum_gate() = default;

    // Apply gate to quantum state
    virtual void apply(quantum_state<T>& state) const = 0;

    // Get gate matrix representation
    virtual std::vector<std::vector<complex<T>>> matrix() const = 0;

    // Gate properties
    size_t num_qubits() const { return num_qubits_; }
    const std::vector<size_t>& targets() const { return target_qubits_; }
    const std::vector<size_t>& controls() const { return control_qubits_; }
};

// Single-qubit gate
template<typename T = double>
class single_qubit_gate : public quantum_gate<T> {
private:
    std::array<std::array<complex<T>, 2>, 2> matrix_;

public:
    single_qubit_gate(size_t target, const std::array<std::array<complex<T>, 2>, 2>& mat)
        : quantum_gate<T>(1, {target}), matrix_(mat) {}

    void apply(quantum_state<T>& state) const override {
        size_t target = this->target_qubits_[0];
        size_t n = state.num_qubits();
        size_t dim = state.dim();

        // Apply gate to each basis state
        std::vector<complex<T>> new_amps(dim);

        for (size_t i = 0; i < dim; ++i) {
            size_t bit = (i >> target) & 1;
            size_t i0 = i & ~(1ULL << target);  // Clear target bit
            size_t i1 = i0 | (1ULL << target);  // Set target bit

            if (bit == 0) {
                new_amps[i0] += matrix_[0][0] * state[i0] + matrix_[0][1] * state[i1];
            } else {
                new_amps[i1] += matrix_[1][0] * state[i0] + matrix_[1][1] * state[i1];
            }
        }

        for (size_t i = 0; i < dim; ++i) {
            state[i] = new_amps[i];
        }
    }

    std::vector<std::vector<complex<T>>> matrix() const override {
        return {{matrix_[0][0], matrix_[0][1]},
                {matrix_[1][0], matrix_[1][1]}};
    }
};

// Common single-qubit gates
template<typename T = double>
class pauli_x : public single_qubit_gate<T> {
public:
    explicit pauli_x(size_t target)
        : single_qubit_gate<T>(target, {{
            {complex<T>{0, 0}, complex<T>{1, 0}},
            {complex<T>{1, 0}, complex<T>{0, 0}}
          }}) {}
};

template<typename T = double>
class pauli_y : public single_qubit_gate<T> {
public:
    explicit pauli_y(size_t target)
        : single_qubit_gate<T>(target, {{
            {complex<T>{0, 0}, complex<T>{0, -1}},
            {complex<T>{0, 1}, complex<T>{0, 0}}
          }}) {}
};

template<typename T = double>
class pauli_z : public single_qubit_gate<T> {
public:
    explicit pauli_z(size_t target)
        : single_qubit_gate<T>(target, {{
            {complex<T>{1, 0}, complex<T>{0, 0}},
            {complex<T>{0, 0}, complex<T>{-1, 0}}
          }}) {}
};

template<typename T = double>
class hadamard : public single_qubit_gate<T> {
public:
    explicit hadamard(size_t target)
        : single_qubit_gate<T>(target, {{
            {complex<T>{1/std::sqrt(T(2)), 0}, complex<T>{1/std::sqrt(T(2)), 0}},
            {complex<T>{1/std::sqrt(T(2)), 0}, complex<T>{-1/std::sqrt(T(2)), 0}}
          }}) {}
};

template<typename T = double>
class phase_gate : public single_qubit_gate<T> {
public:
    phase_gate(size_t target, T phase)
        : single_qubit_gate<T>(target, {{
            {complex<T>{1, 0}, complex<T>{0, 0}},
            {complex<T>{0, 0}, std::exp(complex<T>{0, phase})}
          }}) {}
};

template<typename T = double>
class rotation_x : public single_qubit_gate<T> {
public:
    rotation_x(size_t target, T angle)
        : single_qubit_gate<T>(target, {{
            {complex<T>{std::cos(angle/2), 0}, complex<T>{0, -std::sin(angle/2)}},
            {complex<T>{0, -std::sin(angle/2)}, complex<T>{std::cos(angle/2), 0}}
          }}) {}
};

template<typename T = double>
class rotation_y : public single_qubit_gate<T> {
public:
    rotation_y(size_t target, T angle)
        : single_qubit_gate<T>(target, {{
            {complex<T>{std::cos(angle/2), 0}, complex<T>{-std::sin(angle/2), 0}},
            {complex<T>{std::sin(angle/2), 0}, complex<T>{std::cos(angle/2), 0}}
          }}) {}
};

template<typename T = double>
class rotation_z : public single_qubit_gate<T> {
public:
    rotation_z(size_t target, T angle)
        : single_qubit_gate<T>(target, {{
            {std::exp(complex<T>{0, -angle/2}), complex<T>{0, 0}},
            {complex<T>{0, 0}, std::exp(complex<T>{0, angle/2})}
          }}) {}
};

// Controlled gates
template<typename T = double>
class cnot : public quantum_gate<T> {
public:
    cnot(size_t control, size_t target)
        : quantum_gate<T>(2, {target}, {control}) {}

    void apply(quantum_state<T>& state) const override {
        size_t control = this->control_qubits_[0];
        size_t target = this->target_qubits_[0];
        size_t dim = state.dim();

        for (size_t i = 0; i < dim; ++i) {
            if (i & (1ULL << control)) {  // Control is |1⟩
                size_t flipped = i ^ (1ULL << target);  // Flip target
                if (flipped > i) {  // Avoid double-swapping
                    std::swap(state[i], state[flipped]);
                }
            }
        }
    }

    std::vector<std::vector<complex<T>>> matrix() const override {
        return {{complex<T>{1,0}, complex<T>{0,0}, complex<T>{0,0}, complex<T>{0,0}},
                {complex<T>{0,0}, complex<T>{1,0}, complex<T>{0,0}, complex<T>{0,0}},
                {complex<T>{0,0}, complex<T>{0,0}, complex<T>{0,0}, complex<T>{1,0}},
                {complex<T>{0,0}, complex<T>{0,0}, complex<T>{1,0}, complex<T>{0,0}}};
    }
};

template<typename T = double>
class controlled_z : public quantum_gate<T> {
public:
    controlled_z(size_t control, size_t target)
        : quantum_gate<T>(2, {target}, {control}) {}

    void apply(quantum_state<T>& state) const override {
        size_t control = this->control_qubits_[0];
        size_t target = this->target_qubits_[0];
        size_t dim = state.dim();

        for (size_t i = 0; i < dim; ++i) {
            if ((i & (1ULL << control)) && (i & (1ULL << target))) {
                state[i] *= complex<T>{-1, 0};
            }
        }
    }

    std::vector<std::vector<complex<T>>> matrix() const override {
        return {{complex<T>{1,0}, complex<T>{0,0}, complex<T>{0,0}, complex<T>{0,0}},
                {complex<T>{0,0}, complex<T>{1,0}, complex<T>{0,0}, complex<T>{0,0}},
                {complex<T>{0,0}, complex<T>{0,0}, complex<T>{1,0}, complex<T>{0,0}},
                {complex<T>{0,0}, complex<T>{0,0}, complex<T>{0,0}, complex<T>{-1,0}}};
    }
};

// Toffoli gate (CCNOT)
template<typename T = double>
class toffoli : public quantum_gate<T> {
public:
    toffoli(size_t control1, size_t control2, size_t target)
        : quantum_gate<T>(3, {target}, {control1, control2}) {}

    void apply(quantum_state<T>& state) const override {
        size_t c1 = this->control_qubits_[0];
        size_t c2 = this->control_qubits_[1];
        size_t target = this->target_qubits_[0];
        size_t dim = state.dim();

        for (size_t i = 0; i < dim; ++i) {
            if ((i & (1ULL << c1)) && (i & (1ULL << c2))) {
                size_t flipped = i ^ (1ULL << target);
                if (flipped > i) {
                    std::swap(state[i], state[flipped]);
                }
            }
        }
    }

    std::vector<std::vector<complex<T>>> matrix() const override {
        // 8x8 matrix for 3-qubit gate
        std::vector<std::vector<complex<T>>> m(8, std::vector<complex<T>>(8, {0, 0}));
        for (size_t i = 0; i < 8; ++i) {
            m[i][i] = complex<T>{1, 0};
        }
        std::swap(m[6][6], m[7][7]);  // Swap |110⟩ and |111⟩
        return m;
    }
};

// SWAP gate
template<typename T = double>
class swap : public quantum_gate<T> {
public:
    swap(size_t qubit1, size_t qubit2)
        : quantum_gate<T>(2, {qubit1, qubit2}) {}

    void apply(quantum_state<T>& state) const override {
        size_t q1 = this->target_qubits_[0];
        size_t q2 = this->target_qubits_[1];
        size_t dim = state.dim();

        for (size_t i = 0; i < dim; ++i) {
            bool b1 = (i >> q1) & 1;
            bool b2 = (i >> q2) & 1;

            if (b1 != b2) {  // Need to swap
                size_t swapped = i ^ (1ULL << q1) ^ (1ULL << q2);
                if (swapped > i) {  // Avoid double-swapping
                    std::swap(state[i], state[swapped]);
                }
            }
        }
    }

    std::vector<std::vector<complex<T>>> matrix() const override {
        return {{complex<T>{1,0}, complex<T>{0,0}, complex<T>{0,0}, complex<T>{0,0}},
                {complex<T>{0,0}, complex<T>{0,0}, complex<T>{1,0}, complex<T>{0,0}},
                {complex<T>{0,0}, complex<T>{1,0}, complex<T>{0,0}, complex<T>{0,0}},
                {complex<T>{0,0}, complex<T>{0,0}, complex<T>{0,0}, complex<T>{1,0}}};
    }
};

// Quantum circuit builder
template<typename T = double>
class quantum_circuit {
private:
    size_t num_qubits_;
    std::vector<std::unique_ptr<quantum_gate<T>>> gates_;

public:
    explicit quantum_circuit(size_t num_qubits) : num_qubits_(num_qubits) {}

    // Add gates
    quantum_circuit& x(size_t target) {
        gates_.push_back(std::make_unique<pauli_x<T>>(target));
        return *this;
    }

    quantum_circuit& y(size_t target) {
        gates_.push_back(std::make_unique<pauli_y<T>>(target));
        return *this;
    }

    quantum_circuit& z(size_t target) {
        gates_.push_back(std::make_unique<pauli_z<T>>(target));
        return *this;
    }

    quantum_circuit& h(size_t target) {
        gates_.push_back(std::make_unique<hadamard<T>>(target));
        return *this;
    }

    quantum_circuit& phase(size_t target, T angle) {
        gates_.push_back(std::make_unique<phase_gate<T>>(target, angle));
        return *this;
    }

    quantum_circuit& rx(size_t target, T angle) {
        gates_.push_back(std::make_unique<rotation_x<T>>(target, angle));
        return *this;
    }

    quantum_circuit& ry(size_t target, T angle) {
        gates_.push_back(std::make_unique<rotation_y<T>>(target, angle));
        return *this;
    }

    quantum_circuit& rz(size_t target, T angle) {
        gates_.push_back(std::make_unique<rotation_z<T>>(target, angle));
        return *this;
    }

    quantum_circuit& cx(size_t control, size_t target) {
        gates_.push_back(std::make_unique<cnot<T>>(control, target));
        return *this;
    }

    quantum_circuit& cz(size_t control, size_t target) {
        gates_.push_back(std::make_unique<controlled_z<T>>(control, target));
        return *this;
    }

    quantum_circuit& ccx(size_t c1, size_t c2, size_t target) {
        gates_.push_back(std::make_unique<toffoli<T>>(c1, c2, target));
        return *this;
    }

    quantum_circuit& swap(size_t q1, size_t q2) {
        gates_.push_back(std::make_unique<stepanov::quantum::swap<T>>(q1, q2));
        return *this;
    }

    // Apply circuit to state
    void apply(quantum_state<T>& state) const {
        for (const auto& gate : gates_) {
            gate->apply(state);
        }
    }

    // Execute circuit on initial state
    quantum_state<T> execute(const quantum_state<T>& initial) const {
        quantum_state<T> result = initial;
        apply(result);
        return result;
    }

    // Execute on |00...0⟩
    quantum_state<T> execute() const {
        quantum_state<T> state(num_qubits_);
        apply(state);
        return state;
    }

    // Circuit properties
    size_t num_qubits() const { return num_qubits_; }
    size_t depth() const { return gates_.size(); }

    // Clear circuit
    void clear() { gates_.clear(); }
};

// Quantum Fourier Transform
template<typename T = double>
quantum_circuit<T> qft_circuit(size_t n) {
    quantum_circuit<T> circuit(n);

    for (size_t i = 0; i < n; ++i) {
        circuit.h(i);

        for (size_t j = i + 1; j < n; ++j) {
            T angle = 2 * std::numbers::pi_v<T> / (1ULL << (j - i + 1));
            // Controlled phase rotation
            // This is simplified - proper implementation needs controlled-phase gate
            circuit.phase(j, angle);
        }
    }

    // Swap qubits to get correct order
    for (size_t i = 0; i < n/2; ++i) {
        circuit.swap(i, n - i - 1);
    }

    return circuit;
}

// Grover's algorithm oracle and diffuser
template<typename T = double>
class grover_oracle : public quantum_gate<T> {
private:
    std::function<bool(size_t)> marked_;

public:
    grover_oracle(size_t n, std::function<bool(size_t)> is_marked)
        : quantum_gate<T>(n, {}), marked_(std::move(is_marked)) {}

    void apply(quantum_state<T>& state) const override {
        for (size_t i = 0; i < state.dim(); ++i) {
            if (marked_(i)) {
                state[i] *= complex<T>{-1, 0};
            }
        }
    }

    std::vector<std::vector<complex<T>>> matrix() const override {
        size_t dim = 1ULL << this->num_qubits_;
        std::vector<std::vector<complex<T>>> m(dim, std::vector<complex<T>>(dim, {0, 0}));
        for (size_t i = 0; i < dim; ++i) {
            m[i][i] = marked_(i) ? complex<T>{-1, 0} : complex<T>{1, 0};
        }
        return m;
    }
};

template<typename T = double>
quantum_circuit<T> grover_circuit(size_t n, std::function<bool(size_t)> is_marked,
                                  size_t iterations = 0) {
    quantum_circuit<T> circuit(n);

    // Calculate optimal number of iterations if not specified
    if (iterations == 0) {
        size_t N = 1ULL << n;
        size_t M = 0;  // Count marked items
        for (size_t i = 0; i < N; ++i) {
            if (is_marked(i)) M++;
        }
        if (M > 0) {
            iterations = static_cast<size_t>(
                std::floor(std::numbers::pi_v<T> / 4 * std::sqrt(T(N) / T(M)))
            );
        }
    }

    // Initial superposition
    for (size_t i = 0; i < n; ++i) {
        circuit.h(i);
    }

    // Grover iteration
    for (size_t iter = 0; iter < iterations; ++iter) {
        // Oracle would be applied here
        // In practice, this needs custom gate implementation

        // Diffuser (inversion about average)
        for (size_t i = 0; i < n; ++i) {
            circuit.h(i);
        }
        for (size_t i = 0; i < n; ++i) {
            circuit.x(i);
        }

        // Multi-controlled Z gate (simplified)
        if (n > 1) {
            circuit.h(n-1);
            // Would need multi-controlled NOT here
            circuit.h(n-1);
        }

        for (size_t i = 0; i < n; ++i) {
            circuit.x(i);
        }
        for (size_t i = 0; i < n; ++i) {
            circuit.h(i);
        }
    }

    return circuit;
}

} // namespace stepanov::quantum