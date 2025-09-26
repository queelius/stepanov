// quantum_state.hpp
// Quantum state vectors and operations for quantum computing simulation
// Provides efficient representation and manipulation of quantum states

#pragma once

#include <complex>
#include <vector>
#include <array>
#include <random>
#include <numeric>
#include <cmath>
#include <bit>
#include <functional>

namespace stepanov::quantum {

template<typename T = double>
using complex = std::complex<T>;

// Quantum state vector for n qubits
template<typename T = double>
class quantum_state {
private:
    size_t num_qubits_;
    std::vector<complex<T>> amplitudes_;  // 2^n complex amplitudes
    mutable std::mt19937 rng_{std::random_device{}()};

public:
    // Constructors
    explicit quantum_state(size_t num_qubits)
        : num_qubits_(num_qubits),
          amplitudes_(1ULL << num_qubits) {
        // Initialize to |00...0⟩
        amplitudes_[0] = complex<T>{1, 0};
    }

    // Initialize from computational basis state
    quantum_state(size_t num_qubits, size_t basis_state)
        : num_qubits_(num_qubits),
          amplitudes_(1ULL << num_qubits) {
        if (basis_state >= amplitudes_.size()) {
            throw std::out_of_range("Basis state out of range");
        }
        amplitudes_[basis_state] = complex<T>{1, 0};
    }

    // Initialize from amplitudes
    quantum_state(size_t num_qubits, std::vector<complex<T>> amps)
        : num_qubits_(num_qubits),
          amplitudes_(std::move(amps)) {
        if (amplitudes_.size() != (1ULL << num_qubits)) {
            throw std::invalid_argument("Invalid amplitude vector size");
        }
        normalize();
    }

    // Properties
    size_t num_qubits() const { return num_qubits_; }
    size_t dim() const { return amplitudes_.size(); }

    // Access amplitudes
    const complex<T>& operator[](size_t i) const { return amplitudes_[i]; }
    complex<T>& operator[](size_t i) { return amplitudes_[i]; }

    // Get probability of measuring basis state
    T probability(size_t basis_state) const {
        return std::norm(amplitudes_[basis_state]);
    }

    // Normalize the state
    void normalize() {
        T norm = T(0);
        for (const auto& amp : amplitudes_) {
            norm += std::norm(amp);
        }
        norm = std::sqrt(norm);

        if (norm > std::numeric_limits<T>::epsilon()) {
            for (auto& amp : amplitudes_) {
                amp /= norm;
            }
        }
    }

    // Inner product ⟨ψ|φ⟩
    complex<T> inner_product(const quantum_state& other) const {
        if (num_qubits_ != other.num_qubits_) {
            throw std::invalid_argument("States must have same number of qubits");
        }

        complex<T> result{0, 0};
        for (size_t i = 0; i < amplitudes_.size(); ++i) {
            result += std::conj(amplitudes_[i]) * other.amplitudes_[i];
        }
        return result;
    }

    // Fidelity |⟨ψ|φ⟩|²
    T fidelity(const quantum_state& other) const {
        return std::norm(inner_product(other));
    }

    // Measure a qubit (collapses the state)
    bool measure_qubit(size_t qubit) {
        if (qubit >= num_qubits_) {
            throw std::out_of_range("Qubit index out of range");
        }

        // Calculate probability of measuring |1⟩
        T prob_one = T(0);
        for (size_t i = 0; i < amplitudes_.size(); ++i) {
            if (i & (1ULL << qubit)) {
                prob_one += std::norm(amplitudes_[i]);
            }
        }

        // Random measurement
        std::uniform_real_distribution<T> dist(0, 1);
        bool result = dist(rng_) < prob_one;

        // Collapse the state
        T norm = std::sqrt(result ? prob_one : (T(1) - prob_one));
        for (size_t i = 0; i < amplitudes_.size(); ++i) {
            bool bit_set = i & (1ULL << qubit);
            if (bit_set != result) {
                amplitudes_[i] = complex<T>{0, 0};
            } else if (norm > std::numeric_limits<T>::epsilon()) {
                amplitudes_[i] /= norm;
            }
        }

        return result;
    }

    // Measure all qubits (returns basis state)
    size_t measure() {
        // Calculate cumulative probabilities
        std::vector<T> cumulative(amplitudes_.size());
        T total = T(0);
        for (size_t i = 0; i < amplitudes_.size(); ++i) {
            total += std::norm(amplitudes_[i]);
            cumulative[i] = total;
        }

        // Sample from distribution
        std::uniform_real_distribution<T> dist(0, total);
        T sample = dist(rng_);

        size_t result = 0;
        for (size_t i = 0; i < cumulative.size(); ++i) {
            if (sample <= cumulative[i]) {
                result = i;
                break;
            }
        }

        // Collapse to measured state
        std::fill(amplitudes_.begin(), amplitudes_.end(), complex<T>{0, 0});
        amplitudes_[result] = complex<T>{1, 0};

        return result;
    }

    // Tensor product |ψ⟩ ⊗ |φ⟩
    quantum_state tensor_product(const quantum_state& other) const {
        quantum_state result(num_qubits_ + other.num_qubits_);

        for (size_t i = 0; i < amplitudes_.size(); ++i) {
            for (size_t j = 0; j < other.amplitudes_.size(); ++j) {
                result.amplitudes_[i * other.amplitudes_.size() + j] =
                    amplitudes_[i] * other.amplitudes_[j];
            }
        }

        return result;
    }

    // Partial trace (trace out specified qubits)
    quantum_state partial_trace(const std::vector<size_t>& qubits_to_trace) const {
        size_t remaining = num_qubits_ - qubits_to_trace.size();
        quantum_state result(remaining);

        // Create mask for qubits to keep
        size_t keep_mask = 0;
        size_t trace_mask = 0;
        size_t bit_pos = 0;

        for (size_t i = 0; i < num_qubits_; ++i) {
            bool trace_out = std::find(qubits_to_trace.begin(),
                                      qubits_to_trace.end(), i) != qubits_to_trace.end();
            if (trace_out) {
                trace_mask |= (1ULL << i);
            } else {
                keep_mask |= (1ULL << bit_pos);
                bit_pos++;
            }
        }

        // Sum over traced qubits
        for (size_t i = 0; i < amplitudes_.size(); ++i) {
            // Extract kept qubit values
            size_t kept_state = 0;
            bit_pos = 0;
            for (size_t q = 0; q < num_qubits_; ++q) {
                if (!(trace_mask & (1ULL << q))) {
                    if (i & (1ULL << q)) {
                        kept_state |= (1ULL << bit_pos);
                    }
                    bit_pos++;
                }
            }

            result.amplitudes_[kept_state] += amplitudes_[i];
        }

        result.normalize();
        return result;
    }

    // Create Bell states
    static quantum_state bell_state(size_t which = 0) {
        quantum_state state(2);
        T inv_sqrt2 = T(1) / std::sqrt(T(2));

        switch (which % 4) {
            case 0:  // |Φ+⟩ = (|00⟩ + |11⟩)/√2
                state[0b00] = complex<T>{inv_sqrt2, 0};
                state[0b11] = complex<T>{inv_sqrt2, 0};
                break;
            case 1:  // |Φ-⟩ = (|00⟩ - |11⟩)/√2
                state[0b00] = complex<T>{inv_sqrt2, 0};
                state[0b11] = complex<T>{-inv_sqrt2, 0};
                break;
            case 2:  // |Ψ+⟩ = (|01⟩ + |10⟩)/√2
                state[0b01] = complex<T>{inv_sqrt2, 0};
                state[0b10] = complex<T>{inv_sqrt2, 0};
                break;
            case 3:  // |Ψ-⟩ = (|01⟩ - |10⟩)/√2
                state[0b01] = complex<T>{inv_sqrt2, 0};
                state[0b10] = complex<T>{-inv_sqrt2, 0};
                break;
        }
        return state;
    }

    // Create GHZ state
    static quantum_state ghz_state(size_t n) {
        quantum_state state(n);
        T inv_sqrt2 = T(1) / std::sqrt(T(2));
        state[0] = complex<T>{inv_sqrt2, 0};
        state[(1ULL << n) - 1] = complex<T>{inv_sqrt2, 0};
        return state;
    }

    // Create W state
    static quantum_state w_state(size_t n) {
        quantum_state state(n);
        T inv_sqrtn = T(1) / std::sqrt(T(n));

        for (size_t i = 0; i < n; ++i) {
            state[1ULL << i] = complex<T>{inv_sqrtn, 0};
        }
        return state;
    }

    // Equal superposition state
    static quantum_state superposition(size_t n) {
        quantum_state state(n);
        T amp = T(1) / std::sqrt(T(1ULL << n));

        for (auto& a : state.amplitudes_) {
            a = complex<T>{amp, 0};
        }
        return state;
    }

    // Von Neumann entropy
    T entropy() const {
        T ent = T(0);
        for (const auto& amp : amplitudes_) {
            T p = std::norm(amp);
            if (p > std::numeric_limits<T>::epsilon()) {
                ent -= p * std::log2(p);
            }
        }
        return ent;
    }

    // Entanglement entropy between qubit sets
    T entanglement_entropy(const std::vector<size_t>& subsystem_qubits) const {
        auto reduced = partial_trace(subsystem_qubits);
        return reduced.entropy();
    }

    // Check if state is separable (product state)
    bool is_separable() const {
        if (num_qubits_ == 1) return true;

        // Check if state can be written as |ψ⟩ = |a⟩ ⊗ |b⟩
        // This is a simplified check for 2 qubits
        if (num_qubits_ == 2) {
            // Check if density matrix rank is 1
            complex<T> det = amplitudes_[0] * amplitudes_[3] -
                           amplitudes_[1] * amplitudes_[2];
            return std::abs(det) < std::numeric_limits<T>::epsilon();
        }

        // For larger systems, this is computationally hard
        return false;  // Conservative answer
    }

    // Apply function to amplitudes
    template<typename F>
    void apply(F&& f) {
        for (size_t i = 0; i < amplitudes_.size(); ++i) {
            amplitudes_[i] = f(i, amplitudes_[i]);
        }
        normalize();
    }

    // Get density matrix representation
    std::vector<std::vector<complex<T>>> density_matrix() const {
        size_t n = amplitudes_.size();
        std::vector<std::vector<complex<T>>> rho(n, std::vector<complex<T>>(n));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                rho[i][j] = amplitudes_[i] * std::conj(amplitudes_[j]);
            }
        }

        return rho;
    }

    // Expectation value of observable (diagonal in computational basis)
    T expectation_value(const std::vector<T>& observable) const {
        if (observable.size() != amplitudes_.size()) {
            throw std::invalid_argument("Observable dimension mismatch");
        }

        T result = T(0);
        for (size_t i = 0; i < amplitudes_.size(); ++i) {
            result += observable[i] * std::norm(amplitudes_[i]);
        }
        return result;
    }
};

// Type aliases
using quantum_statef = quantum_state<float>;
using quantum_stated = quantum_state<double>;

} // namespace stepanov::quantum