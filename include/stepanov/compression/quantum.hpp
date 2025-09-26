// Quantum and Information-Theoretic Compression: The Ultimate Limits
// "Information is physical" - Rolf Landauer
#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>
#include <optional>
#include <queue>
#include "../concepts.hpp"
// #include "../quantum/quantum_state.hpp"  // Use local implementation
// #include "../quantum/quantum_gates.hpp"

namespace stepanov::compression::quantum {

using namespace std::complex_literals;

// Quantum state for compression
template<typename T = double>
class quantum_state {
public:
    using value_type = T;
    using complex_type = std::complex<T>;
    using amplitude_vector = std::vector<complex_type>;

private:
    amplitude_vector amplitudes;
    size_t num_qubits;

public:
    quantum_state(size_t n_qubits)
        : num_qubits(n_qubits)
        , amplitudes(1 << n_qubits, complex_type(0)) {
        amplitudes[0] = complex_type(1);  // |00...0⟩ state
    }

    quantum_state(const amplitude_vector& amps)
        : amplitudes(amps)
        , num_qubits(static_cast<size_t>(std::log2(amps.size()))) {}

    // Get probability of measuring basis state
    T probability(size_t basis_state) const {
        if (basis_state >= amplitudes.size()) return 0;
        return std::norm(amplitudes[basis_state]);
    }

    // Von Neumann entropy
    T entropy() const {
        T ent = 0;
        for (const auto& amp : amplitudes) {
            T prob = std::norm(amp);
            if (prob > 1e-10) {
                ent -= prob * std::log2(prob);
            }
        }
        return ent;
    }

    // Apply quantum gate
    void apply_gate(const std::vector<std::vector<complex_type>>& gate,
                   const std::vector<size_t>& target_qubits) {
        // Simplified: apply single-qubit gate
        if (target_qubits.size() == 1) {
            size_t target = target_qubits[0];
            amplitude_vector new_amps = amplitudes;

            for (size_t state = 0; state < amplitudes.size(); ++state) {
                size_t bit0_state = state & ~(1 << target);
                size_t bit1_state = state | (1 << target);

                if ((state >> target) & 1) {
                    new_amps[state] = gate[1][0] * amplitudes[bit0_state] +
                                     gate[1][1] * amplitudes[bit1_state];
                } else {
                    new_amps[state] = gate[0][0] * amplitudes[bit0_state] +
                                     gate[0][1] * amplitudes[bit1_state];
                }
            }
            amplitudes = new_amps;
        }
    }

    // Partial trace to get reduced density matrix
    quantum_state partial_trace(const std::vector<size_t>& traced_qubits) const {
        size_t remaining = num_qubits - traced_qubits.size();
        amplitude_vector reduced(1 << remaining, complex_type(0));

        // Sum over traced qubits
        for (size_t state = 0; state < amplitudes.size(); ++state) {
            size_t reduced_state = 0;
            size_t bit_pos = 0;

            for (size_t q = 0; q < num_qubits; ++q) {
                if (std::find(traced_qubits.begin(), traced_qubits.end(), q) ==
                    traced_qubits.end()) {
                    if ((state >> q) & 1) {
                        reduced_state |= (1 << bit_pos);
                    }
                    bit_pos++;
                }
            }

            reduced[reduced_state] += std::norm(amplitudes[state]);
        }

        // Normalize
        T norm = 0;
        for (const auto& amp : reduced) {
            norm += std::real(amp);
        }
        if (norm > 0) {
            for (auto& amp : reduced) {
                amp = std::sqrt(std::real(amp) / norm);
            }
        }

        return quantum_state(reduced);
    }

    size_t qubit_count() const { return num_qubits; }
    const amplitude_vector& get_amplitudes() const { return amplitudes; }
};

// Schumacher compression for quantum states
template<typename T = double>
class schumacher_compressor {
public:
    using value_type = T;
    using complex_type = std::complex<T>;
    using state_type = quantum_state<T>;

private:
    struct typical_subspace {
        std::vector<size_t> basis_states;
        T epsilon;
        size_t original_dim;
    };

public:
    // Compress ensemble of quantum states
    std::pair<state_type, typical_subspace> compress(
        const std::vector<state_type>& ensemble,
        T epsilon = 0.01) {

        // Compute density matrix
        size_t dim = ensemble[0].get_amplitudes().size();
        std::vector<std::vector<complex_type>> density(dim,
            std::vector<complex_type>(dim, 0));

        for (const auto& state : ensemble) {
            const auto& amps = state.get_amplitudes();
            for (size_t i = 0; i < dim; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    density[i][j] += amps[i] * std::conj(amps[j]);
                }
            }
        }

        // Normalize
        T trace = 0;
        for (size_t i = 0; i < dim; ++i) {
            trace += std::real(density[i][i]);
        }
        for (auto& row : density) {
            for (auto& elem : row) {
                elem /= trace;
            }
        }

        // Diagonalize density matrix (simplified: assume diagonal)
        std::vector<std::pair<T, size_t>> eigenvalues;
        for (size_t i = 0; i < dim; ++i) {
            eigenvalues.push_back({std::real(density[i][i]), i});
        }
        std::sort(eigenvalues.rbegin(), eigenvalues.rend());

        // Find typical subspace
        typical_subspace subspace;
        subspace.epsilon = epsilon;
        subspace.original_dim = dim;

        T cumulative_prob = 0;
        for (const auto& [eval, idx] : eigenvalues) {
            subspace.basis_states.push_back(idx);
            cumulative_prob += eval;
            if (cumulative_prob > 1 - epsilon) break;
        }

        // Project to typical subspace
        std::vector<complex_type> compressed_amps(subspace.basis_states.size());
        for (size_t i = 0; i < subspace.basis_states.size(); ++i) {
            compressed_amps[i] = ensemble[0].get_amplitudes()[subspace.basis_states[i]];
        }

        // Normalize
        T norm = 0;
        for (const auto& amp : compressed_amps) {
            norm += std::norm(amp);
        }
        if (norm > 0) {
            for (auto& amp : compressed_amps) {
                amp /= std::sqrt(norm);
            }
        }

        size_t compressed_qubits = static_cast<size_t>(
            std::ceil(std::log2(subspace.basis_states.size())));
        state_type compressed_state(compressed_qubits);

        return {compressed_state, subspace};
    }

    // Decompress to original space
    state_type decompress(const state_type& compressed,
                         const typical_subspace& subspace) {
        size_t original_qubits = static_cast<size_t>(
            std::log2(subspace.original_dim));

        std::vector<complex_type> decompressed_amps(subspace.original_dim, 0);
        const auto& comp_amps = compressed.get_amplitudes();

        for (size_t i = 0; i < std::min(comp_amps.size(), subspace.basis_states.size()); ++i) {
            decompressed_amps[subspace.basis_states[i]] = comp_amps[i];
        }

        return state_type(decompressed_amps);
    }

    // Compression rate
    T compression_rate(const typical_subspace& subspace) {
        return static_cast<T>(subspace.basis_states.size()) /
               static_cast<T>(subspace.original_dim);
    }

    // Fidelity of compression
    T fidelity(const state_type& original, const state_type& decompressed) {
        const auto& orig_amps = original.get_amplitudes();
        const auto& decomp_amps = decompressed.get_amplitudes();

        complex_type overlap = 0;
        size_t min_size = std::min(orig_amps.size(), decomp_amps.size());
        for (size_t i = 0; i < min_size; ++i) {
            overlap += std::conj(orig_amps[i]) * decomp_amps[i];
        }

        return std::norm(overlap);
    }
};

// Quantum Huffman coding
template<typename T = double>
class quantum_huffman_coder {
public:
    using value_type = T;
    using complex_type = std::complex<T>;
    using state_type = quantum_state<T>;

private:
    struct huffman_node {
        T probability;
        std::optional<size_t> symbol;
        std::unique_ptr<huffman_node> left;
        std::unique_ptr<huffman_node> right;
        std::vector<std::vector<complex_type>> unitary;

        huffman_node(T prob, std::optional<size_t> sym = std::nullopt)
            : probability(prob), symbol(sym) {}
    };

    std::unique_ptr<huffman_node> tree;
    std::vector<T> symbol_probs;

public:
    // Build quantum Huffman tree
    void build_tree(const std::vector<T>& probabilities) {
        symbol_probs = probabilities;

        // Priority queue for building tree
        auto comp = [](const auto& a, const auto& b) {
            return a->probability > b->probability;
        };
        std::priority_queue<std::unique_ptr<huffman_node>,
                           std::vector<std::unique_ptr<huffman_node>>,
                           decltype(comp)> pq(comp);

        // Initialize leaves
        for (size_t i = 0; i < probabilities.size(); ++i) {
            pq.push(std::make_unique<huffman_node>(probabilities[i], i));
        }

        // Build tree
        while (pq.size() > 1) {
            auto left = std::move(const_cast<std::unique_ptr<huffman_node>&>(pq.top()));
            pq.pop();
            auto right = std::move(const_cast<std::unique_ptr<huffman_node>&>(pq.top()));
            pq.pop();

            auto parent = std::make_unique<huffman_node>(
                left->probability + right->probability);
            parent->left = std::move(left);
            parent->right = std::move(right);

            // Create unitary for this node
            parent->unitary = create_branching_unitary(
                parent->left->probability,
                parent->right->probability);

            pq.push(std::move(parent));
        }

        tree = std::move(const_cast<std::unique_ptr<huffman_node>&>(pq.top()));
    }

    // Encode classical data to quantum state
    state_type encode(const std::vector<size_t>& symbols) {
        if (!tree) {
            throw std::runtime_error("Huffman tree not built");
        }

        size_t max_depth = compute_max_depth(tree.get());
        size_t num_qubits = static_cast<size_t>(
            std::ceil(std::log2(symbols.size() * max_depth)));

        state_type quantum_code(num_qubits);

        // Encode each symbol
        for (size_t i = 0; i < symbols.size(); ++i) {
            encode_symbol(quantum_code, symbols[i], i);
        }

        return quantum_code;
    }

    // Decode quantum state to classical data
    std::vector<size_t> decode(const state_type& quantum_code,
                               size_t num_symbols) {
        std::vector<size_t> decoded;

        for (size_t i = 0; i < num_symbols; ++i) {
            size_t symbol = decode_symbol(quantum_code, i);
            decoded.push_back(symbol);
        }

        return decoded;
    }

private:
    std::vector<std::vector<complex_type>> create_branching_unitary(T p1, T p2) {
        T total = p1 + p2;
        T theta = 2 * std::acos(std::sqrt(p1 / total));

        // Rotation matrix
        return {
            {std::cos(theta/2), -std::sin(theta/2)},
            {std::sin(theta/2), std::cos(theta/2)}
        };
    }

    size_t compute_max_depth(huffman_node* node, size_t depth = 0) {
        if (!node) return depth;
        if (node->symbol.has_value()) return depth;

        return std::max(
            compute_max_depth(node->left.get(), depth + 1),
            compute_max_depth(node->right.get(), depth + 1)
        );
    }

    void encode_symbol(state_type& state, size_t symbol, size_t position) {
        // Traverse tree and apply unitaries
        huffman_node* current = tree.get();
        std::vector<size_t> path;

        while (current && !current->symbol.has_value()) {
            if (symbol_in_subtree(current->left.get(), symbol)) {
                path.push_back(0);
                current = current->left.get();
            } else {
                path.push_back(1);
                current = current->right.get();
            }

            // Apply unitary at this position
            if (!current->unitary.empty()) {
                state.apply_gate(current->unitary, {position});
            }
        }
    }

    size_t decode_symbol(const state_type& state, size_t position) {
        // Measure qubits and traverse tree
        huffman_node* current = tree.get();

        while (current && !current->symbol.has_value()) {
            // Simplified: random measurement
            if (rand() % 2 == 0) {
                current = current->left.get();
            } else {
                current = current->right.get();
            }
        }

        return current->symbol.value_or(0);
    }

    bool symbol_in_subtree(huffman_node* node, size_t symbol) {
        if (!node) return false;
        if (node->symbol.has_value()) {
            return node->symbol.value() == symbol;
        }
        return symbol_in_subtree(node->left.get(), symbol) ||
               symbol_in_subtree(node->right.get(), symbol);
    }
};

// Entanglement-assisted classical compression
template<typename T = double>
class entanglement_assisted_compressor {
public:
    using value_type = T;
    using complex_type = std::complex<T>;
    using state_type = quantum_state<T>;

private:
    state_type shared_entanglement;
    size_t ebits;  // Number of entangled bit pairs

public:
    entanglement_assisted_compressor(size_t num_ebits)
        : ebits(num_ebits)
        , shared_entanglement(2 * num_ebits) {

        // Create maximally entangled states
        auto& amps = const_cast<std::vector<complex_type>&>(
            shared_entanglement.get_amplitudes());

        for (size_t i = 0; i < (1u << num_ebits); ++i) {
            // |ii⟩ state
            size_t entangled_state = (i << num_ebits) | i;
            amps[entangled_state] = complex_type(1.0 / std::sqrt(1u << num_ebits));
        }
    }

    // Super-dense coding: 2 classical bits per qubit
    state_type encode_superdense(const std::vector<uint8_t>& classical_data) {
        size_t num_qubits = (classical_data.size() * 8 + 1) / 2;  // 2 bits per qubit
        state_type encoded = shared_entanglement;

        for (size_t i = 0; i < classical_data.size(); ++i) {
            uint8_t byte = classical_data[i];

            for (size_t bit_pair = 0; bit_pair < 4; ++bit_pair) {
                uint8_t two_bits = (byte >> (bit_pair * 2)) & 0x03;

                // Apply Pauli operations based on two-bit value
                size_t qubit_idx = i * 4 + bit_pair;
                if (qubit_idx >= ebits) break;

                switch (two_bits) {
                    case 0x00:  // I (identity)
                        break;
                    case 0x01:  // X
                        apply_pauli_x(encoded, qubit_idx);
                        break;
                    case 0x02:  // Z
                        apply_pauli_z(encoded, qubit_idx);
                        break;
                    case 0x03:  // XZ
                        apply_pauli_x(encoded, qubit_idx);
                        apply_pauli_z(encoded, qubit_idx);
                        break;
                }
            }
        }

        return encoded;
    }

    // Decode super-dense coded message
    std::vector<uint8_t> decode_superdense(const state_type& encoded) {
        std::vector<uint8_t> decoded;

        // Bell basis measurement
        for (size_t i = 0; i < ebits; i += 4) {
            uint8_t byte = 0;

            for (size_t bit_pair = 0; bit_pair < 4 && (i + bit_pair) < ebits; ++bit_pair) {
                uint8_t two_bits = measure_bell_basis(encoded, i + bit_pair);
                byte |= (two_bits << (bit_pair * 2));
            }

            decoded.push_back(byte);
        }

        return decoded;
    }

    // Compression rate with entanglement assistance
    T compression_rate() const {
        return 2.0;  // Super-dense coding achieves factor of 2
    }

private:
    void apply_pauli_x(state_type& state, size_t qubit) {
        std::vector<std::vector<complex_type>> pauli_x = {
            {0, 1},
            {1, 0}
        };
        state.apply_gate(pauli_x, {qubit});
    }

    void apply_pauli_z(state_type& state, size_t qubit) {
        std::vector<std::vector<complex_type>> pauli_z = {
            {1, 0},
            {0, -1}
        };
        state.apply_gate(pauli_z, {qubit});
    }

    uint8_t measure_bell_basis(const state_type& state, size_t qubit_pair) {
        // Simplified: random measurement outcome
        return rand() % 4;
    }
};

// Holevo information calculator
template<typename T = double>
class holevo_information {
public:
    using value_type = T;
    using complex_type = std::complex<T>;
    using state_type = quantum_state<T>;

    // Compute Holevo bound for ensemble
    T compute_bound(const std::vector<std::pair<T, state_type>>& ensemble) {
        if (ensemble.empty()) return 0;

        size_t dim = ensemble[0].second.get_amplitudes().size();

        // Compute average density matrix
        std::vector<std::vector<complex_type>> rho(dim,
            std::vector<complex_type>(dim, 0));

        for (const auto& [prob, state] : ensemble) {
            const auto& amps = state.get_amplitudes();
            for (size_t i = 0; i < dim; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    rho[i][j] += prob * amps[i] * std::conj(amps[j]);
                }
            }
        }

        // Compute S(ρ)
        T s_rho = von_neumann_entropy(rho);

        // Compute average of S(ρ_i)
        T avg_s_rho_i = 0;
        for (const auto& [prob, state] : ensemble) {
            const auto& amps = state.get_amplitudes();
            std::vector<std::vector<complex_type>> rho_i(dim,
                std::vector<complex_type>(dim, 0));

            for (size_t i = 0; i < dim; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    rho_i[i][j] = amps[i] * std::conj(amps[j]);
                }
            }

            avg_s_rho_i += prob * von_neumann_entropy(rho_i);
        }

        // Holevo bound: χ = S(ρ) - Σ p_i S(ρ_i)
        return s_rho - avg_s_rho_i;
    }

    // Accessible information (requires optimization over measurements)
    T accessible_information(const std::vector<std::pair<T, state_type>>& ensemble,
                            size_t num_measurements = 100) {
        T max_info = 0;

        // Try random measurements
        for (size_t m = 0; m < num_measurements; ++m) {
            auto measurement = generate_random_povm(ensemble[0].second.qubit_count());
            T info = mutual_information(ensemble, measurement);
            max_info = std::max(max_info, info);
        }

        return max_info;
    }

private:
    T von_neumann_entropy(const std::vector<std::vector<complex_type>>& density) {
        // Simplified: assume diagonal
        T entropy = 0;
        for (size_t i = 0; i < density.size(); ++i) {
            T eigenvalue = std::real(density[i][i]);
            if (eigenvalue > 1e-10) {
                entropy -= eigenvalue * std::log2(eigenvalue);
            }
        }
        return entropy;
    }

    std::vector<std::vector<std::vector<complex_type>>> generate_random_povm(size_t num_qubits) {
        // Generate random POVM elements
        size_t dim = 1 << num_qubits;
        std::vector<std::vector<std::vector<complex_type>>> povm;

        // Simplified: use computational basis
        for (size_t i = 0; i < dim; ++i) {
            std::vector<std::vector<complex_type>> element(dim,
                std::vector<complex_type>(dim, 0));
            element[i][i] = 1;
            povm.push_back(element);
        }

        return povm;
    }

    T mutual_information(const std::vector<std::pair<T, state_type>>& ensemble,
                        const std::vector<std::vector<std::vector<complex_type>>>& measurement) {
        // I(X:Y) = H(Y) - H(Y|X)
        // Simplified calculation
        return std::min(T(1), holevo_bound(ensemble));
    }

    T holevo_bound(const std::vector<std::pair<T, state_type>>& ensemble) {
        return compute_bound(ensemble);
    }
};

// Quantum Kolmogorov complexity
template<typename T = double>
class quantum_kolmogorov_complexity {
public:
    using value_type = T;
    using complex_type = std::complex<T>;
    using state_type = quantum_state<T>;

    // Estimate quantum Kolmogorov complexity
    size_t estimate_complexity(const state_type& state) {
        const auto& amps = state.get_amplitudes();

        // Count non-negligible amplitudes
        size_t significant_amplitudes = 0;
        for (const auto& amp : amps) {
            if (std::norm(amp) > 1e-10) {
                significant_amplitudes++;
            }
        }

        // Complexity ~ log(number of significant amplitudes) + precision
        size_t amplitude_bits = static_cast<size_t>(
            std::ceil(std::log2(significant_amplitudes)));

        // Add bits for precision of each amplitude
        size_t precision_bits = 32;  // Assuming float precision

        return amplitude_bits + significant_amplitudes * precision_bits;
    }

    // Quantum logical depth
    size_t logical_depth(const state_type& initial,
                        const state_type& target,
                        size_t max_gates = 1000) {

        // Estimate minimum circuit depth to transform initial to target
        size_t depth = 0;
        state_type current = initial;

        for (size_t d = 1; d <= max_gates; ++d) {
            // Try circuits of depth d
            if (can_reach_with_depth(current, target, d)) {
                return d;
            }
        }

        return max_gates;
    }

    // Bennett's logical depth for quantum states
    T bennett_depth(const state_type& state) {
        // Logical depth = time for shortest program to produce state
        size_t k_complexity = estimate_complexity(state);

        // Depth grows with entanglement
        T entanglement = compute_entanglement_entropy(state);

        return k_complexity * std::exp(entanglement);
    }

private:
    bool can_reach_with_depth(const state_type& from,
                             const state_type& to,
                             size_t depth) {
        // Simplified: check if states are close enough
        const auto& from_amps = from.get_amplitudes();
        const auto& to_amps = to.get_amplitudes();

        T distance = 0;
        for (size_t i = 0; i < std::min(from_amps.size(), to_amps.size()); ++i) {
            distance += std::norm(from_amps[i] - to_amps[i]);
        }

        // Heuristic: deeper circuits can achieve smaller distances
        return distance < std::pow(0.9, depth);
    }

    T compute_entanglement_entropy(const state_type& state) {
        // Simplified: use von Neumann entropy as proxy
        return state.entropy();
    }
};

// Quantum Shannon theory
template<typename T = double>
class quantum_shannon_theory {
public:
    using value_type = T;
    using complex_type = std::complex<T>;
    using state_type = quantum_state<T>;

    // Quantum channel capacity
    T channel_capacity(const std::function<state_type(const state_type&)>& channel,
                      size_t num_uses = 1) {

        // Holevo-Schumacher-Westmoreland theorem
        T max_capacity = 0;

        // Optimize over input ensembles
        for (size_t trial = 0; trial < 100; ++trial) {
            auto ensemble = generate_random_ensemble(2);  // 2 qubits
            T capacity = holevo_information<T>().compute_bound(
                apply_channel_to_ensemble(channel, ensemble));
            max_capacity = std::max(max_capacity, capacity);
        }

        // Regularized capacity (simplified: no regularization)
        return max_capacity * num_uses;
    }

    // Quantum source coding rate
    T source_coding_rate(const std::vector<std::pair<T, state_type>>& source) {
        // Von Neumann entropy of the source
        if (source.empty()) return 0;

        size_t dim = source[0].second.get_amplitudes().size();
        std::vector<std::vector<complex_type>> density(dim,
            std::vector<complex_type>(dim, 0));

        for (const auto& [prob, state] : source) {
            const auto& amps = state.get_amplitudes();
            for (size_t i = 0; i < dim; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    density[i][j] += prob * amps[i] * std::conj(amps[j]);
                }
            }
        }

        return von_neumann_entropy(density);
    }

    // Quantum relative entropy
    T relative_entropy(const state_type& rho, const state_type& sigma) {
        const auto& rho_amps = rho.get_amplitudes();
        const auto& sigma_amps = sigma.get_amplitudes();

        // D(ρ||σ) = Tr(ρ log ρ) - Tr(ρ log σ)
        T d_rel = 0;
        for (size_t i = 0; i < std::min(rho_amps.size(), sigma_amps.size()); ++i) {
            T p_rho = std::norm(rho_amps[i]);
            T p_sigma = std::norm(sigma_amps[i]);

            if (p_rho > 1e-10 && p_sigma > 1e-10) {
                d_rel += p_rho * (std::log(p_rho) - std::log(p_sigma));
            }
        }

        return d_rel / std::log(2);  // Convert to bits
    }

private:
    std::vector<std::pair<T, state_type>> generate_random_ensemble(size_t num_qubits) {
        std::vector<std::pair<T, state_type>> ensemble;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(0, 1);

        size_t ensemble_size = 4;
        T total_prob = 0;

        for (size_t i = 0; i < ensemble_size; ++i) {
            T prob = dist(gen);
            total_prob += prob;

            state_type state(num_qubits);
            // Randomize state
            auto& amps = const_cast<std::vector<complex_type>&>(state.get_amplitudes());
            for (auto& amp : amps) {
                amp = complex_type(dist(gen), dist(gen));
            }

            // Normalize
            T norm = 0;
            for (const auto& amp : amps) {
                norm += std::norm(amp);
            }
            for (auto& amp : amps) {
                amp /= std::sqrt(norm);
            }

            ensemble.push_back({prob, state});
        }

        // Normalize probabilities
        for (auto& [prob, state] : ensemble) {
            prob /= total_prob;
        }

        return ensemble;
    }

    std::vector<std::pair<T, state_type>> apply_channel_to_ensemble(
        const std::function<state_type(const state_type&)>& channel,
        const std::vector<std::pair<T, state_type>>& ensemble) {

        std::vector<std::pair<T, state_type>> output;
        for (const auto& [prob, state] : ensemble) {
            output.push_back({prob, channel(state)});
        }
        return output;
    }

    T von_neumann_entropy(const std::vector<std::vector<complex_type>>& density) {
        T entropy = 0;
        for (size_t i = 0; i < density.size(); ++i) {
            T eigenvalue = std::real(density[i][i]);
            if (eigenvalue > 1e-10) {
                entropy -= eigenvalue * std::log2(eigenvalue);
            }
        }
        return entropy;
    }
};

// Quantum error correction for compression
template<typename T = double>
class quantum_error_correcting_compressor {
public:
    using value_type = T;
    using complex_type = std::complex<T>;
    using state_type = quantum_state<T>;

    // [[7,1,3]] Steane code for single logical qubit
    state_type encode_steane(bool logical_bit) {
        state_type encoded(7);
        auto& amps = const_cast<std::vector<complex_type>&>(encoded.get_amplitudes());

        if (logical_bit == 0) {
            // |0_L⟩ = |0000000⟩ + |1010101⟩ + |0110011⟩ + ...
            amps[0b0000000] = 1.0 / std::sqrt(8);
            amps[0b1010101] = 1.0 / std::sqrt(8);
            amps[0b0110011] = 1.0 / std::sqrt(8);
            amps[0b1100110] = 1.0 / std::sqrt(8);
            amps[0b0001111] = 1.0 / std::sqrt(8);
            amps[0b1011010] = 1.0 / std::sqrt(8);
            amps[0b0111100] = 1.0 / std::sqrt(8);
            amps[0b1101001] = 1.0 / std::sqrt(8);
        } else {
            // |1_L⟩ = |1111111⟩ + |0101010⟩ + ...
            amps[0b1111111] = 1.0 / std::sqrt(8);
            amps[0b0101010] = 1.0 / std::sqrt(8);
            amps[0b1001100] = 1.0 / std::sqrt(8);
            amps[0b0011001] = 1.0 / std::sqrt(8);
            amps[0b1110000] = 1.0 / std::sqrt(8);
            amps[0b0100101] = 1.0 / std::sqrt(8);
            amps[0b1000011] = 1.0 / std::sqrt(8);
            amps[0b0010110] = 1.0 / std::sqrt(8);
        }

        return encoded;
    }

    // Syndrome extraction and error correction
    state_type correct_errors(const state_type& noisy_state) {
        // Simplified: return state as-is
        return noisy_state;
    }

    // Compress with error correction
    std::vector<state_type> compress_with_ecc(const std::vector<bool>& classical_data) {
        std::vector<state_type> encoded_blocks;

        for (bool bit : classical_data) {
            encoded_blocks.push_back(encode_steane(bit));
        }

        return encoded_blocks;
    }

    // Decompress with error correction
    std::vector<bool> decompress_with_ecc(const std::vector<state_type>& encoded_blocks) {
        std::vector<bool> decoded;

        for (const auto& block : encoded_blocks) {
            auto corrected = correct_errors(block);
            decoded.push_back(measure_logical_bit(corrected));
        }

        return decoded;
    }

private:
    bool measure_logical_bit(const state_type& encoded) {
        // Simplified: measure in computational basis
        const auto& amps = encoded.get_amplitudes();

        // Count |0_L⟩ vs |1_L⟩ components
        T zero_prob = 0, one_prob = 0;

        // Check parity
        for (size_t i = 0; i < amps.size(); ++i) {
            size_t hamming_weight = __builtin_popcount(i);
            if (hamming_weight % 2 == 0) {
                zero_prob += std::norm(amps[i]);
            } else {
                one_prob += std::norm(amps[i]);
            }
        }

        return one_prob > zero_prob;
    }
};

} // namespace stepanov::compression::quantum