/**
 * Quantum Algorithms on Classical Hardware
 * ==========================================
 *
 * This example demonstrates quantum computing primitives and algorithms
 * running on classical hardware. We show superposition, entanglement,
 * and quantum algorithms like Grover's search and quantum teleportation.
 */

#include <stepanov/quantum/quantum.hpp>
#include <stepanov/math.hpp>
#include <stepanov/algorithms.hpp>
#include <iostream>
#include <complex>
#include <random>
#include <iomanip>

namespace stepanov::examples {

using namespace std::complex_literals;

/**
 * Basic Quantum Operations
 */
namespace quantum_basics {

    void demo_superposition() {
        std::cout << "=== Quantum Superposition ===\n\n";

        // Single qubit in superposition
        stepanov::quantum_register<1> qubit;
        std::cout << "Initial state |0⟩:\n";
        qubit.print_state();

        // Apply Hadamard gate to create superposition
        qubit.hadamard(0);
        std::cout << "\nAfter Hadamard gate (|0⟩ + |1⟩)/√2:\n";
        qubit.print_state();

        // Measure the qubit multiple times to see probability distribution
        std::cout << "\n1000 measurements of superposition:\n";
        int zeros = 0, ones = 0;
        for (int i = 0; i < 1000; ++i) {
            auto q = qubit;  // Copy for measurement
            if (q.measure(0) == 0) zeros++;
            else ones++;
        }
        std::cout << "  |0⟩: " << zeros << " times (" << zeros/10.0 << "%)\n";
        std::cout << "  |1⟩: " << ones << " times (" << ones/10.0 << "%)\n\n";
    }

    void demo_entanglement() {
        std::cout << "=== Quantum Entanglement ===\n\n";

        // Create Bell state (maximally entangled)
        stepanov::quantum_register<2> qubits;
        std::cout << "Initial state |00⟩:\n";
        qubits.print_state();

        // Create Bell state: (|00⟩ + |11⟩)/√2
        qubits.hadamard(0);
        qubits.cnot(0, 1);
        std::cout << "\nBell state (|00⟩ + |11⟩)/√2:\n";
        qubits.print_state();

        // Measure entangled qubits
        std::cout << "\n1000 measurements of entangled pair:\n";
        int both_zero = 0, both_one = 0, different = 0;
        for (int i = 0; i < 1000; ++i) {
            auto q = qubits;  // Copy for measurement
            auto m0 = q.measure(0);
            auto m1 = q.measure(1);
            if (m0 == 0 && m1 == 0) both_zero++;
            else if (m0 == 1 && m1 == 1) both_one++;
            else different++;
        }
        std::cout << "  |00⟩: " << both_zero << " times\n";
        std::cout << "  |11⟩: " << both_one << " times\n";
        std::cout << "  |01⟩ or |10⟩: " << different << " times\n";
        std::cout << "\nNote: Measurements are perfectly correlated!\n\n";
    }

    void demo_phase() {
        std::cout << "=== Quantum Phase ===\n\n";

        stepanov::quantum_register<1> qubit;

        // Create superposition with positive phase
        qubit.hadamard(0);
        std::cout << "State (|0⟩ + |1⟩)/√2:\n";
        qubit.print_state();

        // Apply phase gate
        qubit.phase(0, M_PI);  // Add π phase to |1⟩
        std::cout << "\nAfter π phase on |1⟩: (|0⟩ - |1⟩)/√2:\n";
        qubit.print_state();

        // Apply Hadamard again
        qubit.hadamard(0);
        std::cout << "\nAfter second Hadamard (should be |1⟩):\n";
        qubit.print_state();
        std::cout << "\nPhase matters in quantum computation!\n\n";
    }
}

/**
 * Grover's Search Algorithm
 */
namespace grover {

    template<int N>
    class grover_search {
        stepanov::quantum_register<N> qreg;
        std::function<bool(int)> oracle;
        int target;

    public:
        grover_search(std::function<bool(int)> f) : oracle(f) {}

        int search() {
            // Initialize in uniform superposition
            for (int i = 0; i < N; ++i) {
                qreg.hadamard(i);
            }

            // Optimal number of iterations
            int iterations = static_cast<int>(M_PI/4 * std::sqrt(1 << N));

            std::cout << "Searching " << (1 << N) << " items in "
                      << iterations << " iterations\n";

            for (int iter = 0; iter < iterations; ++iter) {
                // Oracle: flip phase of target state
                apply_oracle();

                // Diffusion operator
                apply_diffusion();
            }

            // Measure to get result
            return qreg.measure_all();
        }

    private:
        void apply_oracle() {
            // For each basis state, check if it's the target
            for (int state = 0; state < (1 << N); ++state) {
                if (oracle(state)) {
                    qreg.conditional_phase(state, -1.0);
                }
            }
        }

        void apply_diffusion() {
            // Apply Hadamard to all qubits
            for (int i = 0; i < N; ++i) {
                qreg.hadamard(i);
            }

            // Conditional phase shift
            for (int state = 1; state < (1 << N); ++state) {
                qreg.conditional_phase(state, -1.0);
            }

            // Apply Hadamard again
            for (int i = 0; i < N; ++i) {
                qreg.hadamard(i);
            }
        }
    };

    void demo_grover() {
        std::cout << "=== Grover's Quantum Search ===\n\n";

        // Search for a specific number in unsorted database
        const int target = 42;
        const int bits = 6;  // 2^6 = 64 items

        std::cout << "Classical search for " << target << " in 64 items:\n";
        std::cout << "  Worst case: 64 operations\n";
        std::cout << "  Average: 32 operations\n\n";

        std::cout << "Quantum search using Grover's algorithm:\n";

        grover_search<bits> search([target](int x) { return x == target; });
        int found = search.search();

        std::cout << "Found: " << found << " (target was " << target << ")\n";
        std::cout << "Speedup: √64 = 8x faster!\n\n";

        // Multiple targets
        std::cout << "Searching for multiple targets (all even numbers):\n";
        grover_search<4> multi_search([](int x) { return x % 2 == 0; });
        std::cout << "Found one of the targets: " << multi_search.search() << "\n\n";
    }
}

/**
 * Quantum Teleportation
 */
namespace teleportation {

    void demo_teleportation() {
        std::cout << "=== Quantum Teleportation ===\n\n";

        // Alice wants to send a qubit state to Bob
        std::cout << "Alice has a qubit in state α|0⟩ + β|1⟩\n";
        std::cout << "She wants to send it to Bob using entanglement\n\n";

        // Create the state to teleport (arbitrary superposition)
        stepanov::quantum_register<1> alice_qubit;
        alice_qubit.rotate_y(0, M_PI/3);  // Create some arbitrary state
        std::cout << "Alice's qubit state to teleport:\n";
        alice_qubit.print_state();

        // Create entangled pair shared between Alice and Bob
        stepanov::quantum_register<2> bell_pair;
        bell_pair.hadamard(0);
        bell_pair.cnot(0, 1);
        std::cout << "\nBell pair shared between Alice and Bob:\n";
        bell_pair.print_state();

        // Combine Alice's qubit with her half of Bell pair
        stepanov::quantum_register<3> full_system;
        full_system.set_state(alice_qubit, bell_pair);

        // Alice performs Bell measurement
        full_system.cnot(0, 1);
        full_system.hadamard(0);

        std::cout << "\nAfter Alice's operations:\n";
        full_system.print_state();

        // Alice measures her qubits
        auto m0 = full_system.measure(0);
        auto m1 = full_system.measure(1);

        std::cout << "\nAlice's measurement results: "
                  << m0 << m1 << "\n";

        // Bob applies corrections based on Alice's measurements
        if (m1 == 1) full_system.pauli_x(2);  // Bit flip
        if (m0 == 1) full_system.pauli_z(2);  // Phase flip

        std::cout << "\nBob's qubit after corrections:\n";
        auto bob_state = full_system.get_qubit_state(2);
        std::cout << "  |0⟩ amplitude: " << bob_state[0] << "\n";
        std::cout << "  |1⟩ amplitude: " << bob_state[1] << "\n";

        std::cout << "\nTeleportation successful! Bob now has Alice's original state.\n";
        std::cout << "Note: Alice's original qubit was destroyed (no cloning theorem).\n\n";
    }
}

/**
 * Quantum Fourier Transform
 */
namespace qft {

    template<int N>
    void quantum_fourier_transform(stepanov::quantum_register<N>& qreg) {
        for (int j = 0; j < N; ++j) {
            // Hadamard on qubit j
            qreg.hadamard(j);

            // Controlled rotations
            for (int k = j + 1; k < N; ++k) {
                double angle = 2 * M_PI / (1 << (k - j + 1));
                qreg.controlled_phase(k, j, angle);
            }
        }

        // Swap qubits (reverse order)
        for (int i = 0; i < N/2; ++i) {
            qreg.swap(i, N - 1 - i);
        }
    }

    void demo_qft() {
        std::cout << "=== Quantum Fourier Transform ===\n\n";

        const int n = 3;
        stepanov::quantum_register<n> qreg;

        // Create state |5⟩ = |101⟩
        qreg.pauli_x(0);  // Set bit 0
        qreg.pauli_x(2);  // Set bit 2

        std::cout << "Initial state |5⟩ = |101⟩:\n";
        qreg.print_state();

        // Apply QFT
        quantum_fourier_transform(qreg);

        std::cout << "\nAfter Quantum Fourier Transform:\n";
        qreg.print_state();

        std::cout << "\nThe QFT creates a superposition with complex phases.\n";
        std::cout << "This is the key component of Shor's factoring algorithm!\n\n";
    }
}

/**
 * Quantum Simulation
 */
namespace simulation {

    void demo_quantum_walk() {
        std::cout << "=== Quantum Random Walk ===\n\n";

        std::cout << "Classical random walk vs Quantum walk on a line:\n\n";

        // Classical random walk
        std::cout << "Classical walk (1000 steps):\n";
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 1);

        int classical_pos = 0;
        for (int i = 0; i < 1000; ++i) {
            classical_pos += dis(gen) ? 1 : -1;
        }
        std::cout << "  Final position: " << classical_pos << "\n";
        std::cout << "  Expected spread: √1000 ≈ 31.6\n\n";

        // Quantum walk (simplified simulation)
        const int positions = 128;  // Number of positions
        std::vector<std::complex<double>> amplitudes(positions, 0);
        amplitudes[positions/2] = 1.0;  // Start in center

        std::cout << "Quantum walk (100 steps):\n";

        for (int step = 0; step < 100; ++step) {
            std::vector<std::complex<double>> new_amp(positions, 0);

            // Simplified quantum walk evolution
            for (int i = 1; i < positions - 1; ++i) {
                new_amp[i-1] += amplitudes[i] * (1.0/std::sqrt(2));
                new_amp[i+1] += amplitudes[i] * (1.0i/std::sqrt(2));
            }

            amplitudes = new_amp;
        }

        // Find probability distribution
        double max_prob = 0;
        int max_pos = 0;
        for (int i = 0; i < positions; ++i) {
            double prob = std::norm(amplitudes[i]);
            if (prob > max_prob) {
                max_prob = prob;
                max_pos = i - positions/2;
            }
        }

        std::cout << "  Peak position: " << max_pos << "\n";
        std::cout << "  Quantum spread: O(n) - Linear in steps!\n";
        std::cout << "\nQuantum walks spread quadratically faster than classical!\n\n";
    }

    void demo_quantum_annealing() {
        std::cout << "=== Quantum Annealing Simulation ===\n\n";

        std::cout << "Finding minimum of energy landscape using quantum annealing:\n\n";

        // Simple optimization problem: find minimum of rugged landscape
        auto energy = [](double x) {
            return std::sin(5*x) + 0.5*std::sin(11*x) + 0.2*std::sin(23*x) + x*x/10;
        };

        // Quantum annealing simulation
        double position = 0;
        double temperature = 10.0;  // Start hot (quantum fluctuations)

        std::cout << "Annealing process:\n";
        for (int step = 0; step < 100; ++step) {
            // Quantum tunneling probability
            double tunnel_prob = std::exp(-1.0/temperature);

            // Try quantum jump
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            std::random_device rd;
            std::mt19937 gen(rd());

            double new_pos = position + tunnel_prob * dis(gen);
            double delta_e = energy(new_pos) - energy(position);

            // Accept or reject based on quantum mechanics
            if (delta_e < 0 || std::exp(-delta_e/temperature) > dis(gen)) {
                position = new_pos;
            }

            // Cool down (reduce quantum fluctuations)
            temperature *= 0.95;

            if (step % 20 == 0) {
                std::cout << "  Step " << step << ": position = "
                          << std::setprecision(4) << position
                          << ", energy = " << energy(position) << "\n";
            }
        }

        std::cout << "\nFinal minimum found at: " << position
                  << " with energy: " << energy(position) << "\n";
        std::cout << "\nQuantum annealing can tunnel through barriers to find\n";
        std::cout << "global minima that classical methods might miss!\n\n";
    }
}

/**
 * Philosophical Implications
 */
void demo_philosophical() {
    std::cout << "=== Quantum Computation: Philosophy & Implications ===\n\n";

    std::cout << "Classical Bit vs Quantum Qubit:\n";
    std::cout << "  Classical: 0 OR 1\n";
    std::cout << "  Quantum:   α|0⟩ + β|1⟩ where |α|² + |β|² = 1\n\n";

    std::cout << "Key Quantum Phenomena:\n";
    std::cout << "  1. Superposition - Being in multiple states simultaneously\n";
    std::cout << "  2. Entanglement - Instant correlation regardless of distance\n";
    std::cout << "  3. Interference - Amplitudes can cancel or reinforce\n";
    std::cout << "  4. Measurement - Observation collapses the wavefunction\n\n";

    std::cout << "Quantum Advantages:\n";
    std::cout << "  • Search: √N speedup (Grover)\n";
    std::cout << "  • Factoring: Exponential speedup (Shor)\n";
    std::cout << "  • Simulation: Natural for quantum systems\n";
    std::cout << "  • Optimization: Quantum tunneling through barriers\n\n";

    std::cout << "Quantum Limitations:\n";
    std::cout << "  • No-cloning theorem - Can't copy unknown quantum states\n";
    std::cout << "  • Decoherence - Quantum states are fragile\n";
    std::cout << "  • Measurement - Destroys superposition\n";
    std::cout << "  • Error rates - Quantum gates are imperfect\n\n";

    // Demonstrate no-cloning theorem
    stepanov::quantum_register<1> original;
    original.rotate_y(0, M_PI/4);  // Create arbitrary state

    std::cout << "No-Cloning Theorem Demonstration:\n";
    std::cout << "Original state: ";
    original.print_state_compact();

    std::cout << "Attempting to clone... ";
    // There's no operation that can copy an arbitrary unknown state
    std::cout << "IMPOSSIBLE!\n";
    std::cout << "This is a fundamental law of quantum mechanics.\n\n";

    std::cout << "The Deeper Meaning:\n";
    std::cout << "Quantum computation isn't just faster classical computation.\n";
    std::cout << "It's a fundamentally different model of information processing.\n";
    std::cout << "It reveals that the universe itself computes quantum mechanically.\n\n";

    std::cout << "Stepanov brings these quantum principles to classical programming,\n";
    std::cout << "preparing us for the quantum future while teaching us that\n";
    std::cout << "computation itself is richer than we imagined.\n";
}

} // namespace stepanov::examples

int main() {
    using namespace stepanov::examples;

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        Quantum Algorithms on Classical Hardware            ║\n";
    std::cout << "║          Superposition, Entanglement, and Beyond           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // Basic quantum mechanics
    quantum_basics::demo_superposition();
    quantum_basics::demo_entanglement();
    quantum_basics::demo_phase();

    // Quantum algorithms
    grover::demo_grover();
    teleportation::demo_teleportation();
    qft::demo_qft();

    // Quantum simulation
    simulation::demo_quantum_walk();
    simulation::demo_quantum_annealing();

    // Philosophy
    demo_philosophical();

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  'Nature isn't classical, dammit, and if you want to      ║\n";
    std::cout << "║   make a simulation of nature, you'd better make it       ║\n";
    std::cout << "║   quantum mechanical.'  - Richard Feynman                 ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║  Stepanov: Quantum thinking for classical programmers.    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    return 0;
}