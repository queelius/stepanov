#pragma once

#include "concepts.hpp"
#include "../concepts.hpp"
#include <random>
#include <cmath>
#include <functional>

namespace stepanov::optimization {

/**
 * Simulated annealing for global optimization
 *
 * Inspired by metallurgical annealing, this algorithm accepts worse solutions
 * with a probability that decreases over time, allowing escape from local minima.
 *
 * Following generic programming principles, works with any state type that
 * supports neighbor generation and any energy/cost function.
 */

// Cooling schedule concepts
template<typename Schedule, typename T>
concept cooling_schedule = requires(Schedule s, size_t iteration) {
    { s(iteration) } -> std::convertible_to<T>;
};

// Neighbor generator concept
template<typename Gen, typename State, typename RNG>
concept neighbor_generator = requires(Gen g, State s, RNG& rng) {
    { g(s, rng) } -> std::convertible_to<State>;
};

// Standard cooling schedules
template<typename T>
    requires field<T>
struct exponential_cooling {
    T initial_temperature;
    T cooling_rate; // Should be between 0 and 1

    exponential_cooling(T init_temp = T(100), T rate = T(0.95))
        : initial_temperature(init_temp), cooling_rate(rate) {}

    T operator()(size_t iteration) const {
        return initial_temperature * std::pow(cooling_rate, T(iteration));
    }
};

template<typename T>
    requires field<T>
struct linear_cooling {
    T initial_temperature;
    T final_temperature;
    size_t max_iterations;

    linear_cooling(T init_temp = T(100), T final_temp = T(0.01), size_t max_iter = 1000)
        : initial_temperature(init_temp), final_temperature(final_temp), max_iterations(max_iter) {}

    T operator()(size_t iteration) const {
        if (iteration >= max_iterations) return final_temperature;
        T progress = T(iteration) / T(max_iterations);
        return initial_temperature + (final_temperature - initial_temperature) * progress;
    }
};

template<typename T>
    requires field<T>
struct logarithmic_cooling {
    T c; // Cooling constant

    explicit logarithmic_cooling(T cooling_constant = T(10))
        : c(cooling_constant) {}

    T operator()(size_t iteration) const {
        return c / std::log(T(iteration + 2));
    }
};

template<typename T>
    requires field<T>
struct adaptive_cooling {
    T current_temperature;
    T acceptance_target;
    T adjustment_factor;

    adaptive_cooling(T init_temp = T(100), T target = T(0.4), T factor = T(1.05))
        : current_temperature(init_temp), acceptance_target(target), adjustment_factor(factor) {}

    T operator()(size_t iteration, T acceptance_rate) {
        // Adjust temperature based on acceptance rate
        if (acceptance_rate > acceptance_target) {
            current_temperature = current_temperature / adjustment_factor;
        } else {
            current_temperature = current_temperature * adjustment_factor;
        }
        return current_temperature;
    }
};

// Simulated annealing result
template<typename State, typename T>
struct annealing_result {
    State best_solution;
    T best_energy;
    State final_solution;
    T final_energy;
    size_t iterations;
    size_t accepted_moves;
    bool converged;

    annealing_result() = default;
    annealing_result(State best, T best_e, State final, T final_e,
                    size_t iter, size_t accepted, bool conv)
        : best_solution(best), best_energy(best_e),
          final_solution(final), final_energy(final_e),
          iterations(iter), accepted_moves(accepted), converged(conv) {}
};

// Basic simulated annealing
template<typename State, typename Energy, typename Neighbor, typename Schedule,
         typename RNG = std::mt19937, typename T = double>
    requires evaluable<Energy, State> &&
             neighbor_generator<Neighbor, State, RNG> &&
             cooling_schedule<Schedule, T>
annealing_result<State, T> simulated_annealing(
    Energy energy_function,
    State initial_state,
    Neighbor neighbor_gen,
    Schedule cooling,
    size_t max_iterations = 10000,
    T min_temperature = T(0.001),
    RNG& rng = *new RNG(std::random_device{}()))
{
    std::uniform_real_distribution<T> uniform(T(0), T(1));

    State current_state = initial_state;
    T current_energy = energy_function(current_state);

    State best_state = current_state;
    T best_energy = current_energy;

    size_t accepted = 0;

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        T temperature = cooling(iter);

        // Check if temperature is too low
        if (temperature < min_temperature) {
            return annealing_result<State, T>{
                best_state, best_energy, current_state, current_energy,
                iter, accepted, true
            };
        }

        // Generate neighbor
        State neighbor = neighbor_gen(current_state, rng);
        T neighbor_energy = energy_function(neighbor);

        // Calculate energy difference
        T delta_energy = neighbor_energy - current_energy;

        // Accept or reject move
        if (delta_energy < T(0) || uniform(rng) < std::exp(-delta_energy / temperature)) {
            current_state = neighbor;
            current_energy = neighbor_energy;
            ++accepted;

            // Update best solution if necessary
            if (current_energy < best_energy) {
                best_state = current_state;
                best_energy = current_energy;
            }
        }
    }

    return annealing_result<State, T>{
        best_state, best_energy, current_state, current_energy,
        max_iterations, accepted, false
    };
}

// Parallel tempering (replica exchange)
template<typename State, typename Energy, typename Neighbor, typename T = double>
    requires evaluable<Energy, State>
class parallel_tempering {
private:
    std::vector<T> temperatures;
    std::vector<State> replicas;
    std::vector<T> energies;
    Energy energy_function;
    Neighbor neighbor_gen;
    std::mt19937 rng;

public:
    parallel_tempering(Energy energy_func, Neighbor neighbor,
                       State initial_state, std::vector<T> temps)
        : temperatures(temps), energy_function(energy_func),
          neighbor_gen(neighbor), rng(std::random_device{}())
    {
        replicas.resize(temperatures.size(), initial_state);
        energies.resize(temperatures.size());

        for (size_t i = 0; i < replicas.size(); ++i) {
            energies[i] = energy_function(replicas[i]);
        }
    }

    annealing_result<State, T> optimize(size_t iterations_per_swap = 100,
                                        size_t num_swaps = 1000)
    {
        std::uniform_real_distribution<T> uniform(T(0), T(1));
        std::uniform_int_distribution<size_t> replica_dist(0, replicas.size() - 2);

        State best_state = replicas[0];
        T best_energy = energies[0];

        for (size_t swap_iter = 0; swap_iter < num_swaps; ++swap_iter) {
            // Run simulated annealing for each replica
            for (size_t i = 0; i < replicas.size(); ++i) {
                for (size_t iter = 0; iter < iterations_per_swap; ++iter) {
                    State neighbor = neighbor_gen(replicas[i], rng);
                    T neighbor_energy = energy_function(neighbor);
                    T delta = neighbor_energy - energies[i];

                    if (delta < T(0) || uniform(rng) < std::exp(-delta / temperatures[i])) {
                        replicas[i] = neighbor;
                        energies[i] = neighbor_energy;

                        if (energies[i] < best_energy) {
                            best_state = replicas[i];
                            best_energy = energies[i];
                        }
                    }
                }
            }

            // Attempt replica exchange
            size_t i = replica_dist(rng);
            size_t j = i + 1;

            T delta = (energies[j] - energies[i]) *
                     (T(1) / temperatures[i] - T(1) / temperatures[j]);

            if (delta < T(0) || uniform(rng) < std::exp(-delta)) {
                std::swap(replicas[i], replicas[j]);
                std::swap(energies[i], energies[j]);
            }
        }

        return annealing_result<State, T>{
            best_state, best_energy, replicas[0], energies[0],
            num_swaps * iterations_per_swap * replicas.size(), 0, true
        };
    }
};

// Quantum-inspired annealing with tunneling
template<typename State, typename Energy, typename Neighbor, typename T = double>
    requires evaluable<Energy, State>
annealing_result<State, T> quantum_annealing(
    Energy energy_function,
    State initial_state,
    Neighbor neighbor_gen,
    T initial_transverse_field = T(10),
    T final_transverse_field = T(0.01),
    size_t max_iterations = 10000,
    std::mt19937& rng = *new std::mt19937(std::random_device{}()))
{
    std::uniform_real_distribution<T> uniform(T(0), T(1));

    State current_state = initial_state;
    T current_energy = energy_function(current_state);

    State best_state = current_state;
    T best_energy = current_energy;

    size_t accepted = 0;

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        // Linear decrease of transverse field (quantum fluctuations)
        T progress = T(iter) / T(max_iterations);
        T transverse_field = initial_transverse_field +
                           (final_transverse_field - initial_transverse_field) * progress;

        // Generate multiple neighbors (quantum superposition)
        const size_t num_candidates = 5;
        std::vector<State> candidates;
        std::vector<T> candidate_energies;

        for (size_t i = 0; i < num_candidates; ++i) {
            State candidate = neighbor_gen(current_state, rng);
            candidates.push_back(candidate);
            candidate_energies.push_back(energy_function(candidate));
        }

        // Select based on quantum tunneling probability
        for (size_t i = 0; i < candidates.size(); ++i) {
            T delta = candidate_energies[i] - current_energy;
            T tunneling_prob = std::exp(-abs(delta) / transverse_field);

            if (delta < T(0) || uniform(rng) < tunneling_prob) {
                current_state = candidates[i];
                current_energy = candidate_energies[i];
                ++accepted;

                if (current_energy < best_energy) {
                    best_state = current_state;
                    best_energy = current_energy;
                }
                break;
            }
        }
    }

    return annealing_result<State, T>{
        best_state, best_energy, current_state, current_energy,
        max_iterations, accepted, true
    };
}

// Simulated annealing with restarts
template<typename State, typename Energy, typename Neighbor, typename Schedule, typename T = double>
    requires evaluable<Energy, State>
annealing_result<State, T> simulated_annealing_with_restarts(
    Energy energy_function,
    State initial_state,
    Neighbor neighbor_gen,
    Schedule cooling,
    size_t num_restarts = 10,
    size_t iterations_per_restart = 1000,
    std::mt19937& rng = *new std::mt19937(std::random_device{}()))
{
    State global_best = initial_state;
    T global_best_energy = energy_function(initial_state);
    size_t total_accepted = 0;

    for (size_t restart = 0; restart < num_restarts; ++restart) {
        // Random restart or use best so far with probability 0.5
        std::uniform_real_distribution<T> uniform(T(0), T(1));
        State start_state = (uniform(rng) < T(0.5)) ? global_best : neighbor_gen(initial_state, rng);

        auto result = simulated_annealing(
            energy_function, start_state, neighbor_gen, cooling,
            iterations_per_restart, T(0.001), rng
        );

        total_accepted += result.accepted_moves;

        if (result.best_energy < global_best_energy) {
            global_best = result.best_solution;
            global_best_energy = result.best_energy;
        }
    }

    return annealing_result<State, T>{
        global_best, global_best_energy, global_best, global_best_energy,
        num_restarts * iterations_per_restart, total_accepted, true
    };
}

} // namespace stepanov::optimization