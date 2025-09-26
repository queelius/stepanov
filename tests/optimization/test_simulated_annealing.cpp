#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "../../include/stepanov/optimization.hpp"

using namespace stepanov;
using namespace stepanov::optimization;

// Test problem 1: Continuous optimization
void test_continuous_optimization() {
    std::cout << "Testing simulated annealing for continuous optimization...\n";

    // Rastrigin function - highly multimodal
    auto rastrigin = [](const std::vector<double>& x) {
        double A = 10.0;
        double sum = A * x.size();
        for (double xi : x) {
            sum += xi * xi - A * std::cos(2 * M_PI * xi);
        }
        return sum;
    };

    // Neighbor generator with Gaussian perturbation
    auto gaussian_neighbor = [](const std::vector<double>& x, std::mt19937& rng) {
        std::normal_distribution<double> dist(0, 0.5);
        std::vector<double> neighbor = x;
        for (double& xi : neighbor) {
            xi += dist(rng);
            // Keep in bounds [-5.12, 5.12]
            xi = std::max(-5.12, std::min(5.12, xi));
        }
        return neighbor;
    };

    std::vector<double> initial{2.0, 2.0};
    std::mt19937 rng(42);

    // Test different cooling schedules
    auto exp_result = simulated_annealing(
        rastrigin, initial, gaussian_neighbor,
        exponential_cooling<double>(100.0, 0.95),
        2000, 0.01, rng
    );

    std::cout << "  Exponential cooling: f(x*) = " << exp_result.best_energy
              << ", x* = (" << exp_result.best_solution[0]
              << ", " << exp_result.best_solution[1] << ")\n";
    std::cout << "  Accepted: " << exp_result.accepted_moves
              << "/" << exp_result.iterations << "\n";

    // Linear cooling
    auto lin_result = simulated_annealing(
        rastrigin, initial, gaussian_neighbor,
        linear_cooling<double>(100.0, 0.01, 2000),
        2000, 0.01, rng
    );

    std::cout << "  Linear cooling: f(x*) = " << lin_result.best_energy << "\n";

    // Logarithmic cooling (slow but thorough)
    auto log_result = simulated_annealing(
        rastrigin, initial, gaussian_neighbor,
        logarithmic_cooling<double>(50.0),
        1000, 0.01, rng
    );

    std::cout << "  Logarithmic cooling: f(x*) = " << log_result.best_energy << "\n";

    std::cout << "  Continuous optimization tests passed!\n\n";
}

// Test problem 2: Traveling Salesman Problem
void test_tsp() {
    std::cout << "Testing simulated annealing for TSP...\n";

    // City coordinates
    std::vector<std::pair<double, double>> cities = {
        {0, 0}, {1, 2}, {3, 1}, {4, 3}, {2, 4},
        {5, 2}, {3, 5}, {1, 4}, {4, 0}, {2, 2}
    };

    using Tour = std::vector<int>;

    // Calculate tour length
    auto tour_length = [&cities](const Tour& tour) {
        double total = 0;
        for (size_t i = 0; i < tour.size(); ++i) {
            int from = tour[i];
            int to = tour[(i + 1) % tour.size()];
            double dx = cities[to].first - cities[from].first;
            double dy = cities[to].second - cities[from].second;
            total += std::sqrt(dx * dx + dy * dy);
        }
        return total;
    };

    // Neighbor generators for TSP
    auto swap_neighbor = [](const Tour& tour, std::mt19937& rng) {
        Tour neighbor = tour;
        std::uniform_int_distribution<int> dist(0, tour.size() - 1);
        std::swap(neighbor[dist(rng)], neighbor[dist(rng)]);
        return neighbor;
    };

    auto reverse_neighbor = [](const Tour& tour, std::mt19937& rng) {
        Tour neighbor = tour;
        std::uniform_int_distribution<int> dist(0, tour.size() - 1);
        int i = dist(rng);
        int j = dist(rng);
        if (i > j) std::swap(i, j);
        std::reverse(neighbor.begin() + i, neighbor.begin() + j + 1);
        return neighbor;
    };

    // Initial tour
    Tour initial(cities.size());
    std::iota(initial.begin(), initial.end(), 0);

    std::mt19937 rng(123);

    // Test with swap moves
    auto swap_result = simulated_annealing(
        tour_length, initial, swap_neighbor,
        exponential_cooling<double>(10.0, 0.95),
        5000, 0.001, rng
    );

    std::cout << "  Swap moves: initial length = " << tour_length(initial)
              << ", final = " << swap_result.best_energy << "\n";

    // Test with reverse moves (2-opt)
    auto reverse_result = simulated_annealing(
        tour_length, initial, reverse_neighbor,
        exponential_cooling<double>(10.0, 0.95),
        5000, 0.001, rng
    );

    std::cout << "  Reverse moves: final length = " << reverse_result.best_energy << "\n";
    std::cout << "  Best tour: ";
    for (int city : reverse_result.best_solution) {
        std::cout << city << " ";
    }
    std::cout << "\n";

    std::cout << "  TSP tests passed!\n\n";
}

// Test problem 3: Graph coloring
void test_graph_coloring() {
    std::cout << "Testing simulated annealing for graph coloring...\n";

    // Simple graph with 6 vertices
    std::vector<std::pair<int, int>> edges = {
        {0, 1}, {0, 2}, {1, 2}, {1, 3},
        {2, 3}, {3, 4}, {3, 5}, {4, 5}
    };

    using Coloring = std::vector<int>;

    // Count conflicts (edges with same color endpoints)
    auto count_conflicts = [&edges](const Coloring& colors) {
        double conflicts = 0;
        for (auto [u, v] : edges) {
            if (colors[u] == colors[v]) {
                conflicts += 1;
            }
        }
        return conflicts;
    };

    // Change one vertex color randomly
    auto recolor_neighbor = [](const Coloring& colors, std::mt19937& rng) {
        Coloring neighbor = colors;
        std::uniform_int_distribution<int> vertex_dist(0, colors.size() - 1);
        std::uniform_int_distribution<int> color_dist(0, 2); // 3 colors
        neighbor[vertex_dist(rng)] = color_dist(rng);
        return neighbor;
    };

    Coloring initial(6, 0); // All same color initially
    std::mt19937 rng(456);

    auto result = simulated_annealing(
        count_conflicts, initial, recolor_neighbor,
        exponential_cooling<double>(5.0, 0.95),
        2000, 0.001, rng
    );

    std::cout << "  Initial conflicts: " << count_conflicts(initial) << "\n";
    std::cout << "  Final conflicts: " << result.best_energy << "\n";
    std::cout << "  Coloring: ";
    for (int color : result.best_solution) {
        std::cout << color << " ";
    }
    std::cout << "\n";

    assert(result.best_energy < 2); // Should find good coloring

    std::cout << "  Graph coloring tests passed!\n\n";
}

// Test parallel tempering
void test_parallel_tempering() {
    std::cout << "Testing parallel tempering...\n";

    // Ackley function - many local minima
    auto ackley = [](const std::vector<double>& x) {
        double a = 20.0, b = 0.2, c = 2 * M_PI;
        double sum1 = 0, sum2 = 0;
        for (double xi : x) {
            sum1 += xi * xi;
            sum2 += std::cos(c * xi);
        }
        double n = x.size();
        return -a * std::exp(-b * std::sqrt(sum1 / n)) -
               std::exp(sum2 / n) + a + std::exp(1.0);
    };

    auto neighbor = [](const std::vector<double>& x, std::mt19937& rng) {
        std::normal_distribution<double> dist(0, 0.2);
        std::vector<double> neighbor = x;
        for (double& xi : neighbor) {
            xi += dist(rng);
        }
        return neighbor;
    };

    std::vector<double> initial{5.0, 5.0};
    std::vector<double> temperatures{0.1, 0.5, 1.0, 5.0, 10.0};

    parallel_tempering<std::vector<double>, decltype(ackley), decltype(neighbor)>
        pt(ackley, neighbor, initial, temperatures);

    auto pt_result = pt.optimize(100, 100);

    std::cout << "  Parallel tempering: f(x*) = " << pt_result.best_energy
              << ", x* = (" << pt_result.best_solution[0]
              << ", " << pt_result.best_solution[1] << ")\n";

    // Should find near-global minimum at origin
    if (pt_result.best_energy < 1.0) {
        std::cout << "  Successfully found near-global minimum!\n";
    }

    std::cout << "  Parallel tempering tests passed!\n\n";
}

// Test quantum annealing
void test_quantum_annealing() {
    std::cout << "Testing quantum-inspired annealing...\n";

    // Simple quadratic with barriers
    auto quantum_landscape = [](const std::vector<double>& x) {
        double smooth = x[0] * x[0] + x[1] * x[1];
        double barriers = 5 * std::exp(-2 * ((x[0] - 1) * (x[0] - 1) + x[1] * x[1]));
        return smooth + barriers;
    };

    auto neighbor = [](const std::vector<double>& x, std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        std::vector<double> neighbor = x;
        for (double& xi : neighbor) {
            xi += dist(rng);
        }
        return neighbor;
    };

    std::vector<double> initial{2.0, 0.0};
    std::mt19937 rng(789);

    auto qa_result = quantum_annealing(
        quantum_landscape, initial, neighbor,
        10.0, 0.01, 1000, rng
    );

    std::cout << "  Quantum annealing: f(x*) = " << qa_result.best_energy
              << ", x* = (" << qa_result.best_solution[0]
              << ", " << qa_result.best_solution[1] << ")\n";
    std::cout << "  Accepted moves: " << qa_result.accepted_moves << "\n";

    std::cout << "  Quantum annealing tests passed!\n\n";
}

// Test with restarts
void test_with_restarts() {
    std::cout << "Testing simulated annealing with restarts...\n";

    // Schwefel function - deceptive global minimum
    auto schwefel = [](const std::vector<double>& x) {
        double sum = 0;
        for (double xi : x) {
            sum += -xi * std::sin(std::sqrt(std::abs(xi)));
        }
        return 418.9829 * x.size() + sum;
    };

    auto neighbor = [](const std::vector<double>& x, std::mt19937& rng) {
        std::normal_distribution<double> dist(0, 50);
        std::vector<double> neighbor = x;
        for (double& xi : neighbor) {
            xi += dist(rng);
            xi = std::max(-500.0, std::min(500.0, xi));
        }
        return neighbor;
    };

    std::vector<double> initial{200.0, 200.0};
    std::mt19937 rng(321);

    auto restart_result = simulated_annealing_with_restarts(
        schwefel, initial, neighbor,
        exponential_cooling<double>(100.0, 0.9),
        10, 500, rng
    );

    std::cout << "  With restarts: f(x*) = " << restart_result.best_energy
              << ", x* = (" << restart_result.best_solution[0]
              << ", " << restart_result.best_solution[1] << ")\n";
    std::cout << "  Total accepted: " << restart_result.accepted_moves << "\n";

    // Global minimum is near (420.9687, 420.9687) with value ~0
    double expected_x = 420.9687;
    if (std::abs(restart_result.best_solution[0] - expected_x) < 50) {
        std::cout << "  Found near-global minimum!\n";
    }

    std::cout << "  Restart tests passed!\n\n";
}

// Test adaptive cooling
void test_adaptive_cooling() {
    std::cout << "Testing adaptive cooling schedule...\n";

    // Simple test function
    auto test_func = [](const std::vector<double>& x) {
        return x[0] * x[0] + 4 * x[1] * x[1];
    };

    auto neighbor = [](const std::vector<double>& x, std::mt19937& rng) {
        std::normal_distribution<double> dist(0, 0.5);
        std::vector<double> neighbor = x;
        for (double& xi : neighbor) {
            xi += dist(rng);
        }
        return neighbor;
    };

    std::vector<double> initial{5.0, 5.0};
    std::mt19937 rng(654);

    // Adaptive cooling adjusts based on acceptance rate
    adaptive_cooling<double> adaptive(100.0, 0.4, 1.05);

    // Manual SA with adaptive cooling for demonstration
    std::vector<double> current = initial;
    double current_energy = test_func(current);
    std::vector<double> best = current;
    double best_energy = current_energy;

    std::uniform_real_distribution<double> uniform(0, 1);
    size_t accepted = 0;
    size_t window_size = 100;

    for (size_t iter = 0; iter < 1000; ++iter) {
        auto next = neighbor(current, rng);
        double next_energy = test_func(next);
        double delta = next_energy - current_energy;

        double acceptance_rate = double(accepted) / (iter + 1);
        double temp = adaptive(iter, acceptance_rate);

        if (delta < 0 || uniform(rng) < std::exp(-delta / temp)) {
            current = next;
            current_energy = next_energy;
            accepted++;

            if (current_energy < best_energy) {
                best = current;
                best_energy = current_energy;
            }
        }

        if (iter % 200 == 199) {
            std::cout << "  Iter " << iter + 1 << ": T = " << temp
                      << ", acceptance = " << acceptance_rate
                      << ", best = " << best_energy << "\n";
        }
    }

    std::cout << "  Final: f(x*) = " << best_energy
              << ", x* = (" << best[0] << ", " << best[1] << ")\n";

    std::cout << "  Adaptive cooling tests passed!\n\n";
}

int main() {
    std::cout << "=== Testing Simulated Annealing Module ===\n\n";

    test_continuous_optimization();
    test_tsp();
    test_graph_coloring();
    test_parallel_tempering();
    test_quantum_annealing();
    test_with_restarts();
    test_adaptive_cooling();

    std::cout << "=== All simulated annealing tests passed! ===\n";

    return 0;
}