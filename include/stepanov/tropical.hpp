#ifndef STEPANOV_TROPICAL_HPP
#define STEPANOV_TROPICAL_HPP

#include <algorithm>
#include <concepts>
#include <limits>
#include <vector>
#include <cmath>
#include <ranges>

namespace stepanov::tropical {

// Tropical mathematics: where addition becomes min/max and multiplication becomes addition
// This creates a semiring structure that linearizes many nonlinear problems
// Applications: shortest paths, scheduling, computational biology, auction theory

// ============================================================================
// Core Tropical Semiring Types
// ============================================================================

template<typename T = double>
struct min_plus {
    // The min-plus semiring: (min, +) with identity elements ∞ and 0
    // Used for shortest path problems and optimization
    using value_type = T;
    T value;

    static constexpr T zero() {
        return std::numeric_limits<T>::infinity(); // additive identity
    }
    static constexpr T one() {
        return T(0); // multiplicative identity
    }

    constexpr min_plus() : value(zero()) {}
    constexpr explicit min_plus(T v) : value(v) {}

    // Tropical addition is minimum
    friend constexpr min_plus operator+(const min_plus& a, const min_plus& b) {
        return min_plus(std::min(a.value, b.value));
    }

    // Tropical multiplication is addition
    friend constexpr min_plus operator*(const min_plus& a, const min_plus& b) {
        if (a.value == zero() || b.value == zero()) return min_plus();
        return min_plus(a.value + b.value);
    }

    min_plus& operator+=(const min_plus& other) {
        value = std::min(value, other.value);
        return *this;
    }

    min_plus& operator*=(const min_plus& other) {
        if (value == zero() || other.value == zero()) {
            value = zero();
        } else {
            value += other.value;
        }
        return *this;
    }

    friend constexpr bool operator==(const min_plus& a, const min_plus& b) = default;
    friend constexpr auto operator<=>(const min_plus& a, const min_plus& b) = default;
};

template<typename T = double>
struct max_plus {
    // The max-plus semiring: (max, +) with identity elements -∞ and 0
    // Used for longest path problems and scheduling
    using value_type = T;
    T value;

    static constexpr T zero() {
        return -std::numeric_limits<T>::infinity(); // additive identity
    }
    static constexpr T one() {
        return T(0); // multiplicative identity
    }

    constexpr max_plus() : value(zero()) {}
    constexpr explicit max_plus(T v) : value(v) {}

    // Tropical addition is maximum
    friend constexpr max_plus operator+(const max_plus& a, const max_plus& b) {
        return max_plus(std::max(a.value, b.value));
    }

    // Tropical multiplication is addition
    friend constexpr max_plus operator*(const max_plus& a, const max_plus& b) {
        if (a.value == zero() || b.value == zero()) return max_plus();
        return max_plus(a.value + b.value);
    }

    max_plus& operator+=(const max_plus& other) {
        value = std::max(value, other.value);
        return *this;
    }

    max_plus& operator*=(const max_plus& other) {
        if (value == zero() || other.value == zero()) {
            value = zero();
        } else {
            value += other.value;
        }
        return *this;
    }

    friend constexpr bool operator==(const max_plus& a, const max_plus& b) = default;
    friend constexpr auto operator<=>(const max_plus& a, const max_plus& b) = default;
};

// Log semiring for probability computations in log space
template<typename T = double>
struct log_semiring {
    // Operations: (log-sum-exp, +) for numerical stability in probability
    using value_type = T;
    T log_value;

    static constexpr T zero() {
        return -std::numeric_limits<T>::infinity();
    }
    static constexpr T one() {
        return T(0);
    }

    constexpr log_semiring() : log_value(zero()) {}
    constexpr explicit log_semiring(T v, bool is_log = true)
        : log_value(is_log ? v : std::log(v)) {}

    // Log-sum-exp for numerical stability: log(exp(a) + exp(b))
    friend log_semiring operator+(const log_semiring& a, const log_semiring& b) {
        if (a.log_value == zero()) return b;
        if (b.log_value == zero()) return a;
        T max_val = std::max(a.log_value, b.log_value);
        return log_semiring(max_val + std::log1p(std::exp(-std::abs(a.log_value - b.log_value))));
    }

    friend log_semiring operator*(const log_semiring& a, const log_semiring& b) {
        if (a.log_value == zero() || b.log_value == zero()) return log_semiring();
        return log_semiring(a.log_value + b.log_value);
    }
};

// ============================================================================
// Tropical Matrix Operations
// ============================================================================

template<typename Semiring>
class tropical_matrix {
    using T = typename Semiring::value_type;
    std::vector<std::vector<Semiring>> data;
    size_t rows_, cols_;

public:
    tropical_matrix(size_t r, size_t c)
        : data(r, std::vector<Semiring>(c)), rows_(r), cols_(c) {}

    tropical_matrix(std::initializer_list<std::initializer_list<T>> init) {
        rows_ = init.size();
        cols_ = rows_ > 0 ? init.begin()->size() : 0;
        data.reserve(rows_);
        for (const auto& row : init) {
            data.emplace_back();
            data.back().reserve(cols_);
            for (auto val : row) {
                data.back().emplace_back(val);
            }
        }
    }

    Semiring& operator()(size_t i, size_t j) { return data[i][j]; }
    const Semiring& operator()(size_t i, size_t j) const { return data[i][j]; }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // Tropical matrix multiplication: the heart of path algorithms
    tropical_matrix operator*(const tropical_matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix dimensions incompatible");
        }

        tropical_matrix result(rows_, other.cols_);

        // This seemingly simple operation computes all shortest paths!
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                Semiring sum = Semiring();  // Initialize with zero (infinity for min-plus)
                for (size_t k = 0; k < cols_; ++k) {
                    sum = sum + (data[i][k] * other.data[k][j]);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    // Compute matrix power - finds k-step shortest paths
    tropical_matrix power(size_t n) const {
        if (rows_ != cols_) {
            throw std::invalid_argument("Matrix must be square");
        }

        if (n == 0) {
            // Identity matrix in tropical sense
            tropical_matrix result(rows_, cols_);
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    result(i, j) = (i == j) ? Semiring(Semiring::one()) : Semiring();
                }
            }
            return result;
        }

        tropical_matrix result = *this;
        tropical_matrix base = *this;
        n--;

        while (n > 0) {
            if (n & 1) result = result * base;
            base = base * base;
            n >>= 1;
        }
        return result;
    }

    // Kleene star: A* = I + A + A² + A³ + ...
    // Computes all shortest paths (transitive closure)
    tropical_matrix kleene_star() const {
        if (rows_ != cols_) {
            throw std::invalid_argument("Matrix must be square");
        }

        tropical_matrix result = *this;

        // Add identity
        for (size_t i = 0; i < rows_; ++i) {
            result(i, i) = result(i, i) + Semiring(Semiring::one());
        }

        // Floyd-Warshall in tropical algebra
        for (size_t k = 0; k < rows_; ++k) {
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    result(i, j) = result(i, j) + (result(i, k) * result(k, j));
                }
            }
        }

        return result;
    }

    // Find tropical eigenvalue (max cycle mean for max-plus)
    // This is the key to solving periodic scheduling problems
    auto tropical_eigenvalue() const -> T {
        if (rows_ != cols_) {
            throw std::invalid_argument("Matrix must be square");
        }

        // Karp's minimum mean-weight cycle algorithm
        const size_t n = rows_;
        std::vector<tropical_matrix> powers;
        powers.reserve(n + 1);

        // Identity
        tropical_matrix id(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                id(i, j) = (i == j) ? Semiring(Semiring::one()) : Semiring();
            }
        }
        powers.push_back(id);

        // Compute powers up to n
        for (size_t k = 1; k <= n; ++k) {
            powers.push_back(powers.back() * (*this));
        }

        // Find the maximum cycle mean
        T lambda = Semiring::zero();

        for (size_t i = 0; i < n; ++i) {
            T max_mean = Semiring::zero();

            for (size_t k = 0; k < n; ++k) {
                T numerator =
                    powers[n](i, i).value - powers[k](i, i).value;
                if (numerator != Semiring::zero()) {
                    T mean = numerator / (n - k);
                    if constexpr (std::same_as<Semiring, max_plus<T>>) {
                        max_mean = std::max(max_mean, mean);
                    } else {
                        max_mean = std::min(max_mean, mean);
                    }
                }
            }

            if constexpr (std::same_as<Semiring, max_plus<T>>) {
                lambda = std::max(lambda, max_mean);
            } else {
                lambda = std::min(lambda, max_mean);
            }
        }

        return lambda;
    }
};

// ============================================================================
// Tropical Polynomials
// ============================================================================

template<typename Semiring>
class tropical_polynomial {
    // Tropical polynomial: f(x) = max/min{aᵢ + i·x}
    // Represented as piecewise linear convex/concave function
    std::vector<std::pair<typename Semiring::value_type, int>> terms;

public:
    tropical_polynomial() = default;

    tropical_polynomial(std::initializer_list<typename Semiring::value_type> coeffs) {
        int degree = 0;
        for (auto coeff : coeffs) {
            if (coeff != Semiring::zero()) {
                terms.emplace_back(coeff, degree);
            }
            degree++;
        }
    }

    // Evaluate polynomial at point x
    Semiring operator()(typename Semiring::value_type x) const {
        if (terms.empty()) return Semiring();

        Semiring result;
        for (const auto& [coeff, degree] : terms) {
            Semiring term(coeff + degree * x);
            result = result + term;  // min or max depending on semiring
        }
        return result;
    }

    // Find tropical roots (breakpoints where the maximum/minimum changes)
    std::vector<typename Semiring::value_type> roots() const {
        std::vector<typename Semiring::value_type> breakpoints;

        // For each pair of terms, find where they intersect
        for (size_t i = 0; i < terms.size(); ++i) {
            for (size_t j = i + 1; j < terms.size(); ++j) {
                auto [a_i, d_i] = terms[i];
                auto [a_j, d_j] = terms[j];

                if (d_i != d_j) {
                    // Intersection point: aᵢ + dᵢ·x = aⱼ + dⱼ·x
                    typename Semiring::value_type x = (a_j - a_i) / (d_i - d_j);
                    breakpoints.push_back(x);
                }
            }
        }

        std::ranges::sort(breakpoints);
        breakpoints.erase(std::unique(breakpoints.begin(), breakpoints.end()),
                         breakpoints.end());
        return breakpoints;
    }

    // Newton polygon: the convex hull of the points (i, aᵢ)
    std::vector<std::pair<int, typename Semiring::value_type>> newton_polygon() const {
        if (terms.empty()) return {};

        // Sort by degree
        auto sorted_terms = terms;
        std::ranges::sort(sorted_terms, [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

        // Compute lower convex hull for min-plus, upper for max-plus
        std::vector<std::pair<int, typename Semiring::value_type>> hull;

        for (const auto& [coeff, degree] : sorted_terms) {
            while (hull.size() >= 2) {
                auto [x1, y1] = hull[hull.size() - 2];
                auto [x2, y2] = hull[hull.size() - 1];

                // Check if we should remove the last point
                double slope1 = double(y2 - y1) / (x2 - x1);
                double slope2 = double(coeff - y2) / (degree - x2);

                bool should_remove;
                if constexpr (std::same_as<Semiring, min_plus<typename Semiring::value_type>>) {
                    should_remove = slope2 < slope1;  // Lower hull
                } else {
                    should_remove = slope2 > slope1;  // Upper hull
                }

                if (should_remove) {
                    hull.pop_back();
                } else {
                    break;
                }
            }
            hull.emplace_back(degree, coeff);
        }

        return hull;
    }

    // Tropical convolution (useful for dynamic programming)
    tropical_polynomial operator*(const tropical_polynomial& other) const {
        tropical_polynomial result;

        for (const auto& [a, i] : terms) {
            for (const auto& [b, j] : other.terms) {
                typename Semiring::value_type coeff = a + b;
                int degree = i + j;

                // Check if we already have this degree
                auto it = std::find_if(result.terms.begin(), result.terms.end(),
                    [degree](const auto& term) { return term.second == degree; });

                if (it != result.terms.end()) {
                    // Combine with existing term
                    if constexpr (std::same_as<Semiring, min_plus<typename Semiring::value_type>>) {
                        it->first = std::min(it->first, coeff);
                    } else {
                        it->first = std::max(it->first, coeff);
                    }
                } else {
                    result.terms.emplace_back(coeff, degree);
                }
            }
        }

        return result;
    }
};

// ============================================================================
// Applications and Algorithms
// ============================================================================

// Solve shortest path problem using tropical linear algebra
template<typename T>
std::vector<T> shortest_paths(const std::vector<std::vector<T>>& graph, size_t source) {
    size_t n = graph.size();
    tropical_matrix<min_plus<T>> adj_matrix(n, n);

    // Convert graph to tropical matrix
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            adj_matrix(i, j) = min_plus<T>(graph[i][j]);
        }
    }

    // Compute transitive closure
    auto closure = adj_matrix.kleene_star();

    // Extract shortest paths from source
    std::vector<T> distances(n);
    for (size_t i = 0; i < n; ++i) {
        distances[i] = closure(source, i).value;
    }

    return distances;
}

// Viterbi algorithm using tropical algebra
template<typename T>
struct viterbi_decoder {
    using tropical_t = max_plus<T>;

    // Find most likely sequence through HMM
    std::vector<int> decode(
        const std::vector<std::vector<T>>& transition,  // log probabilities
        const std::vector<std::vector<T>>& emission,    // log probabilities
        const std::vector<T>& initial,                  // log probabilities
        const std::vector<int>& observations) {

        size_t n_states = transition.size();
        size_t n_obs = observations.size();

        // Dynamic programming table in tropical algebra
        std::vector<std::vector<tropical_t>> dp(n_obs, std::vector<tropical_t>(n_states));
        std::vector<std::vector<int>> backtrack(n_obs, std::vector<int>(n_states));

        // Initialize
        for (size_t s = 0; s < n_states; ++s) {
            dp[0][s] = tropical_t(initial[s] + emission[s][observations[0]]);
        }

        // Forward pass using tropical operations
        for (size_t t = 1; t < n_obs; ++t) {
            for (size_t s = 0; s < n_states; ++s) {
                tropical_t best;
                int best_prev = 0;

                for (size_t prev = 0; prev < n_states; ++prev) {
                    tropical_t score = dp[t-1][prev] *
                        tropical_t(transition[prev][s]) *
                        tropical_t(emission[s][observations[t]]);

                    if (score.value > best.value) {
                        best = score;
                        best_prev = prev;
                    }
                }

                dp[t][s] = best;
                backtrack[t][s] = best_prev;
            }
        }

        // Find best final state
        int best_final = 0;
        tropical_t best_score;
        for (size_t s = 0; s < n_states; ++s) {
            if (dp[n_obs-1][s].value > best_score.value) {
                best_score = dp[n_obs-1][s];
                best_final = s;
            }
        }

        // Backtrack to find path
        std::vector<int> path(n_obs);
        path[n_obs-1] = best_final;
        for (int t = n_obs-2; t >= 0; --t) {
            path[t] = backtrack[t+1][path[t+1]];
        }

        return path;
    }
};

// Optimal scheduling using max-plus algebra
template<typename T>
struct scheduler {
    using tropical_t = max_plus<T>;

    // Find critical path and schedule
    struct schedule_result {
        T cycle_time;           // Minimum achievable period
        std::vector<T> start_times;  // When each task should start
    };

    schedule_result compute_schedule(
        const std::vector<std::vector<T>>& precedence_times,  // Time from task i to j
        const std::vector<T>& processing_times) {             // Duration of each task

        size_t n = processing_times.size();
        tropical_matrix<tropical_t> M(n, n);

        // Build precedence matrix
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (precedence_times[i][j] != std::numeric_limits<T>::infinity()) {
                    M(i, j) = tropical_t(processing_times[i] + precedence_times[i][j]);
                }
            }
        }

        // Find eigenvalue (critical cycle time)
        T lambda = M.tropical_eigenvalue();

        // Compute eigenvector (start times)
        std::vector<T> v(n, T(0));

        // Power iteration in tropical algebra
        for (int iter = 0; iter < 100; ++iter) {
            std::vector<T> v_new(n);

            for (size_t i = 0; i < n; ++i) {
                tropical_t sum;
                for (size_t j = 0; j < n; ++j) {
                    sum = sum + (M(j, i) * tropical_t(v[j]));
                }
                v_new[i] = sum.value - lambda;  // Normalize by eigenvalue
            }

            // Check convergence
            bool converged = true;
            for (size_t i = 0; i < n; ++i) {
                if (std::abs(v_new[i] - v[i]) > 1e-9) {
                    converged = false;
                    break;
                }
            }

            v = v_new;
            if (converged) break;
        }

        return {lambda, v};
    }
};

} // namespace stepanov::tropical

#endif // STEPANOV_TROPICAL_HPP