#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <functional>

// Include all integrated Stepanov components
#include "../include/stepanov/integration/integrators.hpp"
#include "../include/stepanov/intervals/interval_set.hpp"
#include "../include/stepanov/hashing/algebraic_hash.hpp"
#include "../include/stepanov/statistics/accumulators.hpp"
#include "../include/stepanov/concepts.hpp"
#include "../include/stepanov/math.hpp"

using namespace stepanov;

// Showcase: Monte Carlo integration with hash-based sampling and statistics
class monte_carlo_integrator {
private:
    hashing::golden_ratio_hash<std::uint64_t> hasher_;
    statistics::welford_accumulator<double> stats_;
    statistics::minmax_accumulator<double> bounds_;

public:
    monte_carlo_integrator() : hasher_(52) {}  // 52-bit precision for double mantissa

    template<typename F>
    integration::integration_result<double> integrate(
        F&& f,
        const intervals::interval_set<double>& domain,
        std::size_t samples
    ) {
        if (domain.empty()) {
            return {0.0, 0.0, 0};
        }

        // Compute total measure of domain
        double total_measure = 0.0;
        std::vector<double> cumulative_measures;

        for (const auto& interval : domain) {
            if (!interval.is_empty()) {
                total_measure += interval.upper() - interval.lower();
                cumulative_measures.push_back(total_measure);
            }
        }

        // Monte Carlo sampling
        stats_.reset();
        bounds_.reset();

        for (std::size_t i = 0; i < samples; ++i) {
            // Use hash for quasi-random sampling
            double u = static_cast<double>(hasher_(i)) / static_cast<double>(1ULL << 52);
            u *= total_measure;

            // Find which interval contains this point
            auto it = std::lower_bound(cumulative_measures.begin(),
                                       cumulative_measures.end(), u);
            std::size_t interval_idx = std::distance(cumulative_measures.begin(), it);

            // Map to actual point in interval
            double offset = (interval_idx > 0) ? cumulative_measures[interval_idx - 1] : 0.0;
            double local_u = u - offset;

            auto interval_it = domain.begin();
            std::advance(interval_it, interval_idx);
            double x = interval_it->lower() + local_u;

            // Evaluate function
            double y = f(x);
            stats_ += y;
            bounds_ += y;
        }

        double mean = stats_.mean();
        double variance = stats_.variance();
        double error = std::sqrt(variance / samples) * total_measure;

        return {mean * total_measure, error, samples};
    }

    double mean() const { return stats_.mean(); }
    double variance() const { return stats_.variance(); }
    double min_value() const { return bounds_.min(); }
    double max_value() const { return bounds_.max(); }
};

// Showcase: Adaptive integration with interval refinement
class adaptive_interval_integrator {
private:
    integration::simpson_integrator<double> simpson_;
    statistics::histogram_accumulator<double> error_histogram_;

public:
    adaptive_interval_integrator()
        : error_histogram_(-16.0, 0.0, 32) {}  // Log10 error bins

    integration::integration_result<double> integrate_with_refinement(
        std::function<double(double)> f,
        intervals::interval_set<double>& domain,
        double tolerance
    ) {
        statistics::kahan_accumulator<double> total_integral;
        statistics::kahan_accumulator<double> total_error;
        std::size_t total_evaluations = 0;

        // Keep refining intervals with large errors
        bool refined = true;
        int iterations = 0;

        while (refined && iterations < 10) {
            refined = false;
            intervals::interval_set<double> refined_domain;

            for (const auto& interval : domain) {
                if (!interval.is_empty()) {
                    auto result = simpson_(f, interval.lower(), interval.upper(), tolerance);

                    // Log error for analysis
                    if (result.error_estimate > 0) {
                        error_histogram_ += std::log10(result.error_estimate);
                    }

                    if (result.error_estimate > tolerance) {
                        // Refine this interval
                        double mid = (interval.lower() + interval.upper()) / 2.0;
                        refined_domain.insert(interval.lower(), mid);
                        refined_domain.insert(mid, interval.upper());
                        refined = true;
                    } else {
                        // Keep this interval
                        refined_domain.insert(interval);
                        total_integral += result.value;
                        total_error += result.error_estimate;
                        total_evaluations += result.evaluations;
                    }
                }
            }

            domain = refined_domain;
            iterations++;
        }

        // Final integration on refined domain
        if (refined) {
            for (const auto& interval : domain) {
                if (!interval.is_empty()) {
                    auto result = simpson_(f, interval.lower(), interval.upper(), tolerance);
                    total_integral += result.value;
                    total_error += result.error_estimate;
                    total_evaluations += result.evaluations;
                }
            }
        }

        return {total_integral.eval(), total_error.eval(), total_evaluations};
    }

    void print_error_distribution() const {
        std::cout << "Error distribution (log10):" << std::endl;
        for (std::size_t i = 0; i < error_histogram_.num_bins(); ++i) {
            if (error_histogram_.bin_count(i) > 0) {
                std::cout << "[" << error_histogram_.bin_lower(i)
                          << ", " << error_histogram_.bin_upper(i) << "]: "
                          << error_histogram_.bin_count(i) << std::endl;
            }
        }
    }
};

// Showcase: Function fingerprinting with rolling hash
class function_fingerprint {
private:
    hashing::polynomial_hash<std::uint64_t> roller_;
    std::size_t samples_;

public:
    function_fingerprint(std::size_t samples = 1000)
        : samples_(samples) {}

    template<typename F>
    std::uint64_t compute(F&& f, double a, double b) {
        roller_.clear();
        double h = (b - a) / samples_;

        for (std::size_t i = 0; i <= samples_; ++i) {
            double x = a + i * h;
            double y = f(x);

            // Quantize to integer for hashing
            std::uint64_t quantized = static_cast<std::uint64_t>(
                (y + 1000.0) * 1e6  // Shift and scale
            );
            roller_.push_back(quantized);
        }

        return roller_.value();
    }

    template<typename F1, typename F2>
    double similarity(F1&& f1, F2&& f2, double a, double b) {
        std::uint64_t hash1 = compute(f1, a, b);
        std::uint64_t hash2 = compute(f2, a, b);

        // Hamming distance as similarity metric
        std::uint64_t diff = hash1 ^ hash2;
        int distance = __builtin_popcountll(diff);

        return 1.0 - static_cast<double>(distance) / 64.0;
    }
};

int main() {
    std::cout << "=== Stepanov Library Integration Showcase ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    // 1. Complex domain with multiple disjoint intervals
    std::cout << "\n1. Multi-domain Integration with Statistics" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    intervals::interval_set<double> domain;
    domain.insert(intervals::interval<double>::closed(-2, -1));
    domain.insert(intervals::interval<double>::open(0, 1));
    domain.insert(intervals::interval<double>::closed(2, 3));

    // Test function: oscillating exponential
    auto f = [](double x) {
        return std::sin(5 * x) * std::exp(-x * x / 2);
    };

    // Compare different integration methods
    std::cout << "Function: sin(5x) * exp(-x²/2) over disjoint intervals" << std::endl;

    // Method 1: Adaptive Simpson
    integration::simpson_integrator<double> simpson;
    statistics::composite_accumulator<double> simpson_stats;

    for (const auto& interval : domain) {
        if (!interval.is_empty()) {
            auto result = simpson(f, interval.lower(), interval.upper(), 1e-9);
            simpson_stats += result.value;
        }
    }

    std::cout << "\nSimpson integration:" << std::endl;
    std::cout << "  Total: " << simpson_stats.sum() << std::endl;
    std::cout << "  Mean per interval: " << simpson_stats.mean() << std::endl;

    // Method 2: Monte Carlo with hash-based sampling
    monte_carlo_integrator mc;
    auto mc_result = mc.integrate(f, domain, 100000);

    std::cout << "\nMonte Carlo integration (100k samples):" << std::endl;
    std::cout << "  Result: " << mc_result.value << std::endl;
    std::cout << "  Error estimate: " << mc_result.error_estimate << std::endl;
    std::cout << "  Function range: [" << mc.min_value() << ", " << mc.max_value() << "]" << std::endl;

    // 2. Adaptive interval refinement
    std::cout << "\n2. Adaptive Interval Refinement" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    intervals::interval_set<double> initial_domain;
    initial_domain.insert(intervals::interval<double>::closed(-3, 3));

    adaptive_interval_integrator adaptive;
    auto adaptive_result = adaptive.integrate_with_refinement(f, initial_domain, 1e-6);

    std::cout << "Initial intervals: 1" << std::endl;
    std::cout << "Final intervals: " << initial_domain.size() << std::endl;
    std::cout << "Result: " << adaptive_result.value << std::endl;
    std::cout << "Total evaluations: " << adaptive_result.evaluations << std::endl;
    adaptive.print_error_distribution();

    // 3. Function fingerprinting
    std::cout << "\n3. Function Fingerprinting with Rolling Hash" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    function_fingerprint fingerprinter(500);

    auto f1 = [](double x) { return std::sin(x); };
    auto f2 = [](double x) { return std::sin(x + 0.1); };
    auto f3 = [](double x) { return std::cos(x); };
    auto f4 = [](double x) { return x * x; };

    std::cout << "Function similarities on [-π, π]:" << std::endl;
    std::cout << "  sin(x) vs sin(x+0.1): "
              << fingerprinter.similarity(f1, f2, -M_PI, M_PI) << std::endl;
    std::cout << "  sin(x) vs cos(x): "
              << fingerprinter.similarity(f1, f3, -M_PI, M_PI) << std::endl;
    std::cout << "  sin(x) vs x²: "
              << fingerprinter.similarity(f1, f4, -M_PI, M_PI) << std::endl;

    // 4. Statistical analysis of integration methods
    std::cout << "\n4. Statistical Comparison of Integration Methods" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // Test on a difficult integral: ∫ 1/(1+x²) from -∞ to ∞ = π
    auto arctangent_integrand = [](double x) {
        return 1.0 / (1.0 + x * x);
    };

    // Use tanh-sinh for infinite integral
    integration::double_exponential_integrator<double> tanh_sinh;
    auto exact_result = tanh_sinh(arctangent_integrand,
                                  -std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(),
                                  1e-12);

    std::cout << "Testing ∫ 1/(1+x²) dx from -∞ to ∞" << std::endl;
    std::cout << "Exact value: π = " << M_PI << std::endl;
    std::cout << "Tanh-sinh result: " << exact_result.value << std::endl;
    std::cout << "Error: " << std::abs(exact_result.value - M_PI) << std::endl;
    std::cout << "Evaluations: " << exact_result.evaluations << std::endl;

    // 5. Combining all components
    std::cout << "\n5. Combined Analysis Pipeline" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    // Create a piecewise function over intervals
    auto piecewise = [](double x) {
        if (x < -1) return std::exp(x);
        else if (x < 1) return 1.0 + x + x * x;
        else return std::sin(2 * x);
    };

    // Define analysis domain
    intervals::interval_set<double> analysis_domain;
    analysis_domain.insert(intervals::interval<double>::closed(-3, -1));
    analysis_domain.insert(intervals::interval<double>::closed(-1, 1));
    analysis_domain.insert(intervals::interval<double>::closed(1, 3));

    // Compute integrals with statistics
    statistics::composite_accumulator<double> integral_stats;
    hashing::hash_combiner combiner;
    std::uint64_t combined_hash = 0;

    for (const auto& interval : analysis_domain) {
        if (!interval.is_empty()) {
            // Integrate
            auto result = simpson(piecewise, interval.lower(), interval.upper(), 1e-9);
            integral_stats += result.value;

            // Hash the result
            std::uint64_t result_bits = std::bit_cast<std::uint64_t>(result.value);
            combined_hash = combiner.combine(combined_hash, result_bits);

            std::cout << "Interval [" << interval.lower() << ", " << interval.upper()
                      << "]: " << result.value << std::endl;
        }
    }

    std::cout << "\nAnalysis summary:" << std::endl;
    std::cout << "  Total integral: " << integral_stats.sum() << std::endl;
    std::cout << "  Mean value: " << integral_stats.mean() << std::endl;
    std::cout << "  Std deviation: " << integral_stats.stddev() << std::endl;
    std::cout << "  Combined hash: " << std::hex << combined_hash << std::dec << std::endl;

    std::cout << "\n=== Showcase Complete ===" << std::endl;
    return 0;
}