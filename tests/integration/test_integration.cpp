#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>

// Include the new integrated components
#include "../include/stepanov/integration/integrators.hpp"
#include "../include/stepanov/intervals/interval_set.hpp"
#include "../include/stepanov/hashing/algebraic_hash.hpp"
#include "../include/stepanov/statistics/accumulators.hpp"

// Include existing Stepanov components
#include "../include/stepanov/concepts.hpp"
#include "../include/stepanov/math.hpp"

using namespace stepanov;

// Test 1: Integration with statistical accumulation
void test_integration_with_statistics() {
    std::cout << "\n=== Test 1: Integration with Statistical Accumulation ===" << std::endl;

    // Define a function to integrate
    auto f = [](double x) { return std::sin(x) * std::exp(-x * x / 2); };

    // Use adaptive integration
    integration::simpson_integrator<double> integrator;
    statistics::composite_accumulator<double> stats;

    // Perform multiple integrations with different tolerances
    std::vector<double> tolerances = {1e-3, 1e-6, 1e-9, 1e-12};

    for (double tol : tolerances) {
        auto result = integrator(f, -3.0, 3.0, tol);
        stats += result.value;

        std::cout << "Tolerance: " << std::scientific << tol
                  << ", Result: " << result.value
                  << ", Error: " << result.error_estimate
                  << ", Evaluations: " << result.evaluations << std::endl;
    }

    std::cout << "\nStatistics of integration results:" << std::endl;
    std::cout << "Mean: " << stats.mean() << std::endl;
    std::cout << "Std Dev: " << stats.stddev() << std::endl;
    std::cout << "Range: [" << stats.min() << ", " << stats.max() << "]" << std::endl;
}

// Test 2: Interval sets with hashing
void test_intervals_with_hashing() {
    std::cout << "\n=== Test 2: Interval Sets with Hashing ===" << std::endl;

    // Create interval sets
    intervals::interval_set<int> set1;
    set1.insert(intervals::interval<int>::closed(1, 10));
    set1.insert(intervals::interval<int>::closed(20, 30));
    set1.insert(intervals::interval<int>::closed(40, 50));

    intervals::interval_set<int> set2;
    set2.insert(intervals::interval<int>::closed(5, 15));
    set2.insert(intervals::interval<int>::closed(25, 35));

    // Compute set operations
    auto union_set = set1.union_with(set2);
    auto intersect_set = set1.intersect_with(set2);

    // Hash the interval sets for comparison
    hashing::fnv1a_hash<std::uint64_t> hasher;

    // Hash each interval endpoint
    for (const auto& interval : union_set) {
        if (!interval.is_empty()) {
            auto lower = interval.lower();
            auto upper = interval.upper();
            const std::uint8_t* lower_bytes = reinterpret_cast<const std::uint8_t*>(&lower);
            const std::uint8_t* upper_bytes = reinterpret_cast<const std::uint8_t*>(&upper);
            hasher.update(lower_bytes, lower_bytes + sizeof(lower));
            hasher.update(upper_bytes, upper_bytes + sizeof(upper));
        }
    }

    std::cout << "Union set hash: " << std::hex << hasher.value() << std::dec << std::endl;

    hasher.reset();
    for (const auto& interval : intersect_set) {
        if (!interval.is_empty()) {
            auto lower = interval.lower();
            auto upper = interval.upper();
            const std::uint8_t* lower_bytes = reinterpret_cast<const std::uint8_t*>(&lower);
            const std::uint8_t* upper_bytes = reinterpret_cast<const std::uint8_t*>(&upper);
            hasher.update(lower_bytes, lower_bytes + sizeof(lower));
            hasher.update(upper_bytes, upper_bytes + sizeof(upper));
        }
    }

    std::cout << "Intersection set hash: " << std::hex << hasher.value() << std::dec << std::endl;

    std::cout << "Union set size: " << union_set.size() << std::endl;
    std::cout << "Intersection set size: " << intersect_set.size() << std::endl;
}

// Test 3: Polynomial rolling hash with statistical analysis
void test_rolling_hash_statistics() {
    std::cout << "\n=== Test 3: Rolling Hash with Statistical Analysis ===" << std::endl;

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    std::vector<std::uint64_t> data(1000);
    for (auto& val : data) {
        val = dis(gen);
    }

    // Use polynomial rolling hash
    hashing::polynomial_hash<std::uint64_t> roller;
    statistics::welford_accumulator<double> hash_stats;

    // Compute rolling hash over sliding windows
    const std::size_t window_size = 50;

    for (std::size_t i = 0; i < window_size; ++i) {
        roller.push_back(data[i]);
    }

    for (std::size_t i = window_size; i < data.size(); ++i) {
        roller.slide(data[i - window_size], data[i]);
        hash_stats += static_cast<double>(roller.value());
    }

    std::cout << "Rolling hash statistics:" << std::endl;
    std::cout << "Mean hash value: " << hash_stats.mean() << std::endl;
    std::cout << "Hash std dev: " << hash_stats.stddev() << std::endl;
    std::cout << "Number of windows: " << hash_stats.count() << std::endl;

    // Check distribution quality
    double expected_mean = static_cast<double>(hashing::polynomial_hash<std::uint64_t>::default_mod()) / 2.0;
    double deviation_ratio = std::abs(hash_stats.mean() - expected_mean) / expected_mean;
    std::cout << "Deviation from expected mean: " << deviation_ratio * 100 << "%" << std::endl;
}

// Test 4: Integration over interval sets
void test_integration_over_intervals() {
    std::cout << "\n=== Test 4: Integration over Interval Sets ===" << std::endl;

    // Define a piecewise function
    auto f = [](double x) {
        if (x < 0) return std::exp(x);
        else if (x < 2) return 1.0 + x;
        else return std::sin(x);
    };

    // Create disjoint intervals
    intervals::interval_set<double> domain;
    domain.insert(intervals::interval<double>::closed(-2, -1));
    domain.insert(intervals::interval<double>::open(0, 1));
    domain.insert(intervals::interval<double>::closed(2, 4));

    // Integrate over each interval
    integration::gauss_legendre<double, 5> quad;
    statistics::kahan_accumulator<double> total_integral;

    std::cout << "Integrating over disjoint intervals:" << std::endl;
    for (const auto& interval : domain) {
        if (!interval.is_empty()) {
            double result = quad(f, interval.lower(), interval.upper());
            total_integral += result;

            std::cout << "Interval [" << interval.lower() << ", " << interval.upper()
                      << "]: " << result << std::endl;
        }
    }

    std::cout << "Total integral: " << total_integral.eval() << std::endl;
}

// Test 5: Hash-based statistical sampling
void test_hash_sampling_statistics() {
    std::cout << "\n=== Test 5: Hash-based Statistical Sampling ===" << std::endl;

    // Use golden ratio hash for sampling
    hashing::golden_ratio_hash<std::uint64_t> hasher(10);  // 10-bit output

    statistics::histogram_accumulator<double> histogram(0.0, 1024.0, 32);
    statistics::minmax_accumulator<double> minmax;

    const std::size_t num_samples = 10000;
    for (std::size_t i = 0; i < num_samples; ++i) {
        auto hash_val = hasher(i);
        double normalized = static_cast<double>(hash_val);
        histogram += normalized;
        minmax += normalized;
    }

    std::cout << "Hash distribution analysis:" << std::endl;
    std::cout << "Min hash: " << minmax.min() << std::endl;
    std::cout << "Max hash: " << minmax.max() << std::endl;
    std::cout << "Range: " << minmax.range() << std::endl;

    // Check uniformity of distribution
    double expected_per_bin = static_cast<double>(num_samples) / histogram.num_bins();
    double chi_square = 0.0;

    for (std::size_t i = 0; i < histogram.num_bins(); ++i) {
        double observed = static_cast<double>(histogram.bin_count(i));
        double diff = observed - expected_per_bin;
        chi_square += (diff * diff) / expected_per_bin;
    }

    std::cout << "Chi-square statistic: " << chi_square << std::endl;
    std::cout << "Expected for uniform: ~" << histogram.num_bins() << std::endl;
}

// Test 6: Line integral with interval constraints
void test_line_integral_intervals() {
    std::cout << "\n=== Test 6: Line Integral with Interval Constraints ===" << std::endl;

    // Define a parametric curve (circle)
    auto curve = [](double t) -> std::array<double, 2> {
        return {std::cos(t), std::sin(t)};
    };

    auto curve_derivative = [](double t) -> std::array<double, 2> {
        return {-std::sin(t), std::cos(t)};
    };

    // Define a vector field
    auto field = [](std::array<double, 2> p) -> std::array<double, 2> {
        return {-p[1], p[0]};  // Rotation field
    };

    // Create interval set for parameter domain
    intervals::interval_set<double> param_domain;
    param_domain.insert(intervals::interval<double>::closed(0, M_PI/2));
    param_domain.insert(intervals::interval<double>::closed(M_PI, 3*M_PI/2));

    // Compute line integral over each interval
    integration::line_integral<double> line_integrator;
    double total = 0.0;

    for (const auto& interval : param_domain) {
        if (!interval.is_empty()) {
            auto result = line_integrator.vector_field(
                field, curve, curve_derivative,
                interval.lower(), interval.upper()
            );
            total += result.value;

            std::cout << "Integral over [" << interval.lower() << ", " << interval.upper()
                      << "]: " << result.value << std::endl;
        }
    }

    std::cout << "Total line integral: " << total << std::endl;
}

// Test 7: Composite statistics with hashed intervals
void test_composite_statistics() {
    std::cout << "\n=== Test 7: Composite Statistics with Hashed Intervals ===" << std::endl;

    // Generate random intervals
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    intervals::interval_set<double> intervals;
    statistics::composite_accumulator<double> interval_stats;
    hashing::xor_tabulation_hash<std::uint64_t> tabulation;

    for (int i = 0; i < 20; ++i) {
        double a = dis(gen);
        double b = dis(gen);
        if (a > b) std::swap(a, b);

        intervals.insert(stepanov::intervals::interval<double>::closed(a, b));
        interval_stats += (b - a);  // Add interval length
    }

    std::cout << "Interval length statistics:" << std::endl;
    std::cout << "Mean length: " << interval_stats.mean() << std::endl;
    std::cout << "Std dev: " << interval_stats.stddev() << std::endl;
    std::cout << "Total length: " << interval_stats.sum() << std::endl;

    // Hash the final interval set
    std::uint64_t set_hash = 0;
    for (const auto& interval : intervals) {
        if (!interval.is_empty()) {
            auto lower_bits = std::bit_cast<std::uint64_t>(interval.lower());
            auto upper_bits = std::bit_cast<std::uint64_t>(interval.upper());
            set_hash ^= tabulation(lower_bits) ^ tabulation(upper_bits);
        }
    }

    std::cout << "Interval set hash: " << std::hex << set_hash << std::dec << std::endl;
    std::cout << "Number of disjoint intervals: " << intervals.size() << std::endl;
}

int main() {
    std::cout << "=== Stepanov Library Integration Tests ===" << std::endl;
    std::cout << "Testing integration of:" << std::endl;
    std::cout << "- Algebraic Integrators" << std::endl;
    std::cout << "- Disjoint Interval Sets" << std::endl;
    std::cout << "- Algebraic Hashing" << std::endl;
    std::cout << "- Statistical Accumulators" << std::endl;

    try {
        test_integration_with_statistics();
        test_intervals_with_hashing();
        test_rolling_hash_statistics();
        test_integration_over_intervals();
        test_hash_sampling_statistics();
        test_line_integral_intervals();
        test_composite_statistics();

        std::cout << "\n=== All integration tests passed successfully! ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}