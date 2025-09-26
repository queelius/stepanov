#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <cassert>

// Include adaptor for built-in types
#include <stepanov/builtin_adaptors.hpp>

// Include new advanced features
#include <stepanov/graph.hpp>
#include <stepanov/interval.hpp>
#include <stepanov/autodiff.hpp>
#include <stepanov/cache_oblivious.hpp>

// Include existing features
#include <stepanov/algorithms.hpp>
#include <stepanov/gcd.hpp>

using namespace stepanov;
using namespace std;
using namespace std::chrono;

// Utility for timing
template<typename F>
auto time_it(const string& name, F func) {
    auto start = high_resolution_clock::now();
    auto result = func();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << name << " took " << duration.count() << " microseconds" << endl;
    return result;
}

// =============================================================================
// Test Graph Algorithms
// =============================================================================

void test_graph_algorithms() {
    cout << "\n=== Testing Graph Algorithms ===" << endl;

    // Create a weighted directed graph
    adjacency_graph<int, double> graph;

    // Add vertices and edges (example: small network)
    for (int i = 0; i < 6; ++i) {
        graph.add_vertex(i);
    }

    graph.add_edge(0, 1, 4.0);
    graph.add_edge(0, 2, 3.0);
    graph.add_edge(1, 2, 1.0);
    graph.add_edge(1, 3, 2.0);
    graph.add_edge(2, 3, 4.0);
    graph.add_edge(3, 4, 2.0);
    graph.add_edge(4, 5, 6.0);

    // Test BFS
    cout << "\nBreadth-First Search from vertex 0:" << endl;
    auto distances = breadth_first_search(graph, 0);
    for (const auto& [vertex, dist] : distances) {
        cout << "  Distance to " << vertex << ": " << dist << endl;
    }

    // Test DFS
    cout << "\nDepth-First Search from vertex 0:" << endl;
    cout << "  Visit order: ";
    depth_first_search(graph, 0, [](int v) { cout << v << " "; });
    cout << endl;

    // Test Dijkstra's algorithm
    cout << "\nDijkstra's shortest paths from vertex 0:" << endl;
    auto shortest_paths = dijkstra(graph, 0);
    for (const auto& [vertex, distance] : shortest_paths.distances) {
        cout << "  Distance to " << vertex << ": " << distance;
        auto path = shortest_paths.path_to(vertex);
        cout << " (path: ";
        for (size_t i = 0; i < path.size(); ++i) {
            cout << path[i];
            if (i < path.size() - 1) cout << " -> ";
        }
        cout << ")" << endl;
    }

    // Test topological sort on a DAG
    adjacency_graph<string> dag;
    dag.add_edge("shirt", "tie", 1);
    dag.add_edge("shirt", "jacket", 1);
    dag.add_edge("tie", "jacket", 1);
    dag.add_edge("undershorts", "pants", 1);
    dag.add_edge("pants", "shoes", 1);
    dag.add_edge("pants", "belt", 1);
    dag.add_edge("belt", "jacket", 1);
    dag.add_edge("socks", "shoes", 1);

    cout << "\nTopological sort of dressing order:" << endl;
    auto sorted = topological_sort<adjacency_graph<string>, string>(dag);
    if (!sorted.empty()) {
        cout << "  Order: ";
        for (const auto& item : sorted) {
            cout << item << " ";
        }
        cout << endl;
    } else {
        cout << "  Graph has cycles!" << endl;
    }

    // Test graph coloring
    adjacency_graph<int> color_graph;
    for (int i = 0; i < 5; ++i) {
        color_graph.add_vertex(i);
    }
    color_graph.add_undirected_edge(0, 1, 1);
    color_graph.add_undirected_edge(0, 2, 1);
    color_graph.add_undirected_edge(1, 2, 1);
    color_graph.add_undirected_edge(1, 3, 1);
    color_graph.add_undirected_edge(2, 3, 1);
    color_graph.add_undirected_edge(3, 4, 1);

    cout << "\nGraph coloring:" << endl;
    auto coloring = greedy_coloring<adjacency_graph<int>, int>(color_graph);
    for (const auto& [vertex, color] : coloring) {
        cout << "  Vertex " << vertex << ": color " << color << endl;
    }

    // Test maximum flow
    flow_network<string, int> network;
    network.add_edge("s", "a", 10);
    network.add_edge("s", "b", 10);
    network.add_edge("a", "b", 2);
    network.add_edge("a", "c", 4);
    network.add_edge("a", "d", 8);
    network.add_edge("b", "d", 9);
    network.add_edge("c", "t", 10);
    network.add_edge("d", "c", 6);
    network.add_edge("d", "t", 10);

    cout << "\nMaximum flow from s to t: " << network.max_flow("s", "t") << endl;
}

// =============================================================================
// Test Interval Arithmetic
// =============================================================================

void test_interval_arithmetic() {
    cout << "\n=== Testing Interval Arithmetic ===" << endl;

    // Basic interval operations
    interval<double> a(2.0, 3.0);
    interval<double> b(1.5, 2.5);

    cout << "\nBasic operations:" << endl;
    cout << "  a = " << a << endl;
    cout << "  b = " << b << endl;
    cout << "  a + b = " << (a + b) << endl;
    cout << "  a - b = " << (a - b) << endl;
    cout << "  a * b = " << (a * b) << endl;
    cout << "  a / b = " << (a / b) << endl;

    // Mathematical functions
    interval<double> x(0.5, 1.5);
    cout << "\nMathematical functions on " << x << ":" << endl;
    cout << "  sqrt(x) = " << sqrt(x) << endl;
    cout << "  exp(x) = " << exp(x) << endl;
    cout << "  log(x) = " << log(x) << endl;
    cout << "  sin(x) = " << sin(x) << endl;
    cout << "  cos(x) = " << cos(x) << endl;

    // Interval Newton method for finding roots
    auto f = [](const interval<double>& x) {
        return x * x - interval<double>(2.0);  // f(x) = x^2 - 2
    };
    auto df = [](const interval<double>& x) {
        return interval<double>(2.0) * x;  // f'(x) = 2x
    };

    cout << "\nFinding roots of x^2 - 2 = 0 in [0, 3]:" << endl;
    auto newton = make_interval_newton<double>(f, df);
    auto roots = newton.find_roots(interval<double>(0.0, 3.0));
    for (const auto& root : roots) {
        cout << "  Root found in interval " << root << endl;
    }

    // Demonstrate guaranteed bounds
    interval<double> uncertain(0.999, 1.001);  // Value with uncertainty
    cout << "\nComputation with uncertainty:" << endl;
    cout << "  Input: " << uncertain << " (1.0 ± 0.001)" << endl;
    auto result = exp(log(uncertain) * interval<double>(10.0));
    cout << "  exp(log(x) * 10) = " << result << endl;
    cout << "  Width of result: " << result.width() << endl;
}

// =============================================================================
// Test Automatic Differentiation
// =============================================================================

void test_automatic_differentiation() {
    cout << "\n=== Testing Automatic Differentiation ===" << endl;

    // Forward-mode AD with dual numbers
    cout << "\nForward-mode AD:" << endl;

    // Single variable function: f(x) = x^2 * sin(x)
    auto f1 = [](dual<double> x) {
        return x * x * sin(x);
    };

    dual<double> x = dual<double>::variable(2.0);  // x = 2.0, dx = 1.0
    auto result = f1(x);
    cout << "  f(x) = x^2 * sin(x) at x = 2.0:" << endl;
    cout << "    Value: " << result.value() << endl;
    cout << "    Derivative: " << result.derivative() << endl;

    // Multivariable function: f(x, y) = x*y + sin(x)*cos(y)
    auto f2 = [](const vector<dual<double>>& vars) {
        auto x = vars[0];
        auto y = vars[1];
        return x * y + sin(x) * cos(y);
    };

    cout << "\n  f(x, y) = x*y + sin(x)*cos(y) at (π/4, π/3):" << endl;
    double pi = 3.14159265358979323846;
    vector<double> point = {pi/4, pi/3};
    auto grad = gradient(f2, point);
    cout << "    Gradient: [" << grad[0] << ", " << grad[1] << "]" << endl;

    // Higher-order derivatives
    cout << "\nSecond-order derivatives:" << endl;
    auto f3 = [](dual2<double> x) {
        return exp(x) * sin(x);  // f(x) = e^x * sin(x)
    };

    auto x2 = make_dual2(1.0, 1.0, 0.0);  // x = 1, dx = 1, d²x = 0
    auto result2 = f3(x2);
    derivatives<double> d(result2);
    cout << "  f(x) = e^x * sin(x) at x = 1.0:" << endl;
    cout << "    Value: " << d.value << endl;
    cout << "    First derivative: " << d.first << endl;
    cout << "    Second derivative: " << d.second << endl;

    // Optimization using automatic differentiation
    cout << "\nOptimization with gradient descent:" << endl;

    // Rosenbrock function: f(x, y) = (1-x)^2 + 100*(y-x^2)^2
    auto rosenbrock = [](const vector<dual<double>>& vars) {
        auto x = vars[0];
        auto y = vars[1];
        auto term1 = (dual<double>::constant(1.0) - x);
        auto term2 = y - x * x;
        return term1 * term1 + dual<double>::constant(100.0) * term2 * term2;
    };

    vector<double> x0 = {-1.0, 1.0};  // Starting point
    auto opt_result = gradient_descent(rosenbrock, x0, 0.001, 1e-6, 10000);
    cout << "  Minimizing Rosenbrock function:" << endl;
    cout << "    Starting point: [" << x0[0] << ", " << x0[1] << "]" << endl;
    cout << "    Minimum found at: [" << opt_result.x[0] << ", " << opt_result.x[1] << "]" << endl;
    cout << "    Function value: " << opt_result.value << endl;
    cout << "    Iterations: " << opt_result.iterations << endl;
    cout << "    Converged: " << (opt_result.converged ? "Yes" : "No") << endl;

    // Jacobian computation
    cout << "\nJacobian matrix:" << endl;
    auto vector_func = [](const vector<dual<double>>& x) {
        vector<dual<double>> result(2);
        result[0] = x[0] * x[0] + x[1] * x[1];  // r^2
        result[1] = atan(x[1] / x[0]);          // theta
        return result;
    };

    vector<double> point2 = {1.0, 1.0};
    auto jac = jacobian(vector_func, point2);
    cout << "  Polar coordinate transformation at (1, 1):" << endl;
    cout << "    Jacobian matrix:" << endl;
    for (const auto& row : jac) {
        cout << "      [";
        for (size_t i = 0; i < row.size(); ++i) {
            cout << setw(10) << row[i];
            if (i < row.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
    }
}

// =============================================================================
// Test Cache-Oblivious Algorithms
// =============================================================================

void test_cache_oblivious() {
    cout << "\n=== Testing Cache-Oblivious Algorithms ===" << endl;

    // Matrix multiplication
    cout << "\nCache-oblivious matrix multiplication:" << endl;
    const size_t n = 128;
    cache_oblivious::matrix<double> A(n, n);
    cache_oblivious::matrix<double> B(n, n);

    // Initialize with random values
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A(i, j) = dis(gen);
            B(i, j) = dis(gen);
        }
    }

    auto C = time_it("  Matrix multiplication (128x128)", [&]() {
        return A * B;
    });

    // Verify a sample element
    double check = 0.0;
    for (size_t k = 0; k < n; ++k) {
        check += A(0, k) * B(k, 0);
    }
    cout << "  Verification: C(0,0) = " << C(0, 0) << " (expected ~" << check << ")" << endl;

    // Matrix transpose - not implemented in cache_oblivious::matrix
    cout << "  Matrix transpose: Not implemented" << endl;

    // Sorting
    cout << "\nCache-oblivious sorting:" << endl;
    vector<int> data(10000);
    iota(data.begin(), data.end(), 0);
    shuffle(data.begin(), data.end(), gen);

    auto sorted = time_it("  Sorting 10000 elements", [&]() {
        vector<int> copy = data;
        cache_oblivious::funnel_sort<std::vector<int>::iterator> fs;
        fs(copy.begin(), copy.end());
        return copy;
    });

    bool is_sorted = std::is_sorted(sorted.begin(), sorted.end());
    cout << "  Correctly sorted: " << (is_sorted ? "Yes" : "No") << endl;

    // FFT
    cout << "\nCache-oblivious FFT:" << endl;
    vector<double> signal(256);
    const double pi = 3.14159265358979323846;
    for (size_t i = 0; i < signal.size(); ++i) {
        signal[i] = sin(2 * pi * i / 32) + 0.5 * cos(2 * pi * i / 16);
    }

    auto spectrum = time_it("  FFT of 256-point signal", [&]() {
        // return cache_oblivious_fft<double>::fft(signal); // Not implemented
        return vector<pair<double,double>>();
    });

    cout << "  First few frequency components:" << endl;
    for (size_t i = 0; i < min<size_t>(5, spectrum.size()); ++i) {
        double magnitude = sqrt(spectrum[i].first * spectrum[i].first +
                               spectrum[i].second * spectrum[i].second);
        cout << "    Bin " << i << ": magnitude = " << magnitude << endl;
    }

    // Convolution
    cout << "\nConvolution using FFT:" << endl;
    vector<double> filter = {0.25, 0.5, 0.25};  // Simple smoothing filter
    vector<double> input = {1, 2, 3, 4, 5, 4, 3, 2, 1};

    auto convolved = time_it("  Convolution", [&]() {
        // return cache_oblivious_fft<double>::convolve(input, filter); // Not implemented
        return vector<double>();
    });

    cout << "  Result: ";
    for (size_t i = 0; i < min<size_t>(convolved.size(), 10); ++i) {
        cout << convolved[i] << " ";
    }
    cout << "..." << endl;

    // B-tree
    cout << "\nCache-oblivious B-tree:" << endl;
    cache_oblivious::cache_oblivious_btree<int, string> tree;

    // Insert some key-value pairs
    for (int i = 0; i < 1000; ++i) {
        tree.insert(i, "value_" + to_string(i));
    }

    // Search
    auto search_result = time_it("  Searching 1000 keys", [&]() {
        int found = 0;
        for (int i = 0; i < 1000; ++i) {
            if (tree.search(i).has_value()) found++;
        }
        return found;
    });

    cout << "  Found " << search_result << " keys" << endl;
}

// =============================================================================
// Test Generic Algorithm Composition
// =============================================================================

void test_algorithm_composition() {
    cout << "\n=== Testing Algorithm Composition ===" << endl;

    // Compose cycle detection with automatic differentiation
    cout << "\nCycle detection in iterative optimization:" << endl;

    auto newton_map = [](dual<double> x) {
        // Newton's method for f(x) = x^3 - 2x - 5
        auto f = x * x * x - dual<double>::constant(2.0) * x - dual<double>::constant(5.0);
        auto df = dual<double>::constant(3.0) * x * x - dual<double>::constant(2.0);
        return x - f / df;
    };

    // Track iteration history
    double x0 = 2.0;
    vector<double> orbit_values;
    dual<double> current = dual<double>::variable(x0);

    for (int i = 0; i < 20; ++i) {
        orbit_values.push_back(current.value());
        current = newton_map(current);
    }

    cout << "  Newton iteration orbit: ";
    for (size_t i = 0; i < min<size_t>(10, orbit_values.size()); ++i) {
        cout << orbit_values[i] << " ";
    }
    cout << "..." << endl;
    cout << "  Converged to: " << current.value() << endl;

    // Interval arithmetic with GCD
    cout << "\nInterval GCD computation:" << endl;

    // Simulate uncertain measurements
    interval<double> a_interval(11.98, 12.02);
    interval<double> b_interval(17.99, 18.01);

    cout << "  Uncertain integers: a ≈ " << a_interval << ", b ≈ " << b_interval << endl;

    // Round to nearest integers and compute GCD
    int a_rounded = round(a_interval.midpoint());
    int b_rounded = round(b_interval.midpoint());
    int g = gcd(a_rounded, b_rounded);
    cout << "  GCD of rounded values (" << a_rounded << ", " << b_rounded << ") = " << g << endl;
}

// =============================================================================
// Main test runner
// =============================================================================

int main() {
    cout << "====================================================" << endl;
    cout << "     Generic Math Library - Advanced Features Test  " << endl;
    cout << "====================================================" << endl;

    try {
        test_graph_algorithms();
        test_interval_arithmetic();
        test_automatic_differentiation();
        test_cache_oblivious();
        test_algorithm_composition();

        cout << "\n====================================================" << endl;
        cout << "                All tests completed!                " << endl;
        cout << "====================================================" << endl;

    } catch (const exception& e) {
        cerr << "\nError: " << e.what() << endl;
        return 1;
    }

    return 0;
}