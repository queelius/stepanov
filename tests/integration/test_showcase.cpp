#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>

// Core library
#include <stepanov/builtin_adaptors.hpp"
#include <stepanov/algorithms.hpp"
#include <stepanov/gcd.hpp"
#include <stepanov/math.hpp"
#include <stepanov/integer_wrapper.hpp"

// Advanced features
#include <stepanov/graph.hpp"
#include <stepanov/interval.hpp"
#include <stepanov/autodiff.hpp"

using namespace generic_math;
using namespace std;
using namespace std::chrono;

int main() {
    cout << "==========================================" << endl;
    cout << "   Generic Math Library Showcase         " << endl;
    cout << "==========================================" << endl;

    // =========================================
    // 1. Generic Algorithms on Orbits and Cycles
    // =========================================
    cout << "\n1. ORBIT AND CYCLE DETECTION" << endl;
    cout << "-----------------------------" << endl;

    // Collatz conjecture function
    auto collatz = [](int x) {
        return (x % 2 == 0) ? x / 2 : 3 * x + 1;
    };

    cout << "Collatz sequence starting from 27:" << endl;
    auto orbit_27 = orbit(27, 20, collatz);
    cout << "  ";
    for (size_t i = 0; i < min<size_t>(10, orbit_27.size()); ++i) {
        cout << orbit_27[i] << " ";
    }
    cout << "..." << endl;

    // Detect cycle in a simple transformation
    auto cyclic = [](int x) { return (x * 3 + 1) % 10; };
    auto cycle = detect_cycle(5, cyclic);
    cout << "\nCycle detection for f(x) = (3x+1) mod 10:" << endl;
    cout << "  Starting from 5:" << endl;
    cout << "  Cycle length: " << cycle.cycle_length << endl;
    cout << "  Tail length: " << cycle.tail_length << endl;

    // =========================================
    // 2. Graph Algorithms
    // =========================================
    cout << "\n2. GRAPH ALGORITHMS" << endl;
    cout << "--------------------" << endl;

    adjacency_graph<char, int> graph;

    // Create a simple graph
    graph.add_edge('A', 'B', 4);
    graph.add_edge('A', 'C', 2);
    graph.add_edge('B', 'C', 1);
    graph.add_edge('B', 'D', 5);
    graph.add_edge('C', 'D', 8);
    graph.add_edge('C', 'E', 10);
    graph.add_edge('D', 'E', 2);

    cout << "Shortest paths from A (Dijkstra):" << endl;
    auto paths = dijkstra(graph, 'A');
    for (char v = 'B'; v <= 'E'; ++v) {
        if (paths.distances.find(v) != paths.distances.end()) {
            cout << "  A -> " << v << ": distance = " << paths.distances[v];
            auto path = paths.path_to(v);
            cout << " [path: ";
            for (size_t i = 0; i < path.size(); ++i) {
                cout << path[i];
                if (i < path.size() - 1) cout << "->";
            }
            cout << "]" << endl;
        }
    }

    // =========================================
    // 3. Interval Arithmetic
    // =========================================
    cout << "\n3. INTERVAL ARITHMETIC" << endl;
    cout << "-----------------------" << endl;

    interval<double> x(0.99, 1.01);  // 1.0 with 1% uncertainty
    interval<double> y(1.99, 2.01);  // 2.0 with 0.5% uncertainty

    cout << "Computations with uncertainty:" << endl;
    cout << "  x = " << x << " (1.0 ± 1%)" << endl;
    cout << "  y = " << y << " (2.0 ± 0.5%)" << endl;

    auto z = x * y + sqrt(x);
    cout << "  z = x*y + sqrt(x) = " << z << endl;
    cout << "  Relative error: " << (z.width() / z.midpoint()) * 100 << "%" << endl;

    // =========================================
    // 4. Automatic Differentiation
    // =========================================
    cout << "\n4. AUTOMATIC DIFFERENTIATION" << endl;
    cout << "-----------------------------" << endl;

    // Forward-mode AD for single variable
    auto f = [](dual<double> x) {
        return x * x * x - dual<double>::constant(2.0) * x + dual<double>::constant(1.0);
    };

    dual<double> x_ad = dual<double>::variable(3.0);
    auto result = f(x_ad);

    cout << "f(x) = x³ - 2x + 1 at x = 3.0:" << endl;
    cout << "  f(3) = " << result.value() << endl;
    cout << "  f'(3) = " << result.derivative() << endl;

    // Gradient of multivariable function
    auto rosenbrock = [](const vector<dual<double>>& v) {
        auto x = v[0];
        auto y = v[1];
        auto term1 = dual<double>::constant(1.0) - x;
        auto term2 = y - x * x;
        return term1 * term1 + dual<double>::constant(100.0) * term2 * term2;
    };

    vector<double> point = {1.5, 2.0};
    auto grad = gradient(rosenbrock, point);

    cout << "\nRosenbrock function at (1.5, 2.0):" << endl;
    cout << "  Gradient: [" << grad[0] << ", " << grad[1] << "]" << endl;

    // =========================================
    // 5. Generic Number Theory
    // =========================================
    cout << "\n5. NUMBER THEORY WITH GENERIC GCD" << endl;
    cout << "----------------------------------" << endl;

    // Extended GCD using integer wrapper for proper concept modeling
    Int a(240), b(46);
    auto egcd = extended_gcd(a, b);
    cout << "Extended GCD(240, 46):" << endl;
    cout << "  gcd = " << egcd.gcd << endl;
    cout << "  Bézout coefficients: " << egcd.x << " * 240 + " << egcd.y << " * 46 = " << egcd.gcd << endl;
    cout << "  Verification: " << (egcd.x * a + egcd.y * b) << endl;

    // Chinese Remainder Theorem
    vector<Int> remainders = {Int(2), Int(3), Int(2)};
    vector<Int> moduli = {Int(3), Int(5), Int(7)};
    auto crt = chinese_remainder(remainders, moduli);

    if (crt.has_solution) {
        cout << "\nChinese Remainder Theorem:" << endl;
        cout << "  x ≡ 2 (mod 3)" << endl;
        cout << "  x ≡ 3 (mod 5)" << endl;
        cout << "  x ≡ 2 (mod 7)" << endl;
        cout << "  Solution: x ≡ " << crt.solution << " (mod " << crt.modulus << ")" << endl;
    }

    // =========================================
    // 6. Algorithm Composition
    // =========================================
    cout << "\n6. ALGORITHM COMPOSITION" << endl;
    cout << "-------------------------" << endl;

    // Compose accumulate with a transformation
    vector<int> numbers = {1, 2, 3, 4, 5};

    // Sum of squares using accumulate and lambda
    auto sum_of_squares = accumulate(numbers.begin(), numbers.end(), 0,
        [](int acc, int x) { return acc + x * x; });

    cout << "Sum of squares [1,2,3,4,5]: " << sum_of_squares << endl;

    // Three-way partition
    vector<int> data = {3, 7, 3, 3, 2, 9, 1, 3, 8, 3};
    cout << "\nThree-way partition around pivot=3:" << endl;
    cout << "  Original: ";
    for (int x : data) cout << x << " ";
    cout << endl;

    auto [less_end, equal_end] = partition_3way(data.begin(), data.end(), 3);

    cout << "  After:    ";
    for (int x : data) cout << x << " ";
    cout << endl;
    cout << "  Structure: [< 3][== 3][> 3]" << endl;

    // =========================================
    // 7. Mathematical Elegance
    // =========================================
    cout << "\n7. MATHEMATICAL ELEGANCE" << endl;
    cout << "-------------------------" << endl;

    // Power by repeated squaring
    cout << "Computing 2^20 using repeated squaring:" << endl;
    int base = 2, exp = 20;
    int result_power = power_accumulate(base, exp, 1);
    cout << "  2^20 = " << result_power << endl;

    // Function composition
    auto add_one = [](int x) { return x + 1; };
    auto square = [](int x) { return x * x; };
    auto composed = compose(square, add_one);  // square(add_one(x))

    cout << "\nFunction composition f(g(x)) where f=square, g=add_one:" << endl;
    cout << "  f(g(5)) = square(add_one(5)) = " << composed(5) << endl;

    cout << "\n==========================================" << endl;
    cout << "   All demonstrations completed!          " << endl;
    cout << "==========================================" << endl;

    return 0;
}