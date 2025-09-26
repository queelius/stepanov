#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

// Include existing headers first (for the adaptors)
#include <stepanov/builtin_adaptors.hpp"

// Include the new headers
#include <stepanov/continued_fraction.hpp"
#include <stepanov/fenwick_tree.hpp"
#include <stepanov/padic.hpp"
#include <stepanov/combinatorics.hpp"
#include <stepanov/sparse_matrix.hpp"
#include <stepanov/fft.hpp"
#include <stepanov/primality.hpp"

using namespace generic_math;
using namespace std;

// Test continued fractions
void test_continued_fractions() {
    cout << "\n=== Testing Continued Fractions ===" << endl;

    // Convert pi approximation to continued fraction
    auto cf = to_continued_fraction(355, 113);  // 355/113 ≈ π
    cout << "355/113 as continued fraction: [";
    for (size_t i = 0; i < cf.size(); ++i) {
        cout << cf[i];
        if (i < cf.size() - 1) cout << "; ";
    }
    cout << "]" << endl;

    // Compute convergents
    auto convergents = compute_convergents(cf);
    cout << "Convergents: ";
    for (const auto& conv : convergents) {
        cout << conv.p << "/" << conv.q << " ";
    }
    cout << endl;

    // Best rational approximation with small denominator
    auto best = best_rational_approximation(355, 113, 50);
    cout << "Best approximation with denominator <= 50: "
         << best.p << "/" << best.q << endl;

    // Solve Diophantine equation
    auto solution = solve_linear_diophantine(15, 21, 9);
    if (solution.has_solution) {
        cout << "Solution to 15x + 21y = 9: x = " << solution.x0
             << ", y = " << solution.y0 << endl;
        cout << "General solution: x = " << solution.x0 << " + " << solution.dx << "k, "
             << "y = " << solution.y0 << " - " << solution.dy << "k" << endl;
    }
}

// Test Fenwick Tree
void test_fenwick_tree() {
    cout << "\n=== Testing Fenwick Tree ===" << endl;

    // Basic range sum queries
    vector<int> data = {1, 3, 5, 7, 9, 11};
    fenwick_tree<int> ft(data);

    cout << "Array: ";
    for (int x : data) cout << x << " ";
    cout << endl;

    cout << "Prefix sum [0, 3]: " << ft.query_prefix(3) << endl;
    cout << "Range sum [2, 4]: " << ft.query_range(2, 4) << endl;

    ft.update(2, 10);  // Add 10 to index 2
    cout << "After adding 10 to index 2:" << endl;
    cout << "Range sum [2, 4]: " << ft.query_range(2, 4) << endl;

    // 2D Fenwick tree
    fenwick_tree_2d<int> ft2d(3, 3);
    ft2d.update(1, 1, 5);
    ft2d.update(2, 2, 3);
    cout << "2D rectangle sum (0,0) to (2,2): " << ft2d.query_rectangle(0, 0, 2, 2) << endl;
}

// Test p-adic numbers
void test_padic() {
    cout << "\n=== Testing p-adic Numbers ===" << endl;

    // 5-adic representation
    padic<int> x(5, 26);  // 26 in 5-adic
    padic<int> y(5, 14);  // 14 in 5-adic

    auto sum = x + y;
    auto prod = x * y;

    auto [num_x, den_x] = x.to_rational();
    auto [num_y, den_y] = y.to_rational();
    auto [num_sum, den_sum] = sum.to_rational();
    auto [num_prod, den_prod] = prod.to_rational();

    cout << "26 (5-adic) = " << num_x << "/" << den_x << endl;
    cout << "14 (5-adic) = " << num_y << "/" << den_y << endl;
    cout << "Sum = " << num_sum << "/" << den_sum << endl;
    cout << "Product = " << num_prod << "/" << den_prod << endl;

    cout << "5-adic norm of 26: " << x.norm() << endl;
    cout << "5-adic distance between 26 and 14: " << x.distance(y) << endl;

    // p-adic valuation
    cout << "2-adic valuation of 24: " << padic_valuation(24, 2) << endl;
    cout << "3-adic valuation of 54: " << padic_valuation(54, 3) << endl;
}

// Test combinatorics
void test_combinatorics() {
    cout << "\n=== Testing Combinatorics ===" << endl;

    // Binomial coefficients
    binomial_coefficients<int> binom(10);
    cout << "C(10, 3) = " << binom.get(10, 3) << endl;
    cout << "Row 5 of Pascal's triangle: ";
    for (int x : binom.row(5)) cout << x << " ";
    cout << endl;

    // Stirling numbers
    stirling_first<int> s1(8);
    stirling_second<int> s2(8);
    cout << "Stirling first S1(5, 3) = " << s1.get(5, 3) << endl;
    cout << "Stirling second S2(5, 3) = " << s2.get(5, 3) << endl;

    // Catalan numbers
    catalan_numbers<int> catalan(10);
    cout << "First 8 Catalan numbers: ";
    for (size_t i = 0; i <= 7; ++i) {
        cout << catalan.get(i) << " ";
    }
    cout << endl;

    // Bell numbers
    bell_numbers<int> bell(8);
    cout << "First 6 Bell numbers: ";
    for (size_t i = 0; i <= 5; ++i) {
        cout << bell.get(i) << " ";
    }
    cout << endl;

    // Integer partitions
    integer_partitions<int> partitions(10);
    cout << "Number of partitions of 7: " << partitions.total(7) << endl;
    cout << "Partitions of 5: " << endl;
    auto all_partitions = partitions.generate_all(5);
    for (const auto& partition : all_partitions) {
        cout << "  ";
        for (size_t i = 0; i < partition.size(); ++i) {
            cout << partition[i];
            if (i < partition.size() - 1) cout << " + ";
        }
        cout << endl;
    }
}

// Test sparse matrix
void test_sparse_matrix() {
    cout << "\n=== Testing Sparse Matrix ===" << endl;

    // Create sparse matrices
    sparse_matrix<double> A(3, 3);
    A.set(0, 0, 1.0);
    A.set(1, 1, 2.0);
    A.set(2, 2, 3.0);
    A.set(0, 2, 4.0);

    sparse_matrix<double> B(3, 3);
    B.set(0, 0, 2.0);
    B.set(1, 1, 1.0);
    B.set(2, 0, 5.0);

    cout << "Matrix A (nnz=" << A.nnz() << ", sparsity="
         << fixed << setprecision(2) << A.sparsity() * 100 << "%)" << endl;

    // Matrix operations
    auto C = A + B;
    cout << "A + B has " << C.nnz() << " non-zero entries" << endl;

    auto D = A * B;
    cout << "A * B has " << D.nnz() << " non-zero entries" << endl;

    // Matrix-vector multiplication
    vector<double> v = {1.0, 2.0, 3.0};
    auto result = A * v;
    cout << "A * [1, 2, 3] = [";
    for (size_t i = 0; i < result.size(); ++i) {
        cout << result[i];
        if (i < result.size() - 1) cout << ", ";
    }
    cout << "]" << endl;

    // Kronecker product
    sparse_matrix<double> I = sparse_matrix<double>::identity(2);
    auto kron = A.kronecker(I);
    cout << "Kronecker product of A and I(2) has dimensions "
         << kron.rows() << "x" << kron.cols() << " with "
         << kron.nnz() << " non-zeros" << endl;
}

// Test FFT
void test_fft() {
    cout << "\n=== Testing FFT ===" << endl;

    // Polynomial multiplication using FFT
    vector<double> p = {1, 2, 3};  // 1 + 2x + 3x^2
    vector<double> q = {4, 5};     // 4 + 5x

    auto product = complex_fft<double>::multiply_polynomials(p, q);
    cout << "(1 + 2x + 3x^2) * (4 + 5x) = ";
    for (size_t i = 0; i < product.size(); ++i) {
        if (product[i] != 0) {
            if (i > 0) cout << " + ";
            cout << product[i];
            if (i == 1) cout << "x";
            else if (i > 1) cout << "x^" << i;
        }
    }
    cout << endl;

    // Number Theoretic Transform
    const int MOD = 998244353;
    ntt<int> transform(MOD);
    vector<int> a = {1, 2, 3, 4};
    vector<int> b = {5, 6, 7, 8};
    auto ntt_product = transform.multiply_polynomials(a, b);
    cout << "NTT product mod " << MOD << ": ";
    for (int x : ntt_product) cout << x << " ";
    cout << endl;
}

// Test primality
void test_primality() {
    cout << "\n=== Testing Primality ===" << endl;

    miller_rabin<int> mr;
    solovay_strassen<int> ss;
    baillie_psw<int> bpsw;

    vector<int> test_numbers = {17, 91, 97, 561, 1729, 7919};

    cout << "Number | Trial | Miller-Rabin | Solovay-Strassen | Baillie-PSW" << endl;
    cout << "-------|-------|--------------|------------------|------------" << endl;

    for (int n : test_numbers) {
        bool trial = is_prime_trial_division(n);
        bool mr_result = mr.is_prime_deterministic(n);
        bool ss_result = ss.is_prime(n, 20);
        bool bpsw_result = bpsw.is_prime(n);

        cout << setw(6) << n << " | "
             << setw(5) << (trial ? "Yes" : "No") << " | "
             << setw(12) << (mr_result ? "Yes" : "No") << " | "
             << setw(16) << (ss_result ? "Yes" : "No") << " | "
             << setw(10) << (bpsw_result ? "Yes" : "No") << endl;
    }

    // Sieve of Eratosthenes
    auto primes = sieve_of_eratosthenes(100);
    cout << "\nPrimes up to 100: ";
    for (int p : primes) cout << p << " ";
    cout << endl;

    // Pollard's rho factorization
    pollard_rho<int> pr;
    int composite = 9991;  // 991 is 9991
    auto factors = pr.factorize(composite);
    cout << "\nFactorization of " << composite << ": ";
    for (size_t i = 0; i < factors.size(); ++i) {
        cout << factors[i];
        if (i < factors.size() - 1) cout << " * ";
    }
    cout << endl;
}

// Performance comparison
void benchmark_algorithms() {
    cout << "\n=== Performance Benchmarks ===" << endl;

    using namespace chrono;

    // Benchmark Fenwick tree vs naive sum
    const size_t N = 100000;
    vector<int> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = i + 1;

    fenwick_tree<int> ft(data);

    auto start = high_resolution_clock::now();
    long long sum = 0;
    for (size_t i = 0; i < 1000; ++i) {
        sum += ft.query_range(N/4, 3*N/4);
    }
    auto ft_time = duration_cast<microseconds>(high_resolution_clock::now() - start);

    start = high_resolution_clock::now();
    sum = 0;
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t j = N/4; j <= 3*N/4; ++j) {
            sum += data[j];
        }
    }
    auto naive_time = duration_cast<microseconds>(high_resolution_clock::now() - start);

    cout << "Range sum query (N=" << N << "):" << endl;
    cout << "  Fenwick Tree: " << ft_time.count() << " μs" << endl;
    cout << "  Naive sum: " << naive_time.count() << " μs" << endl;
    cout << "  Speedup: " << fixed << setprecision(2)
         << (double)naive_time.count() / ft_time.count() << "x" << endl;

    // Benchmark FFT polynomial multiplication
    vector<double> poly1(512, 1.0), poly2(512, 2.0);

    start = high_resolution_clock::now();
    auto fft_result = complex_fft<double>::multiply_polynomials(poly1, poly2);
    auto fft_mult_time = duration_cast<microseconds>(high_resolution_clock::now() - start);

    cout << "\nPolynomial multiplication (degree 511):" << endl;
    cout << "  FFT: " << fft_mult_time.count() << " μs" << endl;
    cout << "  Theoretical speedup vs naive: O(n log n) vs O(n²)" << endl;
}

int main() {
    cout << "=== Generic Math Library - New Algorithms Demo ===" << endl;
    cout << "Showcasing advanced algorithms following Alex Stepanov's principles" << endl;

    test_continued_fractions();
    test_fenwick_tree();
    test_padic();
    test_combinatorics();
    test_sparse_matrix();
    test_fft();
    test_primality();
    benchmark_algorithms();

    cout << "\n=== All tests completed successfully ===" << endl;
    return 0;
}