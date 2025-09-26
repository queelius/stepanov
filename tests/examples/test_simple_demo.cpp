#include <iostream>
#include <vector>
#include <iomanip>

// Include existing headers first (for the adaptors)
#include <stepanov/builtin_adaptors.hpp"

// Include selected new headers
#include <stepanov/fenwick_tree.hpp"
#include <stepanov/combinatorics.hpp"
#include <stepanov/sparse_matrix.hpp"

using namespace generic_math;
using namespace std;

// Test Fenwick Tree - efficient range queries
void demo_fenwick_tree() {
    cout << "\n=== Fenwick Tree (Binary Indexed Tree) ===" << endl;
    cout << "Efficient range sum queries in O(log n)" << endl << endl;

    vector<int> data = {3, 2, -1, 6, 5, 4, -3, 3, 7, 2};
    fenwick_tree<int> ft(data);

    cout << "Array: ";
    for (int x : data) cout << setw(3) << x << " ";
    cout << endl;

    cout << "Prefix sum [0, 4]: " << ft.query_prefix(4) << endl;
    cout << "Range sum [2, 6]: " << ft.query_range(2, 6) << endl;

    ft.update(3, 10);  // Add 10 to index 3
    cout << "After adding 10 to index 3:" << endl;
    cout << "Range sum [2, 6]: " << ft.query_range(2, 6) << endl;

    // 2D Fenwick tree for matrix queries
    fenwick_tree_2d<int> ft2d(4, 4);
    ft2d.update(1, 1, 5);
    ft2d.update(2, 2, 3);
    ft2d.update(3, 3, 7);
    cout << "\n2D rectangle sum from (1,1) to (3,3): "
         << ft2d.query_rectangle(1, 1, 3, 3) << endl;
}

// Test Combinatorics
void demo_combinatorics() {
    cout << "\n=== Combinatorial Algorithms ===" << endl;
    cout << "Generic implementations working over any ring" << endl << endl;

    // Binomial coefficients - Pascal's triangle
    binomial_coefficients<long long> binom(20);
    cout << "Binomial coefficient C(20, 10) = " << binom.get(20, 10) << endl;

    cout << "Row 6 of Pascal's triangle: ";
    for (auto x : binom.row(6)) cout << x << " ";
    cout << endl;

    // Stirling numbers of both kinds
    stirling_first<long long> s1(10);
    stirling_second<long long> s2(10);
    cout << "\nStirling numbers:" << endl;
    cout << "S1(6, 3) = " << s1.get(6, 3) << " (permutations with 3 cycles)" << endl;
    cout << "S2(6, 3) = " << s2.get(6, 3) << " (partitions into 3 sets)" << endl;

    // Catalan numbers - counting binary trees, parenthesizations, etc.
    catalan_numbers<long long> catalan(15);
    cout << "\nFirst 10 Catalan numbers (binary trees with n internal nodes):" << endl;
    for (size_t i = 0; i < 10; ++i) {
        cout << "C[" << i << "] = " << catalan.get(i) << " ";
        if (i % 5 == 4) cout << endl;
    }

    // Bell numbers - partitions of a set
    bell_numbers<long long> bell(10);
    cout << "\nBell numbers (set partitions): ";
    for (size_t i = 0; i <= 6; ++i) {
        cout << "B[" << i << "] = " << bell.get(i) << " ";
    }
    cout << endl;

    // Integer partitions
    integer_partitions<long long> partitions(15);
    cout << "\nNumber of ways to partition 10 = " << partitions.total(10) << endl;
    cout << "Partitions of 5:" << endl;
    auto all_parts = partitions.generate_all(5);
    int count = 0;
    for (const auto& partition : all_parts) {
        cout << "  ";
        for (size_t i = 0; i < partition.size(); ++i) {
            cout << partition[i];
            if (i < partition.size() - 1) cout << "+";
        }
        count++;
        if (count % 3 == 0) cout << endl;
        else cout << "   ";
    }
    if (count % 3 != 0) cout << endl;

    // Fibonacci numbers (generalized to any ring)
    fibonacci<long long> fib(20);
    cout << "\nFibonacci sequence: ";
    for (size_t i = 0; i <= 12; ++i) {
        cout << fib.get(i) << " ";
    }
    cout << endl;

    // Derangements - permutations with no fixed points
    derangements<long long> derang(10);
    cout << "\nDerangements (no fixed points): ";
    for (size_t i = 0; i <= 6; ++i) {
        cout << "D[" << i << "] = " << derang.get(i) << " ";
    }
    cout << endl;
}

// Test Sparse Matrix
void demo_sparse_matrix() {
    cout << "\n=== Sparse Matrix ===" << endl;
    cout << "Space-efficient matrix operations for mostly-zero matrices" << endl << endl;

    // Create a sparse diagonal matrix
    vector<double> diag_values = {1.0, 2.0, 3.0, 4.0, 5.0};
    sparse_matrix<double> A = sparse_matrix<double>::diagonal(diag_values);
    cout << "Diagonal matrix A (5x5) with " << A.nnz() << " non-zeros" << endl;
    cout << "Sparsity: " << fixed << setprecision(1) << A.sparsity() * 100 << "%" << endl;

    // Create another sparse matrix
    sparse_matrix<double> B(5, 5);
    B.set(0, 4, 1.0);
    B.set(1, 3, 2.0);
    B.set(2, 2, 3.0);
    B.set(3, 1, 4.0);
    B.set(4, 0, 5.0);
    cout << "Anti-diagonal matrix B with " << B.nnz() << " non-zeros" << endl;

    // Matrix operations
    auto C = A + B;
    cout << "\nA + B has " << C.nnz() << " non-zeros" << endl;

    auto D = A * B;
    cout << "A * B has " << D.nnz() << " non-zeros" << endl;

    // Matrix-vector multiplication
    vector<double> v = {1.0, 1.0, 1.0, 1.0, 1.0};
    auto result = A * v;
    cout << "\nA * [1,1,1,1,1] = [";
    for (size_t i = 0; i < result.size(); ++i) {
        cout << result[i];
        if (i < result.size() - 1) cout << ", ";
    }
    cout << "]" << endl;

    // Demonstrate efficiency for large sparse matrices
    cout << "\nLarge sparse matrix example:" << endl;
    sparse_matrix<double> large(1000, 1000);
    // Add only 100 elements to a 1000x1000 matrix
    for (int i = 0; i < 100; ++i) {
        large.set(i * 10, i * 10, i + 1.0);
    }
    cout << "1000x1000 matrix with only " << large.nnz() << " non-zeros" << endl;
    cout << "Sparsity: " << fixed << setprecision(3) << large.sparsity() * 100 << "%" << endl;
    cout << "Memory usage: ~" << large.nnz() * 16 << " bytes vs "
         << 1000 * 1000 * 8 << " bytes for dense" << endl;

    // Power operation for adjacency matrices
    sparse_matrix<int> adj(4, 4);
    adj.set(0, 1, 1);
    adj.set(1, 2, 1);
    adj.set(2, 3, 1);
    adj.set(3, 0, 1);
    adj.set(0, 2, 1);

    auto adj2 = adj.pow(2);
    cout << "\nGraph with 4 nodes, adjacency matrix squared (2-step paths):" << endl;
    cout << "Number of 2-step paths from node 0 to node 3: " << adj2.get(0, 3) << endl;
}

// Demonstrate the design philosophy
void demonstrate_generic_programming() {
    cout << "\n=== Generic Programming Principles ===" << endl;
    cout << "These algorithms follow Alex Stepanov's design philosophy:" << endl;
    cout << "1. Work with concepts, not concrete types" << endl;
    cout << "2. Minimize requirements on types" << endl;
    cout << "3. Provide maximal functionality" << endl;
    cout << "4. Achieve efficiency through abstraction\n" << endl;

    // Example: Combinatorial algorithms work over any ring
    cout << "Example: Bell numbers modulo 1000007 (large prime):" << endl;

    // Custom modular arithmetic type would work here
    // For now, we compute and take modulo manually
    bell_numbers<long long> bell_mod(15);
    const long long MOD = 1000007;
    cout << "Bell numbers mod " << MOD << ": ";
    for (size_t i = 0; i <= 10; ++i) {
        cout << bell_mod.get(i) % MOD << " ";
    }
    cout << endl;

    cout << "\nKey insights from these implementations:" << endl;
    cout << "- Fenwick Tree: Generic over any group (not just addition)" << endl;
    cout << "- Combinatorics: Work over any ring (integers, polynomials, matrices)" << endl;
    cout << "- Sparse Matrix: Optimize for common case while maintaining abstraction" << endl;
}

int main() {
    cout << "==================================================================" << endl;
    cout << "     GENERIC MATH LIBRARY - ADVANCED ALGORITHMS SHOWCASE" << endl;
    cout << "==================================================================" << endl;
    cout << "Demonstrating elegant, composable algorithms following" << endl;
    cout << "Alex Stepanov's principles of generic programming" << endl;

    demo_fenwick_tree();
    demo_combinatorics();
    demo_sparse_matrix();
    demonstrate_generic_programming();

    cout << "\n==================================================================" << endl;
    cout << "These implementations showcase:" << endl;
    cout << "- Mathematical elegance through proper abstractions" << endl;
    cout << "- Efficiency without sacrificing generality" << endl;
    cout << "- Composability through well-defined interfaces" << endl;
    cout << "- Trade-offs explicitly documented and justified" << endl;
    cout << "==================================================================" << endl;

    return 0;
}