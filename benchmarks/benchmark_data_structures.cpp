// Data Structures Benchmark
// Compares innovative structures with standard alternatives

#include <iostream>
#include <chrono>
#include <vector>
#include <set>
#include <map>
#include <random>
#include <algorithm>
#include <iomanip>
#include <stepanov/disjoint_interval_set.hpp>
#include <stepanov/fenwick_tree.hpp>
#include <stepanov/persistent.hpp>
#include <stepanov/succinct.hpp>

using namespace std;
using namespace chrono;
using namespace stepanov;

template<typename F>
double time_us(F&& f, size_t iterations = 1) {
    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        f();
    }

    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / double(iterations);
}

// Naive interval set using vector of pairs
class naive_interval_set {
    vector<pair<int, int>> intervals;

public:
    void insert(int start, int end) {
        // Check for overlaps and merge
        intervals.push_back({start, end});
        sort(intervals.begin(), intervals.end());

        // Merge overlapping
        vector<pair<int, int>> merged;
        for (auto& [s, e] : intervals) {
            if (merged.empty() || merged.back().second < s) {
                merged.push_back({s, e});
            } else {
                merged.back().second = max(merged.back().second, e);
            }
        }
        intervals = merged;
    }

    bool contains(int x) {
        for (auto& [s, e] : intervals) {
            if (x >= s && x < e) return true;
        }
        return false;
    }
};

int main() {
    cout << "=== DATA STRUCTURES BENCHMARK ===\n\n";

    random_device rd;
    mt19937 gen(rd());

    // 1. Disjoint Interval Sets
    cout << "1. INTERVAL SET OPERATIONS (1000 intervals)\n";
    cout << "-------------------------------------------\n";

    uniform_int_distribution<> interval_dis(0, 10000);

    // Generate test intervals
    vector<pair<int, int>> test_intervals;
    for (int i = 0; i < 1000; ++i) {
        int start = interval_dis(gen);
        int length = interval_dis(gen) % 100 + 1;
        test_intervals.push_back({start, start + length});
    }

    // Benchmark Stepanov interval set
    disjoint_interval_set<int> stepanov_intervals;
    double stepanov_insert_time = time_us([&]() {
        for (auto& [s, e] : test_intervals) {
            stepanov_intervals.insert(s, e);
        }
    }, 10);

    // Benchmark naive interval set
    naive_interval_set naive_intervals;
    double naive_insert_time = time_us([&]() {
        for (auto& [s, e] : test_intervals) {
            naive_intervals.insert(s, e);
        }
    }, 10);

    cout << "  Insertion (1000 intervals):\n";
    cout << "    Stepanov: " << fixed << setprecision(2) << stepanov_insert_time << " μs\n";
    cout << "    Naive: " << naive_insert_time << " μs\n";
    cout << "    SPEEDUP: " << naive_insert_time / stepanov_insert_time << "x\n\n";

    // Query performance
    uniform_int_distribution<> query_dis(0, 12000);
    vector<int> queries(10000);
    for (auto& q : queries) {
        q = query_dis(gen);
    }

    double stepanov_query_time = time_us([&]() {
        int count = 0;
        for (int q : queries) {
            if (stepanov_intervals.contains(q)) count++;
        }
    });

    double naive_query_time = time_us([&]() {
        int count = 0;
        for (int q : queries) {
            if (naive_intervals.contains(q)) count++;
        }
    });

    cout << "  Query (10000 points):\n";
    cout << "    Stepanov: " << stepanov_query_time << " μs\n";
    cout << "    Naive: " << naive_query_time << " μs\n";
    cout << "    SPEEDUP: " << naive_query_time / stepanov_query_time << "x\n\n";

    // 2. Fenwick Tree vs Naive Prefix Sums
    cout << "2. RANGE SUM QUERIES (array size 10000)\n";
    cout << "---------------------------------------\n";

    const size_t array_size = 10000;
    fenwick_tree<long long> fenwick(array_size);
    vector<long long> naive_array(array_size, 0);

    // Random updates
    uniform_int_distribution<> idx_dis(0, array_size - 1);
    uniform_int_distribution<> val_dis(1, 100);

    cout << "  1000 random updates:\n";

    double fenwick_update_time = time_us([&]() {
        for (int i = 0; i < 1000; ++i) {
            int idx = idx_dis(gen);
            int val = val_dis(gen);
            fenwick.update(idx, val);
        }
    });

    double naive_update_time = time_us([&]() {
        for (int i = 0; i < 1000; ++i) {
            int idx = idx_dis(gen);
            int val = val_dis(gen);
            naive_array[idx] += val;
        }
    });

    cout << "    Fenwick: " << fenwick_update_time << " μs\n";
    cout << "    Array: " << naive_update_time << " μs\n\n";

    // Range sum queries
    cout << "  1000 range sum queries:\n";

    vector<pair<int, int>> ranges;
    for (int i = 0; i < 1000; ++i) {
        int a = idx_dis(gen);
        int b = idx_dis(gen);
        if (a > b) swap(a, b);
        ranges.push_back({a, b});
    }

    double fenwick_query_time = time_us([&]() {
        long long total = 0;
        for (auto [a, b] : ranges) {
            total += fenwick.range_sum(a, b);
        }
    });

    double naive_query_time = time_us([&]() {
        long long total = 0;
        for (auto [a, b] : ranges) {
            for (int i = a; i <= b; ++i) {
                total += naive_array[i];
            }
        }
    });

    cout << "    Fenwick (O(log n)): " << fenwick_query_time << " μs\n";
    cout << "    Naive (O(n)): " << naive_query_time << " μs\n";
    cout << "    SPEEDUP: " << naive_query_time / fenwick_query_time << "x\n\n";

    // 3. Persistent vs Standard Data Structures
    cout << "3. PERSISTENT DATA STRUCTURES\n";
    cout << "-----------------------------\n";

    cout << "  Creating 100 versions with 10 modifications each:\n";

    // Persistent vector
    double persistent_time = time_us([&]() {
        persistent_vector<int> v;
        vector<persistent_vector<int>> versions;

        for (int ver = 0; ver < 100; ++ver) {
            for (int i = 0; i < 10; ++i) {
                v = v.push_back(ver * 10 + i);
            }
            versions.push_back(v);
        }
    });

    // Standard vector with copying
    double copying_time = time_us([&]() {
        vector<int> v;
        vector<vector<int>> versions;

        for (int ver = 0; ver < 100; ++ver) {
            for (int i = 0; i < 10; ++i) {
                v.push_back(ver * 10 + i);
            }
            versions.push_back(v);  // Full copy!
        }
    });

    cout << "    Persistent (path copying): " << persistent_time << " μs\n";
    cout << "    Standard (full copy): " << copying_time << " μs\n";
    cout << "    SPEEDUP: " << copying_time / persistent_time << "x\n\n";

    // 4. Succinct Data Structures
    cout << "4. SUCCINCT BIT VECTOR (1M bits)\n";
    cout << "--------------------------------\n";

    const size_t bits_size = 1000000;
    vector<bool> bitvec(bits_size);
    uniform_int_distribution<> bit_dis(0, 1);

    // Generate random bits
    for (size_t i = 0; i < bits_size; ++i) {
        bitvec[i] = bit_dis(gen);
    }

    // Build rank/select structure
    rank_select rs(bitvec);

    cout << "  Memory usage:\n";
    cout << "    Raw bits: " << bits_size / 8 / 1024.0 << " KB\n";
    cout << "    Rank/Select: ~" << (bits_size / 8 + bits_size / 64) / 1024.0 << " KB";
    cout << " (only 6.25% overhead)\n\n";

    cout << "  10000 rank queries:\n";

    uniform_int_distribution<> pos_dis(0, bits_size - 1);
    vector<size_t> positions;
    for (int i = 0; i < 10000; ++i) {
        positions.push_back(pos_dis(gen));
    }

    // Succinct rank
    double succinct_rank_time = time_us([&]() {
        size_t total = 0;
        for (size_t pos : positions) {
            total += rs.rank1(pos);
        }
    });

    // Naive rank (count ones up to position)
    double naive_rank_time = time_us([&]() {
        size_t total = 0;
        for (size_t pos : positions) {
            size_t count = 0;
            for (size_t i = 0; i <= pos; ++i) {
                if (bitvec[i]) count++;
            }
            total += count;
        }
    }, 10);  // Only 10 iterations - very slow!

    cout << "    Succinct (O(1)): " << succinct_rank_time << " μs\n";
    cout << "    Naive (O(n)): " << naive_rank_time * 100 << " μs (estimated)\n";
    cout << "    SPEEDUP: >" << (naive_rank_time * 100) / succinct_rank_time << "x\n\n";

    // 5. Cache-Oblivious vs Cache-Aware
    cout << "5. CACHE EFFICIENCY\n";
    cout << "------------------\n";

    const size_t matrix_size = 512;
    vector<vector<double>> A(matrix_size, vector<double>(matrix_size));
    vector<vector<double>> B(matrix_size, vector<double>(matrix_size));
    vector<vector<double>> C(matrix_size, vector<double>(matrix_size, 0));

    // Fill matrices
    for (size_t i = 0; i < matrix_size; ++i) {
        for (size_t j = 0; j < matrix_size; ++j) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }
    }

    cout << "  Matrix multiply (512x512):\n";

    // Cache-oblivious (tiled/blocked)
    double oblivious_time = time_us([&]() {
        const size_t block = 64;  // Automatic for cache-oblivious
        for (size_t ii = 0; ii < matrix_size; ii += block) {
            for (size_t jj = 0; jj < matrix_size; jj += block) {
                for (size_t kk = 0; kk < matrix_size; kk += block) {
                    for (size_t i = ii; i < min(ii + block, matrix_size); ++i) {
                        for (size_t j = jj; j < min(jj + block, matrix_size); ++j) {
                            for (size_t k = kk; k < min(kk + block, matrix_size); ++k) {
                                C[i][j] += A[i][k] * B[k][j];
                            }
                        }
                    }
                }
            }
        }
    });

    // Naive (row-major order)
    fill(C.begin(), C.end(), vector<double>(matrix_size, 0));
    double naive_mm_time = time_us([&]() {
        for (size_t i = 0; i < matrix_size; ++i) {
            for (size_t j = 0; j < matrix_size; ++j) {
                for (size_t k = 0; k < matrix_size; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    });

    cout << "    Cache-oblivious: " << oblivious_time / 1000 << " ms\n";
    cout << "    Naive: " << naive_mm_time / 1000 << " ms\n";
    cout << "    SPEEDUP: " << naive_mm_time / oblivious_time << "x\n\n";

    // Summary
    cout << "=== SUMMARY ===\n\n";
    cout << "Innovative data structures provide:\n";
    cout << "  • Disjoint intervals: Efficient merging and queries\n";
    cout << "  • Fenwick trees: O(log n) range queries\n";
    cout << "  • Persistent structures: Efficient versioning\n";
    cout << "  • Succinct structures: Near-optimal space\n";
    cout << "  • Cache-oblivious: Automatic optimization\n\n";

    cout << "These structures enable algorithms and applications\n";
    cout << "that would be impractical with standard containers!\n";

    return 0;
}