// Advanced Data Structures Demo
// Showcases innovative and efficient data structures

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <stepanov/disjoint_interval_set.hpp>
#include <stepanov/fenwick_tree.hpp>
#include <stepanov/bounded_nat.hpp>
#include <stepanov/succinct.hpp>
#include <stepanov/persistent.hpp>

using namespace stepanov;
using namespace std;

int main() {
    cout << "=== Advanced Data Structures Demo ===\n\n";

    // 1. Disjoint Interval Sets
    cout << "1. DISJOINT INTERVAL SETS\n";
    cout << "-------------------------\n";
    cout << "Efficiently manages non-overlapping intervals:\n\n";

    disjoint_interval_set<int> intervals;

    // Insert intervals (automatically merges overlapping)
    intervals.insert(10, 20);
    intervals.insert(30, 40);
    intervals.insert(15, 35);  // Merges with [10,20] and [30,40]

    cout << "After inserting [10,20], [30,40], [15,35]:\n";
    cout << "  Intervals: ";
    for (const auto& [start, end] : intervals) {
        cout << "[" << start << "," << end << ") ";
    }
    cout << "\n";

    // Query operations
    cout << "  Contains 25? " << (intervals.contains(25) ? "Yes" : "No") << "\n";
    cout << "  Contains 45? " << (intervals.contains(45) ? "Yes" : "No") << "\n";
    cout << "  Total coverage: " << intervals.total_length() << "\n\n";

    // 2. Fenwick Trees (Binary Indexed Trees)
    cout << "2. FENWICK TREES\n";
    cout << "----------------\n";
    cout << "O(log n) range sum queries and updates:\n\n";

    fenwick_tree<int> tree(10);

    // Update some values
    tree.update(0, 3);
    tree.update(2, 5);
    tree.update(5, 2);
    tree.update(7, 4);

    cout << "Array after updates: ";
    for (int i = 0; i < 10; ++i) {
        cout << tree.point_query(i) << " ";
    }
    cout << "\n";

    cout << "  Range sum [0,5]: " << tree.range_sum(0, 5) << "\n";
    cout << "  Range sum [2,7]: " << tree.range_sum(2, 7) << "\n";

    // Dynamic updates
    cout << "After incrementing position 3 by 10:\n";
    tree.update(3, 10);
    cout << "  Range sum [0,5]: " << tree.range_sum(0, 5) << "\n\n";

    // 3. Bounded Natural Numbers
    cout << "3. BOUNDED NATURAL NUMBERS\n";
    cout << "--------------------------\n";
    cout << "Fixed-size arbitrary precision arithmetic:\n\n";

    using uint256 = bounded_nat<4>;  // 256-bit unsigned integer

    uint256 a = uint256::from_string("123456789012345678901234567890");
    uint256 b = uint256::from_string("987654321098765432109876543210");

    cout << "256-bit arithmetic:\n";
    cout << "  a = " << a.to_string() << "\n";
    cout << "  b = " << b.to_string() << "\n";
    cout << "  a + b = " << (a + b).to_string() << "\n";
    cout << "  a * 2 = " << (a * uint256(2)).to_string() << "\n\n";

    // 4. Succinct Data Structures
    cout << "4. SUCCINCT DATA STRUCTURES\n";
    cout << "---------------------------\n";
    cout << "Near-optimal space with fast operations:\n\n";

    // Rank/Select structure for bit vectors
    vector<bool> bits = {1,0,1,1,0,1,0,0,1,1};
    rank_select rs(bits);

    cout << "Bit vector: ";
    for (bool b : bits) cout << b;
    cout << "\n";

    cout << "  rank(5) = " << rs.rank1(5) << " (ones up to position 5)\n";
    cout << "  select(3) = " << rs.select1(3) << " (position of 3rd one)\n";

    // Succinct tree representation
    cout << "\nSuccinct tree (2n bits for n nodes):\n";
    succinct_tree tree_repr("(()(()()))");  // Balanced parentheses

    cout << "  Tree: (()(()()))\n";
    cout << "  Nodes: " << tree_repr.size() << "\n";
    cout << "  Depth of node 3: " << tree_repr.depth(3) << "\n";
    cout << "  Parent of node 4: " << tree_repr.parent(4) << "\n\n";

    // 5. Persistent Data Structures
    cout << "5. PERSISTENT DATA STRUCTURES\n";
    cout << "-----------------------------\n";
    cout << "Immutable with efficient updates:\n\n";

    // Persistent vector (path copying)
    persistent_vector<int> v1;
    v1 = v1.push_back(10);
    v1 = v1.push_back(20);
    v1 = v1.push_back(30);

    cout << "Version 1: ";
    for (size_t i = 0; i < v1.size(); ++i) {
        cout << v1[i] << " ";
    }
    cout << "\n";

    // Create new version by updating
    auto v2 = v1.set(1, 99);

    cout << "Version 2 (after v1.set(1, 99)): ";
    for (size_t i = 0; i < v2.size(); ++i) {
        cout << v2[i] << " ";
    }
    cout << "\n";

    cout << "Version 1 unchanged: ";
    for (size_t i = 0; i < v1.size(); ++i) {
        cout << v1[i] << " ";
    }
    cout << "\n\n";

    // Persistent map (using path copying)
    persistent_map<string, int> map1;
    map1 = map1.insert("apple", 5);
    map1 = map1.insert("banana", 3);

    auto map2 = map1.insert("cherry", 7);
    auto map3 = map2.erase("apple");

    cout << "Persistent map versions:\n";
    cout << "  Version 1 has 'apple': " << (map1.contains("apple") ? "Yes" : "No") << "\n";
    cout << "  Version 3 has 'apple': " << (map3.contains("apple") ? "Yes" : "No") << "\n";
    cout << "  All versions have 'banana': Yes\n\n";

    // 6. Cache-Oblivious Algorithms
    cout << "6. CACHE-OBLIVIOUS STRUCTURES\n";
    cout << "-----------------------------\n";
    cout << "Optimal cache performance without tuning:\n\n";

    cout << "Van Emde Boas layout for binary trees:\n";
    cout << "  Recursive subdivision minimizes cache misses\n";
    cout << "  Works optimally for ANY cache size\n";
    cout << "  No parameters to tune!\n\n";

    cout << "Key Insight: These structures push the boundaries\n";
    cout << "of space/time tradeoffs and enable algorithms\n";
    cout << "that were previously impractical.\n";

    return 0;
}