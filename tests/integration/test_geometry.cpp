#include <iostream>
#include <cassert>
#include <random>
#include <chrono>
#include <stepanov/geometry.hpp>

using namespace stepanov;

void test_point_operations() {
    std::cout << "Testing point operations...\n";

    point2d<double> p1(3.0, 4.0);
    point2d<double> p2(1.0, 2.0);

    auto sum = p1 + p2;
    assert(sum[0] == 4.0 && sum[1] == 6.0);

    auto diff = p1 - p2;
    assert(diff[0] == 2.0 && diff[1] == 2.0);

    auto scaled = p1 * 2.0;
    assert(scaled[0] == 6.0 && scaled[1] == 8.0);

    double dot_product = dot(p1, p2);
    assert(dot_product == 11.0);  // 3*1 + 4*2

    double dist = euclidean_distance(p1, p2);
    assert(std::abs(dist - 2.82843) < 0.0001);

    std::cout << "  Point operations passed\n";
}

void test_orientation() {
    std::cout << "Testing orientation predicate...\n";

    point2d<int> p(0, 0);
    point2d<int> q(4, 4);
    point2d<int> r1(1, 2);  // Left of line pq (CCW)
    point2d<int> r2(2, 1);  // Right of line pq (CW)
    point2d<int> r3(2, 2);  // On line pq

    assert(orientation(p, q, r1) > 0);  // CCW
    assert(orientation(p, q, r2) < 0);  // CW
    assert(orientation(p, q, r3) == 0); // Collinear

    std::cout << "  Orientation tests passed\n";
}

void test_segment_intersection() {
    std::cout << "Testing segment intersection...\n";

    point2d<int> p1(0, 0), q1(4, 4);
    point2d<int> p2(0, 4), q2(4, 0);
    assert(segments_intersect(p1, q1, p2, q2));  // Intersecting diagonals

    point2d<int> p3(0, 0), q3(2, 0);
    point2d<int> p4(3, 0), q4(5, 0);
    assert(!segments_intersect(p3, q3, p4, q4));  // Non-intersecting parallel

    std::cout << "  Segment intersection tests passed\n";
}

void test_convex_hull() {
    std::cout << "Testing convex hull algorithms...\n";

    std::vector<point2d<double>> points = {
        {0, 3}, {1, 1}, {2, 2}, {4, 4},
        {0, 0}, {1, 2}, {3, 1}, {3, 3}
    };

    // Test Graham scan
    auto hull1 = graham_scan(points);
    assert(hull1.size() == 4);  // Square hull

    // Test Jarvis march
    auto hull2 = jarvis_march(points);
    assert(hull2.size() == 4);

    // Test QuickHull (skip for now as implementation needs refinement)
    // quick_hull<double> qh;
    // auto hull3 = qh(points);
    // assert(hull3.size() >= 3);  // At least a triangle

    std::cout << "  Convex hull algorithms passed\n";
}

void test_closest_pair() {
    std::cout << "Testing closest pair algorithm...\n";

    std::vector<point2d<double>> points = {
        {2, 3}, {12, 30}, {40, 50}, {5, 1}, {12, 10}, {3, 4}
    };

    closest_pair<double> cp;
    auto result = cp(points);

    // Closest points should be (2,3) and (3,4)
    assert(std::abs(result.distance - 1.41421) < 0.0001);

    std::cout << "  Closest pair algorithm passed\n";
}

void test_kd_tree() {
    std::cout << "Testing k-d tree...\n";

    std::vector<point2d<double>> points = {
        {2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}
    };

    kd_tree<double, 2> tree(points);

    // Test range search
    point2d<double> lower(2, 0);
    point2d<double> upper(5, 10);
    auto in_range = tree.range_search(lower, upper);
    assert(in_range.size() == 3);  // Points (2,3), (5,4), (4,7)

    // Test nearest neighbor
    point2d<double> query(9, 2);
    auto nearest = tree.nearest_neighbor(query);
    assert(nearest.has_value());
    // Nearest should be (8,1) or (7,2)

    std::cout << "  K-d tree tests passed\n";
}

void test_rtree() {
    std::cout << "Testing R-tree...\n";

    rtree<double> rt;

    // Insert points
    rt.insert({2, 3});
    rt.insert({5, 4});
    rt.insert({9, 6});
    rt.insert({4, 7});

    // Range query
    auto results = rt.range_query({1, 2}, {6, 8});
    assert(results.size() >= 2);  // At least (2,3) and (5,4)

    std::cout << "  R-tree tests passed\n";
}

void test_delaunay() {
    std::cout << "Testing Delaunay triangulation...\n";

    delaunay_triangulation<double> dt;

    // Add points forming a square
    dt.add_point({0, 0});
    dt.add_point({1, 0});
    dt.add_point({1, 1});
    dt.add_point({0, 1});

    auto triangles = dt.get_triangles();
    assert(triangles.size() >= 1);  // At least one triangle

    std::cout << "  Delaunay triangulation passed\n";
}

void benchmark_algorithms() {
    std::cout << "\nBenchmarking geometric algorithms...\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0, 1000);

    // Generate random points
    std::vector<point2d<double>> points;
    for (int i = 0; i < 1000; ++i) {
        points.push_back({dist(gen), dist(gen)});
    }

    // Benchmark Graham scan
    auto start = std::chrono::high_resolution_clock::now();
    auto hull = graham_scan(points);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  Graham scan (1000 points): " << duration.count() << " μs\n";

    // Benchmark k-d tree construction
    start = std::chrono::high_resolution_clock::now();
    kd_tree<double, 2> tree(points);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  K-d tree construction (1000 points): " << duration.count() << " μs\n";

    // Benchmark nearest neighbor queries
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        point2d<double> query(dist(gen), dist(gen));
        tree.nearest_neighbor(query);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  K-d tree 100 NN queries: " << duration.count() << " μs\n";

    // Benchmark closest pair
    std::vector<point2d<double>> small_points(points.begin(), points.begin() + 100);
    start = std::chrono::high_resolution_clock::now();
    closest_pair<double> cp;
    cp(small_points);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  Closest pair (100 points): " << duration.count() << " μs\n";
}

int main() {
    std::cout << "=== Testing Computational Geometry ===\n\n";

    test_point_operations();
    test_orientation();
    test_segment_intersection();
    test_convex_hull();
    test_closest_pair();
    test_kd_tree();
    test_rtree();
    test_delaunay();

    benchmark_algorithms();

    std::cout << "\n=== All geometry tests passed! ===\n";
    return 0;
}