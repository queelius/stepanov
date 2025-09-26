#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <numeric>
#include <vector>
#include <stack>
#include <queue>
#include <set>
#include <map>
#include <optional>
#include <limits>
#include <array>
#include "concepts.hpp"
#include "iterators.hpp"
#include "ranges.hpp"

namespace stepanov {

// Generic point type with N dimensions
template<typename T, std::size_t N>
    requires std::regular<T>
struct point {
    std::array<T, N> coords;

    point() = default;

    template<typename... Args>
        requires (sizeof...(Args) == N) && (std::convertible_to<Args, T> && ...)
    point(Args&&... args) : coords{static_cast<T>(args)...} {}

    T& operator[](std::size_t i) { return coords[i]; }
    const T& operator[](std::size_t i) const { return coords[i]; }

    auto begin() { return coords.begin(); }
    auto end() { return coords.end(); }
    auto begin() const { return coords.begin(); }
    auto end() const { return coords.end(); }

    friend bool operator==(const point& a, const point& b) = default;
    friend auto operator<=>(const point& a, const point& b) = default;
};

// Convenience aliases
template<typename T> using point2d = point<T, 2>;
template<typename T> using point3d = point<T, 3>;

// Vector operations
template<typename T, std::size_t N>
point<T, N> operator+(const point<T, N>& a, const point<T, N>& b) {
    point<T, N> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<>{});
    return result;
}

template<typename T, std::size_t N>
point<T, N> operator-(const point<T, N>& a, const point<T, N>& b) {
    point<T, N> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<>{});
    return result;
}

template<typename T, std::size_t N>
point<T, N> operator*(const point<T, N>& p, const T& scalar) {
    point<T, N> result;
    std::transform(p.begin(), p.end(), result.begin(),
                   [scalar](const T& x) { return x * scalar; });
    return result;
}

// Dot product
template<typename T, std::size_t N>
T dot(const point<T, N>& a, const point<T, N>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), T{0});
}

// Cross product for 2D (returns scalar)
template<typename T>
T cross2d(const point2d<T>& a, const point2d<T>& b) {
    return a[0] * b[1] - a[1] * b[0];
}

// Cross product for 3D
template<typename T>
point3d<T> cross3d(const point3d<T>& a, const point3d<T>& b) {
    return point3d<T>{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

// Distance functions
template<typename T, std::size_t N>
    requires std::floating_point<T>
T euclidean_distance(const point<T, N>& a, const point<T, N>& b) {
    T sum{0};
    for (std::size_t i = 0; i < N; ++i) {
        T diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

template<typename T, std::size_t N>
T manhattan_distance(const point<T, N>& a, const point<T, N>& b) {
    T sum{0};
    for (std::size_t i = 0; i < N; ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

template<typename T, std::size_t N>
T squared_distance(const point<T, N>& a, const point<T, N>& b) {
    T sum{0};
    for (std::size_t i = 0; i < N; ++i) {
        T diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Geometric predicates

// Orientation test: returns positive if ccw, negative if cw, 0 if collinear
template<typename T>
T orientation(const point2d<T>& p, const point2d<T>& q, const point2d<T>& r) {
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
}

// Check if point r is on segment pq
template<typename T>
bool on_segment(const point2d<T>& p, const point2d<T>& q, const point2d<T>& r) {
    return r[0] <= std::max(p[0], q[0]) && r[0] >= std::min(p[0], q[0]) &&
           r[1] <= std::max(p[1], q[1]) && r[1] >= std::min(p[1], q[1]);
}

// Line segment intersection
template<typename T>
bool segments_intersect(const point2d<T>& p1, const point2d<T>& q1,
                        const point2d<T>& p2, const point2d<T>& q2) {
    T o1 = orientation(p1, q1, p2);
    T o2 = orientation(p1, q1, q2);
    T o3 = orientation(p2, q2, p1);
    T o4 = orientation(p2, q2, q1);

    // General case
    if (o1 * o2 < 0 && o3 * o4 < 0) return true;

    // Special cases (collinear points)
    if (o1 == 0 && on_segment(p1, q1, p2)) return true;
    if (o2 == 0 && on_segment(p1, q1, q2)) return true;
    if (o3 == 0 && on_segment(p2, q2, p1)) return true;
    if (o4 == 0 && on_segment(p2, q2, q1)) return true;

    return false;
}

// Convex hull algorithms

// Graham scan algorithm for convex hull
template<typename T>
std::vector<point2d<T>> graham_scan(std::vector<point2d<T>> points) {
    if (points.size() < 3) return points;

    // Find bottom-most point (and leftmost if tie)
    auto pivot = std::min_element(points.begin(), points.end(),
        [](const auto& a, const auto& b) {
            return a[1] < b[1] || (a[1] == b[1] && a[0] < b[0]);
        });

    std::swap(*points.begin(), *pivot);
    const auto& p0 = points[0];

    // Sort by polar angle with respect to pivot
    std::sort(points.begin() + 1, points.end(),
        [&p0](const auto& a, const auto& b) {
            T orient = orientation(p0, a, b);
            if (orient == 0) {
                return squared_distance(p0, a) < squared_distance(p0, b);
            }
            return orient > 0;
        });

    // Remove collinear points
    std::size_t m = 1;
    for (std::size_t i = 1; i < points.size(); ++i) {
        while (i < points.size() - 1 &&
               orientation(p0, points[i], points[i + 1]) == 0) {
            ++i;
        }
        points[m++] = points[i];
    }

    if (m < 3) return points;

    std::vector<point2d<T>> hull;
    hull.push_back(points[0]);
    hull.push_back(points[1]);
    hull.push_back(points[2]);

    for (std::size_t i = 3; i < m; ++i) {
        while (hull.size() > 1 &&
               orientation(hull[hull.size() - 2], hull.back(), points[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    return hull;
}

// Jarvis march (gift wrapping) algorithm
template<typename T>
std::vector<point2d<T>> jarvis_march(const std::vector<point2d<T>>& points) {
    if (points.size() < 3) return points;

    std::vector<point2d<T>> hull;

    // Find leftmost point
    std::size_t l = 0;
    for (std::size_t i = 1; i < points.size(); ++i) {
        if (points[i][0] < points[l][0]) l = i;
    }

    std::size_t p = l;
    do {
        hull.push_back(points[p]);
        std::size_t q = (p + 1) % points.size();

        for (std::size_t i = 0; i < points.size(); ++i) {
            if (orientation(points[p], points[i], points[q]) > 0) {
                q = i;
            }
        }
        p = q;
    } while (p != l);

    return hull;
}

// QuickHull algorithm
template<typename T>
class quick_hull {
private:
    std::vector<point2d<T>> hull;

    T distance_from_line(const point2d<T>& p,
                         const point2d<T>& a,
                         const point2d<T>& b) {
        return std::abs((b[0] - a[0]) * (a[1] - p[1]) -
                       (a[0] - p[0]) * (b[1] - a[1]));
    }

    void find_hull(const std::vector<point2d<T>>& points,
                   const point2d<T>& a,
                   const point2d<T>& b,
                   int side) {
        int ind = -1;
        T max_dist = 0;

        // Find point with maximum distance from line ab
        for (std::size_t i = 0; i < points.size(); ++i) {
            T dist = distance_from_line(points[i], a, b);
            if (orientation(a, b, points[i]) == side && dist > max_dist) {
                ind = i;
                max_dist = dist;
            }
        }

        if (ind == -1) {
            hull.push_back(a);
            hull.push_back(b);
            return;
        }

        // Recursively find points on both sides
        find_hull(points, points[ind], a, -orientation(points[ind], a, b));
        find_hull(points, points[ind], b, -orientation(points[ind], b, a));
    }

public:
    std::vector<point2d<T>> operator()(const std::vector<point2d<T>>& points) {
        if (points.size() < 3) return points;

        // Find leftmost and rightmost points
        auto [left, right] = std::minmax_element(points.begin(), points.end(),
            [](const auto& a, const auto& b) { return a[0] < b[0]; });

        // Divide points into two sets and process
        find_hull(points, *left, *right, 1);
        find_hull(points, *left, *right, -1);

        // Sort hull points
        std::sort(hull.begin(), hull.end());
        hull.erase(std::unique(hull.begin(), hull.end()), hull.end());

        return graham_scan(hull);  // Final cleanup
    }
};

// Closest pair of points using divide & conquer
template<typename T>
struct closest_pair_result {
    point2d<T> p1, p2;
    T distance;
};

template<typename T>
class closest_pair {
private:
    T brute_force(std::vector<point2d<T>>& points, std::size_t l, std::size_t r) {
        T min_dist = std::numeric_limits<T>::max();
        for (std::size_t i = l; i < r; ++i) {
            for (std::size_t j = i + 1; j < r; ++j) {
                min_dist = std::min(min_dist, squared_distance(points[i], points[j]));
            }
        }
        return min_dist;
    }

    T strip_closest(std::vector<point2d<T>>& strip, T d) {
        T min_dist = d;

        std::sort(strip.begin(), strip.end(),
                  [](const auto& a, const auto& b) { return a[1] < b[1]; });

        for (std::size_t i = 0; i < strip.size(); ++i) {
            for (std::size_t j = i + 1;
                 j < strip.size() && (strip[j][1] - strip[i][1]) * (strip[j][1] - strip[i][1]) < min_dist;
                 ++j) {
                min_dist = std::min(min_dist, squared_distance(strip[i], strip[j]));
            }
        }

        return min_dist;
    }

    T closest_util(std::vector<point2d<T>>& px,
                   std::vector<point2d<T>>& py,
                   std::size_t l, std::size_t r) {
        if (r - l <= 3) {
            return brute_force(px, l, r);
        }

        std::size_t mid = l + (r - l) / 2;
        point2d<T> midpoint = px[mid];

        std::vector<point2d<T>> pyl, pyr;
        for (const auto& p : py) {
            if (p[0] <= midpoint[0]) pyl.push_back(p);
            else pyr.push_back(p);
        }

        T dl = closest_util(px, pyl, l, mid);
        T dr = closest_util(px, pyr, mid, r);

        T d = std::min(dl, dr);

        std::vector<point2d<T>> strip;
        for (const auto& p : py) {
            if (std::abs(p[0] - midpoint[0]) * std::abs(p[0] - midpoint[0]) < d) {
                strip.push_back(p);
            }
        }

        return std::min(d, strip_closest(strip, d));
    }

public:
    closest_pair_result<T> operator()(std::vector<point2d<T>> points) {
        auto px = points;
        auto py = points;

        std::sort(px.begin(), px.end(),
                  [](const auto& a, const auto& b) { return a[0] < b[0]; });
        std::sort(py.begin(), py.end(),
                  [](const auto& a, const auto& b) { return a[1] < b[1]; });

        T min_dist_sq = closest_util(px, py, 0, points.size());

        // Find the actual pair
        closest_pair_result<T> result;
        result.distance = std::sqrt(min_dist_sq);

        for (std::size_t i = 0; i < points.size(); ++i) {
            for (std::size_t j = i + 1; j < points.size(); ++j) {
                if (squared_distance(points[i], points[j]) == min_dist_sq) {
                    result.p1 = points[i];
                    result.p2 = points[j];
                    return result;
                }
            }
        }

        return result;
    }
};

// K-d tree for spatial indexing
template<typename T, std::size_t K>
class kd_tree {
private:
    struct node {
        point<T, K> pt;
        std::unique_ptr<node> left;
        std::unique_ptr<node> right;

        node(const point<T, K>& p) : pt(p) {}
    };

    std::unique_ptr<node> root;

    std::unique_ptr<node> build(std::vector<point<T, K>>& points,
                                std::size_t l, std::size_t r, std::size_t depth) {
        if (l >= r) return nullptr;

        std::size_t axis = depth % K;
        std::size_t mid = l + (r - l) / 2;

        std::nth_element(points.begin() + l, points.begin() + mid, points.begin() + r,
            [axis](const auto& a, const auto& b) { return a[axis] < b[axis]; });

        auto n = std::make_unique<node>(points[mid]);
        n->left = build(points, l, mid, depth + 1);
        n->right = build(points, mid + 1, r, depth + 1);

        return n;
    }

    void range_search_impl(const node* n,
                          const point<T, K>& lower,
                          const point<T, K>& upper,
                          std::vector<point<T, K>>& result,
                          std::size_t depth) const {
        if (!n) return;

        bool in_range = true;
        for (std::size_t i = 0; i < K; ++i) {
            if (n->pt[i] < lower[i] || n->pt[i] > upper[i]) {
                in_range = false;
                break;
            }
        }

        if (in_range) {
            result.push_back(n->pt);
        }

        std::size_t axis = depth % K;

        if (lower[axis] <= n->pt[axis]) {
            range_search_impl(n->left.get(), lower, upper, result, depth + 1);
        }
        if (upper[axis] >= n->pt[axis]) {
            range_search_impl(n->right.get(), lower, upper, result, depth + 1);
        }
    }

    void nearest_impl(const node* n,
                     const point<T, K>& target,
                     point<T, K>& best,
                     T& best_dist,
                     std::size_t depth) const {
        if (!n) return;

        T dist = squared_distance(n->pt, target);
        if (dist < best_dist) {
            best_dist = dist;
            best = n->pt;
        }

        std::size_t axis = depth % K;
        T diff = target[axis] - n->pt[axis];

        node* good_side = diff < 0 ? n->left.get() : n->right.get();
        node* bad_side = diff < 0 ? n->right.get() : n->left.get();

        nearest_impl(good_side, target, best, best_dist, depth + 1);

        if (diff * diff < best_dist) {
            nearest_impl(bad_side, target, best, best_dist, depth + 1);
        }
    }

public:
    kd_tree() = default;

    explicit kd_tree(std::vector<point<T, K>> points) {
        if (!points.empty()) {
            root = build(points, 0, points.size(), 0);
        }
    }

    std::vector<point<T, K>> range_search(const point<T, K>& lower,
                                          const point<T, K>& upper) const {
        std::vector<point<T, K>> result;
        range_search_impl(root.get(), lower, upper, result, 0);
        return result;
    }

    std::optional<point<T, K>> nearest_neighbor(const point<T, K>& target) const {
        if (!root) return std::nullopt;

        point<T, K> best = root->pt;
        T best_dist = squared_distance(root->pt, target);
        nearest_impl(root.get(), target, best, best_dist, 0);

        return best;
    }
};

// R-tree for range queries (simplified 2D version)
template<typename T>
class rtree {
private:
    struct rect {
        point2d<T> min, max;

        bool contains(const point2d<T>& p) const {
            return p[0] >= min[0] && p[0] <= max[0] &&
                   p[1] >= min[1] && p[1] <= max[1];
        }

        bool intersects(const rect& r) const {
            return !(r.max[0] < min[0] || r.min[0] > max[0] ||
                    r.max[1] < min[1] || r.min[1] > max[1]);
        }

        T area() const {
            return (max[0] - min[0]) * (max[1] - min[1]);
        }

        rect mbr_with(const rect& r) const {
            return rect{
                point2d<T>{std::min(min[0], r.min[0]), std::min(min[1], r.min[1])},
                point2d<T>{std::max(max[0], r.max[0]), std::max(max[1], r.max[1])}
            };
        }
    };

    struct node {
        rect mbr;
        std::vector<std::unique_ptr<node>> children;
        std::vector<point2d<T>> points;  // For leaf nodes
        bool is_leaf = true;

        static constexpr std::size_t max_entries = 4;
        static constexpr std::size_t min_entries = 2;
    };

    std::unique_ptr<node> root;

    void query_impl(const node* n, const rect& r, std::vector<point2d<T>>& result) const {
        if (!n || !n->mbr.intersects(r)) return;

        if (n->is_leaf) {
            for (const auto& p : n->points) {
                if (r.contains(p)) {
                    result.push_back(p);
                }
            }
        } else {
            for (const auto& child : n->children) {
                query_impl(child.get(), r, result);
            }
        }
    }

public:
    rtree() : root(std::make_unique<node>()) {}

    void insert(const point2d<T>& p) {
        if (root->is_leaf) {
            root->points.push_back(p);

            // Update MBR
            if (root->points.size() == 1) {
                root->mbr = rect{p, p};
            } else {
                root->mbr.min[0] = std::min(root->mbr.min[0], p[0]);
                root->mbr.min[1] = std::min(root->mbr.min[1], p[1]);
                root->mbr.max[0] = std::max(root->mbr.max[0], p[0]);
                root->mbr.max[1] = std::max(root->mbr.max[1], p[1]);
            }

            // Split if necessary (simplified)
            if (root->points.size() > node::max_entries) {
                // In a real implementation, this would split the node
                // For simplicity, we just keep all points in the root
            }
        }
        // Simplified: only handle single-level tree
    }

    std::vector<point2d<T>> range_query(const point2d<T>& min, const point2d<T>& max) const {
        std::vector<point2d<T>> result;
        query_impl(root.get(), rect{min, max}, result);
        return result;
    }
};

// In-circle test for Delaunay triangulation
template<typename T>
    requires std::floating_point<T>
bool in_circle(const point2d<T>& a, const point2d<T>& b,
               const point2d<T>& c, const point2d<T>& d) {
    T ax = a[0] - d[0];
    T ay = a[1] - d[1];
    T bx = b[0] - d[0];
    T by = b[1] - d[1];
    T cx = c[0] - d[0];
    T cy = c[1] - d[1];

    T det = (ax * ax + ay * ay) * (bx * cy - cx * by) -
            (bx * bx + by * by) * (ax * cy - cx * ay) +
            (cx * cx + cy * cy) * (ax * by - bx * ay);

    return det > 0;
}

// Simple Delaunay triangulation using incremental algorithm
template<typename T>
class delaunay_triangulation {
public:
    struct triangle {
        std::size_t a, b, c;

        bool contains_vertex(std::size_t v) const {
            return a == v || b == v || c == v;
        }

        bool operator==(const triangle& other) const {
            return a == other.a && b == other.b && c == other.c;
        }
    };

private:
    std::vector<point2d<T>> points;
    std::vector<triangle> triangles;

public:
    void add_point(const point2d<T>& p) {
        std::size_t idx = points.size();
        points.push_back(p);

        if (points.size() == 3) {
            // First triangle
            triangles.push_back({0, 1, 2});
            return;
        }

        // Find triangles whose circumcircle contains the new point
        std::vector<triangle> bad_triangles;
        for (const auto& tri : triangles) {
            if (in_circle(points[tri.a], points[tri.b], points[tri.c], p)) {
                bad_triangles.push_back(tri);
            }
        }

        // Find the boundary of the polygonal hole
        std::set<std::pair<std::size_t, std::size_t>> polygon;
        for (const auto& tri : bad_triangles) {
            std::pair<std::size_t, std::size_t> edges[3] = {
                {tri.a, tri.b}, {tri.b, tri.c}, {tri.c, tri.a}
            };

            for (const auto& edge : edges) {
                bool shared = false;
                for (const auto& other : bad_triangles) {
                    if (&other == &tri) continue;

                    if ((other.contains_vertex(edge.first) && other.contains_vertex(edge.second))) {
                        shared = true;
                        break;
                    }
                }

                if (!shared) {
                    polygon.insert(edge);
                }
            }
        }

        // Remove bad triangles
        triangles.erase(
            std::remove_if(triangles.begin(), triangles.end(),
                [&bad_triangles](const triangle& t) {
                    return std::find(bad_triangles.begin(), bad_triangles.end(), t) != bad_triangles.end();
                }),
            triangles.end());

        // Re-triangulate the polygonal hole
        for (const auto& edge : polygon) {
            triangles.push_back({edge.first, edge.second, idx});
        }
    }

    const std::vector<triangle>& get_triangles() const { return triangles; }
    const std::vector<point2d<T>>& get_points() const { return points; }
};

// Line segment representation
template<typename T>
struct segment {
    point2d<T> start, end;

    segment(const point2d<T>& s, const point2d<T>& e) : start(s), end(e) {}
};

// Bentley-Ottmann algorithm for line segment intersection
template<typename T>
class bentley_ottmann {
private:
    enum class event_type { start, end, intersection };

    struct event {
        point2d<T> point;
        event_type type;
        std::size_t segment_id;
        std::optional<std::size_t> other_segment_id;  // For intersection events

        bool operator<(const event& other) const {
            if (point[0] != other.point[0]) return point[0] < other.point[0];
            if (point[1] != other.point[1]) return point[1] < other.point[1];
            return type < other.type;
        }
    };

public:
    std::vector<point2d<T>> find_intersections(const std::vector<segment<T>>& segments) {
        std::vector<point2d<T>> intersections;
        std::priority_queue<event, std::vector<event>, std::greater<>> event_queue;

        // Initialize with segment endpoints
        for (std::size_t i = 0; i < segments.size(); ++i) {
            event_queue.push({segments[i].start, event_type::start, i});
            event_queue.push({segments[i].end, event_type::end, i});
        }

        // Process events (simplified version)
        while (!event_queue.empty()) {
            event e = event_queue.top();
            event_queue.pop();

            if (e.type == event_type::intersection) {
                intersections.push_back(e.point);
            }

            // In a complete implementation, we would maintain a sweep line status
            // and check for new intersections
        }

        return intersections;
    }
};

} // namespace stepanov