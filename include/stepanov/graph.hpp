#pragma once

#include <vector>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <functional>
#include <limits>
#include <algorithm>
#include <span>
#include <ranges>
#include "concepts.hpp"

namespace stepanov {

/**
 * Generic graph algorithms following Stepanov's principles
 *
 * These algorithms work with any graph representation that models
 * the required concepts. The graph doesn't need to be a specific
 * data structure - it just needs to provide the required operations.
 */

// =============================================================================
// Graph concepts
// =============================================================================

template<typename G>
concept vertex_enumerable = requires(G g) {
    { g.vertices() } -> std::ranges::range;
};

template<typename G>
concept edge_enumerable = requires(G g) {
    { g.edges() } -> std::ranges::range;
};

template<typename G, typename V>
concept adjacency_queryable = requires(G g, V v) {
    { g.adjacent(v) } -> std::ranges::range;
};

template<typename G, typename V>
concept has_vertex_count = requires(G g) {
    { g.vertex_count() } -> std::integral;
};

template<typename E>
concept edge_like = requires(E e) {
    typename E::vertex_type;
    { e.source() } -> std::convertible_to<typename E::vertex_type>;
    { e.target() } -> std::convertible_to<typename E::vertex_type>;
};

template<typename E>
concept weighted_edge = edge_like<E> && requires(E e) {
    typename E::weight_type;
    { e.weight() } -> std::convertible_to<typename E::weight_type>;
};

// =============================================================================
// Generic graph adaptor for adjacency list representation
// =============================================================================

template<typename V, typename W = void>
class adjacency_graph {
public:
    using vertex_type = V;
    using weight_type = std::conditional_t<std::is_void_v<W>, int, W>;

    struct edge {
        using vertex_type = V;
        using weight_type = adjacency_graph::weight_type;

        V src;
        V tgt;
        weight_type w;

        V source() const { return src; }
        V target() const { return tgt; }
        weight_type weight() const { return w; }
    };

private:
    std::unordered_map<V, std::vector<std::pair<V, weight_type>>> adj_list;
    std::unordered_set<V> vertex_set;

public:
    void add_vertex(const V& v) {
        vertex_set.insert(v);
    }

    void add_edge(const V& src, const V& tgt, const weight_type& w = weight_type(1)) {
        vertex_set.insert(src);
        vertex_set.insert(tgt);
        adj_list[src].emplace_back(tgt, w);
    }

    void add_undirected_edge(const V& src, const V& tgt, const weight_type& w = weight_type(1)) {
        add_edge(src, tgt, w);
        add_edge(tgt, src, w);
    }

    auto vertices() const {
        return vertex_set;
    }

    auto edges() const {
        std::vector<edge> result;
        for (const auto& [v, neighbors] : adj_list) {
            for (const auto& [u, w] : neighbors) {
                result.push_back({v, u, w});
            }
        }
        return result;
    }

    std::vector<V> adjacent(const V& v) const {
        std::vector<V> result;
        auto it = adj_list.find(v);
        if (it != adj_list.end()) {
            for (const auto& [neighbor, weight] : it->second) {
                result.push_back(neighbor);
            }
        }
        return result;
    }

    std::vector<std::pair<V, weight_type>> adjacent_weighted(const V& v) const {
        auto it = adj_list.find(v);
        if (it != adj_list.end()) {
            return it->second;
        }
        return {};
    }

    size_t vertex_count() const {
        return vertex_set.size();
    }

    bool has_edge(const V& src, const V& tgt) const {
        auto it = adj_list.find(src);
        if (it != adj_list.end()) {
            return std::any_of(it->second.begin(), it->second.end(),
                             [&tgt](const auto& p) { return p.first == tgt; });
        }
        return false;
    }
};

// =============================================================================
// Generic traversal algorithms
// =============================================================================

/**
 * Generic depth-first search
 * Works with any graph type that provides adjacent() operation
 */
template<typename G, typename V, typename Visitor>
    requires adjacency_queryable<G, V>
void depth_first_search(const G& graph, const V& start, Visitor visit) {
    std::unordered_set<V> visited;
    std::stack<V> stack;
    stack.push(start);

    while (!stack.empty()) {
        V current = stack.top();
        stack.pop();

        if (visited.insert(current).second) {
            visit(current);

            for (const auto& neighbor : graph.adjacent(current)) {
                if (visited.find(neighbor) == visited.end()) {
                    stack.push(neighbor);
                }
            }
        }
    }
}

/**
 * Generic breadth-first search
 * Returns distances from start vertex
 */
template<typename G, typename V>
    requires adjacency_queryable<G, V>
std::unordered_map<V, int> breadth_first_search(const G& graph, const V& start) {
    std::unordered_map<V, int> distances;
    std::queue<V> queue;

    distances[start] = 0;
    queue.push(start);

    while (!queue.empty()) {
        V current = queue.front();
        queue.pop();

        for (const auto& neighbor : graph.adjacent(current)) {
            if (distances.find(neighbor) == distances.end()) {
                distances[neighbor] = distances[current] + 1;
                queue.push(neighbor);
            }
        }
    }

    return distances;
}

// =============================================================================
// Shortest path algorithms
// =============================================================================

/**
 * Dijkstra's algorithm - generic implementation
 * Works with any graph that provides weighted adjacency information
 */
template<typename G, typename V, typename W = double>
struct shortest_path_result {
    std::unordered_map<V, W> distances;
    std::unordered_map<V, V> predecessors;

    std::vector<V> path_to(const V& target) const {
        std::vector<V> path;
        if (distances.find(target) == distances.end()) {
            return path;  // No path exists
        }

        V current = target;
        while (predecessors.find(current) != predecessors.end()) {
            path.push_back(current);
            current = predecessors.at(current);
        }
        path.push_back(current);
        std::reverse(path.begin(), path.end());
        return path;
    }
};

template<typename G, typename V>
auto dijkstra(const G& graph, const V& start) {
    using W = typename decltype(graph.adjacent_weighted(start))::value_type::second_type;

    shortest_path_result<V, W> result;
    auto& distances = result.distances;
    auto& predecessors = result.predecessors;

    using pq_pair = std::pair<W, V>;
    std::priority_queue<pq_pair, std::vector<pq_pair>, std::greater<>> pq;

    distances[start] = W(0);
    pq.emplace(W(0), start);

    while (!pq.empty()) {
        auto [dist, u] = pq.top();
        pq.pop();

        if (dist > distances[u]) continue;

        for (const auto& [v, weight] : graph.adjacent_weighted(u)) {
            W new_dist = distances[u] + weight;

            if (distances.find(v) == distances.end() || new_dist < distances[v]) {
                distances[v] = new_dist;
                predecessors[v] = u;
                pq.emplace(new_dist, v);
            }
        }
    }

    return result;
}

/**
 * Bellman-Ford algorithm for graphs with negative weights
 */
template<typename G, typename V>
    requires edge_enumerable<G> && vertex_enumerable<G>
auto bellman_ford(const G& graph, const V& start) {
    using E = typename decltype(graph.edges())::value_type;
    using W = typename E::weight_type;

    shortest_path_result<V, W> result;
    auto& distances = result.distances;
    auto& predecessors = result.predecessors;

    // Initialize distances
    for (const auto& v : graph.vertices()) {
        distances[v] = std::numeric_limits<W>::max();
    }
    distances[start] = W(0);

    // Relax edges V-1 times
    size_t n = graph.vertex_count();
    for (size_t i = 0; i < n - 1; ++i) {
        for (const auto& edge : graph.edges()) {
            V u = edge.source();
            V v = edge.target();
            W w = edge.weight();

            if (distances[u] != std::numeric_limits<W>::max() &&
                distances[u] + w < distances[v]) {
                distances[v] = distances[u] + w;
                predecessors[v] = u;
            }
        }
    }

    // Check for negative cycles
    for (const auto& edge : graph.edges()) {
        V u = edge.source();
        V v = edge.target();
        W w = edge.weight();

        if (distances[u] != std::numeric_limits<W>::max() &&
            distances[u] + w < distances[v]) {
            throw std::runtime_error("Graph contains negative-weight cycle");
        }
    }

    return result;
}

// =============================================================================
// Minimum spanning tree algorithms
// =============================================================================

/**
 * Kruskal's algorithm using disjoint set union
 */
template<typename V>
class disjoint_set {
private:
    std::unordered_map<V, V> parent;
    std::unordered_map<V, int> rank;

public:
    void make_set(const V& v) {
        if (parent.find(v) == parent.end()) {
            parent[v] = v;
            rank[v] = 0;
        }
    }

    V find(const V& v) {
        if (parent[v] != v) {
            parent[v] = find(parent[v]);  // Path compression
        }
        return parent[v];
    }

    bool unite(const V& a, const V& b) {
        V root_a = find(a);
        V root_b = find(b);

        if (root_a == root_b) return false;

        // Union by rank
        if (rank[root_a] < rank[root_b]) {
            parent[root_a] = root_b;
        } else if (rank[root_a] > rank[root_b]) {
            parent[root_b] = root_a;
        } else {
            parent[root_b] = root_a;
            rank[root_a]++;
        }
        return true;
    }
};

template<typename G>
    requires edge_enumerable<G> && vertex_enumerable<G>
auto kruskal_mst(const G& graph) {
    using E = typename decltype(graph.edges())::value_type;
    using V = typename E::vertex_type;
    using W = typename E::weight_type;

    std::vector<E> mst_edges;
    W total_weight = W(0);

    // Get all edges and sort by weight
    auto edges = graph.edges();
    std::sort(edges.begin(), edges.end(),
              [](const E& a, const E& b) { return a.weight() < b.weight(); });

    disjoint_set<V> ds;
    for (const auto& v : graph.vertices()) {
        ds.make_set(v);
    }

    for (const auto& edge : edges) {
        if (ds.unite(edge.source(), edge.target())) {
            mst_edges.push_back(edge);
            total_weight = total_weight + edge.weight();
        }
    }

    return std::make_pair(mst_edges, total_weight);
}

// =============================================================================
// Topological sorting
// =============================================================================

/**
 * Topological sort using Kahn's algorithm
 * Returns empty vector if graph has cycles
 */
template<typename G, typename V>
    requires adjacency_queryable<G, V> && vertex_enumerable<G>
std::vector<V> topological_sort(const G& graph) {
    std::unordered_map<V, int> in_degree;
    std::vector<V> result;
    std::queue<V> queue;

    // Calculate in-degrees
    for (const auto& v : graph.vertices()) {
        in_degree[v] = 0;
    }

    for (const auto& v : graph.vertices()) {
        for (const auto& u : graph.adjacent(v)) {
            in_degree[u]++;
        }
    }

    // Find vertices with no incoming edges
    for (const auto& v : graph.vertices()) {
        if (in_degree[v] == 0) {
            queue.push(v);
        }
    }

    // Process vertices
    while (!queue.empty()) {
        V current = queue.front();
        queue.pop();
        result.push_back(current);

        for (const auto& neighbor : graph.adjacent(current)) {
            if (--in_degree[neighbor] == 0) {
                queue.push(neighbor);
            }
        }
    }

    // Check if all vertices were processed (no cycles)
    if (result.size() != graph.vertex_count()) {
        return {};  // Graph has cycles
    }

    return result;
}

// =============================================================================
// Strongly connected components
// =============================================================================

/**
 * Tarjan's algorithm for finding strongly connected components
 */
template<typename G, typename V>
    requires adjacency_queryable<G, V> && vertex_enumerable<G>
class tarjan_scc {
private:
    const G& graph;
    int index_counter = 0;
    std::stack<V> stack;
    std::unordered_map<V, int> index;
    std::unordered_map<V, int> lowlink;
    std::unordered_set<V> on_stack;
    std::vector<std::vector<V>> sccs;

    void strongconnect(const V& v) {
        index[v] = index_counter;
        lowlink[v] = index_counter;
        index_counter++;
        stack.push(v);
        on_stack.insert(v);

        for (const auto& w : graph.adjacent(v)) {
            if (index.find(w) == index.end()) {
                strongconnect(w);
                lowlink[v] = std::min(lowlink[v], lowlink[w]);
            } else if (on_stack.count(w)) {
                lowlink[v] = std::min(lowlink[v], index[w]);
            }
        }

        if (lowlink[v] == index[v]) {
            std::vector<V> scc;
            V w;
            do {
                w = stack.top();
                stack.pop();
                on_stack.erase(w);
                scc.push_back(w);
            } while (w != v);
            sccs.push_back(std::move(scc));
        }
    }

public:
    explicit tarjan_scc(const G& g) : graph(g) {}

    std::vector<std::vector<V>> find_sccs() {
        for (const auto& v : graph.vertices()) {
            if (index.find(v) == index.end()) {
                strongconnect(v);
            }
        }
        return sccs;
    }
};

template<typename G>
auto strongly_connected_components(const G& graph) {
    tarjan_scc scc_finder(graph);
    return scc_finder.find_sccs();
}

// =============================================================================
// Graph coloring
// =============================================================================

/**
 * Greedy graph coloring algorithm
 * Returns a map from vertices to colors (integers)
 */
template<typename G, typename V>
    requires adjacency_queryable<G, V> && vertex_enumerable<G>
std::unordered_map<V, int> greedy_coloring(const G& graph) {
    std::unordered_map<V, int> colors;

    for (const auto& v : graph.vertices()) {
        std::unordered_set<int> neighbor_colors;

        for (const auto& u : graph.adjacent(v)) {
            if (colors.find(u) != colors.end()) {
                neighbor_colors.insert(colors[u]);
            }
        }

        int color = 0;
        while (neighbor_colors.count(color)) {
            color++;
        }

        colors[v] = color;
    }

    return colors;
}

// =============================================================================
// Maximum flow algorithms
// =============================================================================

/**
 * Ford-Fulkerson algorithm using Edmonds-Karp implementation
 */
template<typename V, typename W = int>
class flow_network {
private:
    std::unordered_map<V, std::unordered_map<V, W>> capacity;
    std::unordered_map<V, std::unordered_map<V, W>> flow;
    std::unordered_set<V> vertices;

public:
    void add_edge(const V& u, const V& v, W cap) {
        capacity[u][v] = cap;
        flow[u][v] = W(0);
        flow[v][u] = W(0);
        vertices.insert(u);
        vertices.insert(v);
    }

    W max_flow(const V& source, const V& sink) {
        W total_flow = W(0);

        while (true) {
            // BFS to find augmenting path
            std::unordered_map<V, V> parent;
            std::queue<std::pair<V, W>> q;
            q.push({source, std::numeric_limits<W>::max()});

            while (!q.empty()) {
                auto [u, flow_val] = q.front();
                q.pop();

                for (const auto& v : vertices) {
                    if (parent.find(v) == parent.end() && v != source) {
                        W residual = capacity[u][v] - flow[u][v];
                        if (residual > W(0)) {
                            parent[v] = u;
                            W new_flow = std::min(flow_val, residual);

                            if (v == sink) {
                                // Augment flow along path
                                total_flow = total_flow + new_flow;
                                V current = sink;
                                while (current != source) {
                                    V prev = parent[current];
                                    flow[prev][current] = flow[prev][current] + new_flow;
                                    flow[current][prev] = flow[current][prev] - new_flow;
                                    current = prev;
                                }
                                goto next_iteration;
                            }

                            q.push({v, new_flow});
                        }
                    }
                }
            }

            break;  // No augmenting path found
            next_iteration:;
        }

        return total_flow;
    }
};

} // namespace stepanov