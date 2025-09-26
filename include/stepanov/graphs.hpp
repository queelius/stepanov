// graphs.hpp - Innovative Graph Algorithms
// Algebraic path problems, spectral methods, dynamic graphs, and more
// Mathematical elegance meets practical power

#ifndef STEPANOV_GRAPHS_HPP
#define STEPANOV_GRAPHS_HPP

#include <concepts>
#include <vector>
#include <unordered_map>
#include <queue>
#include <limits>
#include <algorithm>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
// #include <Eigen/Dense>  // External dependency - not included
// #include <Eigen/Sparse>
#include <complex>
#include <random>
#include <chrono>

namespace stepanov::graphs {

// ============================================================================
// Algebraic Path Problems - Generalize Shortest Path to Any Semiring
// ============================================================================

// Semiring concept for path problems
template<typename T>
concept Semiring = requires(T a, T b) {
    { a + b } -> std::same_as<T>;  // Additive operation (path combination)
    { a * b } -> std::same_as<T>;  // Multiplicative operation (edge extension)
    { T::zero() } -> std::same_as<T>;  // Additive identity
    { T::one() } -> std::same_as<T>;   // Multiplicative identity
};

// Tropical (min-plus) semiring for shortest paths
template<typename T = double>
struct tropical_semiring {
    T value;

    tropical_semiring(T v = std::numeric_limits<T>::infinity()) : value(v) {}

    tropical_semiring operator+(const tropical_semiring& other) const {
        return tropical_semiring(std::min(value, other.value));
    }

    tropical_semiring operator*(const tropical_semiring& other) const {
        if (value == std::numeric_limits<T>::infinity() ||
            other.value == std::numeric_limits<T>::infinity()) {
            return tropical_semiring(std::numeric_limits<T>::infinity());
        }
        return tropical_semiring(value + other.value);
    }

    static tropical_semiring zero() {
        return tropical_semiring(std::numeric_limits<T>::infinity());
    }

    static tropical_semiring one() {
        return tropical_semiring(T{0});
    }

    bool operator<(const tropical_semiring& other) const {
        return value < other.value;
    }
};

// Max-plus semiring for longest paths
template<typename T = double>
struct maxplus_semiring {
    T value;

    maxplus_semiring(T v = -std::numeric_limits<T>::infinity()) : value(v) {}

    maxplus_semiring operator+(const maxplus_semiring& other) const {
        return maxplus_semiring(std::max(value, other.value));
    }

    maxplus_semiring operator*(const maxplus_semiring& other) const {
        if (value == -std::numeric_limits<T>::infinity() ||
            other.value == -std::numeric_limits<T>::infinity()) {
            return maxplus_semiring(-std::numeric_limits<T>::infinity());
        }
        return maxplus_semiring(value + other.value);
    }

    static maxplus_semiring zero() {
        return maxplus_semiring(-std::numeric_limits<T>::infinity());
    }

    static maxplus_semiring one() {
        return maxplus_semiring(T{0});
    }
};

// Boolean semiring for reachability
struct boolean_semiring {
    bool value;

    boolean_semiring(bool v = false) : value(v) {}

    boolean_semiring operator+(const boolean_semiring& other) const {
        return boolean_semiring(value || other.value);
    }

    boolean_semiring operator*(const boolean_semiring& other) const {
        return boolean_semiring(value && other.value);
    }

    static boolean_semiring zero() { return boolean_semiring(false); }
    static boolean_semiring one() { return boolean_semiring(true); }
};

// Counting semiring for path counting
template<typename T = uint64_t>
struct counting_semiring {
    T value;

    counting_semiring(T v = 0) : value(v) {}

    counting_semiring operator+(const counting_semiring& other) const {
        return counting_semiring(value + other.value);
    }

    counting_semiring operator*(const counting_semiring& other) const {
        return counting_semiring(value * other.value);
    }

    static counting_semiring zero() { return counting_semiring(0); }
    static counting_semiring one() { return counting_semiring(1); }
};

// Generic algebraic path solver
template<typename Node, typename Weight>
    requires Semiring<Weight>
class algebraic_path_solver {
    std::unordered_map<Node, std::unordered_map<Node, Weight>> adjacency_;

public:
    void add_edge(Node from, Node to, Weight weight) {
        adjacency_[from][to] = weight;
    }

    // Bellman-Ford generalized to any semiring
    std::unordered_map<Node, Weight> single_source_paths(Node source) const {
        std::unordered_map<Node, Weight> dist;

        // Initialize distances
        for (const auto& [node, _] : adjacency_) {
            dist[node] = Weight::zero();
        }
        dist[source] = Weight::one();

        // Relax edges |V|-1 times
        size_t n = adjacency_.size();
        for (size_t i = 0; i < n - 1; ++i) {
            for (const auto& [u, edges] : adjacency_) {
                for (const auto& [v, w] : edges) {
                    dist[v] = dist[v] + (dist[u] * w);
                }
            }
        }

        return dist;
    }

    // Floyd-Warshall generalized to any semiring
    std::unordered_map<Node, std::unordered_map<Node, Weight>> all_pairs_paths() const {
        std::unordered_map<Node, std::unordered_map<Node, Weight>> dist;

        // Initialize
        for (const auto& [i, edges] : adjacency_) {
            for (const auto& [j, _] : adjacency_) {
                if (i == j) {
                    dist[i][j] = Weight::one();
                } else {
                    dist[i][j] = Weight::zero();
                }
            }
            for (const auto& [j, w] : edges) {
                dist[i][j] = w;
            }
        }

        // Dynamic programming
        for (const auto& [k, _] : adjacency_) {
            for (const auto& [i, _] : adjacency_) {
                for (const auto& [j, _] : adjacency_) {
                    dist[i][j] = dist[i][j] + (dist[i][k] * dist[k][j]);
                }
            }
        }

        return dist;
    }

    // Matrix multiplication approach for paths of exact length k
    std::unordered_map<Node, std::unordered_map<Node, Weight>>
    paths_of_length(size_t k) const {
        if (k == 0) {
            std::unordered_map<Node, std::unordered_map<Node, Weight>> identity;
            for (const auto& [node, _] : adjacency_) {
                identity[node][node] = Weight::one();
            }
            return identity;
        }

        if (k == 1) {
            return adjacency_;
        }

        // Matrix exponentiation by squaring
        if (k % 2 == 0) {
            auto half = paths_of_length(k / 2);
            return matrix_multiply(half, half);
        } else {
            auto half = paths_of_length(k / 2);
            auto squared = matrix_multiply(half, half);
            return matrix_multiply(squared, adjacency_);
        }
    }

private:
    std::unordered_map<Node, std::unordered_map<Node, Weight>>
    matrix_multiply(const std::unordered_map<Node, std::unordered_map<Node, Weight>>& a,
                   const std::unordered_map<Node, std::unordered_map<Node, Weight>>& b) const {
        std::unordered_map<Node, std::unordered_map<Node, Weight>> result;

        for (const auto& [i, row_a] : a) {
            for (const auto& [j, _] : adjacency_) {
                result[i][j] = Weight::zero();
                for (const auto& [k, a_ik] : row_a) {
                    if (b.count(k) && b.at(k).count(j)) {
                        result[i][j] = result[i][j] + (a_ik * b.at(k).at(j));
                    }
                }
            }
        }

        return result;
    }
};

// ============================================================================
// Spectral Graph Algorithms - Eigenvalue-based Methods
// ============================================================================

template<typename Node>
class spectral_graph {
    using Matrix = Eigen::MatrixXd;
    using SparseMatrix = Eigen::SparseMatrix<double>;

    std::vector<Node> nodes_;
    std::unordered_map<Node, size_t> node_index_;
    SparseMatrix adjacency_matrix_;
    SparseMatrix laplacian_matrix_;

    void compute_laplacian() {
        size_t n = nodes_.size();
        std::vector<Eigen::Triplet<double>> triplets;

        // Degree matrix - Adjacency matrix
        Eigen::VectorXd degrees = adjacency_matrix_ * Eigen::VectorXd::Ones(n);

        for (size_t i = 0; i < n; ++i) {
            triplets.emplace_back(i, i, degrees(i));
        }

        laplacian_matrix_.resize(n, n);
        laplacian_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        laplacian_matrix_ -= adjacency_matrix_;
    }

public:
    void add_edge(Node from, Node to, double weight = 1.0) {
        if (!node_index_.count(from)) {
            node_index_[from] = nodes_.size();
            nodes_.push_back(from);
        }
        if (!node_index_.count(to)) {
            node_index_[to] = nodes_.size();
            nodes_.push_back(to);
        }

        size_t i = node_index_[from];
        size_t j = node_index_[to];

        adjacency_matrix_.coeffRef(i, j) = weight;
        adjacency_matrix_.coeffRef(j, i) = weight;  // Undirected
    }

    void finalize() {
        size_t n = nodes_.size();
        adjacency_matrix_.resize(n, n);
        adjacency_matrix_.makeCompressed();
        compute_laplacian();
    }

    // Spectral clustering using k smallest eigenvectors
    std::vector<std::vector<Node>> spectral_clustering(size_t k) const {
        // Compute eigenvectors of Laplacian
        Eigen::SelfAdjointEigenSolver<Matrix> solver(laplacian_matrix_);
        Matrix eigenvectors = solver.eigenvectors();

        // Take first k eigenvectors (smallest eigenvalues)
        Matrix embedding = eigenvectors.leftCols(k);

        // K-means clustering in eigenspace
        return kmeans_cluster(embedding, k);
    }

    // Fiedler vector for graph partitioning
    std::pair<std::vector<Node>, std::vector<Node>> fiedler_partition() const {
        Eigen::SelfAdjointEigenSolver<Matrix> solver(laplacian_matrix_);
        Eigen::VectorXd fiedler = solver.eigenvectors().col(1);  // Second smallest

        std::vector<Node> partition1, partition2;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (fiedler(i) < 0) {
                partition1.push_back(nodes_[i]);
            } else {
                partition2.push_back(nodes_[i]);
            }
        }

        return {partition1, partition2};
    }

    // PageRank using power iteration
    std::unordered_map<Node, double> pagerank(double damping = 0.85, size_t iterations = 100) const {
        size_t n = nodes_.size();
        Eigen::VectorXd rank = Eigen::VectorXd::Ones(n) / n;

        // Transition matrix
        SparseMatrix transition = adjacency_matrix_;
        for (size_t i = 0; i < n; ++i) {
            double row_sum = transition.row(i).sum();
            if (row_sum > 0) {
                transition.row(i) /= row_sum;
            }
        }

        // Power iteration
        for (size_t iter = 0; iter < iterations; ++iter) {
            Eigen::VectorXd new_rank = damping * transition.transpose() * rank +
                                       (1 - damping) / n * Eigen::VectorXd::Ones(n);
            rank = new_rank;
        }

        std::unordered_map<Node, double> result;
        for (size_t i = 0; i < n; ++i) {
            result[nodes_[i]] = rank(i);
        }

        return result;
    }

    // Cheeger constant approximation
    double cheeger_constant() const {
        auto [part1, part2] = fiedler_partition();

        // Count edges between partitions
        double cut_edges = 0;
        for (const Node& u : part1) {
            for (const Node& v : part2) {
                size_t i = node_index_.at(u);
                size_t j = node_index_.at(v);
                cut_edges += adjacency_matrix_.coeff(i, j);
            }
        }

        // Volume of partitions
        double vol1 = 0, vol2 = 0;
        for (const Node& u : part1) {
            size_t i = node_index_.at(u);
            vol1 += adjacency_matrix_.row(i).sum();
        }
        for (const Node& v : part2) {
            size_t j = node_index_.at(v);
            vol2 += adjacency_matrix_.row(j).sum();
        }

        return cut_edges / std::min(vol1, vol2);
    }

private:
    std::vector<std::vector<Node>> kmeans_cluster(const Matrix& data, size_t k) const {
        // Simplified k-means implementation
        size_t n = data.rows();
        size_t d = data.cols();

        // Random initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n - 1);

        Matrix centers(k, d);
        for (size_t i = 0; i < k; ++i) {
            centers.row(i) = data.row(dis(gen));
        }

        std::vector<size_t> assignments(n);

        // Lloyd's algorithm
        for (size_t iter = 0; iter < 100; ++iter) {
            // Assign points to nearest center
            for (size_t i = 0; i < n; ++i) {
                double min_dist = std::numeric_limits<double>::infinity();
                for (size_t j = 0; j < k; ++j) {
                    double dist = (data.row(i) - centers.row(j)).squaredNorm();
                    if (dist < min_dist) {
                        min_dist = dist;
                        assignments[i] = j;
                    }
                }
            }

            // Update centers
            Matrix new_centers = Matrix::Zero(k, d);
            std::vector<size_t> counts(k, 0);

            for (size_t i = 0; i < n; ++i) {
                new_centers.row(assignments[i]) += data.row(i);
                counts[assignments[i]]++;
            }

            for (size_t j = 0; j < k; ++j) {
                if (counts[j] > 0) {
                    centers.row(j) = new_centers.row(j) / counts[j];
                }
            }
        }

        // Build result
        std::vector<std::vector<Node>> clusters(k);
        for (size_t i = 0; i < n; ++i) {
            clusters[assignments[i]].push_back(nodes_[i]);
        }

        return clusters;
    }
};

// ============================================================================
// Incremental/Decremental Dynamic Graph Algorithms
// ============================================================================

template<typename Node>
class dynamic_graph {
    struct edge {
        Node to;
        double weight;
        size_t timestamp;
    };

    std::unordered_map<Node, std::vector<edge>> adjacency_;
    size_t current_time_ = 0;

    // Incremental strongly connected components
    std::unordered_map<Node, size_t> scc_id_;
    size_t scc_counter_ = 0;

    void update_sccs_after_edge(Node from, Node to) {
        // Tarjan's incremental SCC algorithm
        if (scc_id_.count(from) && scc_id_.count(to)) {
            if (scc_id_[from] != scc_id_[to]) {
                // Merge SCCs
                size_t old_id = scc_id_[to];
                size_t new_id = scc_id_[from];

                for (auto& [node, id] : scc_id_) {
                    if (id == old_id) {
                        id = new_id;
                    }
                }
            }
        } else if (!scc_id_.count(from) && !scc_id_.count(to)) {
            // New SCC
            scc_id_[from] = scc_id_[to] = scc_counter_++;
        } else if (scc_id_.count(from)) {
            scc_id_[to] = scc_id_[from];
        } else {
            scc_id_[from] = scc_id_[to];
        }
    }

public:
    // Add edge with automatic timestamp
    void add_edge(Node from, Node to, double weight = 1.0) {
        adjacency_[from].push_back({to, weight, current_time_++});
        update_sccs_after_edge(from, to);
    }

    // Remove edge (mark as deleted)
    void remove_edge(Node from, Node to) {
        auto& edges = adjacency_[from];
        edges.erase(
            std::remove_if(edges.begin(), edges.end(),
                          [&to](const edge& e) { return e.to == to; }),
            edges.end()
        );
    }

    // Query graph at specific time
    std::vector<std::pair<Node, double>> neighbors_at_time(Node node, size_t time) const {
        std::vector<std::pair<Node, double>> result;

        if (adjacency_.count(node)) {
            for (const auto& e : adjacency_.at(node)) {
                if (e.timestamp <= time) {
                    result.emplace_back(e.to, e.weight);
                }
            }
        }

        return result;
    }

    // Incremental shortest paths
    std::unordered_map<Node, double> incremental_dijkstra(Node source) const {
        std::unordered_map<Node, double> dist;
        std::priority_queue<std::pair<double, Node>,
                           std::vector<std::pair<double, Node>>,
                           std::greater<>> pq;

        dist[source] = 0;
        pq.emplace(0, source);

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();

            if (d > dist[u]) continue;

            if (adjacency_.count(u)) {
                for (const auto& e : adjacency_.at(u)) {
                    double new_dist = dist[u] + e.weight;
                    if (!dist.count(e.to) || new_dist < dist[e.to]) {
                        dist[e.to] = new_dist;
                        pq.emplace(new_dist, e.to);
                    }
                }
            }
        }

        return dist;
    }

    // Get strongly connected component ID
    std::optional<size_t> get_scc_id(Node node) const {
        if (scc_id_.count(node)) {
            return scc_id_.at(node);
        }
        return std::nullopt;
    }

    // Sliding window queries
    std::vector<edge> edges_in_window(size_t start_time, size_t end_time) const {
        std::vector<edge> result;

        for (const auto& [from, edges] : adjacency_) {
            for (const auto& e : edges) {
                if (e.timestamp >= start_time && e.timestamp <= end_time) {
                    result.push_back(e);
                }
            }
        }

        return result;
    }
};

// ============================================================================
// Temporal Graphs - Graphs that Evolve Over Time
// ============================================================================

template<typename Node, typename Time = std::chrono::steady_clock::time_point>
class temporal_graph {
    struct temporal_edge {
        Node to;
        double weight;
        Time start_time;
        Time end_time;
    };

    std::unordered_map<Node, std::vector<temporal_edge>> adjacency_;

public:
    void add_temporal_edge(Node from, Node to, double weight, Time start, Time end) {
        adjacency_[from].push_back({to, weight, start, end});
    }

    // Find earliest arrival path
    std::unordered_map<Node, Time> earliest_arrival(Node source, Time start_time) const {
        std::unordered_map<Node, Time> arrival;
        std::priority_queue<std::pair<Time, Node>,
                           std::vector<std::pair<Time, Node>>,
                           std::greater<>> pq;

        arrival[source] = start_time;
        pq.emplace(start_time, source);

        while (!pq.empty()) {
            auto [time, u] = pq.top();
            pq.pop();

            if (time > arrival[u]) continue;

            if (adjacency_.count(u)) {
                for (const auto& e : adjacency_.at(u)) {
                    if (time >= e.start_time && time <= e.end_time) {
                        Time new_arrival = time + std::chrono::duration_cast<Time::duration>(
                            std::chrono::duration<double>(e.weight)
                        );

                        if (!arrival.count(e.to) || new_arrival < arrival[e.to]) {
                            arrival[e.to] = new_arrival;
                            pq.emplace(new_arrival, e.to);
                        }
                    }
                }
            }
        }

        return arrival;
    }

    // Find latest departure path
    std::unordered_map<Node, Time> latest_departure(Node destination, Time arrival_time) const {
        // Reverse computation from destination
        std::unordered_map<Node, Time> departure;
        std::priority_queue<std::pair<Time, Node>> pq;  // Max heap

        departure[destination] = arrival_time;
        pq.emplace(arrival_time, destination);

        while (!pq.empty()) {
            auto [time, v] = pq.top();
            pq.pop();

            // Find all edges leading to v
            for (const auto& [u, edges] : adjacency_) {
                for (const auto& e : edges) {
                    if (e.to == v && e.end_time <= time) {
                        Time new_departure = e.start_time;

                        if (!departure.count(u) || new_departure > departure[u]) {
                            departure[u] = new_departure;
                            pq.emplace(new_departure, u);
                        }
                    }
                }
            }
        }

        return departure;
    }

    // Temporal betweenness centrality
    std::unordered_map<Node, double> temporal_betweenness(Time window_start, Time window_end) const {
        std::unordered_map<Node, double> centrality;

        // For each pair of nodes
        std::vector<Node> all_nodes;
        for (const auto& [node, _] : adjacency_) {
            all_nodes.push_back(node);
        }

        for (const Node& s : all_nodes) {
            for (const Node& t : all_nodes) {
                if (s == t) continue;

                // Find all temporal paths from s to t
                auto paths = find_temporal_paths(s, t, window_start, window_end);

                // Count paths through each intermediate node
                for (const auto& path : paths) {
                    for (size_t i = 1; i < path.size() - 1; ++i) {
                        centrality[path[i]] += 1.0 / paths.size();
                    }
                }
            }
        }

        return centrality;
    }

private:
    std::vector<std::vector<Node>> find_temporal_paths(
        Node source, Node target, Time start, Time end) const {
        // BFS for temporal paths
        std::vector<std::vector<Node>> paths;

        struct state {
            Node node;
            Time time;
            std::vector<Node> path;
        };

        std::queue<state> q;
        q.push({source, start, {source}});

        while (!q.empty()) {
            auto [node, time, path] = q.front();
            q.pop();

            if (node == target) {
                paths.push_back(path);
                continue;
            }

            if (adjacency_.count(node)) {
                for (const auto& e : adjacency_.at(node)) {
                    if (time >= e.start_time && time <= e.end_time && time + e.weight <= end) {
                        auto new_path = path;
                        new_path.push_back(e.to);
                        q.push({e.to, time + e.weight, new_path});
                    }
                }
            }
        }

        return paths;
    }
};

// ============================================================================
// Graph Grammars - Generate Graphs from Rules
// ============================================================================

template<typename Node>
class graph_grammar {
    struct rule {
        std::function<bool(const std::vector<Node>&)> condition;
        std::function<std::vector<std::pair<Node, Node>>(const std::vector<Node>&)> production;
    };

    std::vector<rule> rules_;
    std::unordered_map<Node, std::unordered_set<Node>> graph_;

public:
    void add_rule(rule r) {
        rules_.push_back(std::move(r));
    }

    // L-system style rule for tree generation
    void add_branching_rule(size_t max_depth) {
        add_rule({
            [max_depth](const std::vector<Node>& nodes) {
                return nodes.size() < max_depth;
            },
            [](const std::vector<Node>& nodes) {
                std::vector<std::pair<Node, Node>> edges;
                Node parent = nodes.back();
                Node left_child = Node(parent * 2);
                Node right_child = Node(parent * 2 + 1);
                edges.emplace_back(parent, left_child);
                edges.emplace_back(parent, right_child);
                return edges;
            }
        });
    }

    // Preferential attachment rule for scale-free networks
    void add_preferential_attachment_rule() {
        add_rule({
            [](const std::vector<Node>& nodes) { return true; },
            [this](const std::vector<Node>& nodes) {
                std::vector<std::pair<Node, Node>> edges;

                if (nodes.empty()) return edges;

                // Compute degree distribution
                std::vector<double> probabilities;
                double total = 0;

                for (const Node& n : nodes) {
                    double degree = graph_[n].size() + 1;  // +1 to avoid zero
                    probabilities.push_back(degree);
                    total += degree;
                }

                // Normalize
                for (double& p : probabilities) {
                    p /= total;
                }

                // Select node with probability proportional to degree
                std::random_device rd;
                std::mt19937 gen(rd());
                std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());

                Node new_node = Node(nodes.size() + 1);
                Node target = nodes[dist(gen)];

                edges.emplace_back(new_node, target);
                return edges;
            }
        });
    }

    // Generate graph by applying rules
    std::unordered_map<Node, std::unordered_set<Node>> generate(
        Node seed, size_t iterations) {

        graph_.clear();
        std::vector<Node> nodes = {seed};

        for (size_t iter = 0; iter < iterations; ++iter) {
            for (const auto& rule : rules_) {
                if (rule.condition(nodes)) {
                    auto edges = rule.production(nodes);

                    for (const auto& [from, to] : edges) {
                        graph_[from].insert(to);
                        graph_[to].insert(from);  // Undirected

                        // Add new nodes to list
                        if (std::find(nodes.begin(), nodes.end(), from) == nodes.end()) {
                            nodes.push_back(from);
                        }
                        if (std::find(nodes.begin(), nodes.end(), to) == nodes.end()) {
                            nodes.push_back(to);
                        }
                    }
                }
            }
        }

        return graph_;
    }

    // Stochastic graph grammar
    std::unordered_map<Node, std::unordered_set<Node>> generate_stochastic(
        Node seed, size_t iterations, double rule_probability = 0.7) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        graph_.clear();
        std::vector<Node> nodes = {seed};

        for (size_t iter = 0; iter < iterations; ++iter) {
            for (const auto& rule : rules_) {
                if (dis(gen) < rule_probability && rule.condition(nodes)) {
                    auto edges = rule.production(nodes);

                    for (const auto& [from, to] : edges) {
                        graph_[from].insert(to);

                        if (std::find(nodes.begin(), nodes.end(), from) == nodes.end()) {
                            nodes.push_back(from);
                        }
                        if (std::find(nodes.begin(), nodes.end(), to) == nodes.end()) {
                            nodes.push_back(to);
                        }
                    }
                }
            }
        }

        return graph_;
    }
};

} // namespace stepanov::graphs

#endif // STEPANOV_GRAPHS_HPP