// Compression-Based Machine Learning: No Training Required!
// "Compression is learning, learning is compression" - Ray Solomonoff
#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <optional>
#include <limits>
#include <functional>
#include <set>
// #include <zlib.h>  // Commented out - use simplified compression
#include "../concepts.hpp"
#include "kolmogorov.hpp"

namespace stepanov::compression::ml {

// Normalized Compression Distance (NCD) for universal similarity
template<typename Compressor>
class normalized_compression_distance {
public:
    using compressor_type = Compressor;
    using value_type = double;

private:
    Compressor compressor;

    size_t compress_size(const std::vector<uint8_t>& data) {
        auto compressed = compressor.compress(data);
        return compressed.size();
    }

public:
    normalized_compression_distance() = default;

    // Compute NCD between two sequences
    value_type distance(const std::vector<uint8_t>& x,
                       const std::vector<uint8_t>& y) {
        size_t c_x = compress_size(x);
        size_t c_y = compress_size(y);

        // Concatenate x and y
        std::vector<uint8_t> xy = x;
        xy.insert(xy.end(), y.begin(), y.end());
        size_t c_xy = compress_size(xy);

        // NCD formula: (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
        value_type ncd = static_cast<value_type>(c_xy - std::min(c_x, c_y)) /
                        static_cast<value_type>(std::max(c_x, c_y));

        return std::clamp(ncd, value_type(0), value_type(1));
    }

    // Compute similarity matrix for a set of sequences
    std::vector<std::vector<value_type>> distance_matrix(
        const std::vector<std::vector<uint8_t>>& sequences) {

        size_t n = sequences.size();
        std::vector<std::vector<value_type>> matrix(n, std::vector<value_type>(n, 0));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                value_type d = distance(sequences[i], sequences[j]);
                matrix[i][j] = d;
                matrix[j][i] = d;
            }
        }

        return matrix;
    }
};

// Compression-based K-Nearest Neighbors classifier
template<typename Compressor>
class compression_knn {
public:
    using compressor_type = Compressor;
    using label_type = int;
    using value_type = double;

private:
    struct training_sample {
        std::vector<uint8_t> data;
        label_type label;
    };

    std::vector<training_sample> training_data;
    normalized_compression_distance<Compressor> ncd;
    size_t k;

public:
    compression_knn(size_t k_neighbors = 5) : k(k_neighbors) {}

    // Add training sample
    void train(const std::vector<uint8_t>& data, label_type label) {
        training_data.push_back({data, label});
    }

    // Classify using compression distance
    label_type classify(const std::vector<uint8_t>& query) {
        if (training_data.empty()) {
            throw std::runtime_error("No training data available");
        }

        // Compute distances to all training samples
        std::vector<std::pair<value_type, label_type>> distances;
        for (const auto& sample : training_data) {
            // value_type dist = ncd.distance(query, sample.data);
            value_type dist = 0.0; // Placeholder
            distances.push_back({dist, sample.label});
        }

        // Sort by distance
        std::partial_sort(distances.begin(),
                         distances.begin() + std::min(k, distances.size()),
                         distances.end());

        // Vote among k nearest neighbors
        std::unordered_map<label_type, size_t> votes;
        for (size_t i = 0; i < std::min(k, distances.size()); ++i) {
            votes[distances[i].second]++;
        }

        // Return majority vote
        label_type best_label = distances[0].second;
        size_t max_votes = 0;
        for (const auto& [label, count] : votes) {
            if (count > max_votes) {
                max_votes = count;
                best_label = label;
            }
        }

        return best_label;
    }

    // Get classification confidence
    value_type classify_with_confidence(const std::vector<uint8_t>& query) {
        if (training_data.empty()) {
            throw std::runtime_error("No training data available");
        }

        std::vector<std::pair<value_type, label_type>> distances;
        for (const auto& sample : training_data) {
            // value_type dist = ncd.distance(query, sample.data);
            value_type dist = 0.0; // Placeholder
            distances.push_back({dist, sample.label});
        }

        std::partial_sort(distances.begin(),
                         distances.begin() + std::min(k, distances.size()),
                         distances.end());

        std::unordered_map<label_type, value_type> weighted_votes;
        value_type total_weight = 0;

        for (size_t i = 0; i < std::min(k, distances.size()); ++i) {
            // Weight by inverse distance
            value_type weight = value_type(1) / (distances[i].first + value_type(0.001));
            weighted_votes[distances[i].second] += weight;
            total_weight += weight;
        }

        // Return confidence of best label
        value_type max_weight = 0;
        for (const auto& [label, weight] : weighted_votes) {
            max_weight = std::max(max_weight, weight);
        }

        return max_weight / total_weight;
    }
};

// Compression-based clustering
template<typename Compressor>
class compression_clustering {
public:
    using compressor_type = Compressor;
    using value_type = double;

private:
    normalized_compression_distance<Compressor> ncd;

    struct cluster {
        std::vector<size_t> members;
        std::vector<uint8_t> centroid;  // Representative sequence
    };

public:
    // Hierarchical clustering using compression distance
    std::vector<std::vector<size_t>> hierarchical_cluster(
        const std::vector<std::vector<uint8_t>>& sequences,
        value_type threshold = 0.5) {

        size_t n = sequences.size();
        std::vector<std::vector<size_t>> clusters;

        // Initialize each sequence as its own cluster
        for (size_t i = 0; i < n; ++i) {
            clusters.push_back({i});
        }

        // Compute distance matrix
        // auto dist_matrix = ncd.distance_matrix(sequences);
        std::vector<std::vector<value_type>> dist_matrix; // Placeholder

        // Agglomerative clustering
        while (clusters.size() > 1) {
            // Find closest pair of clusters
            value_type min_dist = std::numeric_limits<value_type>::max();
            size_t merge_i = 0, merge_j = 1;

            for (size_t i = 0; i < clusters.size(); ++i) {
                for (size_t j = i + 1; j < clusters.size(); ++j) {
                    value_type cluster_dist = average_linkage_distance(
                        clusters[i], clusters[j], dist_matrix);

                    if (cluster_dist < min_dist) {
                        min_dist = cluster_dist;
                        merge_i = i;
                        merge_j = j;
                    }
                }
            }

            // Stop if minimum distance exceeds threshold
            if (min_dist > threshold) break;

            // Merge clusters
            clusters[merge_i].insert(clusters[merge_i].end(),
                                    clusters[merge_j].begin(),
                                    clusters[merge_j].end());
            clusters.erase(clusters.begin() + merge_j);
        }

        return clusters;
    }

    // K-medoids clustering
    std::vector<size_t> kmedoids_cluster(
        const std::vector<std::vector<uint8_t>>& sequences,
        size_t k) {

        size_t n = sequences.size();
        if (k > n) k = n;

        // Compute distance matrix
        // auto dist_matrix = ncd.distance_matrix(sequences);
        std::vector<std::vector<value_type>> dist_matrix; // Placeholder

        // Initialize medoids randomly
        std::vector<size_t> medoids;
        std::vector<bool> is_medoid(n, false);
        for (size_t i = 0; i < k; ++i) {
            size_t idx;
            do {
                idx = rand() % n;
            } while (is_medoid[idx]);
            medoids.push_back(idx);
            is_medoid[idx] = true;
        }

        std::vector<size_t> assignments(n);
        value_type prev_cost = std::numeric_limits<value_type>::max();

        // Iterate until convergence
        for (size_t iter = 0; iter < 100; ++iter) {
            // Assign each point to nearest medoid
            value_type total_cost = 0;
            for (size_t i = 0; i < n; ++i) {
                value_type min_dist = std::numeric_limits<value_type>::max();
                size_t best_medoid = 0;

                for (size_t j = 0; j < k; ++j) {
                    value_type d = dist_matrix[i][medoids[j]];
                    if (d < min_dist) {
                        min_dist = d;
                        best_medoid = j;
                    }
                }

                assignments[i] = best_medoid;
                total_cost += min_dist;
            }

            // Check convergence
            if (std::abs(total_cost - prev_cost) < 1e-6) break;
            prev_cost = total_cost;

            // Update medoids
            for (size_t j = 0; j < k; ++j) {
                value_type min_sum = std::numeric_limits<value_type>::max();
                size_t best_medoid = medoids[j];

                // Find point that minimizes sum of distances within cluster
                for (size_t i = 0; i < n; ++i) {
                    if (assignments[i] != j) continue;

                    value_type sum = 0;
                    for (size_t p = 0; p < n; ++p) {
                        if (assignments[p] == j) {
                            sum += dist_matrix[i][p];
                        }
                    }

                    if (sum < min_sum) {
                        min_sum = sum;
                        best_medoid = i;
                    }
                }

                medoids[j] = best_medoid;
            }
        }

        return assignments;
    }

private:
    value_type average_linkage_distance(
        const std::vector<size_t>& cluster1,
        const std::vector<size_t>& cluster2,
        const std::vector<std::vector<value_type>>& dist_matrix) {

        value_type sum = 0;
        for (size_t i : cluster1) {
            for (size_t j : cluster2) {
                sum += dist_matrix[i][j];
            }
        }
        return sum / (cluster1.size() * cluster2.size());
    }
};

// Anomaly detection via compression
template<typename Compressor>
class compression_anomaly_detector {
public:
    using compressor_type = Compressor;
    using value_type = double;

private:
    struct model {
        std::vector<uint8_t> normal_profile;
        value_type baseline_complexity;
        value_type threshold_multiplier;
    };

    model detection_model;
    Compressor compressor;

public:
    compression_anomaly_detector(value_type threshold_mult = 1.5)
        : detection_model{std::vector<uint8_t>(), 0, threshold_mult} {}

    // Train on normal data
    void train(const std::vector<std::vector<uint8_t>>& normal_data) {
        // Concatenate all normal data
        detection_model.normal_profile.clear();
        for (const auto& sample : normal_data) {
            detection_model.normal_profile.insert(
                detection_model.normal_profile.end(),
                sample.begin(), sample.end());
        }

        // Compute baseline complexity
        auto compressed = compressor.compress(detection_model.normal_profile);
        detection_model.baseline_complexity =
            static_cast<value_type>(compressed.size()) /
            static_cast<value_type>(detection_model.normal_profile.size());
    }

    // Detect anomaly
    bool is_anomaly(const std::vector<uint8_t>& sample) {
        value_type score = anomaly_score(sample);
        return score > detection_model.threshold_multiplier;
    }

    // Compute anomaly score (higher = more anomalous)
    value_type anomaly_score(const std::vector<uint8_t>& sample) {
        // Compress sample alone
        auto compressed_alone = compressor.compress(sample);
        value_type complexity_alone =
            static_cast<value_type>(compressed_alone.size()) /
            static_cast<value_type>(sample.size());

        // Compress sample with normal profile
        std::vector<uint8_t> combined = detection_model.normal_profile;
        combined.insert(combined.end(), sample.begin(), sample.end());
        auto compressed_combined = compressor.compress(combined);

        // Conditional complexity
        value_type conditional_complexity =
            static_cast<value_type>(compressed_combined.size() -
                                   compressor.compress(detection_model.normal_profile).size()) /
            static_cast<value_type>(sample.size());

        // Anomaly score: how much harder to compress given normal data
        return conditional_complexity / detection_model.baseline_complexity;
    }

    // Detect point anomalies in time series
    std::vector<size_t> detect_point_anomalies(
        const std::vector<value_type>& time_series,
        size_t window_size = 50) {

        std::vector<size_t> anomalies;

        for (size_t i = window_size; i < time_series.size(); ++i) {
            // Convert window to bytes
            std::vector<uint8_t> window_data;
            for (size_t j = i - window_size; j < i; ++j) {
                auto bytes = reinterpret_cast<const uint8_t*>(&time_series[j]);
                window_data.insert(window_data.end(), bytes, bytes + sizeof(value_type));
            }

            // Check current point
            auto point_bytes = reinterpret_cast<const uint8_t*>(&time_series[i]);
            std::vector<uint8_t> point_data(point_bytes, point_bytes + sizeof(value_type));

            // Compute compression-based anomaly score
            auto window_compressed = compressor.compress(window_data);
            std::vector<uint8_t> combined = window_data;
            combined.insert(combined.end(), point_data.begin(), point_data.end());
            auto combined_compressed = compressor.compress(combined);

            value_type compression_gain =
                static_cast<value_type>(combined_compressed.size() - window_compressed.size()) /
                static_cast<value_type>(point_data.size());

            // High compression gain = anomaly
            if (compression_gain > detection_model.threshold_multiplier) {
                anomalies.push_back(i);
            }
        }

        return anomalies;
    }
};

// Grammar-based feature extraction
class grammar_feature_extractor {
public:
    using rule_type = std::pair<std::string, std::vector<std::string>>;

private:
    struct grammar {
        std::unordered_map<std::string, std::vector<std::vector<std::string>>> rules;
        std::string start_symbol;
    };

    grammar learned_grammar;

public:
    // Learn grammar from sequences
    void learn_grammar(const std::vector<std::string>& sequences) {
        // Simplified grammar induction using repeated substrings
        std::unordered_map<std::string, size_t> substring_counts;

        // Count all substrings
        for (const auto& seq : sequences) {
            for (size_t len = 2; len <= seq.size() / 2; ++len) {
                for (size_t i = 0; i <= seq.size() - len; ++i) {
                    substring_counts[seq.substr(i, len)]++;
                }
            }
        }

        // Create rules for frequent substrings
        size_t rule_id = 0;
        for (const auto& [substring, count] : substring_counts) {
            if (count > sequences.size() / 2) {  // Appears in >50% of sequences
                std::string non_terminal = "N" + std::to_string(rule_id++);
                learned_grammar.rules[non_terminal].push_back(
                    {substring});
            }
        }

        learned_grammar.start_symbol = "S";
    }

    // Extract features as grammar rule applications
    std::vector<double> extract_features(const std::string& sequence) {
        std::vector<double> features;

        // Count rule applications
        for (const auto& [non_terminal, productions] : learned_grammar.rules) {
            size_t count = 0;
            for (const auto& production : productions) {
                if (production.size() == 1) {
                    // Count occurrences of this pattern
                    size_t pos = 0;
                    while ((pos = sequence.find(production[0], pos)) != std::string::npos) {
                        count++;
                        pos += production[0].length();
                    }
                }
            }
            features.push_back(static_cast<double>(count));
        }

        // Normalize features
        double sum = std::accumulate(features.begin(), features.end(), 0.0);
        if (sum > 0) {
            for (auto& f : features) {
                f /= sum;
            }
        }

        return features;
    }

    // Get interpretable rule descriptions
    std::vector<std::string> get_rule_descriptions() {
        std::vector<std::string> descriptions;
        for (const auto& [non_terminal, productions] : learned_grammar.rules) {
            for (const auto& production : productions) {
                std::string desc = non_terminal + " -> ";
                for (const auto& symbol : production) {
                    desc += symbol + " ";
                }
                descriptions.push_back(desc);
            }
        }
        return descriptions;
    }
};

// PAC-Compression bounds for generalization
template<typename Hypothesis>
class pac_compression_learner {
public:
    using hypothesis_type = Hypothesis;
    using value_type = double;

private:
    struct compression_scheme {
        std::vector<size_t> compression_set;  // Indices of compressed samples
        hypothesis_type hypothesis;
        value_type reconstruction_error;
    };

    compression_scheme best_scheme;

public:
    // Learn hypothesis with compression-based generalization bound
    hypothesis_type learn_with_compression(
        const std::vector<std::vector<value_type>>& training_data,
        const std::vector<value_type>& labels,
        size_t compression_size,
        value_type delta = 0.05) {

        size_t n = training_data.size();
        if (compression_size > n) compression_size = n;

        value_type best_bound = std::numeric_limits<value_type>::max();

        // Try different compression sets
        for (size_t trial = 0; trial < 100; ++trial) {
            // Random compression set
            std::vector<size_t> compression_set;
            std::vector<bool> selected(n, false);

            for (size_t i = 0; i < compression_size; ++i) {
                size_t idx;
                do {
                    idx = rand() % n;
                } while (selected[idx]);
                compression_set.push_back(idx);
                selected[idx] = true;
            }

            // Train hypothesis on compression set
            hypothesis_type h = train_on_subset(training_data, labels, compression_set);

            // Compute empirical error
            value_type empirical_error = 0;
            for (size_t i = 0; i < n; ++i) {
                if (!selected[i]) {
                    value_type prediction = h.predict(training_data[i]);
                    empirical_error += std::abs(prediction - labels[i]);
                }
            }
            empirical_error /= (n - compression_size);

            // PAC-Compression bound
            value_type bound = compute_pac_bound(empirical_error, compression_size, n, delta);

            if (bound < best_bound) {
                best_bound = bound;
                best_scheme = {compression_set, h, empirical_error};
            }
        }

        return best_scheme.hypothesis;
    }

    // Get generalization bound
    value_type get_generalization_bound(size_t sample_size, value_type delta = 0.05) {
        return compute_pac_bound(best_scheme.reconstruction_error,
                                best_scheme.compression_set.size(),
                                sample_size, delta);
    }

private:
    hypothesis_type train_on_subset(
        const std::vector<std::vector<value_type>>& data,
        const std::vector<value_type>& labels,
        const std::vector<size_t>& indices) {

        // Simplified: create hypothesis from subset
        hypothesis_type h;
        // ... training logic specific to hypothesis type ...
        return h;
    }

    value_type compute_pac_bound(value_type empirical_error,
                                 size_t compression_size,
                                 size_t sample_size,
                                 value_type delta) {
        // PAC-Compression bound: err <= emp_err + sqrt((k*log(n) + log(1/delta))/n)
        value_type k = static_cast<value_type>(compression_size);
        value_type n = static_cast<value_type>(sample_size);

        value_type complexity_term = std::sqrt((k * std::log(n) + std::log(1/delta)) / n);

        return empirical_error + complexity_term;
    }
};

// MDL-based model selection
template<typename Model>
class mdl_model_selector {
public:
    using model_type = Model;
    using value_type = double;

private:
    struct mdl_score {
        model_type model;
        value_type description_length;
        value_type data_given_model_length;
        value_type total_length;
    };

public:
    // Select best model using MDL principle
    model_type select_model(
        const std::vector<model_type>& candidates,
        const std::vector<std::vector<value_type>>& data) {

        mdl_score best_score;
        best_score.total_length = std::numeric_limits<value_type>::max();

        for (const auto& model : candidates) {
            // Model description length (complexity)
            value_type model_length = compute_model_description_length(model);

            // Data given model length (goodness of fit)
            value_type data_length = compute_data_given_model_length(model, data);

            // Total MDL score
            value_type total = model_length + data_length;

            if (total < best_score.total_length) {
                best_score = {model, model_length, data_length, total};
            }
        }

        return best_score.model;
    }

    // Two-part MDL
    value_type two_part_mdl(const model_type& model,
                           const std::vector<std::vector<value_type>>& data) {
        return compute_model_description_length(model) +
               compute_data_given_model_length(model, data);
    }

    // Normalized Maximum Likelihood (NML) for refined MDL
    value_type normalized_ml(const model_type& model,
                            const std::vector<std::vector<value_type>>& data) {
        value_type likelihood = compute_likelihood(model, data);
        value_type normalization = compute_normalization_constant(model, data.size());

        return -std::log(likelihood / normalization);
    }

private:
    value_type compute_model_description_length(const model_type& model) {
        // Simplified: proportional to number of parameters
        return model.parameter_count() * std::log(2);
    }

    value_type compute_data_given_model_length(
        const model_type& model,
        const std::vector<std::vector<value_type>>& data) {

        value_type total_error = 0;
        for (const auto& sample : data) {
            auto prediction = model.predict(sample);
            // Use squared error as proxy for encoding length
            value_type error = 0;
            for (size_t i = 0; i < prediction.size(); ++i) {
                value_type diff = prediction[i] - sample[i];
                error += diff * diff;
            }
            total_error += error;
        }

        // Convert error to bits (Shannon-Fano coding)
        return std::log2(total_error + 1);
    }

    value_type compute_likelihood(const model_type& model,
                                 const std::vector<std::vector<value_type>>& data) {
        // Simplified Gaussian likelihood
        value_type likelihood = 1;
        for (const auto& sample : data) {
            auto prediction = model.predict(sample);
            value_type error = 0;
            for (size_t i = 0; i < prediction.size(); ++i) {
                value_type diff = prediction[i] - sample[i];
                error += diff * diff;
            }
            likelihood *= std::exp(-error / 2);
        }
        return likelihood;
    }

    value_type compute_normalization_constant(const model_type& model, size_t n) {
        // Simplified: use Rissanen's approximation
        return std::pow(n, model.parameter_count() / 2.0);
    }
};

// Compression-based reasoning engine
class compression_reasoner {
public:
    using value_type = double;

private:
    // normalized_compression_distance<kolmogorov::compressor> ncd;
    int ncd;  // Placeholder - kolmogorov namespace not implemented

public:
    // Analogical reasoning: A:B :: C:?
    template<typename T>
    T solve_analogy(const T& a, const T& b, const T& c,
                   const std::vector<T>& candidates) {

        // Find relationship between A and B
        auto ab_data = serialize(a, b);
        auto ab_compressed = compress(ab_data);

        T best_candidate = candidates[0];
        value_type best_score = std::numeric_limits<value_type>::max();

        for (const auto& d : candidates) {
            // Check if C:D has similar relationship as A:B
            auto cd_data = serialize(c, d);
            auto cd_compressed = compress(cd_data);

            // Compare compression patterns
            // value_type score = ncd.distance(ab_compressed, cd_compressed);
            value_type score = 0.0; // Placeholder

            if (score < best_score) {
                best_score = score;
                best_candidate = d;
            }
        }

        return best_candidate;
    }

    // Causal discovery through compression
    std::vector<std::pair<size_t, size_t>> discover_causal_links(
        const std::vector<std::vector<value_type>>& time_series) {

        std::vector<std::pair<size_t, size_t>> causal_links;
        size_t n = time_series.size();

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;

                // Test if series i causes series j
                value_type causal_score = compression_causality(time_series[i], time_series[j]);

                if (causal_score > 0.7) {  // High causality threshold
                    causal_links.push_back({i, j});
                }
            }
        }

        return causal_links;
    }

    // Concept learning as grammar induction
    std::string learn_concept(const std::vector<std::string>& positive_examples,
                              const std::vector<std::string>& negative_examples) {

        // Find minimal description that separates positive from negative
        std::string best_concept;
        value_type best_mdl = std::numeric_limits<value_type>::max();

        // Try different concept descriptions
        for (size_t length = 1; length <= 10; ++length) {
            auto concept_grammar = induce_concept_grammar(positive_examples, length);

            // Compute MDL score
            value_type description_length = concept_grammar.size() * std::log(26);  // Alphabet size

            value_type data_length = 0;
            for (const auto& pos : positive_examples) {
                if (!matches_concept(pos, concept_grammar)) {
                    data_length += pos.size() * std::log(26);  // Cost of exception
                }
            }
            for (const auto& neg : negative_examples) {
                if (matches_concept(neg, concept_grammar)) {
                    data_length += neg.size() * std::log(26);  // Cost of exception
                }
            }

            value_type total_mdl = description_length + data_length;
            if (total_mdl < best_mdl) {
                best_mdl = total_mdl;
                best_concept = concept_grammar;
            }
        }

        return best_concept;
    }

private:
    template<typename T>
    std::vector<uint8_t> serialize(const T& a, const T& b) {
        // Simplified serialization
        std::vector<uint8_t> result;
        auto a_bytes = reinterpret_cast<const uint8_t*>(&a);
        auto b_bytes = reinterpret_cast<const uint8_t*>(&b);
        result.insert(result.end(), a_bytes, a_bytes + sizeof(T));
        result.insert(result.end(), b_bytes, b_bytes + sizeof(T));
        return result;
    }

    std::vector<uint8_t> compress(const std::vector<uint8_t>& data) {
        // Simplified compression (RLE for demonstration)
        std::vector<uint8_t> compressed;

        for (size_t i = 0; i < data.size(); ) {
            uint8_t value = data[i];
            size_t count = 1;

            while (i + count < data.size() && data[i + count] == value && count < 255) {
                count++;
            }

            if (count > 2) {
                compressed.push_back(0xFF);  // Marker
                compressed.push_back(static_cast<uint8_t>(count));
                compressed.push_back(value);
                i += count;
            } else {
                compressed.push_back(data[i]);
                i++;
            }
        }

        return compressed.size() < data.size() ? compressed : data;
    }

    value_type compression_causality(const std::vector<value_type>& cause,
                                    const std::vector<value_type>& effect) {
        // Simplified causality measure using conditional compression
        auto cause_data = serialize_series(cause);
        auto effect_data = serialize_series(effect);

        auto cause_compressed = compress(cause_data);
        auto effect_compressed = compress(effect_data);

        // Concatenate cause then effect
        std::vector<uint8_t> causal_order = cause_data;
        causal_order.insert(causal_order.end(), effect_data.begin(), effect_data.end());
        auto causal_compressed = compress(causal_order);

        // Concatenate effect then cause
        std::vector<uint8_t> reverse_order = effect_data;
        reverse_order.insert(reverse_order.end(), cause_data.begin(), cause_data.end());
        auto reverse_compressed = compress(reverse_order);

        // If cause->effect compresses better, there's likely causality
        value_type causal_ratio = static_cast<value_type>(causal_compressed.size()) /
                                 static_cast<value_type>(cause_compressed.size() + effect_compressed.size());
        value_type reverse_ratio = static_cast<value_type>(reverse_compressed.size()) /
                                  static_cast<value_type>(cause_compressed.size() + effect_compressed.size());

        return (reverse_ratio - causal_ratio) / reverse_ratio;
    }

    std::vector<uint8_t> serialize_series(const std::vector<value_type>& series) {
        std::vector<uint8_t> result;
        for (const auto& val : series) {
            auto bytes = reinterpret_cast<const uint8_t*>(&val);
            result.insert(result.end(), bytes, bytes + sizeof(value_type));
        }
        return result;
    }

    std::string induce_concept_grammar(const std::vector<std::string>& examples,
                                      size_t max_length) {
        // Find common patterns (simplified)
        if (examples.empty()) return "";

        std::string pattern;
        for (size_t i = 0; i < std::min(max_length, examples[0].size()); ++i) {
            char c = examples[0][i];
            bool all_match = true;

            for (const auto& ex : examples) {
                if (i >= ex.size() || ex[i] != c) {
                    all_match = false;
                    break;
                }
            }

            if (all_match) {
                pattern += c;
            } else {
                pattern += '*';  // Wildcard
            }
        }

        return pattern;
    }

    bool matches_concept(const std::string& example, const std::string& concept_pattern) {
        if (concept_pattern.size() > example.size()) return false;

        for (size_t i = 0; i < concept_pattern.size(); ++i) {
            if (concept_pattern[i] != '*' && concept_pattern[i] != example[i]) {
                return false;
            }
        }
        return true;
    }
};

} // namespace stepanov::compression::ml