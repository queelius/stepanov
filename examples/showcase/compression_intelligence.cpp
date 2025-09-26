/**
 * Compression as Intelligence
 * ============================
 *
 * This example demonstrates the profound connection between
 * compression and intelligence, showing that understanding,
 * learning, and prediction are all forms of compression.
 */

#include <stepanov/compression.hpp>
#include <stepanov/algorithms.hpp>
#include <stepanov/lazy.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <map>

namespace stepanov::examples {

/**
 * Kolmogorov Complexity and Algorithmic Information Theory
 */
namespace kolmogorov {

    // Estimate Kolmogorov complexity via compression
    double estimate_complexity(const std::string& data) {
        auto compressed = stepanov::compress(data);
        return static_cast<double>(compressed.size()) / data.size();
    }

    // Measure randomness (incompressibility)
    double randomness_score(const std::string& data) {
        return 1.0 - stepanov::compression_ratio(data,
            stepanov::compress(data));
    }

    void demo_kolmogorov() {
        std::cout << "=== Kolmogorov Complexity ===\n\n";

        std::vector<std::pair<std::string, std::string>> samples = {
            {"0000000000000000", "Repetitive"},
            {"0101010101010101", "Pattern"},
            {"0110100110010110", "Thue-Morse"},
            {"1101001110100101", "Random-like"},
            {"The quick brown fox", "English"},
            {"aaabbbcccdddeee", "Run-length"},
            {"x7k3m9p2q5w8n1", "High entropy"}
        };

        std::cout << "String                  | Complexity | Randomness | Type\n";
        std::cout << "------------------------|------------|------------|------------\n";

        for (const auto& [str, type] : samples) {
            double complexity = estimate_complexity(str);
            double randomness = randomness_score(str);

            std::cout << std::left << std::setw(23) << str.substr(0, 20) << " | "
                      << std::fixed << std::setprecision(3) << std::setw(10) << complexity << " | "
                      << std::setw(10) << randomness << " | "
                      << type << "\n";
        }

        std::cout << "\nInterpretation:\n";
        std::cout << "  • Low complexity = High compressibility = Pattern exists\n";
        std::cout << "  • High complexity = Low compressibility = Random/complex\n";
        std::cout << "  • Kolmogorov complexity measures the shortest program\n";
        std::cout << "    that generates the string\n\n";
    }

    // Algorithmic probability
    double algorithmic_probability(const std::string& data,
                                  const std::vector<std::string>& hypotheses) {
        double min_length = std::numeric_limits<double>::max();

        for (const auto& hypothesis : hypotheses) {
            auto combined = hypothesis + data;
            auto compressed = stepanov::compress(combined);
            min_length = std::min(min_length, static_cast<double>(compressed.size()));
        }

        // Solomonoff's universal prior: P(x) ∝ 2^(-K(x))
        return std::pow(2.0, -min_length);
    }
}

/**
 * Minimum Description Length (MDL) Principle
 */
namespace mdl {

    struct model {
        std::string description;
        std::function<std::string(const std::string&)> encode;
        double complexity;
    };

    // Select best model using MDL
    model select_model(const std::string& data,
                       const std::vector<model>& models) {
        double best_score = std::numeric_limits<double>::max();
        model best_model = models[0];

        for (const auto& m : models) {
            // MDL = model complexity + data complexity given model
            std::string encoded = m.encode(data);
            auto compressed = stepanov::compress(encoded);
            double data_length = compressed.size();
            double total_length = m.complexity + data_length;

            std::cout << "Model: " << m.description << "\n";
            std::cout << "  Model complexity: " << m.complexity << "\n";
            std::cout << "  Data given model: " << data_length << "\n";
            std::cout << "  Total (MDL): " << total_length << "\n\n";

            if (total_length < best_score) {
                best_score = total_length;
                best_model = m;
            }
        }

        return best_model;
    }

    void demo_mdl() {
        std::cout << "=== Minimum Description Length Principle ===\n\n";

        // Data to model
        std::string data = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";

        std::cout << "Data: " << data << "\n\n";
        std::cout << "Comparing models:\n\n";

        std::vector<model> models = {
            {
                "Raw data",
                [](const std::string& d) { return d; },
                1.0  // Minimal model
            },
            {
                "Arithmetic sequence",
                [](const std::string& d) {
                    // Encode as "start,step,count"
                    return "1,1,15";
                },
                10.0  // Model description cost
            },
            {
                "Polynomial",
                [](const std::string& d) {
                    // Encode as coefficients
                    return "0,1,0";  // f(x) = x
                },
                20.0  // More complex model
            }
        };

        auto best = select_model(data, models);
        std::cout << "Best model (MDL): " << best.description << "\n\n";

        std::cout << "Insight: MDL balances model complexity with data fit.\n";
        std::cout << "Simple patterns need simple models (Occam's Razor).\n\n";
    }
}

/**
 * Prediction via Compression
 */
namespace prediction {

    // Context-tree weighting for sequence prediction
    class context_tree_predictor {
        struct node {
            std::map<char, int> counts;
            std::map<char, std::unique_ptr<node>> children;
            int total = 0;
        };

        std::unique_ptr<node> root;
        int max_depth;

    public:
        context_tree_predictor(int depth = 5)
            : root(std::make_unique<node>()), max_depth(depth) {}

        void update(const std::string& context, char symbol) {
            node* current = root.get();

            // Update all context lengths
            for (size_t len = 0; len <= std::min(static_cast<size_t>(max_depth),
                                                 context.size()); ++len) {
                current->counts[symbol]++;
                current->total++;

                if (len < context.size()) {
                    char ctx = context[context.size() - 1 - len];
                    if (!current->children[ctx]) {
                        current->children[ctx] = std::make_unique<node>();
                    }
                    current = current->children[ctx].get();
                }
            }
        }

        char predict(const std::string& context) {
            node* current = root.get();

            // Use longest matching context
            for (int i = context.size() - 1; i >= 0 && current; --i) {
                if (current->children.count(context[i])) {
                    current = current->children[context[i]].get();
                } else {
                    break;
                }
            }

            // Find most likely next symbol
            char best = '\0';
            int best_count = 0;
            for (const auto& [symbol, count] : current->counts) {
                if (count > best_count) {
                    best_count = count;
                    best = symbol;
                }
            }

            return best;
        }

        double predict_probability(const std::string& context, char symbol) {
            node* current = root.get();

            for (int i = context.size() - 1; i >= 0 && current; --i) {
                if (current->children.count(context[i])) {
                    current = current->children[context[i]].get();
                } else {
                    break;
                }
            }

            if (current->total == 0) return 0.0;
            return static_cast<double>(current->counts[symbol]) / current->total;
        }
    };

    void demo_prediction() {
        std::cout << "=== Prediction via Compression ===\n\n";

        // Train on sequence with patterns
        std::string training = "abcabcabcdefdefdefabcabcdefdef";
        context_tree_predictor predictor(3);

        std::cout << "Training on: " << training << "\n\n";

        for (size_t i = 0; i < training.size() - 1; ++i) {
            std::string context = training.substr(0, i);
            predictor.update(context, training[i]);
        }

        // Test predictions
        std::vector<std::pair<std::string, std::string>> tests = {
            {"ab", "Next after 'ab'"},
            {"abc", "Next after 'abc'"},
            {"de", "Next after 'de'"},
            {"def", "Next after 'def'"},
            {"abcab", "Next after 'abcab'"}
        };

        std::cout << "Predictions:\n";
        for (const auto& [context, description] : tests) {
            char predicted = predictor.predict(context);
            double prob = predictor.predict_probability(context, predicted);

            std::cout << "  " << description << ": '" << predicted << "' "
                      << "(confidence: " << std::fixed << std::setprecision(2)
                      << prob * 100 << "%)\n";
        }

        std::cout << "\nNote: The predictor learned the patterns through compression!\n\n";
    }

    // Sequence generation via compression
    std::string generate_sequence(const std::string& seed, int length) {
        std::string result = seed;

        for (int i = 0; i < length; ++i) {
            // Find the continuation that compresses best
            char best_char = 'a';
            double best_ratio = 0;

            for (char c = 'a'; c <= 'z'; ++c) {
                std::string candidate = result + c;
                auto compressed = stepanov::compress(candidate);
                double ratio = stepanov::compression_ratio(candidate, compressed);

                if (ratio > best_ratio) {
                    best_ratio = ratio;
                    best_char = c;
                }
            }

            result += best_char;
        }

        return result;
    }
}

/**
 * Normalized Information Distance (NID)
 */
namespace similarity {

    // Compute normalized compression distance
    double ncd(const std::string& x, const std::string& y) {
        auto cx = stepanov::compress(x);
        auto cy = stepanov::compress(y);
        auto cxy = stepanov::compress(x + y);
        auto cyx = stepanov::compress(y + x);

        double ncd_score = (std::min(cxy.size(), cyx.size()) -
                           std::min(cx.size(), cy.size())) /
                          static_cast<double>(std::max(cx.size(), cy.size()));

        return ncd_score;
    }

    // Cluster data using compression distance
    std::vector<std::vector<int>> cluster_by_compression(
        const std::vector<std::string>& data,
        double threshold = 0.5) {

        int n = data.size();
        std::vector<std::vector<double>> distances(n, std::vector<double>(n, 0));

        // Compute pairwise distances
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                distances[i][j] = distances[j][i] = ncd(data[i], data[j]);
            }
        }

        // Simple clustering
        std::vector<int> cluster_id(n, -1);
        int next_cluster = 0;

        for (int i = 0; i < n; ++i) {
            if (cluster_id[i] == -1) {
                cluster_id[i] = next_cluster++;

                for (int j = i + 1; j < n; ++j) {
                    if (cluster_id[j] == -1 && distances[i][j] < threshold) {
                        cluster_id[j] = cluster_id[i];
                    }
                }
            }
        }

        // Group by cluster
        std::map<int, std::vector<int>> clusters;
        for (int i = 0; i < n; ++i) {
            clusters[cluster_id[i]].push_back(i);
        }

        std::vector<std::vector<int>> result;
        for (const auto& [_, indices] : clusters) {
            result.push_back(indices);
        }

        return result;
    }

    void demo_similarity() {
        std::cout << "=== Similarity via Compression ===\n\n";

        std::vector<std::string> documents = {
            "The cat sat on the mat",           // 0
            "The dog sat on the log",           // 1
            "Cats and dogs are pets",           // 2
            "import numpy as np",                // 3
            "from sklearn import svm",          // 4
            "Machine learning with Python",      // 5
            "The feline rested on the rug",     // 6
            "def function(): return None"        // 7
        };

        std::cout << "Documents:\n";
        for (size_t i = 0; i < documents.size(); ++i) {
            std::cout << "  " << i << ": \"" << documents[i] << "\"\n";
        }
        std::cout << "\n";

        // Compute similarity matrix
        std::cout << "Normalized Compression Distance Matrix:\n";
        std::cout << "    ";
        for (size_t i = 0; i < documents.size(); ++i) {
            std::cout << std::setw(5) << i;
        }
        std::cout << "\n";

        for (size_t i = 0; i < documents.size(); ++i) {
            std::cout << std::setw(2) << i << ": ";
            for (size_t j = 0; j < documents.size(); ++j) {
                if (i == j) {
                    std::cout << std::setw(5) << "0.00";
                } else {
                    double dist = ncd(documents[i], documents[j]);
                    std::cout << std::fixed << std::setprecision(2)
                              << std::setw(5) << dist;
                }
            }
            std::cout << "\n";
        }

        // Cluster documents
        auto clusters = cluster_by_compression(documents, 0.6);

        std::cout << "\nClusters (threshold = 0.6):\n";
        for (size_t i = 0; i < clusters.size(); ++i) {
            std::cout << "  Cluster " << i << ": ";
            for (int idx : clusters[i]) {
                std::cout << idx << " ";
            }
            std::cout << "\n";
        }

        std::cout << "\nInterpretation:\n";
        std::cout << "  • Similar content compresses well together\n";
        std::cout << "  • NCD discovers semantic similarity\n";
        std::cout << "  • No features or domain knowledge required!\n\n";
    }
}

/**
 * Hutter Prize and Universal Intelligence
 */
namespace universal {

    // Approximate universal intelligence via compression
    class universal_agent {
        std::string knowledge_base;
        context_tree_predictor predictor;

    public:
        universal_agent() : predictor(10) {}

        // Learn from observation
        void observe(const std::string& data) {
            knowledge_base += data;

            // Update predictor
            for (size_t i = 0; i < data.size() - 1; ++i) {
                predictor.update(data.substr(0, i), data[i]);
            }
        }

        // Act based on compression
        std::string act(const std::string& situation) {
            // Find the action that makes the situation most compressible
            // with our knowledge base
            std::vector<std::string> possible_actions = {
                "move_forward", "turn_left", "turn_right", "interact", "wait"
            };

            std::string best_action;
            double best_score = std::numeric_limits<double>::max();

            for (const auto& action : possible_actions) {
                std::string combined = knowledge_base + situation + action;
                auto compressed = stepanov::compress(combined);
                double score = compressed.size();

                if (score < best_score) {
                    best_score = score;
                    best_action = action;
                }
            }

            return best_action;
        }

        // Predict future
        std::string predict(const std::string& context, int steps) {
            std::string prediction = context;
            for (int i = 0; i < steps; ++i) {
                char next = predictor.predict(prediction);
                if (next == '\0') break;
                prediction += next;
            }
            return prediction.substr(context.size());
        }

        double intelligence_score() {
            // Intelligence ∝ compression ability
            auto compressed = stepanov::compress(knowledge_base);
            return static_cast<double>(knowledge_base.size()) / compressed.size();
        }
    };

    void demo_universal() {
        std::cout << "=== Universal Intelligence ===\n\n";

        universal_agent agent;

        std::cout << "Training universal agent...\n";

        // Teach patterns
        agent.observe("if see danger then avoid");
        agent.observe("if see food then approach");
        agent.observe("if see friend then greet");
        agent.observe("danger leads to harm");
        agent.observe("food leads to energy");
        agent.observe("friend leads to help");

        std::cout << "Agent intelligence score: "
                  << agent.intelligence_score() << "\n\n";

        // Test agent
        std::vector<std::string> situations = {
            "see danger",
            "see food",
            "see friend",
            "see unknown"
        };

        std::cout << "Agent responses:\n";
        for (const auto& situation : situations) {
            std::string action = agent.act(situation);
            std::cout << "  Situation: " << situation
                      << " -> Action: " << action << "\n";
        }

        std::cout << "\n";

        // Test prediction
        std::string context = "if see ";
        std::string prediction = agent.predict(context, 10);
        std::cout << "Prediction after '" << context << "': "
                  << prediction << "\n\n";

        std::cout << "Note: The agent learned and acts purely through compression!\n\n";
    }

    void demo_hutter_prize() {
        std::cout << "=== The Hutter Prize Connection ===\n\n";

        std::cout << "The Hutter Prize offers 500,000 euros for compressing\n";
        std::cout << "human knowledge (Wikipedia) better than current methods.\n\n";

        std::cout << "Why? Because Marcus Hutter proved mathematically that:\n\n";

        std::cout << "  Compression = Learning = Prediction = Intelligence\n\n";

        std::cout << "The proof (simplified):\n";
        std::cout << "  1. Learning: Finding patterns in data\n";
        std::cout << "  2. Patterns: Allow shorter descriptions (compression)\n";
        std::cout << "  3. Compression: Enables prediction (patterns continue)\n";
        std::cout << "  4. Prediction: Core of intelligent behavior\n\n";

        std::cout << "Therefore:\n";
        std::cout << "  Better compression → Better learning → Higher intelligence\n\n";

        std::cout << "Stepanov implements this principle:\n";
        std::cout << "  • Every compression algorithm is a learning algorithm\n";
        std::cout << "  • Every predictor is a compressor\n";
        std::cout << "  • Every intelligent system finds patterns\n\n";

        // Demonstrate with actual compression
        std::string knowledge = "Mathematics is the language of patterns. "
                               "Patterns lead to compression. "
                               "Compression leads to understanding. "
                               "Understanding leads to intelligence. "
                               "Intelligence leads to better compression.";

        auto compressed = stepanov::compress(knowledge);
        double ratio = stepanov::compression_ratio(knowledge, compressed);

        std::cout << "Example knowledge base:\n";
        std::cout << "  Original size: " << knowledge.size() << " bytes\n";
        std::cout << "  Compressed size: " << compressed.size() << " bytes\n";
        std::cout << "  Compression ratio: " << std::fixed << std::setprecision(2)
                  << ratio * 100 << "%\n";
        std::cout << "  Information content: " << (1.0 - ratio) * 100 << "%\n\n";

        std::cout << "The better we compress, the more we understand.\n";
        std::cout << "The more we understand, the more intelligent we are.\n\n";
    }
}

/**
 * Philosophical Implications
 */
void demo_philosophical() {
    std::cout << "=== The Deep Truth: Compression IS Intelligence ===\n\n";

    std::cout << "Traditional AI (Statistical Learning):\n";
    std::cout << "  • Millions of parameters\n";
    std::cout << "  • Gradient descent\n";
    std::cout << "  • Black box models\n";
    std::cout << "  • Domain-specific\n\n";

    std::cout << "Compression-Based AI (Algorithmic Information Theory):\n";
    std::cout << "  • No parameters\n";
    std::cout << "  • Information theory\n";
    std::cout << "  • Explainable by design\n";
    std::cout << "  • Universal\n\n";

    std::cout << "The Fundamental Equation:\n";
    std::cout << "  K(x) = min{|p| : U(p) = x}\n";
    std::cout << "  (Kolmogorov complexity = shortest program generating x)\n\n";

    std::cout << "Implications:\n";
    std::cout << "  1. Understanding = Finding short descriptions\n";
    std::cout << "  2. Learning = Improving compression over time\n";
    std::cout << "  3. Creativity = Generating compressible patterns\n";
    std::cout << "  4. Intelligence = Optimal compression\n\n";

    std::cout << "Real-World Applications:\n";
    std::cout << "  • DNA is compressed information (3 billion bases → 750MB)\n";
    std::cout << "  • Physics equations compress universe's behavior\n";
    std::cout << "  • Music theory compresses sound patterns\n";
    std::cout << "  • Mathematics compresses logical relationships\n\n";

    std::cout << "Stepanov's Vision:\n";
    std::cout << "By making compression a first-class operation, we enable:\n";
    std::cout << "  • Universal learning without training\n";
    std::cout << "  • Pattern discovery without features\n";
    std::cout << "  • Similarity without metrics\n";
    std::cout << "  • Prediction without models\n";
    std::cout << "  • Intelligence without parameters\n\n";

    std::cout << "This is not just theory. This is computation at its purest:\n";
    std::cout << "Finding the shortest program that explains the data.\n\n";

    std::cout << "Welcome to the future of AI:\n";
    std::cout << "Not bigger models, but better compression.\n";
    std::cout << "Not more parameters, but deeper understanding.\n";
    std::cout << "Not artificial intelligence, but algorithmic intelligence.\n";
}

} // namespace stepanov::examples

int main() {
    using namespace stepanov::examples;

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║             Compression as Intelligence                     ║\n";
    std::cout << "║     Understanding = Learning = Prediction = Compression    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // Demonstrate core concepts
    kolmogorov::demo_kolmogorov();
    mdl::demo_mdl();
    prediction::demo_prediction();
    similarity::demo_similarity();
    universal::demo_universal();
    universal::demo_hutter_prize();

    // Philosophy
    demo_philosophical();

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  'Compression is the ultimate form of understanding.'      ║\n";
    std::cout << "║                                    - Gregory Chaitin       ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║  'The ability to compress well is closely related to      ║\n";
    std::cout << "║   acting intelligently.'           - Marcus Hutter         ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║  Stepanov: Where information theory becomes intelligence.  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    return 0;
}