// Neural and Learned Compression: Where Compression Meets Intelligence
// "Compression is comprehension" - Gregory Chaitin
#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>
#include <unordered_map>
#include "../concepts.hpp"
#include "../autodiff.hpp"

// autodiff types are already included from autodiff.hpp

namespace stepanov::compression::neural {

// Neural network layer for compression
template<typename T = double>
struct neural_layer {
    using value_type = T;
    std::vector<std::vector<T>> weights;
    std::vector<T> biases;
    std::function<T(T)> activation;
    std::function<T(T)> activation_derivative;

    neural_layer(size_t input_dim, size_t output_dim,
                 std::function<T(T)> act = [](T x) { return std::tanh(x); },
                 std::function<T(T)> act_deriv = [](T x) {
                     T t = std::tanh(x);
                     return T(1) - t * t;
                 })
        : weights(output_dim, std::vector<T>(input_dim))
        , biases(output_dim)
        , activation(act)
        , activation_derivative(act_deriv) {

        // Xavier initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        T scale = std::sqrt(T(2) / T(input_dim + output_dim));
        std::normal_distribution<T> dist(T(0), scale);

        for (auto& row : weights) {
            for (auto& w : row) {
                w = dist(gen);
            }
        }
        for (auto& b : biases) {
            b = dist(gen);
        }
    }

    std::vector<T> forward(const std::vector<T>& input) const {
        std::vector<T> output(biases.size());
        for (size_t i = 0; i < output.size(); ++i) {
            T sum = biases[i];
            for (size_t j = 0; j < input.size(); ++j) {
                sum += weights[i][j] * input[j];
            }
            output[i] = activation(sum);
        }
        return output;
    }
};

// Neural Arithmetic Coding: Neural networks predict symbol probabilities
template<typename Symbol = uint8_t>
class neural_arithmetic_coder {
public:
    using symbol_type = Symbol;
    using prob_type = double;

private:
    // LSTM cell for sequence modeling
    struct lstm_cell {
        neural_layer<prob_type> forget_gate;
        neural_layer<prob_type> input_gate;
        neural_layer<prob_type> output_gate;
        neural_layer<prob_type> candidate;
        std::vector<prob_type> hidden_state;
        std::vector<prob_type> cell_state;

        lstm_cell(size_t input_dim, size_t hidden_dim)
            : forget_gate(input_dim + hidden_dim, hidden_dim,
                         [](prob_type x) { return prob_type(1) / (prob_type(1) + std::exp(-x)); })
            , input_gate(input_dim + hidden_dim, hidden_dim,
                        [](prob_type x) { return prob_type(1) / (prob_type(1) + std::exp(-x)); })
            , output_gate(input_dim + hidden_dim, hidden_dim,
                         [](prob_type x) { return prob_type(1) / (prob_type(1) + std::exp(-x)); })
            , candidate(input_dim + hidden_dim, hidden_dim)
            , hidden_state(hidden_dim, prob_type(0))
            , cell_state(hidden_dim, prob_type(0)) {}

        std::vector<prob_type> forward(const std::vector<prob_type>& input) {
            // Concatenate input with hidden state
            std::vector<prob_type> combined;
            combined.insert(combined.end(), input.begin(), input.end());
            combined.insert(combined.end(), hidden_state.begin(), hidden_state.end());

            // Compute gates
            auto f = forget_gate.forward(combined);
            auto i = input_gate.forward(combined);
            auto o = output_gate.forward(combined);
            auto c_tilde = candidate.forward(combined);

            // Update cell state
            for (size_t j = 0; j < cell_state.size(); ++j) {
                cell_state[j] = f[j] * cell_state[j] + i[j] * c_tilde[j];
                hidden_state[j] = o[j] * std::tanh(cell_state[j]);
            }

            return hidden_state;
        }
    };

    size_t symbol_count;
    size_t hidden_dim;
    size_t context_length;
    lstm_cell lstm;
    neural_layer<prob_type> output_layer;
    std::vector<Symbol> context;

public:
    neural_arithmetic_coder(size_t symbols = 256, size_t hidden = 128, size_t ctx_len = 32)
        : symbol_count(symbols)
        , hidden_dim(hidden)
        , context_length(ctx_len)
        , lstm(symbols, hidden)
        , output_layer(hidden, symbols,
                      [](prob_type x) { return x; })  // Linear for logits
        , context() {}

    // Get probability distribution for next symbol
    std::vector<prob_type> predict_distribution() {
        // One-hot encode context
        std::vector<prob_type> input(symbol_count, prob_type(0));
        if (!context.empty()) {
            input[context.back()] = prob_type(1);
        }

        // Forward through LSTM
        auto hidden = lstm.forward(input);
        auto logits = output_layer.forward(hidden);

        // Softmax for probabilities
        prob_type max_logit = *std::max_element(logits.begin(), logits.end());
        std::vector<prob_type> probs(logits.size());
        prob_type sum = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(logits[i] - max_logit);
            sum += probs[i];
        }
        for (auto& p : probs) {
            p /= sum;
        }

        return probs;
    }

    // Update context with new symbol
    void update_context(Symbol s) {
        context.push_back(s);
        if (context.size() > context_length) {
            context.erase(context.begin());
        }
    }

    // Encode sequence using neural predictions
    std::vector<bool> encode(const std::vector<Symbol>& data) {
        std::vector<bool> output;
        prob_type low = 0.0, high = 1.0;

        for (const auto& symbol : data) {
            auto probs = predict_distribution();

            // Compute cumulative probabilities
            std::vector<prob_type> cum_probs(probs.size() + 1, 0);
            for (size_t i = 0; i < probs.size(); ++i) {
                cum_probs[i + 1] = cum_probs[i] + probs[i];
            }

            // Update range
            prob_type range = high - low;
            high = low + range * cum_probs[symbol + 1];
            low = low + range * cum_probs[symbol];

            // Output bits when possible
            while (true) {
                if (high < 0.5) {
                    output.push_back(false);
                    low *= 2;
                    high *= 2;
                } else if (low >= 0.5) {
                    output.push_back(true);
                    low = 2 * (low - 0.5);
                    high = 2 * (high - 0.5);
                } else {
                    break;
                }
            }

            update_context(symbol);
        }

        // Final bits
        if (low < 0.25) {
            output.push_back(false);
        } else {
            output.push_back(true);
        }

        return output;
    }
};

// Variational Autoencoder for lossy compression
template<typename T = double>
class variational_autoencoder {
public:
    using value_type = T;

private:
    // Encoder network
    std::vector<neural_layer<T>> encoder_layers;
    neural_layer<T> mu_layer;
    neural_layer<T> log_var_layer;

    // Decoder network
    std::vector<neural_layer<T>> decoder_layers;

    size_t input_dim;
    size_t latent_dim;
    std::mt19937 rng;

public:
    variational_autoencoder(size_t input_dimension, size_t latent_dimension,
                           std::vector<size_t> hidden_dims = {512, 256})
        : input_dim(input_dimension)
        , latent_dim(latent_dimension)
        , rng(std::random_device{}()) {

        // Build encoder
        size_t last_dim = input_dim;
        for (size_t dim : hidden_dims) {
            encoder_layers.emplace_back(last_dim, dim);
            last_dim = dim;
        }
        mu_layer = neural_layer<T>(last_dim, latent_dim,
                                   [](T x) { return x; });  // Linear
        log_var_layer = neural_layer<T>(last_dim, latent_dim,
                                       [](T x) { return x; });  // Linear

        // Build decoder (reverse architecture)
        last_dim = latent_dim;
        std::vector<size_t> rev_dims(hidden_dims.rbegin(), hidden_dims.rend());
        for (size_t dim : rev_dims) {
            decoder_layers.emplace_back(last_dim, dim);
            last_dim = dim;
        }
        decoder_layers.emplace_back(last_dim, input_dim,
                                   [](T x) { return T(1) / (T(1) + std::exp(-x)); });  // Sigmoid
    }

    // Encode to latent space with reparameterization trick
    std::pair<std::vector<T>, std::vector<T>> encode(const std::vector<T>& input) {
        std::vector<T> hidden = input;
        for (const auto& layer : encoder_layers) {
            hidden = layer.forward(hidden);
        }

        auto mu = mu_layer.forward(hidden);
        auto log_var = log_var_layer.forward(hidden);

        // Reparameterization trick
        std::normal_distribution<T> dist(T(0), T(1));
        std::vector<T> z(latent_dim);
        for (size_t i = 0; i < latent_dim; ++i) {
            T epsilon = dist(rng);
            z[i] = mu[i] + std::exp(T(0.5) * log_var[i]) * epsilon;
        }

        return {z, log_var};
    }

    // Decode from latent space
    std::vector<T> decode(const std::vector<T>& z) {
        std::vector<T> hidden = z;
        for (const auto& layer : decoder_layers) {
            hidden = layer.forward(hidden);
        }
        return hidden;
    }

    // Compress data to latent representation
    std::vector<T> compress(const std::vector<T>& data) {
        auto [z, log_var] = encode(data);
        return z;
    }

    // Decompress from latent representation
    std::vector<T> decompress(const std::vector<T>& z) {
        return decode(z);
    }

    // Compute ELBO loss for training
    T elbo_loss(const std::vector<T>& input) {
        auto [z, log_var] = encode(input);
        auto mu = mu_layer.forward(input);
        auto reconstructed = decode(z);

        // Reconstruction loss (binary cross-entropy)
        T recon_loss = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            recon_loss -= input[i] * std::log(reconstructed[i] + T(1e-10))
                        + (T(1) - input[i]) * std::log(T(1) - reconstructed[i] + T(1e-10));
        }

        // KL divergence
        T kl_loss = 0;
        for (size_t i = 0; i < latent_dim; ++i) {
            kl_loss += T(0.5) * (mu[i] * mu[i] + std::exp(log_var[i]) - log_var[i] - T(1));
        }

        return recon_loss + kl_loss;
    }
};

// Learned Index Structure for compression
template<typename Key, typename Value>
class learned_index {
public:
    using key_type = Key;
    using value_type = Value;
    using model_type = neural_layer<double>;

private:
    struct segment {
        model_type model;
        std::vector<std::pair<Key, Value>> data;
        Key min_key, max_key;

        segment(size_t input_dim = 1, size_t hidden_dim = 32)
            : model(input_dim, 1,  // Single output for position prediction
                   [](double x) { return x; })  // Linear output
            , data()
            , min_key()
            , max_key() {}

        size_t predict_position(Key k) const {
            if (data.empty()) return 0;

            // Normalize key to [0, 1]
            double normalized = static_cast<double>(k - min_key) /
                              static_cast<double>(max_key - min_key + 1);

            auto prediction = model.forward({normalized});
            size_t pos = static_cast<size_t>(prediction[0] * data.size());
            return std::min(pos, data.size() - 1);
        }
    };

    std::vector<segment> segments;
    size_t max_segment_size;

public:
    learned_index(size_t max_seg_size = 1000)
        : max_segment_size(max_seg_size) {
        segments.emplace_back();
    }

    // Insert key-value pair
    void insert(Key k, Value v) {
        // Find appropriate segment
        auto& seg = segments.back();

        // Insert into segment
        seg.data.push_back({k, v});
        if (seg.data.size() == 1) {
            seg.min_key = seg.max_key = k;
        } else {
            seg.min_key = std::min(seg.min_key, k);
            seg.max_key = std::max(seg.max_key, k);
        }

        // Split if segment too large
        if (seg.data.size() > max_segment_size) {
            // Sort data
            std::sort(seg.data.begin(), seg.data.end());

            // Train model on sorted data
            train_segment_model(seg);

            // Create new segment
            segments.emplace_back();
        }
    }

    // Lookup with learned index
    std::optional<Value> lookup(Key k) const {
        for (const auto& seg : segments) {
            if (seg.data.empty()) continue;
            if (k < seg.min_key || k > seg.max_key) continue;

            // Use model to predict position
            size_t predicted_pos = seg.predict_position(k);

            // Local search around prediction
            size_t search_radius = std::sqrt(seg.data.size());
            size_t start = (predicted_pos > search_radius) ?
                          predicted_pos - search_radius : 0;
            size_t end = std::min(predicted_pos + search_radius, seg.data.size());

            for (size_t i = start; i < end; ++i) {
                if (seg.data[i].first == k) {
                    return seg.data[i].second;
                }
            }
        }
        return std::nullopt;
    }

    // Compress using learned CDF
    std::vector<uint8_t> compress() const {
        std::vector<uint8_t> compressed;

        for (const auto& seg : segments) {
            if (seg.data.empty()) continue;

            // Store segment metadata
            compressed.push_back(static_cast<uint8_t>(seg.data.size() >> 8));
            compressed.push_back(static_cast<uint8_t>(seg.data.size() & 0xFF));

            // Store model parameters (simplified)
            for (const auto& w : seg.model.weights[0]) {
                uint32_t encoded = *reinterpret_cast<const uint32_t*>(&w);
                compressed.push_back(encoded >> 24);
                compressed.push_back((encoded >> 16) & 0xFF);
                compressed.push_back((encoded >> 8) & 0xFF);
                compressed.push_back(encoded & 0xFF);
            }

            // Delta-encode sorted keys
            Key prev = seg.min_key;
            for (const auto& [k, v] : seg.data) {
                Key delta = k - prev;
                // Variable-length encoding of delta
                while (delta >= 128) {
                    compressed.push_back(0x80 | (delta & 0x7F));
                    delta >>= 7;
                }
                compressed.push_back(delta);
                prev = k;
            }
        }

        return compressed;
    }

private:
    void train_segment_model(segment& seg) {
        // Simple gradient descent to train model
        double learning_rate = 0.01;
        size_t epochs = 100;

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0;

            for (size_t i = 0; i < seg.data.size(); ++i) {
                Key k = seg.data[i].first;
                double normalized_key = static_cast<double>(k - seg.min_key) /
                                      static_cast<double>(seg.max_key - seg.min_key + 1);
                double target_pos = static_cast<double>(i) / seg.data.size();

                auto prediction = seg.model.forward({normalized_key});
                double error = prediction[0] - target_pos;
                total_loss += error * error;

                // Gradient update (simplified)
                seg.model.weights[0][0] -= learning_rate * error * normalized_key;
                seg.model.biases[0] -= learning_rate * error;
            }
        }
    }
};

// Differentiable compression with rate-distortion optimization
template<typename T = double>
class differentiable_compressor {
public:
    using value_type = T;
    using autodiff_type = stepanov::dual<T>;

private:
    struct rate_distortion_optimizer {
        T lambda;  // Lagrange multiplier for rate-distortion trade-off
        neural_layer<autodiff_type> encoder;
        neural_layer<autodiff_type> decoder;
        neural_layer<autodiff_type> quantizer;

        rate_distortion_optimizer(size_t input_dim, size_t code_dim, T lambda_param = 0.01)
            : lambda(lambda_param)
            , encoder(input_dim, code_dim)
            , decoder(code_dim, input_dim)
            , quantizer(code_dim, code_dim,
                       [](autodiff_type x) {
                           // Differentiable quantization using tanh
                           return autodiff_type(std::round(x.real())) +
                                  (std::tanh(x) - autodiff_type(std::round(x.real())));
                       }) {}

        std::vector<autodiff_type> forward(const std::vector<autodiff_type>& input) {
            auto encoded = encoder.forward(input);
            auto quantized = quantizer.forward(encoded);
            auto decoded = decoder.forward(quantized);
            return decoded;
        }

        autodiff_type loss(const std::vector<autodiff_type>& input,
                          const std::vector<autodiff_type>& output) {
            // Distortion: MSE
            autodiff_type distortion(0);
            for (size_t i = 0; i < input.size(); ++i) {
                auto diff = input[i] - output[i];
                distortion = distortion + diff * diff;
            }

            // Rate: Entropy of quantized codes
            auto encoded = encoder.forward(input);
            auto quantized = quantizer.forward(encoded);
            autodiff_type rate(0);
            for (const auto& q : quantized) {
                // Approximate entropy using sigmoid
                auto p = autodiff_type(1) / (autodiff_type(1) + stepanov::exp(-q));
                rate = rate - p * stepanov::log(p + autodiff_type(1e-10))
                           - (autodiff_type(1) - p) * stepanov::log(autodiff_type(1) - p + autodiff_type(1e-10));
            }

            return distortion + autodiff_type(lambda) * rate;
        }
    };

    rate_distortion_optimizer optimizer;

public:
    differentiable_compressor(size_t input_dim, size_t code_dim, T lambda = 0.01)
        : optimizer(input_dim, code_dim, lambda) {}

    // Compress with gradient-based optimization
    std::vector<T> compress(const std::vector<T>& data) {
        std::vector<autodiff_type> input;
        for (const auto& x : data) {
            input.emplace_back(x, 0);  // Create dual number
        }

        // Forward pass
        auto encoded = optimizer.encoder.forward(input);
        auto quantized = optimizer.quantizer.forward(encoded);

        // Extract real parts
        std::vector<T> compressed;
        for (const auto& q : quantized) {
            compressed.push_back(q.real());
        }

        return compressed;
    }

    // Decompress
    std::vector<T> decompress(const std::vector<T>& compressed) {
        std::vector<autodiff_type> input;
        for (const auto& x : compressed) {
            input.emplace_back(x, 0);
        }

        auto decoded = optimizer.decoder.forward(input);

        std::vector<T> decompressed;
        for (const auto& d : decoded) {
            decompressed.push_back(d.real());
        }

        return decompressed;
    }

    // Train compressor on data
    void train(const std::vector<std::vector<T>>& training_data,
              size_t epochs = 100,
              T learning_rate = 0.001) {

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            T total_loss = 0;

            for (const auto& sample : training_data) {
                // Convert to autodiff type
                std::vector<autodiff_type> input;
                for (size_t i = 0; i < sample.size(); ++i) {
                    input.emplace_back(sample[i], i == 0 ? 1.0 : 0.0);  // Seed gradient
                }

                // Forward pass
                auto output = optimizer.forward(input);

                // Compute loss
                auto loss = optimizer.loss(input, output);
                total_loss += loss.real();

                // Backward pass (gradients in dual part)
                // Update weights using gradients
                updateWeights(optimizer.encoder, loss.dual(), learning_rate);
                updateWeights(optimizer.decoder, loss.dual(), learning_rate);
            }
        }
    }

private:
    void updateWeights(neural_layer<autodiff_type>& layer,
                      T gradient,
                      T learning_rate) {
        // Simplified weight update
        for (auto& row : layer.weights) {
            for (auto& w : row) {
                w = w - autodiff_type(learning_rate * gradient, 0);
            }
        }
        for (auto& b : layer.biases) {
            b = b - autodiff_type(learning_rate * gradient, 0);
        }
    }
};

// Transformer-based context model for compression
template<typename T = double>
class transformer_compressor {
public:
    using value_type = T;

private:
    struct attention_head {
        size_t d_model;
        size_t d_key;
        neural_layer<T> query;
        neural_layer<T> key;
        neural_layer<T> value;

        attention_head(size_t model_dim, size_t key_dim)
            : d_model(model_dim)
            , d_key(key_dim)
            , query(model_dim, key_dim, [](T x) { return x; })
            , key(model_dim, key_dim, [](T x) { return x; })
            , value(model_dim, key_dim, [](T x) { return x; }) {}

        std::vector<T> forward(const std::vector<std::vector<T>>& input) {
            size_t seq_len = input.size();
            std::vector<std::vector<T>> Q, K, V;

            // Compute Q, K, V
            for (const auto& x : input) {
                Q.push_back(query.forward(x));
                K.push_back(key.forward(x));
                V.push_back(value.forward(x));
            }

            // Scaled dot-product attention
            std::vector<T> output(d_key, T(0));
            T scale = T(1) / std::sqrt(static_cast<T>(d_key));

            for (size_t i = 0; i < seq_len; ++i) {
                std::vector<T> scores(seq_len);
                T max_score = std::numeric_limits<T>::lowest();

                // Compute attention scores
                for (size_t j = 0; j <= i; ++j) {  // Causal mask
                    T score = 0;
                    for (size_t k = 0; k < d_key; ++k) {
                        score += Q[i][k] * K[j][k];
                    }
                    scores[j] = score * scale;
                    max_score = std::max(max_score, scores[j]);
                }

                // Softmax
                T sum_exp = 0;
                for (size_t j = 0; j <= i; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                for (size_t j = 0; j <= i; ++j) {
                    scores[j] /= sum_exp;
                }

                // Weighted sum of values
                for (size_t j = 0; j <= i; ++j) {
                    for (size_t k = 0; k < d_key; ++k) {
                        output[k] += scores[j] * V[j][k];
                    }
                }
            }

            return output;
        }
    };

    struct transformer_block {
        std::vector<attention_head> heads;
        neural_layer<T> output_projection;
        neural_layer<T> ffn1;
        neural_layer<T> ffn2;

        transformer_block(size_t d_model, size_t num_heads, size_t d_ff)
            : output_projection(d_model, d_model, [](T x) { return x; })
            , ffn1(d_model, d_ff)
            , ffn2(d_ff, d_model, [](T x) { return x; }) {

            size_t d_key = d_model / num_heads;
            for (size_t i = 0; i < num_heads; ++i) {
                heads.emplace_back(d_model, d_key);
            }
        }

        std::vector<T> forward(const std::vector<std::vector<T>>& input) {
            // Multi-head attention
            std::vector<T> concat_heads;
            for (auto& head : heads) {
                auto head_output = head.forward(input);
                concat_heads.insert(concat_heads.end(),
                                  head_output.begin(), head_output.end());
            }

            // Output projection
            auto attended = output_projection.forward(concat_heads);

            // Add & Norm
            std::vector<T> normed1 = layer_norm(add_residual(input.back(), attended));

            // Feed-forward
            auto ff_out = ffn2.forward(ffn1.forward(normed1));

            // Add & Norm
            return layer_norm(add_residual(normed1, ff_out));
        }

    private:
        std::vector<T> add_residual(const std::vector<T>& x, const std::vector<T>& y) {
            std::vector<T> result(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                result[i] = x[i] + y[i];
            }
            return result;
        }

        std::vector<T> layer_norm(const std::vector<T>& x) {
            T mean = std::accumulate(x.begin(), x.end(), T(0)) / x.size();
            T variance = 0;
            for (const auto& xi : x) {
                T diff = xi - mean;
                variance += diff * diff;
            }
            variance /= x.size();
            T std_dev = std::sqrt(variance + T(1e-6));

            std::vector<T> normalized(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                normalized[i] = (x[i] - mean) / std_dev;
            }
            return normalized;
        }
    };

    std::vector<transformer_block> blocks;
    size_t d_model;
    size_t vocab_size;

public:
    transformer_compressor(size_t model_dim = 512,
                         size_t num_blocks = 6,
                         size_t num_heads = 8,
                         size_t vocab = 256)
        : d_model(model_dim)
        , vocab_size(vocab) {

        for (size_t i = 0; i < num_blocks; ++i) {
            blocks.emplace_back(model_dim, num_heads, model_dim * 4);
        }
    }

    // Predict next token probabilities
    std::vector<T> predict_next(const std::vector<size_t>& context) {
        // Embed tokens
        std::vector<std::vector<T>> embeddings;
        for (size_t token : context) {
            std::vector<T> embedding(d_model, T(0));
            embedding[token % d_model] = T(1);  // Simple embedding
            embeddings.push_back(embedding);
        }

        // Forward through transformer blocks
        std::vector<T> output = embeddings.back();
        for (auto& block : blocks) {
            output = block.forward(embeddings);
        }

        // Project to vocabulary
        std::vector<T> logits(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            logits[i] = output[i % d_model];  // Simple projection
        }

        // Softmax
        T max_logit = *std::max_element(logits.begin(), logits.end());
        std::vector<T> probs(vocab_size);
        T sum = 0;
        for (size_t i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(logits[i] - max_logit);
            sum += probs[i];
        }
        for (auto& p : probs) {
            p /= sum;
        }

        return probs;
    }

    // Compress using transformer predictions
    std::vector<bool> compress(const std::vector<size_t>& data) {
        std::vector<bool> compressed;
        std::vector<size_t> context;

        for (size_t token : data) {
            auto probs = predict_next(context);

            // Arithmetic coding with predicted probabilities
            T low = 0, high = 1;
            T cum_prob = 0;
            for (size_t i = 0; i < token; ++i) {
                cum_prob += probs[i];
            }

            T range = high - low;
            high = low + range * (cum_prob + probs[token]);
            low = low + range * cum_prob;

            // Output bits
            while (high < 0.5 || low >= 0.5) {
                if (high < 0.5) {
                    compressed.push_back(false);
                    low *= 2;
                    high *= 2;
                } else {
                    compressed.push_back(true);
                    low = 2 * (low - 0.5);
                    high = 2 * (high - 0.5);
                }
            }

            context.push_back(token);
        }

        return compressed;
    }
};

// Progressive neural compression for scalable quality
template<typename T = double>
class progressive_neural_compressor {
    struct quality_layer {
        neural_layer<T> encoder;
        neural_layer<T> decoder;
        T quality_level;

        quality_layer(size_t input_dim, size_t code_dim, T quality)
            : encoder(input_dim, code_dim)
            , decoder(code_dim, input_dim)
            , quality_level(quality) {}
    };

    std::vector<quality_layer> layers;

public:
    progressive_neural_compressor(size_t input_dim,
                                 std::vector<std::pair<size_t, T>> layer_specs) {
        for (const auto& [code_dim, quality] : layer_specs) {
            layers.emplace_back(input_dim, code_dim, quality);
        }
    }

    // Compress at specified quality level
    std::vector<T> compress(const std::vector<T>& data, T target_quality) {
        std::vector<T> compressed;
        std::vector<T> residual = data;

        for (const auto& layer : layers) {
            if (layer.quality_level > target_quality) break;

            auto encoded = layer.encoder.forward(residual);
            compressed.insert(compressed.end(), encoded.begin(), encoded.end());

            auto reconstructed = layer.decoder.forward(encoded);
            for (size_t i = 0; i < residual.size(); ++i) {
                residual[i] -= reconstructed[i];
            }
        }

        return compressed;
    }

    // Decompress progressively
    std::vector<T> decompress(const std::vector<T>& compressed,
                             size_t num_layers = std::numeric_limits<size_t>::max()) {
        std::vector<T> reconstructed;
        size_t offset = 0;

        for (size_t i = 0; i < std::min(num_layers, layers.size()); ++i) {
            const auto& layer = layers[i];
            size_t code_dim = layer.encoder.biases.size();

            if (offset + code_dim > compressed.size()) break;

            std::vector<T> code(compressed.begin() + offset,
                               compressed.begin() + offset + code_dim);
            offset += code_dim;

            auto decoded = layer.decoder.forward(code);
            if (reconstructed.empty()) {
                reconstructed = decoded;
            } else {
                for (size_t j = 0; j < decoded.size(); ++j) {
                    reconstructed[j] += decoded[j];
                }
            }
        }

        return reconstructed;
    }
};

} // namespace stepanov::compression::neural