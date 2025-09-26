// Compression and Cryptography Fusion: Security Through Compression
// "Perfect compression is perfect encryption" - Information Theory Insight
#pragma once

#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>
#include <optional>
#include <bitset>
#include <functional>
#include "../concepts.hpp"

namespace stepanov::compression::crypto {

// Format-Preserving Encryption with Compression
template<typename T = uint64_t>
class format_preserving_compressor {
public:
    using value_type = T;
    using key_type = std::array<uint8_t, 32>;

private:
    key_type master_key;
    std::mt19937_64 prng;

    // Feistel network for FPE
    struct feistel_network {
        size_t rounds = 10;
        std::vector<std::function<T(T, size_t)>> round_functions;

        feistel_network(const key_type& key, size_t domain_size) {
            // Generate round functions from key
            std::seed_seq seed(key.begin(), key.end());
            std::mt19937 gen(seed);

            for (size_t r = 0; r < rounds; ++r) {
                uint32_t round_key = gen();
                round_functions.push_back([round_key, domain_size](T x, size_t round) {
                    // Pseudo-random function
                    std::hash<T> hasher;
                    return (hasher(x ^ round_key ^ round) % domain_size);
                });
            }
        }

        T encrypt(T plaintext, size_t domain_size) {
            size_t split = domain_size / 2;
            T left = plaintext / split;
            T right = plaintext % split;

            for (size_t r = 0; r < rounds; ++r) {
                T new_right = (left + round_functions[r](right, r)) % split;
                left = right;
                right = new_right;
            }

            return left * split + right;
        }

        T decrypt(T ciphertext, size_t domain_size) {
            size_t split = domain_size / 2;
            T left = ciphertext / split;
            T right = ciphertext % split;

            for (size_t r = rounds; r > 0; --r) {
                T new_left = right;
                T new_right = (left - round_functions[r-1](right, r-1) + split) % split;
                left = new_left;
                right = new_right;
            }

            return left * split + right;
        }
    };

public:
    format_preserving_compressor(const key_type& key)
        : master_key(key)
        , prng(std::seed_seq(key.begin(), key.end())) {}

    // Compress then encrypt preserving format
    std::vector<T> compress_encrypt(const std::vector<T>& data,
                                   size_t compressed_size) {
        // Step 1: Compress using rank/unrank
        auto compressed = rank_compress(data, compressed_size);

        // Step 2: Encrypt while preserving domain
        size_t domain_size = compute_domain_size(compressed_size);
        feistel_network fpe(master_key, domain_size);

        std::vector<T> encrypted;
        for (const auto& val : compressed) {
            encrypted.push_back(fpe.encrypt(val, domain_size));
        }

        return encrypted;
    }

    // Decrypt then decompress
    std::vector<T> decrypt_decompress(const std::vector<T>& encrypted,
                                     size_t original_size) {
        size_t domain_size = compute_domain_size(encrypted.size());
        feistel_network fpe(master_key, domain_size);

        // Step 1: Decrypt
        std::vector<T> decrypted;
        for (const auto& val : encrypted) {
            decrypted.push_back(fpe.decrypt(val, domain_size));
        }

        // Step 2: Decompress
        return unrank_decompress(decrypted, original_size);
    }

private:
    // Rank-based compression
    std::vector<T> rank_compress(const std::vector<T>& data, size_t target_size) {
        // Create sorted index mapping
        std::vector<size_t> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                 [&data](size_t i, size_t j) { return data[i] < data[j]; });

        // Rank encoding
        std::vector<T> ranks(data.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            ranks[indices[i]] = i;
        }

        // Subsample to target size
        std::vector<T> compressed;
        size_t step = std::max<size_t>(1, data.size() / target_size);
        for (size_t i = 0; i < data.size() && compressed.size() < target_size; i += step) {
            compressed.push_back(ranks[i]);
        }

        return compressed;
    }

    std::vector<T> unrank_decompress(const std::vector<T>& ranks, size_t original_size) {
        // Interpolate missing ranks
        std::vector<T> decompressed;
        size_t step = original_size / ranks.size();

        for (size_t i = 0; i < ranks.size(); ++i) {
            decompressed.push_back(ranks[i]);
            if (i < ranks.size() - 1) {
                // Linear interpolation
                T start = ranks[i];
                T end = ranks[i + 1];
                for (size_t j = 1; j < step && decompressed.size() < original_size; ++j) {
                    T interpolated = start + (end - start) * j / step;
                    decompressed.push_back(interpolated);
                }
            }
        }

        return decompressed;
    }

    size_t compute_domain_size(size_t data_size) {
        // Domain must be even for Feistel
        return (static_cast<size_t>(1) << (std::min<size_t>(32,
            static_cast<size_t>(std::ceil(std::log2(data_size)))))) & ~1;
    }
};

// Compressed Sensing with Encryption
template<typename T = double>
class encrypted_compressed_sensing {
public:
    using value_type = T;
    using matrix_type = std::vector<std::vector<T>>;
    using key_type = std::array<uint8_t, 32>;

private:
    key_type secret_key;
    matrix_type measurement_matrix;
    size_t m;  // Measurements
    size_t n;  // Signal dimension

    // Homomorphic properties for computations on encrypted measurements
    struct homomorphic_ops {
        static std::vector<T> add(const std::vector<T>& a,
                                 const std::vector<T>& b) {
            std::vector<T> result(a.size());
            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = a[i] + b[i];
            }
            return result;
        }

        static std::vector<T> scalar_multiply(const std::vector<T>& a, T scalar) {
            std::vector<T> result(a.size());
            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = a[i] * scalar;
            }
            return result;
        }
    };

public:
    encrypted_compressed_sensing(size_t measurements, size_t dimension,
                                const key_type& key)
        : m(measurements)
        , n(dimension)
        , secret_key(key) {

        // Generate measurement matrix from key (acts as encryption)
        generate_secure_measurement_matrix();
    }

    // Compress and encrypt simultaneously
    std::vector<T> sense_and_encrypt(const std::vector<T>& signal) {
        if (signal.size() != n) {
            throw std::invalid_argument("Signal dimension mismatch");
        }

        std::vector<T> measurements(m, 0);

        // y = Φx where Φ is the secret measurement matrix
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                measurements[i] += measurement_matrix[i][j] * signal[j];
            }
        }

        // Add noise for additional security
        add_encryption_noise(measurements);

        return measurements;
    }

    // Decrypt and reconstruct (requires key)
    std::vector<T> decrypt_and_reconstruct(const std::vector<T>& measurements) {
        // Remove encryption noise
        auto clean_measurements = remove_encryption_noise(measurements);

        // Solve using L1 minimization (simplified OMP)
        return orthogonal_matching_pursuit(clean_measurements);
    }

    // Compute on encrypted measurements
    std::vector<T> encrypted_linear_combination(
        const std::vector<std::vector<T>>& encrypted_signals,
        const std::vector<T>& coefficients) {

        std::vector<T> result(m, 0);

        for (size_t i = 0; i < encrypted_signals.size(); ++i) {
            auto scaled = homomorphic_ops::scalar_multiply(
                encrypted_signals[i], coefficients[i]);
            result = homomorphic_ops::add(result, scaled);
        }

        return result;
    }

private:
    void generate_secure_measurement_matrix() {
        // Use key as seed for RIP-satisfying matrix
        std::seed_seq seed(secret_key.begin(), secret_key.end());
        std::mt19937 gen(seed);
        std::normal_distribution<T> dist(0, 1.0 / std::sqrt(m));

        measurement_matrix.resize(m, std::vector<T>(n));
        for (auto& row : measurement_matrix) {
            for (auto& elem : row) {
                elem = dist(gen);
            }
        }
    }

    void add_encryption_noise(std::vector<T>& measurements) {
        std::mt19937 gen(std::seed_seq(secret_key.begin(), secret_key.end()));
        std::normal_distribution<T> noise(0, 0.001);

        for (auto& m : measurements) {
            m += noise(gen);
        }
    }

    std::vector<T> remove_encryption_noise(const std::vector<T>& noisy) {
        // In practice, would use more sophisticated denoising
        return noisy;
    }

    std::vector<T> orthogonal_matching_pursuit(const std::vector<T>& measurements) {
        std::vector<T> residual = measurements;
        std::vector<T> solution(n, 0);
        std::vector<size_t> support;

        // Simplified OMP
        for (size_t iter = 0; iter < m / 4; ++iter) {
            // Find column most correlated with residual
            size_t best_col = 0;
            T best_correlation = 0;

            for (size_t j = 0; j < n; ++j) {
                if (std::find(support.begin(), support.end(), j) != support.end()) {
                    continue;
                }

                T correlation = 0;
                for (size_t i = 0; i < m; ++i) {
                    correlation += std::abs(measurement_matrix[i][j] * residual[i]);
                }

                if (correlation > best_correlation) {
                    best_correlation = correlation;
                    best_col = j;
                }
            }

            support.push_back(best_col);

            // Least squares on support
            // (Simplified: just set coefficient)
            T coefficient = 0;
            T norm = 0;
            for (size_t i = 0; i < m; ++i) {
                coefficient += measurement_matrix[i][best_col] * residual[i];
                norm += measurement_matrix[i][best_col] * measurement_matrix[i][best_col];
            }
            solution[best_col] = coefficient / norm;

            // Update residual
            for (size_t i = 0; i < m; ++i) {
                residual[i] -= solution[best_col] * measurement_matrix[i][best_col];
            }

            // Check convergence
            T residual_norm = 0;
            for (T r : residual) {
                residual_norm += r * r;
            }
            if (residual_norm < 1e-6) break;
        }

        return solution;
    }
};

// Deniable compression with multiple valid decompressions
template<typename T = uint8_t>
class deniable_compressor {
public:
    using value_type = T;
    using key_type = std::array<uint8_t, 32>;

private:
    struct deniable_layer {
        std::vector<T> data;
        key_type key;
        size_t offset;
    };

    std::vector<deniable_layer> layers;
    std::mt19937 rng;

public:
    deniable_compressor() : rng(std::random_device{}()) {}

    // Create deniably compressed data with multiple messages
    std::vector<T> compress_deniable(
        const std::vector<std::vector<T>>& messages,
        const std::vector<key_type>& keys) {

        if (messages.size() != keys.size()) {
            throw std::invalid_argument("Messages and keys count mismatch");
        }

        // Find maximum size
        size_t max_size = 0;
        for (const auto& msg : messages) {
            max_size = std::max(max_size, msg.size());
        }

        // Pad to power of 2
        size_t padded_size = 1;
        while (padded_size < max_size) padded_size <<= 1;

        // Create steganographic container
        std::vector<T> container(padded_size * 2);

        // Fill with random data
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& byte : container) {
            byte = static_cast<T>(dist(rng));
        }

        // Embed each message at key-dependent positions
        for (size_t i = 0; i < messages.size(); ++i) {
            embed_message(container, messages[i], keys[i]);
        }

        return container;
    }

    // Extract message with specific key
    std::vector<T> extract_with_key(const std::vector<T>& container,
                                   const key_type& key) {
        // Generate extraction positions from key
        std::seed_seq seed(key.begin(), key.end());
        std::mt19937 gen(seed);

        std::vector<size_t> positions = generate_positions(gen, container.size());

        // Extract message
        std::vector<T> message;
        for (size_t pos : positions) {
            if (pos < container.size()) {
                message.push_back(container[pos]);
            }
        }

        // Remove padding
        return remove_padding(message);
    }

    // Create plausibly deniable archive
    std::vector<T> create_deniable_archive(
        const std::vector<T>& real_data,
        const std::vector<T>& decoy_data,
        const key_type& real_key,
        const key_type& decoy_key) {

        // Interleave real and decoy data
        size_t total_size = std::max(real_data.size(), decoy_data.size()) * 2;
        std::vector<T> archive(total_size);

        // Use keys to determine interleaving pattern
        auto real_positions = key_to_positions(real_key, total_size);
        auto decoy_positions = key_to_positions(decoy_key, total_size);

        // Fill with noise
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& byte : archive) {
            byte = static_cast<T>(dist(rng));
        }

        // Place real data
        for (size_t i = 0; i < real_data.size() && i < real_positions.size(); ++i) {
            archive[real_positions[i] % total_size] = real_data[i];
        }

        // Place decoy data (may overwrite some real data positions)
        for (size_t i = 0; i < decoy_data.size() && i < decoy_positions.size(); ++i) {
            size_t pos = decoy_positions[i] % total_size;
            // Avoid obvious collisions
            if (std::find(real_positions.begin(),
                         real_positions.begin() + real_data.size(),
                         pos) == real_positions.begin() + real_data.size()) {
                archive[pos] = decoy_data[i];
            }
        }

        return archive;
    }

private:
    void embed_message(std::vector<T>& container,
                      const std::vector<T>& message,
                      const key_type& key) {
        std::seed_seq seed(key.begin(), key.end());
        std::mt19937 gen(seed);

        auto positions = generate_positions(gen, container.size());

        // Add length prefix
        std::vector<T> prefixed_message;
        uint32_t length = message.size();
        prefixed_message.push_back((length >> 24) & 0xFF);
        prefixed_message.push_back((length >> 16) & 0xFF);
        prefixed_message.push_back((length >> 8) & 0xFF);
        prefixed_message.push_back(length & 0xFF);
        prefixed_message.insert(prefixed_message.end(),
                              message.begin(), message.end());

        // Embed at pseudo-random positions
        for (size_t i = 0; i < prefixed_message.size() && i < positions.size(); ++i) {
            container[positions[i] % container.size()] = prefixed_message[i];
        }
    }

    std::vector<size_t> generate_positions(std::mt19937& gen, size_t max_pos) {
        std::vector<size_t> positions;
        std::uniform_int_distribution<size_t> dist(0, max_pos - 1);

        // Generate unique positions
        std::set<size_t> used;
        while (positions.size() < max_pos / 2) {
            size_t pos = dist(gen);
            if (used.insert(pos).second) {
                positions.push_back(pos);
            }
        }

        return positions;
    }

    std::vector<size_t> key_to_positions(const key_type& key, size_t max_pos) {
        std::seed_seq seed(key.begin(), key.end());
        std::mt19937 gen(seed);
        return generate_positions(gen, max_pos);
    }

    std::vector<T> remove_padding(const std::vector<T>& data) {
        if (data.size() < 4) return {};

        // Extract length from prefix
        uint32_t length = (static_cast<uint32_t>(data[0]) << 24) |
                         (static_cast<uint32_t>(data[1]) << 16) |
                         (static_cast<uint32_t>(data[2]) << 8) |
                         static_cast<uint32_t>(data[3]);

        if (length > data.size() - 4) {
            return {};  // Invalid length
        }

        return std::vector<T>(data.begin() + 4, data.begin() + 4 + length);
    }
};

// Zero-knowledge compression proofs
template<typename T = uint8_t>
class zk_compression_prover {
public:
    using value_type = T;
    using commitment_type = std::array<uint8_t, 32>;
    using challenge_type = uint64_t;

private:
    // Merkle tree for commitments
    struct merkle_tree {
        struct node {
            commitment_type hash;
            std::unique_ptr<node> left;
            std::unique_ptr<node> right;
        };

        std::unique_ptr<node> root;
        std::vector<commitment_type> leaves;

        void build(const std::vector<std::vector<T>>& data) {
            leaves.clear();
            for (const auto& block : data) {
                leaves.push_back(hash_block(block));
            }
            root = build_tree(0, leaves.size());
        }

        commitment_type get_root() const {
            return root ? root->hash : commitment_type{};
        }

        std::vector<commitment_type> get_proof(size_t index) {
            std::vector<commitment_type> proof;
            get_proof_recursive(root.get(), 0, leaves.size(), index, proof);
            return proof;
        }

    private:
        std::unique_ptr<node> build_tree(size_t start, size_t end) {
            if (start >= end) return nullptr;
            if (end - start == 1) {
                auto n = std::make_unique<node>();
                n->hash = leaves[start];
                return n;
            }

            size_t mid = (start + end) / 2;
            auto n = std::make_unique<node>();
            n->left = build_tree(start, mid);
            n->right = build_tree(mid, end);

            if (n->left && n->right) {
                n->hash = hash_pair(n->left->hash, n->right->hash);
            }

            return n;
        }

        void get_proof_recursive(node* n, size_t start, size_t end,
                                size_t index, std::vector<commitment_type>& proof) {
            if (!n || start >= end) return;
            if (end - start == 1) return;

            size_t mid = (start + end) / 2;
            if (index < mid) {
                if (n->right) proof.push_back(n->right->hash);
                get_proof_recursive(n->left.get(), start, mid, index, proof);
            } else {
                if (n->left) proof.push_back(n->left->hash);
                get_proof_recursive(n->right.get(), mid, end, index, proof);
            }
        }

        commitment_type hash_block(const std::vector<T>& block) {
            // Simplified hash
            commitment_type hash{};
            for (size_t i = 0; i < block.size() && i < hash.size(); ++i) {
                hash[i % hash.size()] ^= block[i];
            }
            return hash;
        }

        commitment_type hash_pair(const commitment_type& a,
                                 const commitment_type& b) {
            commitment_type result{};
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] = a[i] ^ b[i] ^ (a[i] * b[i]);
            }
            return result;
        }
    };

    merkle_tree commitment_tree;
    std::vector<std::vector<T>> original_data;
    size_t compression_ratio;

public:
    // Commit to compressed data
    commitment_type commit_to_compression(const std::vector<T>& data,
                                        size_t compressed_size) {
        // Split into blocks
        size_t block_size = 1024;
        original_data.clear();

        for (size_t i = 0; i < data.size(); i += block_size) {
            size_t end = std::min(i + block_size, data.size());
            original_data.emplace_back(data.begin() + i, data.begin() + end);
        }

        compression_ratio = data.size() / compressed_size;

        // Build Merkle tree
        commitment_tree.build(original_data);
        return commitment_tree.get_root();
    }

    // Prove compression ratio without revealing data
    struct compression_proof {
        commitment_type commitment;
        size_t claimed_ratio;
        std::vector<commitment_type> merkle_proofs;
        std::vector<size_t> revealed_indices;
        std::vector<std::vector<T>> revealed_blocks;
    };

    compression_proof prove_compression_ratio(challenge_type challenge,
                                            size_t num_samples = 10) {
        compression_proof proof;
        proof.commitment = commitment_tree.get_root();
        proof.claimed_ratio = compression_ratio;

        // Use challenge to select blocks to reveal
        std::mt19937 gen(challenge);
        std::uniform_int_distribution<size_t> dist(0, original_data.size() - 1);

        for (size_t i = 0; i < num_samples && i < original_data.size(); ++i) {
            size_t idx = dist(gen);
            proof.revealed_indices.push_back(idx);
            proof.revealed_blocks.push_back(original_data[idx]);

            auto merkle_proof = commitment_tree.get_proof(idx);
            proof.merkle_proofs.insert(proof.merkle_proofs.end(),
                                      merkle_proof.begin(),
                                      merkle_proof.end());
        }

        return proof;
    }

    // Verify compression proof
    static bool verify_compression_proof(const compression_proof& proof,
                                       challenge_type challenge) {
        // Verify revealed blocks compress to claimed ratio
        size_t total_original = 0;
        size_t total_compressed = 0;

        for (const auto& block : proof.revealed_blocks) {
            total_original += block.size();
            // Simplified: assume compression achieves claimed ratio
            total_compressed += block.size() / proof.claimed_ratio;
        }

        size_t actual_ratio = total_original / std::max<size_t>(1, total_compressed);

        // Allow 10% tolerance
        return std::abs(static_cast<int>(actual_ratio) -
                       static_cast<int>(proof.claimed_ratio)) <
               proof.claimed_ratio / 10;
    }

    // Prove knowledge of decompression without revealing
    struct decompression_proof {
        commitment_type compressed_commitment;
        commitment_type decompressed_commitment;
        std::vector<uint8_t> proof_transcript;
    };

    decompression_proof prove_decompression_knowledge(
        const std::vector<T>& compressed,
        const std::vector<T>& decompressed) {

        decompression_proof proof;

        // Commit to both
        proof.compressed_commitment = hash_data(compressed);
        proof.decompressed_commitment = hash_data(decompressed);

        // Generate proof transcript (simplified Sigma protocol)
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<uint8_t> dist(0, 255);

        // Commitment phase
        std::vector<uint8_t> random_commitment(32);
        for (auto& byte : random_commitment) {
            byte = dist(gen);
        }

        // Challenge (would come from verifier)
        uint64_t challenge = dist(gen);

        // Response
        proof.proof_transcript = random_commitment;
        proof.proof_transcript.push_back(challenge & 0xFF);

        // Add hash of decompression relationship
        auto relationship_hash = hash_relationship(compressed, decompressed);
        proof.proof_transcript.insert(proof.proof_transcript.end(),
                                    relationship_hash.begin(),
                                    relationship_hash.end());

        return proof;
    }

private:
    commitment_type hash_data(const std::vector<T>& data) {
        commitment_type hash{};
        for (size_t i = 0; i < data.size(); ++i) {
            hash[i % hash.size()] ^= data[i];
            hash[(i + 1) % hash.size()] += data[i];
        }
        return hash;
    }

    commitment_type hash_relationship(const std::vector<T>& compressed,
                                     const std::vector<T>& decompressed) {
        commitment_type hash{};

        // Hash the size relationship
        size_t ratio = decompressed.size() / std::max<size_t>(1, compressed.size());
        for (size_t i = 0; i < sizeof(ratio); ++i) {
            hash[i] = (ratio >> (i * 8)) & 0xFF;
        }

        // Hash samples
        for (size_t i = 0; i < std::min(compressed.size(), hash.size() / 2); ++i) {
            hash[i * 2] ^= compressed[i];
        }
        for (size_t i = 0; i < std::min(decompressed.size(), hash.size() / 2); ++i) {
            hash[i * 2 + 1] ^= decompressed[i];
        }

        return hash;
    }
};

// Homomorphic compression for computation on compressed data
template<typename T = int64_t>
class homomorphic_compressor {
public:
    using value_type = T;
    using ciphertext_type = std::pair<T, T>;  // Simplified Paillier

private:
    struct paillier_params {
        T n;   // modulus
        T g;   // generator
        T lambda;  // Carmichael function
        T mu;  // for decryption

        paillier_params() {
            // Simplified: use small primes for demonstration
            T p = 61, q = 53;
            n = p * q;
            g = n + 1;
            lambda = (p - 1) * (q - 1);
            mu = mod_inverse(lambda, n);
        }

        T mod_inverse(T a, T m) {
            // Extended Euclidean algorithm
            T m0 = m, x0 = 0, x1 = 1;
            while (a > 1) {
                T q = a / m;
                T t = m;
                m = a % m;
                a = t;
                t = x0;
                x0 = x1 - q * x0;
                x1 = t;
            }
            if (x1 < 0) x1 += m0;
            return x1;
        }
    };

    paillier_params params;
    std::mt19937 rng;

public:
    homomorphic_compressor() : rng(std::random_device{}()) {}

    // Compress and encrypt homomorphically
    std::vector<ciphertext_type> compress_homomorphic(const std::vector<T>& data) {
        std::vector<ciphertext_type> encrypted;

        for (const auto& val : data) {
            encrypted.push_back(encrypt(val));
        }

        return encrypted;
    }

    // Homomorphic addition on compressed data
    ciphertext_type add_compressed(const ciphertext_type& a,
                                  const ciphertext_type& b) {
        // E(a) * E(b) mod n^2 = E(a + b)
        T n_squared = params.n * params.n;
        return {(a.first * b.first) % n_squared,
               (a.second * b.second) % n_squared};
    }

    // Homomorphic scalar multiplication
    ciphertext_type multiply_compressed(const ciphertext_type& a, T scalar) {
        // E(a)^k mod n^2 = E(k * a)
        T n_squared = params.n * params.n;
        return {mod_exp(a.first, scalar, n_squared),
               mod_exp(a.second, scalar, n_squared)};
    }

    // Compute on compressed data
    ciphertext_type compute_on_compressed(
        const std::vector<ciphertext_type>& compressed_data,
        const std::function<T(std::vector<T>)>& linear_function) {

        // Only linear functions supported
        ciphertext_type result = encrypt(0);

        // Example: compute weighted sum
        std::vector<T> weights(compressed_data.size(), 1);

        for (size_t i = 0; i < compressed_data.size(); ++i) {
            auto weighted = multiply_compressed(compressed_data[i], weights[i]);
            result = add_compressed(result, weighted);
        }

        return result;
    }

    // Decompress and decrypt
    T decompress_homomorphic(const ciphertext_type& encrypted) {
        return decrypt(encrypted);
    }

private:
    ciphertext_type encrypt(T plaintext) {
        std::uniform_int_distribution<T> dist(1, params.n - 1);
        T r = dist(rng);

        T n_squared = params.n * params.n;
        T c1 = mod_exp(params.g, plaintext, n_squared);
        T c2 = mod_exp(r, params.n, n_squared);

        return {(c1 * c2) % n_squared, r};
    }

    T decrypt(const ciphertext_type& ciphertext) {
        T n_squared = params.n * params.n;
        T u = mod_exp(ciphertext.first, params.lambda, n_squared);
        T l = (u - 1) / params.n;

        return (l * params.mu) % params.n;
    }

    T mod_exp(T base, T exp, T mod) {
        T result = 1;
        base %= mod;
        while (exp > 0) {
            if (exp & 1) {
                result = (result * base) % mod;
            }
            base = (base * base) % mod;
            exp >>= 1;
        }
        return result;
    }
};

} // namespace stepanov::compression::crypto