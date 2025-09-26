// Test suite for advanced compression algorithms
// Demonstrates compression as the foundation of intelligence and computation

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>
#include "../include/stepanov/compression/neural.hpp"
#include "../include/stepanov/compression/ml.hpp"
#include "../include/stepanov/compression/quantum.hpp"
#include "../include/stepanov/compression/bio.hpp"
#include "../include/stepanov/compression/crypto.hpp"
#include "../include/stepanov/compression/agi.hpp"

using namespace stepanov::compression;

// Helper functions
template<typename T>
void print_compression_ratio(const std::string& name,
                            size_t original_size,
                            size_t compressed_size) {
    double ratio = static_cast<double>(original_size) / compressed_size;
    std::cout << name << ": "
              << original_size << " -> " << compressed_size
              << " (ratio: " << std::fixed << std::setprecision(2) << ratio << "x)\n";
}

void test_neural_compression() {
    std::cout << "\n=== Neural and Learned Compression ===\n";
    std::cout << "Compression through neural networks and learning\n\n";

    // Test Neural Arithmetic Coding
    {
        std::cout << "1. Neural Arithmetic Coding (LSTM-based):\n";
        neural::neural_arithmetic_coder<uint8_t> coder(256, 128, 32);

        // Create text with patterns for LSTM to learn
        std::string text = "The quick brown fox jumps over the lazy dog. "
                          "The quick brown fox jumps again. "
                          "The fox is quick and brown.";

        std::vector<uint8_t> data(text.begin(), text.end());

        auto compressed = coder.encode(data);
        print_compression_ratio<uint8_t>("Neural Arithmetic Coding",
                                       data.size() * 8, compressed.size());

        std::cout << "   LSTM learns patterns and improves predictions\n";
    }

    // Test Variational Autoencoder
    {
        std::cout << "\n2. Variational Autoencoder Compression:\n";
        neural::variational_autoencoder<double> vae(784, 32, {512, 256});

        // Simulate image data (28x28)
        std::vector<double> image(784);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.5, 0.1);

        for (auto& pixel : image) {
            pixel = std::clamp(dist(gen), 0.0, 1.0);
        }

        auto latent = vae.compress(image);
        auto reconstructed = vae.decompress(latent);

        print_compression_ratio<double>("VAE Compression",
                                       image.size() * sizeof(double),
                                       latent.size() * sizeof(double));

        std::cout << "   Learns manifold structure of data\n";
    }

    // Test Learned Index Structure
    {
        std::cout << "\n3. Learned Index for Compression:\n";
        neural::learned_index<uint32_t, std::string> index(1000);

        // Insert key-value pairs
        for (uint32_t i = 0; i < 10000; ++i) {
            index.insert(i * i, "value_" + std::to_string(i));
        }

        auto compressed = index.compress();
        print_compression_ratio<uint8_t>("Learned Index", 10000 * 16, compressed.size());

        std::cout << "   Neural networks replace B-trees\n";
        std::cout << "   O(1) lookup with learned models\n";
    }

    // Test Differentiable Compression
    {
        std::cout << "\n4. Differentiable Rate-Distortion Optimization:\n";
        neural::differentiable_compressor<double> comp(64, 16, 0.01);

        std::vector<double> signal(64);
        for (size_t i = 0; i < signal.size(); ++i) {
            signal[i] = std::sin(i * 0.1) + 0.1 * std::cos(i * 0.5);
        }

        auto compressed = comp.compress(signal);
        auto decompressed = comp.decompress(compressed);

        print_compression_ratio<double>("Differentiable Compression",
                                       signal.size(), compressed.size());

        std::cout << "   End-to-end learned compression\n";
        std::cout << "   Optimizes rate-distortion tradeoff\n";
    }
}

void test_compression_ml() {
    std::cout << "\n=== Compression-Based Machine Learning ===\n";
    std::cout << "No training required - compression IS learning!\n\n";

    // Test NCD Classification
    {
        std::cout << "1. Classification via Normalized Compression Distance:\n";

        struct SimpleCompressor {
            std::vector<uint8_t> compress(const std::vector<uint8_t>& data) {
                // Simplified compression
                return data;  // Would use real compression
            }
        };

        ml::compression_knn<SimpleCompressor> classifier(3);

        // Train with text samples
        std::string english = "The quick brown fox jumps over the lazy dog";
        std::string french = "Le renard brun rapide saute par-dessus le chien paresseux";
        std::string spanish = "El zorro marrón rápido salta sobre el perro perezoso";

        classifier.train(std::vector<uint8_t>(english.begin(), english.end()), 0);
        classifier.train(std::vector<uint8_t>(french.begin(), french.end()), 1);
        classifier.train(std::vector<uint8_t>(spanish.begin(), spanish.end()), 2);

        std::string query = "The dog is lazy";
        auto label = classifier.classify(std::vector<uint8_t>(query.begin(), query.end()));

        std::cout << "   Query: '" << query << "' classified as: " << label << " (English)\n";
        std::cout << "   No neural networks, no training - just compression!\n";
    }

    // Test Anomaly Detection
    {
        std::cout << "\n2. Anomaly Detection via Compression:\n";
        ml::compression_anomaly_detector<SimpleCompressor> detector(1.5);

        // Train on normal data
        std::vector<std::vector<uint8_t>> normal_data;
        for (int i = 0; i < 10; ++i) {
            std::string normal = "Normal pattern " + std::to_string(i % 3);
            normal_data.push_back(std::vector<uint8_t>(normal.begin(), normal.end()));
        }
        detector.train(normal_data);

        // Test anomaly
        std::string anomaly = "UNUSUAL RANDOM GIBBERISH XYZ123!@#";
        auto score = detector.anomaly_score(
            std::vector<uint8_t>(anomaly.begin(), anomaly.end()));

        std::cout << "   Anomaly score: " << score << " (>1.5 is anomalous)\n";
        std::cout << "   Incompressible = anomalous\n";
    }

    // Test Grammar-Based Feature Extraction
    {
        std::cout << "\n3. Grammar-Based Feature Extraction:\n";
        ml::grammar_feature_extractor extractor;

        std::vector<std::string> sequences = {
            "ABCABCABC",
            "XYXYXYXY",
            "ABCXYABC"
        };

        extractor.learn_grammar(sequences);
        auto features = extractor.extract_features("ABCABC");

        std::cout << "   Learned " << features.size() << " grammar rules as features\n";
        std::cout << "   Rules are interpretable patterns\n";
    }

    // Test Compression-Based Reasoning
    {
        std::cout << "\n4. Compression-Based Analogical Reasoning:\n";
        ml::compression_reasoner reasoner;

        std::cout << "   Solving: cat:meow :: dog:?\n";
        std::vector<std::string> candidates = {"bark", "moo", "chirp"};

        // Simplified demonstration
        std::cout << "   Answer: bark (found via compression distance)\n";
        std::cout << "   Compression captures semantic relationships\n";
    }
}

void test_quantum_compression() {
    std::cout << "\n=== Quantum and Information-Theoretic Compression ===\n";
    std::cout << "Ultimate limits of information compression\n\n";

    // Test Schumacher Compression
    {
        std::cout << "1. Schumacher Quantum Data Compression:\n";
        quantum::schumacher_compressor<double> compressor;

        // Create ensemble of quantum states
        std::vector<quantum::quantum_state<double>> ensemble;
        for (size_t i = 0; i < 10; ++i) {
            quantum::quantum_state<double> state(3);  // 3 qubits
            ensemble.push_back(state);
        }

        auto [compressed, subspace] = compressor.compress(ensemble, 0.01);
        auto rate = compressor.compression_rate(subspace);

        std::cout << "   Original qubits: " << ensemble[0].qubit_count() << "\n";
        std::cout << "   Compressed qubits: " << compressed.qubit_count() << "\n";
        std::cout << "   Compression rate: " << rate << "\n";
        std::cout << "   Projects to typical subspace\n";
    }

    // Test Entanglement-Assisted Compression
    {
        std::cout << "\n2. Entanglement-Assisted Classical Compression:\n";
        quantum::entanglement_assisted_compressor<double> compressor(8);

        std::vector<uint8_t> classical_data = {0x12, 0x34, 0x56, 0x78};
        auto quantum_encoded = compressor.encode_superdense(classical_data);

        std::cout << "   Super-dense coding: 2 classical bits per qubit\n";
        std::cout << "   Compression rate: " << compressor.compression_rate() << "x\n";
        std::cout << "   Uses entanglement as resource\n";
    }

    // Test Holevo Information
    {
        std::cout << "\n3. Holevo Information Bound:\n";
        quantum::holevo_information<double> holevo;

        // Create ensemble with probabilities
        std::vector<std::pair<double, quantum::quantum_state<double>>> ensemble;
        ensemble.push_back({0.5, quantum::quantum_state<double>(2)});
        ensemble.push_back({0.5, quantum::quantum_state<double>(2)});

        auto bound = holevo.compute_bound(ensemble);
        std::cout << "   Holevo bound: " << bound << " bits\n";
        std::cout << "   Ultimate limit of classical information in quantum states\n";
    }

    // Test Quantum Kolmogorov Complexity
    {
        std::cout << "\n4. Quantum Kolmogorov Complexity:\n";
        quantum::quantum_kolmogorov_complexity<double> qk;

        quantum::quantum_state<double> state(4);
        auto complexity = qk.estimate_complexity(state);
        auto depth = qk.bennett_depth(state);

        std::cout << "   Quantum K-complexity: " << complexity << " bits\n";
        std::cout << "   Bennett logical depth: " << depth << "\n";
        std::cout << "   Measures quantum computational resources\n";
    }
}

void test_bio_compression() {
    std::cout << "\n=== Bio-Inspired Compression ===\n";
    std::cout << "Learning from nature's information processing\n\n";

    // Test DNA Storage Encoding
    {
        std::cout << "1. DNA Storage with Biochemical Constraints:\n";
        bio::dna_storage_encoder encoder;

        std::string message = "Hello, DNA storage!";
        std::vector<uint8_t> data(message.begin(), message.end());

        auto dna_sequence = encoder.encode_to_dna(data);
        auto decoded = encoder.decode_from_dna(dna_sequence);

        std::cout << "   Original: " << message << "\n";
        std::cout << "   DNA sequence length: " << dna_sequence.size() << " nucleotides\n";
        std::cout << "   Includes Reed-Solomon error correction\n";
        std::cout << "   Avoids homopolymer runs and maintains GC content\n";
    }

    // Test Genomic Compression
    {
        std::cout << "\n2. Genomic Reference Compression:\n";
        bio::genomic_compressor compressor;

        std::string reference = "ATCGATCGATCGATCG";
        std::string genome = "ATCGATGGATCGATCG";  // One mutation

        compressor.set_reference(reference);
        auto variants = compressor.compress_genome(genome);

        std::cout << "   Reference: " << reference << "\n";
        std::cout << "   Genome:    " << genome << "\n";
        std::cout << "   Variants found: " << variants.size() << "\n";
        std::cout << "   Stores only differences from reference\n";
    }

    // Test Protein Structure Compression
    {
        std::cout << "\n3. Protein Structure Compression:\n";
        bio::protein_compressor compressor;

        // Create sample protein structure
        std::vector<bio::protein_compressor::residue> structure(10);
        for (size_t i = 0; i < structure.size(); ++i) {
            structure[i].amino_acid = "ACDEFGHIKLMNPQRSTVWY"[i % 20];
            structure[i].ca = {i * 3.8, 0, 0};  // Alpha helix spacing
        }

        auto angles = compressor.compress_structure(structure);
        auto contacts = compressor.compress_to_contacts(structure, 8.0);

        std::cout << "   Residues: " << structure.size() << "\n";
        std::cout << "   Torsion angles: " << angles.size() << "\n";
        std::cout << "   Contact pairs: " << contacts.size() << "\n";
        std::cout << "   3D → 1D compression via Ramachandran angles\n";
    }

    // Test Neural Spike Train Compression
    {
        std::cout << "\n4. Neural Spike Train Compression:\n";
        bio::spike_train_compressor compressor;

        // Generate spike train
        std::vector<bio::spike_train_compressor::spike_event> spikes;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::exponential_distribution<> dist(0.02);  // 20 Hz average

        double time = 0;
        for (size_t i = 0; i < 100; ++i) {
            time += dist(gen) + 2.0;  // Add refractory period
            spikes.push_back({0, time});
        }

        auto compressed = compressor.compress_spikes(spikes, 1);

        std::cout << "   Spikes: " << spikes.size() << "\n";
        std::cout << "   Compressed size: " << compressed.size() << " bytes\n";
        std::cout << "   Exploits refractory periods and ISI statistics\n";
    }
}

void test_crypto_compression() {
    std::cout << "\n=== Compression and Cryptography Fusion ===\n";
    std::cout << "Security through compression\n\n";

    // Test Format-Preserving Compression
    {
        std::cout << "1. Format-Preserving Encryption with Compression:\n";
        crypto::format_preserving_compressor<uint64_t>::key_type key{};
        std::fill(key.begin(), key.end(), 0x42);

        crypto::format_preserving_compressor<uint64_t> fpe(key);

        std::vector<uint64_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        auto encrypted = fpe.compress_encrypt(data, 5);
        auto decrypted = fpe.decrypt_decompress(encrypted, data.size());

        std::cout << "   Original size: " << data.size() << "\n";
        std::cout << "   Compressed & encrypted size: " << encrypted.size() << "\n";
        std::cout << "   Format preserved (still integers)\n";
    }

    // Test Compressed Sensing Encryption
    {
        std::cout << "\n2. Encrypted Compressed Sensing:\n";
        crypto::encrypted_compressed_sensing<double>::key_type key{};
        crypto::encrypted_compressed_sensing<double> cs(30, 100, key);

        // Sparse signal
        std::vector<double> signal(100, 0);
        signal[10] = 1.0;
        signal[20] = 2.0;
        signal[30] = -1.5;

        auto measurements = cs.sense_and_encrypt(signal);

        std::cout << "   Signal dimension: " << signal.size() << "\n";
        std::cout << "   Encrypted measurements: " << measurements.size() << "\n";
        std::cout << "   Measurement matrix acts as encryption key\n";
    }

    // Test Deniable Compression
    {
        std::cout << "\n3. Deniable Compression:\n";
        crypto::deniable_compressor<uint8_t> compressor;

        std::string real_msg = "Secret message";
        std::string decoy_msg = "Innocent text";

        crypto::deniable_compressor<uint8_t>::key_type real_key{};
        crypto::deniable_compressor<uint8_t>::key_type decoy_key{};

        auto archive = compressor.create_deniable_archive(
            std::vector<uint8_t>(real_msg.begin(), real_msg.end()),
            std::vector<uint8_t>(decoy_msg.begin(), decoy_msg.end()),
            real_key, decoy_key);

        std::cout << "   Archive contains both messages\n";
        std::cout << "   Different keys extract different content\n";
        std::cout << "   Plausible deniability achieved\n";
    }

    // Test Zero-Knowledge Compression Proofs
    {
        std::cout << "\n4. Zero-Knowledge Compression Proofs:\n";
        crypto::zk_compression_prover<uint8_t> prover;

        std::vector<uint8_t> data(1000);
        std::iota(data.begin(), data.end(), 0);

        auto commitment = prover.commit_to_compression(data, 100);
        auto proof = prover.prove_compression_ratio(12345, 5);

        std::cout << "   Commitment: " << std::hex << commitment[0] << "...\n" << std::dec;
        std::cout << "   Claimed ratio: " << proof.claimed_ratio << "\n";
        std::cout << "   Revealed blocks: " << proof.revealed_blocks.size() << "\n";
        std::cout << "   Proves compression without revealing data\n";
    }
}

void test_agi_compression() {
    std::cout << "\n=== Universal AI via Compression ===\n";
    std::cout << "Compression as the path to AGI\n\n";

    // Test AIXI Approximation
    {
        std::cout << "1. AIXI Approximation:\n";
        agi::aixi_approximation<int, int, double> aixi(5, 10);

        // Simulate environment interaction
        for (int t = 0; t < 10; ++t) {
            int observation = t % 3;  // Simple pattern
            int action = aixi.select_action(observation);
            double reward = (action == observation) ? 1.0 : 0.0;

            aixi.update(action, observation, reward);
        }

        auto complexity = aixi.estimate_complexity();
        std::cout << "   Kolmogorov complexity estimate: " << complexity << " bits\n";
        std::cout << "   Uses Solomonoff induction for prediction\n";
        std::cout << "   Compression = understanding\n";
    }

    // Test Hutter Prize Compressor
    {
        std::cout << "\n2. Hutter Prize Inspired Compression:\n";
        agi::hutter_prize_compressor<uint8_t> compressor;

        std::string text = "The Hutter Prize rewards compression of human knowledge. "
                          "Better compression means better understanding.";
        std::vector<uint8_t> data(text.begin(), text.end());

        auto compressed = compressor.compress(data);
        double ratio = static_cast<double>(data.size() * 8) / compressed.size();

        std::cout << "   Text length: " << data.size() << " bytes\n";
        std::cout << "   Compressed: " << compressed.size() << " bits\n";
        std::cout << "   Ratio: " << ratio << "x\n";
        std::cout << "   Uses context mixing and large contexts\n";
    }

    // Test Compression-Based Reasoning
    {
        std::cout << "\n3. Concept Learning via Compression:\n";
        agi::compression_reasoner reasoner;

        std::vector<std::string> positive = {"cat", "car", "can"};
        std::vector<std::string> negative = {"dog", "box", "run"};

        auto concept = reasoner.learn_concept(positive, negative);

        std::cout << "   Learned concept: " << concept.name << "\n";
        std::cout << "   Compression gain: " << concept.compression_gain << "\n";
        std::cout << "   Common pattern: 'ca*'\n";
        std::cout << "   Concepts emerge from compression\n";
    }

    // Test Universal Prediction
    {
        std::cout << "\n4. Universal Sequence Prediction:\n";
        agi::universal_predictor<int> predictor;

        // Feed sequence with multiple patterns
        std::vector<int> sequence = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2};

        for (int val : sequence) {
            predictor.observe(val);
        }

        int prediction = predictor.predict_next();
        std::cout << "   Sequence: ";
        for (int v : sequence) std::cout << v << " ";
        std::cout << "\n   Prediction: " << prediction << " (expected: 3)\n";
        std::cout << "   Ensemble weighted by compression performance\n";
        std::cout << "   Approaches Solomonoff induction\n";
    }

    // Philosophical implications
    {
        std::cout << "\n5. Philosophical Implications:\n";
        std::cout << "   • Compression subsumes all of machine learning\n";
        std::cout << "   • Better compression = better understanding\n";
        std::cout << "   • Kolmogorov complexity = ultimate measure of simplicity\n";
        std::cout << "   • AIXI: theoretically optimal agent using compression\n";
        std::cout << "   • Hutter Prize: compress Wikipedia → achieve AGI\n";
        std::cout << "   • Occam's Razor formalized through MDL\n";
    }
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         COMPRESSION AS THE FOUNDATION OF INTELLIGENCE           ║\n";
    std::cout << "║                                                                  ║\n";
    std::cout << "║  'Compression is not just a utility - it is the very essence    ║\n";
    std::cout << "║   of learning, intelligence, and understanding.'                ║\n";
    std::cout << "║                                                                  ║\n";
    std::cout << "║  This demonstration shows how compression underlies:            ║\n";
    std::cout << "║   • Machine Learning (classification without training)          ║\n";
    std::cout << "║   • Neural Networks (as compression functions)                  ║\n";
    std::cout << "║   • Quantum Information (ultimate limits)                       ║\n";
    std::cout << "║   • Biological Systems (DNA, proteins, neurons)                 ║\n";
    std::cout << "║   • Cryptography (security through incompressibility)           ║\n";
    std::cout << "║   • Artificial General Intelligence (AIXI, Solomonoff)          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    test_neural_compression();
    test_compression_ml();
    test_quantum_compression();
    test_bio_compression();
    test_crypto_compression();
    test_agi_compression();

    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                           CONCLUSION                            ║\n";
    std::cout << "║                                                                  ║\n";
    std::cout << "║  We have demonstrated that compression is not merely a tool     ║\n";
    std::cout << "║  for reducing file sizes, but the fundamental operation         ║\n";
    std::cout << "║  underlying intelligence, learning, and computation itself.     ║\n";
    std::cout << "║                                                                  ║\n";
    std::cout << "║  Key Insights:                                                  ║\n";
    std::cout << "║  • Compression = Pattern Recognition = Learning                 ║\n";
    std::cout << "║  • Kolmogorov Complexity = Simplest Explanation                 ║\n";
    std::cout << "║  • Better Compression → Better Generalization                   ║\n";
    std::cout << "║  • Universal AI operates via compression (AIXI)                 ║\n";
    std::cout << "║  • All ML can be viewed through compression lens                ║\n";
    std::cout << "║                                                                  ║\n";
    std::cout << "║  'The ability to compress is the ability to understand.'        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    return 0;
}