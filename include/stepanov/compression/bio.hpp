// Bio-Inspired Compression: Learning from Nature's Information Processing
// "Biology is the most sophisticated information processing system" - DNA as code
#pragma once

#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <numeric>
#include <bitset>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <random>
#include "../concepts.hpp"

namespace stepanov::compression::bio {

// DNA storage encoding with biochemical constraints
class dna_storage_encoder {
public:
    using nucleotide = char;  // A, C, G, T
    using codon = std::array<nucleotide, 3>;

private:
    // Biochemical constraints
    struct synthesis_constraints {
        size_t max_homopolymer_length = 3;      // Max consecutive same nucleotide
        double gc_content_min = 0.4;            // Min GC content for stability
        double gc_content_max = 0.6;            // Max GC content
        size_t min_hamming_distance = 3;        // Between sequences
        bool avoid_restriction_sites = true;     // Avoid enzyme cut sites
    };

    synthesis_constraints constraints;

    // Error correction using Reed-Solomon codes
    struct reed_solomon_dna {
        size_t block_size = 255;
        size_t data_size = 223;
        size_t parity_size = 32;

        std::vector<uint8_t> encode_block(const std::vector<uint8_t>& data) {
            // Simplified RS encoding
            std::vector<uint8_t> encoded = data;

            // Add parity symbols
            for (size_t i = 0; i < parity_size; ++i) {
                uint8_t parity = 0;
                for (size_t j = 0; j < data.size(); ++j) {
                    parity ^= galois_multiply(data[j], i + 1);
                }
                encoded.push_back(parity);
            }

            return encoded;
        }

        uint8_t galois_multiply(uint8_t a, uint8_t b) {
            // GF(256) multiplication
            uint16_t result = 0;
            uint16_t temp_a = a;
            uint16_t temp_b = b;

            while (temp_b) {
                if (temp_b & 1) {
                    result ^= temp_a;
                }
                temp_a <<= 1;
                if (temp_a & 0x100) {
                    temp_a ^= 0x11B;  // Primitive polynomial
                }
                temp_b >>= 1;
            }

            return static_cast<uint8_t>(result);
        }
    };

    reed_solomon_dna error_corrector;

public:
    // Encode binary data to DNA sequence
    std::string encode_to_dna(const std::vector<uint8_t>& binary_data) {
        // Apply error correction
        std::vector<uint8_t> protected_data;
        for (size_t i = 0; i < binary_data.size(); i += error_corrector.data_size) {
            size_t chunk_size = std::min(error_corrector.data_size,
                                        binary_data.size() - i);
            std::vector<uint8_t> chunk(binary_data.begin() + i,
                                      binary_data.begin() + i + chunk_size);

            // Pad if necessary
            chunk.resize(error_corrector.data_size, 0);

            auto encoded = error_corrector.encode_block(chunk);
            protected_data.insert(protected_data.end(),
                                encoded.begin(), encoded.end());
        }

        // Convert to quaternary (base 4) then to DNA
        std::string dna_sequence;
        for (uint8_t byte : protected_data) {
            // Each byte becomes 4 nucleotides (2 bits each)
            for (int i = 0; i < 4; ++i) {
                uint8_t two_bits = (byte >> (6 - 2*i)) & 0x03;
                dna_sequence += quaternary_to_nucleotide(two_bits);
            }
        }

        // Apply biochemical constraints
        return apply_constraints(dna_sequence);
    }

    // Decode DNA sequence to binary data
    std::vector<uint8_t> decode_from_dna(const std::string& dna_sequence) {
        // Remove constraint encoding
        std::string raw_dna = remove_constraints(dna_sequence);

        // Convert DNA to bytes
        std::vector<uint8_t> encoded_data;
        for (size_t i = 0; i < raw_dna.size(); i += 4) {
            uint8_t byte = 0;
            for (int j = 0; j < 4 && i + j < raw_dna.size(); ++j) {
                uint8_t two_bits = nucleotide_to_quaternary(raw_dna[i + j]);
                byte |= (two_bits << (6 - 2*j));
            }
            encoded_data.push_back(byte);
        }

        // Apply error correction decoding
        std::vector<uint8_t> decoded_data;
        size_t total_block_size = error_corrector.data_size + error_corrector.parity_size;

        for (size_t i = 0; i < encoded_data.size(); i += total_block_size) {
            size_t chunk_size = std::min(total_block_size, encoded_data.size() - i);
            std::vector<uint8_t> chunk(encoded_data.begin() + i,
                                      encoded_data.begin() + i + chunk_size);

            // Extract data portion (simplified - no actual error correction)
            for (size_t j = 0; j < error_corrector.data_size && j < chunk.size(); ++j) {
                decoded_data.push_back(chunk[j]);
            }
        }

        return decoded_data;
    }

private:
    nucleotide quaternary_to_nucleotide(uint8_t q) {
        const char map[4] = {'A', 'C', 'G', 'T'};
        return map[q & 0x03];
    }

    uint8_t nucleotide_to_quaternary(nucleotide n) {
        switch (n) {
            case 'A': return 0;
            case 'C': return 1;
            case 'G': return 2;
            case 'T': return 3;
            default: return 0;
        }
    }

    std::string apply_constraints(const std::string& dna) {
        std::string constrained;

        for (size_t i = 0; i < dna.size(); ++i) {
            constrained += dna[i];

            // Check homopolymer runs
            if (i >= constraints.max_homopolymer_length - 1) {
                bool is_homopolymer = true;
                for (size_t j = 1; j < constraints.max_homopolymer_length; ++j) {
                    if (constrained[constrained.size() - j] !=
                        constrained[constrained.size() - j - 1]) {
                        is_homopolymer = false;
                        break;
                    }
                }

                if (is_homopolymer) {
                    // Insert different nucleotide to break run
                    constrained += get_different_nucleotide(dna[i]);
                }
            }
        }

        // Adjust GC content if needed
        double gc_content = compute_gc_content(constrained);
        if (gc_content < constraints.gc_content_min) {
            constrained = increase_gc_content(constrained);
        } else if (gc_content > constraints.gc_content_max) {
            constrained = decrease_gc_content(constrained);
        }

        return constrained;
    }

    std::string remove_constraints(const std::string& dna) {
        // Simplified: assume constraints were markers that can be removed
        return dna;
    }

    double compute_gc_content(const std::string& dna) {
        size_t gc_count = std::count_if(dna.begin(), dna.end(),
            [](char n) { return n == 'G' || n == 'C'; });
        return static_cast<double>(gc_count) / dna.size();
    }

    char get_different_nucleotide(char n) {
        const char nucleotides[4] = {'A', 'C', 'G', 'T'};
        for (char alt : nucleotides) {
            if (alt != n) return alt;
        }
        return 'N';
    }

    std::string increase_gc_content(const std::string& dna) {
        std::string adjusted = dna;
        // Replace some A/T with G/C
        for (size_t i = 0; i < adjusted.size(); i += 10) {
            if (adjusted[i] == 'A') adjusted[i] = 'G';
            else if (adjusted[i] == 'T') adjusted[i] = 'C';
        }
        return adjusted;
    }

    std::string decrease_gc_content(const std::string& dna) {
        std::string adjusted = dna;
        // Replace some G/C with A/T
        for (size_t i = 0; i < adjusted.size(); i += 10) {
            if (adjusted[i] == 'G') adjusted[i] = 'A';
            else if (adjusted[i] == 'C') adjusted[i] = 'T';
        }
        return adjusted;
    }
};

// Genomic reference compression
class genomic_compressor {
public:
    using position = size_t;
    using variant = std::pair<position, std::string>;

private:
    std::string reference_genome;

    // Variant representation
    struct genomic_variant {
        position pos;
        std::string ref_allele;
        std::string alt_allele;
        uint8_t quality;
    };

public:
    void set_reference(const std::string& reference) {
        reference_genome = reference;
    }

    // Compress genome as differences from reference
    std::vector<genomic_variant> compress_genome(const std::string& genome) {
        std::vector<genomic_variant> variants;

        // Find differences using suffix arrays for efficiency
        size_t i = 0, j = 0;
        while (i < reference_genome.size() && j < genome.size()) {
            if (reference_genome[i] != genome[j]) {
                // Find extent of variation
                size_t ref_end = i + 1;
                size_t alt_end = j + 1;

                // Extend to find full variant
                while (ref_end < reference_genome.size() &&
                       alt_end < genome.size() &&
                       reference_genome[ref_end] != genome[alt_end]) {
                    ref_end++;
                    alt_end++;
                }

                variants.push_back({
                    i,
                    reference_genome.substr(i, ref_end - i),
                    genome.substr(j, alt_end - j),
                    255  // Max quality
                });

                i = ref_end;
                j = alt_end;
            } else {
                i++;
                j++;
            }
        }

        return variants;
    }

    // Decompress using reference
    std::string decompress_genome(const std::vector<genomic_variant>& variants) {
        std::string genome = reference_genome;

        // Apply variants in reverse order to maintain positions
        for (auto it = variants.rbegin(); it != variants.rend(); ++it) {
            genome.replace(it->pos, it->ref_allele.size(), it->alt_allele);
        }

        return genome;
    }

    // Population-based compression using haplotypes
    std::vector<uint8_t> compress_population(
        const std::vector<std::string>& genomes) {

        // Find common haplotype blocks
        std::vector<std::string> haplotypes = extract_haplotypes(genomes);

        std::vector<uint8_t> compressed;

        // Encode haplotype identifiers for each genome
        for (const auto& genome : genomes) {
            auto hap_ids = identify_haplotypes(genome, haplotypes);
            for (uint32_t id : hap_ids) {
                // Variable length encoding
                while (id >= 128) {
                    compressed.push_back(0x80 | (id & 0x7F));
                    id >>= 7;
                }
                compressed.push_back(id);
            }
        }

        return compressed;
    }

private:
    std::vector<std::string> extract_haplotypes(
        const std::vector<std::string>& genomes) {

        std::vector<std::string> haplotypes;

        // Simplified: extract fixed-size blocks
        size_t block_size = 1000;
        std::unordered_map<std::string, size_t> block_counts;

        for (const auto& genome : genomes) {
            for (size_t i = 0; i < genome.size(); i += block_size) {
                std::string block = genome.substr(i, block_size);
                block_counts[block]++;
            }
        }

        // Keep frequent haplotypes
        for (const auto& [block, count] : block_counts) {
            if (count > genomes.size() / 10) {  // >10% frequency
                haplotypes.push_back(block);
            }
        }

        return haplotypes;
    }

    std::vector<uint32_t> identify_haplotypes(
        const std::string& genome,
        const std::vector<std::string>& haplotypes) {

        std::vector<uint32_t> ids;

        for (size_t i = 0; i < genome.size(); i += 1000) {
            std::string block = genome.substr(i, 1000);

            // Find matching haplotype
            auto it = std::find(haplotypes.begin(), haplotypes.end(), block);
            if (it != haplotypes.end()) {
                ids.push_back(std::distance(haplotypes.begin(), it));
            } else {
                ids.push_back(0xFFFFFFFF);  // Special marker for novel block
            }
        }

        return ids;
    }
};

// Protein structure compression
class protein_compressor {
public:
    // 3D coordinates
    struct atom_position {
        double x, y, z;
    };

    struct residue {
        char amino_acid;
        atom_position ca;  // Alpha carbon
        atom_position n;   // Nitrogen
        atom_position c;   // Carbon
        atom_position o;   // Oxygen
    };

private:
    // Ramachandran angles for backbone
    struct torsion_angles {
        double phi;    // C(-1) - N - CA - C
        double psi;    // N - CA - C - N(+1)
        double omega;  // CA - C - N(+1) - CA(+1)
    };

public:
    // Compress 3D structure to torsion angles
    std::vector<torsion_angles> compress_structure(
        const std::vector<residue>& structure) {

        std::vector<torsion_angles> angles;

        for (size_t i = 1; i < structure.size() - 1; ++i) {
            torsion_angles ta;

            // Calculate phi
            ta.phi = dihedral_angle(
                structure[i-1].c,
                structure[i].n,
                structure[i].ca,
                structure[i].c
            );

            // Calculate psi
            ta.psi = dihedral_angle(
                structure[i].n,
                structure[i].ca,
                structure[i].c,
                structure[i+1].n
            );

            // Omega is usually ~180 degrees (trans)
            ta.omega = dihedral_angle(
                structure[i].ca,
                structure[i].c,
                structure[i+1].n,
                structure[i+1].ca
            );

            angles.push_back(ta);
        }

        return angles;
    }

    // Reconstruct structure from angles
    std::vector<residue> decompress_structure(
        const std::string& sequence,
        const std::vector<torsion_angles>& angles) {

        std::vector<residue> structure;

        // Start with standard geometry
        residue first;
        first.amino_acid = sequence[0];
        first.n = {0, 0, 0};
        first.ca = {1.458, 0, 0};
        first.c = {2.428, 1.326, 0};
        first.o = {2.059, 2.483, 0};
        structure.push_back(first);

        // Build rest using angles
        for (size_t i = 0; i < angles.size() && i + 1 < sequence.size(); ++i) {
            residue next = build_next_residue(
                structure.back(),
                sequence[i + 1],
                angles[i]
            );
            structure.push_back(next);
        }

        return structure;
    }

    // Contact map compression
    std::vector<std::pair<uint16_t, uint16_t>> compress_to_contacts(
        const std::vector<residue>& structure,
        double threshold = 8.0) {

        std::vector<std::pair<uint16_t, uint16_t>> contacts;

        for (size_t i = 0; i < structure.size(); ++i) {
            for (size_t j = i + 4; j < structure.size(); ++j) {
                double dist = distance(structure[i].ca, structure[j].ca);
                if (dist < threshold) {
                    contacts.push_back({i, j});
                }
            }
        }

        return contacts;
    }

    // Secondary structure compression
    std::string compress_secondary_structure(
        const std::vector<torsion_angles>& angles) {

        std::string ss_string;

        for (const auto& ta : angles) {
            // Classify based on Ramachandran regions
            if (is_alpha_helix(ta.phi, ta.psi)) {
                ss_string += 'H';
            } else if (is_beta_sheet(ta.phi, ta.psi)) {
                ss_string += 'E';
            } else {
                ss_string += 'C';  // Coil
            }
        }

        // Run-length encode
        return run_length_encode(ss_string);
    }

private:
    double dihedral_angle(const atom_position& a1,
                         const atom_position& a2,
                         const atom_position& a3,
                         const atom_position& a4) {
        // Calculate dihedral angle between 4 atoms
        // Vector calculations
        double v1x = a2.x - a1.x, v1y = a2.y - a1.y, v1z = a2.z - a1.z;
        double v2x = a3.x - a2.x, v2y = a3.y - a2.y, v2z = a3.z - a2.z;
        double v3x = a4.x - a3.x, v3y = a4.y - a3.y, v3z = a4.z - a3.z;

        // Cross products
        double n1x = v1y * v2z - v1z * v2y;
        double n1y = v1z * v2x - v1x * v2z;
        double n1z = v1x * v2y - v1y * v2x;

        double n2x = v2y * v3z - v2z * v3y;
        double n2y = v2z * v3x - v2x * v3z;
        double n2z = v2x * v3y - v2y * v3x;

        // Dot product
        double dot = n1x * n2x + n1y * n2y + n1z * n2z;

        // Magnitudes
        double mag1 = std::sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
        double mag2 = std::sqrt(n2x * n2x + n2y * n2y + n2z * n2z);

        return std::acos(dot / (mag1 * mag2)) * 180.0 / M_PI;
    }

    double distance(const atom_position& a, const atom_position& b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        double dz = a.z - b.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    bool is_alpha_helix(double phi, double psi) {
        return (phi > -80 && phi < -50) && (psi > -60 && psi < -30);
    }

    bool is_beta_sheet(double phi, double psi) {
        return (phi > -140 && phi < -90) && (psi > 90 && psi < 150);
    }

    residue build_next_residue(const residue& prev,
                              char amino_acid,
                              const torsion_angles& angles) {
        residue next;
        next.amino_acid = amino_acid;

        // Simplified: place atoms based on angles and standard bond lengths
        // Standard bond lengths
        double bond_ca_c = 1.525;
        double bond_c_n = 1.329;
        double bond_n_ca = 1.458;

        // Use transformation matrices based on angles
        // This is simplified - real implementation would use proper 3D rotations
        next.n = {prev.c.x + bond_c_n, prev.c.y, prev.c.z};
        next.ca = {next.n.x + bond_n_ca, next.n.y, next.n.z};
        next.c = {next.ca.x + bond_ca_c, next.ca.y, next.ca.z};
        next.o = {next.c.x, next.c.y + 1.231, next.c.z};

        return next;
    }

    std::string run_length_encode(const std::string& ss) {
        std::string encoded;

        for (size_t i = 0; i < ss.size(); ) {
            char current = ss[i];
            size_t count = 1;

            while (i + count < ss.size() && ss[i + count] == current) {
                count++;
            }

            if (count > 1) {
                encoded += std::to_string(count);
            }
            encoded += current;

            i += count;
        }

        return encoded;
    }
};

// Neural spike train compression
class spike_train_compressor {
public:
    using spike_time = double;  // Time in milliseconds
    using neuron_id = uint32_t;

private:
    struct spike_event {
        neuron_id neuron;
        spike_time time;
    };

    // ISI (Inter-Spike Interval) statistics
    struct isi_model {
        double mean_rate;
        double cv;  // Coefficient of variation
        double refractory_period;

        // Fit exponential distribution
        double lambda() const {
            return 1.0 / (mean_rate - refractory_period);
        }
    };

public:
    // Compress spike trains using ISI distributions
    std::vector<uint8_t> compress_spikes(
        const std::vector<spike_event>& spikes,
        size_t num_neurons) {

        // Group by neuron
        std::vector<std::vector<spike_time>> trains(num_neurons);
        for (const auto& spike : spikes) {
            trains[spike.neuron].push_back(spike.time);
        }

        std::vector<uint8_t> compressed;

        for (size_t n = 0; n < num_neurons; ++n) {
            auto& train = trains[n];
            if (train.empty()) continue;

            // Sort spike times
            std::sort(train.begin(), train.end());

            // Fit ISI model
            auto model = fit_isi_model(train);

            // Encode model parameters
            encode_double(compressed, model.mean_rate);
            encode_double(compressed, model.cv);
            encode_double(compressed, model.refractory_period);

            // Encode spike count
            encode_varint(compressed, train.size());

            // Delta encode ISIs with predictive coding
            double last_time = 0;
            for (spike_time t : train) {
                double isi = t - last_time;
                double predicted = 1.0 / model.lambda();
                double residual = isi - predicted;

                // Quantize and encode residual
                int16_t quantized = static_cast<int16_t>(residual * 100);
                compressed.push_back(quantized >> 8);
                compressed.push_back(quantized & 0xFF);

                last_time = t;
            }
        }

        return compressed;
    }

    // Population vector compression
    std::vector<uint8_t> compress_population_vector(
        const std::vector<std::vector<double>>& firing_rates,
        double time_bin_ms = 10.0) {

        std::vector<uint8_t> compressed;
        size_t num_neurons = firing_rates.size();
        size_t num_bins = firing_rates[0].size();

        // PCA on firing rates
        auto [components, scores] = pca(firing_rates, 10);  // Keep top 10 components

        // Encode PCA components
        for (const auto& comp : components) {
            for (double val : comp) {
                encode_double(compressed, val);
            }
        }

        // Encode scores with temporal correlation
        for (size_t t = 0; t < scores[0].size(); ++t) {
            for (size_t c = 0; c < scores.size(); ++c) {
                if (t == 0) {
                    encode_double(compressed, scores[c][t]);
                } else {
                    // Delta encoding
                    double delta = scores[c][t] - scores[c][t-1];
                    int16_t quantized = static_cast<int16_t>(delta * 1000);
                    compressed.push_back(quantized >> 8);
                    compressed.push_back(quantized & 0xFF);
                }
            }
        }

        return compressed;
    }

    // Compress using refractory periods
    std::vector<bool> compress_with_refractory(
        const std::vector<spike_event>& spikes,
        double refractory_ms = 2.0) {

        std::vector<bool> binary;

        // Group by neuron
        std::unordered_map<neuron_id, spike_time> last_spike;

        for (const auto& spike : spikes) {
            auto it = last_spike.find(spike.neuron);

            if (it != last_spike.end()) {
                double isi = spike.time - it->second;

                // Can't spike during refractory period
                if (isi < refractory_ms) {
                    binary.push_back(false);  // Error flag
                    continue;
                }

                // Encode ISI minus refractory period
                double effective_isi = isi - refractory_ms;
                encode_exponential(binary, effective_isi);
            } else {
                // First spike
                binary.push_back(true);  // Start flag
                encode_time(binary, spike.time);
            }

            last_spike[spike.neuron] = spike.time;
        }

        return binary;
    }

private:
    isi_model fit_isi_model(const std::vector<spike_time>& train) {
        isi_model model;

        if (train.size() < 2) {
            model.mean_rate = 0;
            model.cv = 0;
            model.refractory_period = 2.0;  // Default
            return model;
        }

        // Calculate ISIs
        std::vector<double> isis;
        for (size_t i = 1; i < train.size(); ++i) {
            isis.push_back(train[i] - train[i-1]);
        }

        // Statistics
        double mean = std::accumulate(isis.begin(), isis.end(), 0.0) / isis.size();
        double variance = 0;
        for (double isi : isis) {
            variance += (isi - mean) * (isi - mean);
        }
        variance /= isis.size();

        model.mean_rate = 1000.0 / mean;  // Convert to Hz
        model.cv = std::sqrt(variance) / mean;
        model.refractory_period = *std::min_element(isis.begin(), isis.end());

        return model;
    }

    void encode_double(std::vector<uint8_t>& buffer, double value) {
        auto bytes = reinterpret_cast<const uint8_t*>(&value);
        buffer.insert(buffer.end(), bytes, bytes + sizeof(double));
    }

    void encode_varint(std::vector<uint8_t>& buffer, size_t value) {
        while (value >= 128) {
            buffer.push_back(0x80 | (value & 0x7F));
            value >>= 7;
        }
        buffer.push_back(value);
    }

    void encode_exponential(std::vector<bool>& binary, double value) {
        // Simplified exponential Golomb coding
        int k = static_cast<int>(std::log2(value + 1));

        // Unary code for k
        for (int i = 0; i < k; ++i) {
            binary.push_back(false);
        }
        binary.push_back(true);

        // Binary representation
        int remainder = static_cast<int>(value - (1 << k) + 1);
        for (int i = k - 1; i >= 0; --i) {
            binary.push_back((remainder >> i) & 1);
        }
    }

    void encode_time(std::vector<bool>& binary, spike_time time) {
        // Quantize to microseconds
        uint32_t quantized = static_cast<uint32_t>(time * 1000);
        for (int i = 31; i >= 0; --i) {
            binary.push_back((quantized >> i) & 1);
        }
    }

    std::pair<std::vector<std::vector<double>>,
              std::vector<std::vector<double>>> pca(
        const std::vector<std::vector<double>>& data,
        size_t num_components) {

        // Simplified PCA
        size_t n = data.size();
        size_t m = data[0].size();

        // Center data
        std::vector<double> means(n, 0);
        for (size_t i = 0; i < n; ++i) {
            means[i] = std::accumulate(data[i].begin(), data[i].end(), 0.0) / m;
        }

        std::vector<std::vector<double>> centered(n, std::vector<double>(m));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                centered[i][j] = data[i][j] - means[i];
            }
        }

        // Compute covariance (simplified - should use SVD)
        std::vector<std::vector<double>> components(num_components,
                                                   std::vector<double>(n, 1.0/std::sqrt(n)));
        std::vector<std::vector<double>> scores(num_components,
                                               std::vector<double>(m, 0));

        // Project onto components
        for (size_t c = 0; c < num_components; ++c) {
            for (size_t j = 0; j < m; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    scores[c][j] += components[c][i] * centered[i][j];
                }
            }
        }

        return {components, scores};
    }
};

// Evolutionary compression using genetic algorithms
class evolutionary_compressor {
public:
    using genome_type = std::vector<uint8_t>;
    using fitness_type = double;

private:
    struct individual {
        genome_type genome;
        fitness_type fitness;
    };

    struct evolution_params {
        size_t population_size = 100;
        double mutation_rate = 0.01;
        double crossover_rate = 0.7;
        size_t elite_size = 10;
    };

    evolution_params params;
    std::mt19937 rng;

public:
    evolutionary_compressor() : rng(std::random_device{}()) {}

    // Evolve compression dictionary
    std::vector<std::string> evolve_dictionary(
        const std::vector<std::string>& training_data,
        size_t dict_size,
        size_t generations = 100) {

        // Initialize population
        std::vector<individual> population;
        for (size_t i = 0; i < params.population_size; ++i) {
            population.push_back({
                random_genome(dict_size * 10),  // Dictionary as genome
                0.0
            });
        }

        // Evolution loop
        for (size_t gen = 0; gen < generations; ++gen) {
            // Evaluate fitness
            for (auto& ind : population) {
                ind.fitness = evaluate_compression(ind.genome, training_data);
            }

            // Sort by fitness
            std::sort(population.begin(), population.end(),
                     [](const auto& a, const auto& b) {
                         return a.fitness > b.fitness;
                     });

            // Create next generation
            std::vector<individual> next_gen;

            // Elite selection
            for (size_t i = 0; i < params.elite_size; ++i) {
                next_gen.push_back(population[i]);
            }

            // Crossover and mutation
            while (next_gen.size() < params.population_size) {
                auto parent1 = tournament_select(population);
                auto parent2 = tournament_select(population);

                auto child = crossover(parent1.genome, parent2.genome);
                mutate(child);

                next_gen.push_back({child, 0.0});
            }

            population = next_gen;
        }

        // Extract dictionary from best individual
        return genome_to_dictionary(population[0].genome, dict_size);
    }

private:
    genome_type random_genome(size_t size) {
        genome_type genome(size);
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        for (auto& gene : genome) {
            gene = dist(rng);
        }
        return genome;
    }

    fitness_type evaluate_compression(const genome_type& genome,
                                     const std::vector<std::string>& data) {
        auto dictionary = genome_to_dictionary(genome, 100);

        // Measure compression ratio
        size_t original_size = 0;
        size_t compressed_size = 0;

        for (const auto& text : data) {
            original_size += text.size();
            compressed_size += compress_with_dictionary(text, dictionary).size();
        }

        return static_cast<double>(original_size) / compressed_size;
    }

    std::vector<std::string> genome_to_dictionary(const genome_type& genome,
                                                 size_t dict_size) {
        std::vector<std::string> dictionary;

        // Extract substrings from genome
        for (size_t i = 0; i < dict_size && i < genome.size() / 4; ++i) {
            size_t start = i * 4;
            size_t length = std::min<size_t>(genome[start] % 16 + 1,
                                            genome.size() - start - 1);

            std::string entry;
            for (size_t j = 0; j < length; ++j) {
                entry += static_cast<char>(genome[start + 1 + j]);
            }

            if (!entry.empty()) {
                dictionary.push_back(entry);
            }
        }

        return dictionary;
    }

    std::string compress_with_dictionary(const std::string& text,
                                        const std::vector<std::string>& dict) {
        std::string compressed;
        size_t pos = 0;

        while (pos < text.size()) {
            // Find longest match in dictionary
            size_t best_match = 0;
            size_t best_length = 0;

            for (size_t i = 0; i < dict.size(); ++i) {
                if (text.substr(pos, dict[i].size()) == dict[i]) {
                    if (dict[i].size() > best_length) {
                        best_match = i;
                        best_length = dict[i].size();
                    }
                }
            }

            if (best_length > 0) {
                compressed += static_cast<char>(0x80 | best_match);
                pos += best_length;
            } else {
                compressed += text[pos];
                pos++;
            }
        }

        return compressed;
    }

    individual tournament_select(const std::vector<individual>& population) {
        std::uniform_int_distribution<size_t> dist(0, population.size() - 1);

        size_t idx1 = dist(rng);
        size_t idx2 = dist(rng);

        return population[idx1].fitness > population[idx2].fitness ?
               population[idx1] : population[idx2];
    }

    genome_type crossover(const genome_type& parent1,
                         const genome_type& parent2) {
        std::uniform_real_distribution<double> dist(0, 1);

        if (dist(rng) > params.crossover_rate) {
            return parent1;
        }

        genome_type child(parent1.size());
        size_t crossover_point = dist(rng) * parent1.size();

        for (size_t i = 0; i < child.size(); ++i) {
            child[i] = (i < crossover_point) ? parent1[i] : parent2[i];
        }

        return child;
    }

    void mutate(genome_type& genome) {
        std::uniform_real_distribution<double> dist(0, 1);
        std::uniform_int_distribution<uint8_t> byte_dist(0, 255);

        for (auto& gene : genome) {
            if (dist(rng) < params.mutation_rate) {
                gene = byte_dist(rng);
            }
        }
    }
};

} // namespace stepanov::compression::bio