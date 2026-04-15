/**
 * @file odds_ratio.hpp
 * @brief Odds-Ratio Transform - Bayesian Computation Without Normalization
 * @author CBT Framework Contributors
 * @date 2024
 * 
 * @details
 * The odds-ratio transform converts probabilities to odds, transforming Bayesian
 * updates from complex normalizations into simple multiplications. This is the
 * foundation of many practical inference systems.
 * 
 * Mathematical Foundation:
 * - Transform: φ(p) = p/(1-p)
 * - Inverse: φ⁻¹(o) = o/(1+o)
 * - Homomorphism: posterior_odds = prior_odds × likelihood_ratio
 * 
 * Trade-offs:
 * - Gains:
 *   - Bayesian updates without normalization
 *   - Sequential inference via multiplication
 *   - Natural representation for likelihood ratios
 *   - Numerical stability for extreme probabilities (via log-odds)
 * - Losses:
 *   - Cannot directly add probabilities
 *   - Marginalization becomes complex
 *   - Domain restricted to [0,1] for probabilities
 * 
 * @see log_odds for numerical stability with extreme probabilities
 * @see logarithmic.hpp for related log-domain computations
 */

#pragma once
#include <cmath>
#include <iostream>
#include <limits>

namespace mutatio {

/**
 * @class odds_ratio
 * @brief Transforms probabilities to odds for efficient Bayesian computation
 * @tparam T Underlying floating-point type (float, double, long double)
 * 
 * @details
 * The odds_ratio class implements the odds domain where Bayesian updates
 * become simple multiplications. This is particularly useful for:
 * - Medical diagnosis systems
 * - Spam filtering algorithms
 * - Sequential decision making
 * - Information retrieval ranking
 * 
 * <b>Example Usage:</b>
 * @code
 * // Disease diagnosis with 1% prevalence
 * auto prior = odds_ratio<double>::from_probability(0.01);
 * 
 * // Test with 90% sensitivity, 95% specificity
 * double likelihood_ratio = 0.90 / (1 - 0.95);  // LR+ = 18
 * 
 * // Positive test result updates odds
 * auto posterior = prior * odds_ratio<double>(likelihood_ratio);
 * 
 * // Convert back to probability
 * double prob = posterior.to_probability();  // ≈ 0.154
 * @endcode
 */
template<typename T>
class odds_ratio {
    static_assert(std::is_floating_point_v<T>, "odds_ratio requires floating-point type");
    
private:
    T odds_;  ///< Internal odds representation (p/(1-p))
    
public:
    /// @brief Default constructor - equal odds (50% probability)
    constexpr odds_ratio() : odds_(1) {}
    
    /// @brief Construct from odds value
    /// @param odds The odds value (ratio form, not percentage)
    /// @note odds = 2 means 2:1 odds, or 66.7% probability
    explicit constexpr odds_ratio(T odds) : odds_(odds) {}
    
    /// @brief Factory method to create from probability
    /// @param prob Probability value in [0,1]
    /// @return odds_ratio instance
    /// @note Handles edge cases: prob=0 → odds=0, prob=1 → odds=∞
    /// @warning Values outside [0,1] are clamped to valid range
    static constexpr odds_ratio from_probability(T prob) {
        if (prob <= 0) return odds_ratio(0);
        if (prob >= 1) return odds_ratio(std::numeric_limits<T>::infinity());
        return odds_ratio(prob / (1 - prob));
    }
    
    /// @brief Convert odds back to probability
    /// @return Probability in [0,1]
    /// @note Handles infinite odds (returns 1.0)
    /// @details Uses formula: p = odds/(1+odds)
    constexpr T to_probability() const {
        if (std::isinf(odds_)) return 1;
        return odds_ / (1 + odds_);
    }
    
    /// @brief Get the raw odds value
    /// @return The odds as a ratio (not percentage)
    constexpr T value() const { return odds_; }
    
    /// @brief Convert to log-odds representation
    /// @return Natural logarithm of the odds
    /// @note Useful for numerical stability with extreme probabilities
    constexpr T to_log_odds() const {
        return std::log(odds_);
    }
    
    /// @brief Bayesian update via multiplication
    /// @param likelihood_ratio The likelihood ratio (LR+ or LR-)
    /// @return Updated posterior odds
    /// @details Implements Bayes' theorem: posterior_odds = prior_odds × LR
    /// @note This is the key operation that makes Bayesian inference efficient
    /// 
    /// @code
    /// // Example: Medical test with LR+ = 10
    /// auto prior = odds_ratio<double>::from_probability(0.1);
    /// auto posterior = prior * odds_ratio<double>(10.0);
    /// @endcode
    constexpr odds_ratio operator*(const odds_ratio& likelihood_ratio) const {
        return odds_ratio(odds_ * likelihood_ratio.odds_);
    }
    
    /// @brief Inverse Bayesian update
    /// @param other Odds to divide by
    /// @return Quotient of odds
    /// @note Useful for removing evidence or computing relative odds
    constexpr odds_ratio operator/(const odds_ratio& other) const {
        return odds_ratio(odds_ / other.odds_);
    }
    
    // Comparison operators (full set for consistency)
    constexpr bool operator==(const odds_ratio& other) const {
        return odds_ == other.odds_;
    }
    
    constexpr bool operator!=(const odds_ratio& other) const {
        return !(*this == other);
    }
    
    constexpr bool operator<(const odds_ratio& other) const {
        return odds_ < other.odds_;
    }
    
    constexpr bool operator<=(const odds_ratio& other) const {
        return odds_ <= other.odds_;
    }
    
    constexpr bool operator>(const odds_ratio& other) const {
        return odds_ > other.odds_;
    }
    
    constexpr bool operator>=(const odds_ratio& other) const {
        return odds_ >= other.odds_;
    }
    
    /// @brief Stream output operator
    /// @param os Output stream
    /// @param o Odds ratio to output
    /// @return Modified output stream
    /// @note Format: "X:1" where X is the odds value
    friend std::ostream& operator<<(std::ostream& os, const odds_ratio& o) {
        return os << o.odds_ << ":1";
    }
};

/**
 * @class log_odds
 * @brief Log-odds representation for numerical stability with extreme probabilities
 * @tparam T Underlying floating-point type
 * 
 * @details
 * The log_odds class combines logarithmic and odds-ratio transforms for
 * numerical stability when working with very small or very large probabilities.
 * This is essential for:
 * - Deep learning (prevents underflow in deep networks)
 * - Sequential inference with many updates
 * - Extreme probability calculations
 * 
 * Mathematical relationship:
 * - log_odds = log(p/(1-p)) = log(p) - log(1-p)
 * - Also known as the logit function
 * 
 * <b>Example Usage:</b>
 * @code
 * // Very small probability that would underflow
 * auto prior = log_odds<double>::from_probability(1e-100);
 * 
 * // Sequential updates in log space (addition instead of multiplication)
 * for(int i = 0; i < 1000; ++i) {
 *     prior = prior + log_odds<double>(0.1);  // Add log likelihood ratio
 * }
 * 
 * // Convert back (may still be extreme)
 * double prob = prior.to_probability();
 * @endcode
 */
template<typename T>
class log_odds {
    static_assert(std::is_floating_point_v<T>, "log_odds requires floating-point type");
    
private:
    T log_odds_;  ///< Natural logarithm of odds
    
public:
    /// @brief Default constructor - log(1) = 0 (equal odds)
    constexpr log_odds() : log_odds_(0) {}
    
    /// @brief Construct from log-odds value
    /// @param log_val Natural logarithm of odds
    explicit constexpr log_odds(T log_val) : log_odds_(log_val) {}
    
    /// @brief Factory method to create from probability
    /// @param prob Probability in [0,1]
    /// @return log_odds instance
    /// @note Handles edge cases: prob=0 → -∞, prob=1 → +∞
    /// @details Uses numerically stable formulation
    static constexpr log_odds from_probability(T prob) {
        if (prob <= 0) return log_odds(-std::numeric_limits<T>::infinity());
        if (prob >= 1) return log_odds(std::numeric_limits<T>::infinity());
        return log_odds(std::log(prob / (1 - prob)));
    }
    
    /// @brief Factory method to create from odds
    /// @param odds Odds value (not in log space)
    /// @return log_odds instance
    static constexpr log_odds from_odds(T odds) {
        return log_odds(std::log(odds));
    }
    
    /// @brief Convert to probability using sigmoid function
    /// @return Probability in [0,1]
    /// @note Uses numerically stable sigmoid: 1/(1+e^(-x))
    /// @warning May return 0 or 1 for extreme log-odds values
    constexpr T to_probability() const {
        if (log_odds_ > 0) {
            T exp_neg = std::exp(-log_odds_);
            return 1 / (1 + exp_neg);
        } else {
            T exp_val = std::exp(log_odds_);
            return exp_val / (1 + exp_val);
        }
    }
    
    /// @brief Get the raw log-odds value
    /// @return Natural logarithm of odds
    constexpr T value() const { return log_odds_; }
    
    /// @brief Bayesian update via addition (multiplication in odds space)
    /// @param log_lr Log of the likelihood ratio
    /// @return Updated posterior log-odds
    /// @details In log space: log(posterior_odds) = log(prior_odds) + log(LR)
    /// @note This avoids numerical issues with very small/large probabilities
    constexpr log_odds operator+(const log_odds& log_lr) const {
        return log_odds(log_odds_ + log_lr.log_odds_);
    }
    
    /// @brief Inverse update via subtraction
    /// @param other Log-odds to subtract
    /// @return Difference of log-odds
    /// @note Equivalent to division in odds space
    constexpr log_odds operator-(const log_odds& other) const {
        return log_odds(log_odds_ - other.log_odds_);
    }
    
    friend std::ostream& operator<<(std::ostream& os, const log_odds& lo) {
        return os << "log_odds(" << lo.log_odds_ << ")";
    }
};

} // namespace mutatio