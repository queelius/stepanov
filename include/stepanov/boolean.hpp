#pragma once

#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <optional>
#include <algorithm>
#include <sstream>
#include <bitset>
#include <queue>
#include <stack>
#include <set>

namespace stepanov::boolean {

// Forward declarations
class expression;
using expr_ptr = std::shared_ptr<expression>;

// =============================================================================
// Boolean Expression AST
// =============================================================================

// Base expression node
class expression {
public:
    virtual ~expression() = default;

    // Core operations
    virtual bool evaluate(const std::unordered_map<std::string, bool>& vars) const = 0;
    virtual expr_ptr simplify() const = 0;
    virtual expr_ptr clone() const = 0;
    virtual std::string to_string() const = 0;

    // Analysis
    virtual int complexity() const = 0;
    virtual std::vector<std::string> variables() const = 0;
    virtual bool is_constant() const { return false; }

    // Transformations
    virtual expr_ptr to_nnf() const = 0;  // Negation normal form
    virtual expr_ptr to_cnf() const = 0;  // Conjunctive normal form
    virtual expr_ptr to_dnf() const = 0;  // Disjunctive normal form
};

// Variable node
class variable : public expression {
    std::string name_;

public:
    explicit variable(std::string name) : name_(std::move(name)) {}

    bool evaluate(const std::unordered_map<std::string, bool>& vars) const override {
        auto it = vars.find(name_);
        if (it == vars.end()) {
            throw std::runtime_error("Undefined variable: " + name_);
        }
        return it->second;
    }

    expr_ptr simplify() const override {
        return clone();
    }

    expr_ptr clone() const override {
        return std::make_shared<variable>(name_);
    }

    std::string to_string() const override {
        return name_;
    }

    int complexity() const override { return 1; }

    std::vector<std::string> variables() const override {
        return {name_};
    }

    expr_ptr to_nnf() const override { return clone(); }
    expr_ptr to_cnf() const override { return clone(); }
    expr_ptr to_dnf() const override { return clone(); }

    const std::string& name() const { return name_; }
};

// Constant node
class constant : public expression {
    bool value_;

public:
    explicit constant(bool value) : value_(value) {}

    bool evaluate(const std::unordered_map<std::string, bool>&) const override {
        return value_;
    }

    expr_ptr simplify() const override {
        return clone();
    }

    expr_ptr clone() const override {
        return std::make_shared<constant>(value_);
    }

    std::string to_string() const override {
        return value_ ? "true" : "false";
    }

    int complexity() const override { return 1; }

    std::vector<std::string> variables() const override {
        return {};
    }

    bool is_constant() const override { return true; }

    expr_ptr to_nnf() const override { return clone(); }
    expr_ptr to_cnf() const override { return clone(); }
    expr_ptr to_dnf() const override { return clone(); }

    bool value() const { return value_; }
};

// NOT expression
class not_expr : public expression {
    expr_ptr operand_;

public:
    explicit not_expr(expr_ptr operand) : operand_(std::move(operand)) {}

    bool evaluate(const std::unordered_map<std::string, bool>& vars) const override {
        return !operand_->evaluate(vars);
    }

    expr_ptr simplify() const override;

    expr_ptr clone() const override {
        return std::make_shared<not_expr>(operand_->clone());
    }

    std::string to_string() const override {
        return "¬(" + operand_->to_string() + ")";
    }

    int complexity() const override {
        return 1 + operand_->complexity();
    }

    std::vector<std::string> variables() const override {
        return operand_->variables();
    }

    expr_ptr to_nnf() const override;
    expr_ptr to_cnf() const override;
    expr_ptr to_dnf() const override;

    expr_ptr operand() const { return operand_; }
};

// AND expression
class and_expr : public expression {
    expr_ptr left_;
    expr_ptr right_;

public:
    and_expr(expr_ptr left, expr_ptr right)
        : left_(std::move(left)), right_(std::move(right)) {}

    bool evaluate(const std::unordered_map<std::string, bool>& vars) const override {
        return left_->evaluate(vars) && right_->evaluate(vars);
    }

    expr_ptr simplify() const override;

    expr_ptr clone() const override {
        return std::make_shared<and_expr>(left_->clone(), right_->clone());
    }

    std::string to_string() const override {
        return "(" + left_->to_string() + " ∧ " + right_->to_string() + ")";
    }

    int complexity() const override {
        return 1 + left_->complexity() + right_->complexity();
    }

    std::vector<std::string> variables() const override {
        auto left_vars = left_->variables();
        auto right_vars = right_->variables();
        left_vars.insert(left_vars.end(), right_vars.begin(), right_vars.end());
        std::sort(left_vars.begin(), left_vars.end());
        left_vars.erase(std::unique(left_vars.begin(), left_vars.end()), left_vars.end());
        return left_vars;
    }

    expr_ptr to_nnf() const override;
    expr_ptr to_cnf() const override;
    expr_ptr to_dnf() const override;

    expr_ptr left() const { return left_; }
    expr_ptr right() const { return right_; }
};

// OR expression
class or_expr : public expression {
    expr_ptr left_;
    expr_ptr right_;

public:
    or_expr(expr_ptr left, expr_ptr right)
        : left_(std::move(left)), right_(std::move(right)) {}

    bool evaluate(const std::unordered_map<std::string, bool>& vars) const override {
        return left_->evaluate(vars) || right_->evaluate(vars);
    }

    expr_ptr simplify() const override;

    expr_ptr clone() const override {
        return std::make_shared<or_expr>(left_->clone(), right_->clone());
    }

    std::string to_string() const override {
        return "(" + left_->to_string() + " ∨ " + right_->to_string() + ")";
    }

    int complexity() const override {
        return 1 + left_->complexity() + right_->complexity();
    }

    std::vector<std::string> variables() const override {
        auto left_vars = left_->variables();
        auto right_vars = right_->variables();
        left_vars.insert(left_vars.end(), right_vars.begin(), right_vars.end());
        std::sort(left_vars.begin(), left_vars.end());
        left_vars.erase(std::unique(left_vars.begin(), left_vars.end()), left_vars.end());
        return left_vars;
    }

    expr_ptr to_nnf() const override;
    expr_ptr to_cnf() const override;
    expr_ptr to_dnf() const override;

    expr_ptr left() const { return left_; }
    expr_ptr right() const { return right_; }
};

// XOR expression
class xor_expr : public expression {
    expr_ptr left_;
    expr_ptr right_;

public:
    xor_expr(expr_ptr left, expr_ptr right)
        : left_(std::move(left)), right_(std::move(right)) {}

    bool evaluate(const std::unordered_map<std::string, bool>& vars) const override {
        return left_->evaluate(vars) != right_->evaluate(vars);
    }

    expr_ptr simplify() const override;

    expr_ptr clone() const override {
        return std::make_shared<xor_expr>(left_->clone(), right_->clone());
    }

    std::string to_string() const override {
        return "(" + left_->to_string() + " ⊕ " + right_->to_string() + ")";
    }

    int complexity() const override {
        return 1 + left_->complexity() + right_->complexity();
    }

    std::vector<std::string> variables() const override {
        auto left_vars = left_->variables();
        auto right_vars = right_->variables();
        left_vars.insert(left_vars.end(), right_vars.begin(), right_vars.end());
        std::sort(left_vars.begin(), left_vars.end());
        left_vars.erase(std::unique(left_vars.begin(), left_vars.end()), left_vars.end());
        return left_vars;
    }

    expr_ptr to_nnf() const override;
    expr_ptr to_cnf() const override;
    expr_ptr to_dnf() const override;

    expr_ptr left() const { return left_; }
    expr_ptr right() const { return right_; }
};

// IMPLIES expression
class implies_expr : public expression {
    expr_ptr left_;
    expr_ptr right_;

public:
    implies_expr(expr_ptr left, expr_ptr right)
        : left_(std::move(left)), right_(std::move(right)) {}

    bool evaluate(const std::unordered_map<std::string, bool>& vars) const override {
        return !left_->evaluate(vars) || right_->evaluate(vars);
    }

    expr_ptr simplify() const override;

    expr_ptr clone() const override {
        return std::make_shared<implies_expr>(left_->clone(), right_->clone());
    }

    std::string to_string() const override {
        return "(" + left_->to_string() + " → " + right_->to_string() + ")";
    }

    int complexity() const override {
        return 1 + left_->complexity() + right_->complexity();
    }

    std::vector<std::string> variables() const override {
        auto left_vars = left_->variables();
        auto right_vars = right_->variables();
        left_vars.insert(left_vars.end(), right_vars.begin(), right_vars.end());
        std::sort(left_vars.begin(), left_vars.end());
        left_vars.erase(std::unique(left_vars.begin(), left_vars.end()), left_vars.end());
        return left_vars;
    }

    expr_ptr to_nnf() const override;
    expr_ptr to_cnf() const override;
    expr_ptr to_dnf() const override;

    expr_ptr left() const { return left_; }
    expr_ptr right() const { return right_; }
};

// =============================================================================
// Factory functions for building expressions
// =============================================================================

inline expr_ptr var(const std::string& name) {
    return std::make_shared<variable>(name);
}

inline expr_ptr const_(bool value) {
    return std::make_shared<constant>(value);
}

inline expr_ptr true_() {
    return std::make_shared<constant>(true);
}

inline expr_ptr false_() {
    return std::make_shared<constant>(false);
}

inline expr_ptr not_(expr_ptr e) {
    return std::make_shared<not_expr>(std::move(e));
}

inline expr_ptr and_(expr_ptr left, expr_ptr right) {
    return std::make_shared<and_expr>(std::move(left), std::move(right));
}

inline expr_ptr or_(expr_ptr left, expr_ptr right) {
    return std::make_shared<or_expr>(std::move(left), std::move(right));
}

inline expr_ptr xor_(expr_ptr left, expr_ptr right) {
    return std::make_shared<xor_expr>(std::move(left), std::move(right));
}

inline expr_ptr implies(expr_ptr left, expr_ptr right) {
    return std::make_shared<implies_expr>(std::move(left), std::move(right));
}

inline expr_ptr iff(expr_ptr left, expr_ptr right) {
    // A ↔ B ≡ (A → B) ∧ (B → A)
    return and_(implies(left, right), implies(right, left));
}

// Variadic AND
template<typename... Args>
expr_ptr and_all(expr_ptr first, Args... rest) {
    if constexpr (sizeof...(rest) == 0) {
        return first;
    } else {
        return and_(std::move(first), and_all(std::move(rest)...));
    }
}

// Variadic OR
template<typename... Args>
expr_ptr or_all(expr_ptr first, Args... rest) {
    if constexpr (sizeof...(rest) == 0) {
        return first;
    } else {
        return or_(std::move(first), or_all(std::move(rest)...));
    }
}

// =============================================================================
// Simplification rules implementation
// =============================================================================

inline expr_ptr not_expr::simplify() const {
    auto simplified = operand_->simplify();

    // Double negation: ¬¬A → A
    if (auto not_op = std::dynamic_pointer_cast<not_expr>(simplified)) {
        return not_op->operand()->simplify();
    }

    // ¬true → false, ¬false → true
    if (auto const_op = std::dynamic_pointer_cast<constant>(simplified)) {
        return const_(!const_op->value());
    }

    // De Morgan's laws
    if (auto and_op = std::dynamic_pointer_cast<and_expr>(simplified)) {
        // ¬(A ∧ B) → ¬A ∨ ¬B
        return or_(not_(and_op->left()), not_(and_op->right()))->simplify();
    }

    if (auto or_op = std::dynamic_pointer_cast<or_expr>(simplified)) {
        // ¬(A ∨ B) → ¬A ∧ ¬B
        return and_(not_(or_op->left()), not_(or_op->right()))->simplify();
    }

    return not_(simplified);
}

inline expr_ptr and_expr::simplify() const {
    auto left_simplified = left_->simplify();
    auto right_simplified = right_->simplify();

    // Identity: A ∧ true → A
    if (auto right_const = std::dynamic_pointer_cast<constant>(right_simplified)) {
        if (right_const->value()) return left_simplified;
        // Domination: A ∧ false → false
        return false_();
    }

    if (auto left_const = std::dynamic_pointer_cast<constant>(left_simplified)) {
        if (left_const->value()) return right_simplified;
        return false_();
    }

    // Idempotent: A ∧ A → A
    if (left_simplified->to_string() == right_simplified->to_string()) {
        return left_simplified;
    }

    // Contradiction: A ∧ ¬A → false
    if (auto right_not = std::dynamic_pointer_cast<not_expr>(right_simplified)) {
        if (left_simplified->to_string() == right_not->operand()->to_string()) {
            return false_();
        }
    }

    if (auto left_not = std::dynamic_pointer_cast<not_expr>(left_simplified)) {
        if (right_simplified->to_string() == left_not->operand()->to_string()) {
            return false_();
        }
    }

    return and_(left_simplified, right_simplified);
}

inline expr_ptr or_expr::simplify() const {
    auto left_simplified = left_->simplify();
    auto right_simplified = right_->simplify();

    // Identity: A ∨ false → A
    if (auto right_const = std::dynamic_pointer_cast<constant>(right_simplified)) {
        if (!right_const->value()) return left_simplified;
        // Domination: A ∨ true → true
        return true_();
    }

    if (auto left_const = std::dynamic_pointer_cast<constant>(left_simplified)) {
        if (!left_const->value()) return right_simplified;
        return true_();
    }

    // Idempotent: A ∨ A → A
    if (left_simplified->to_string() == right_simplified->to_string()) {
        return left_simplified;
    }

    // Tautology: A ∨ ¬A → true
    if (auto right_not = std::dynamic_pointer_cast<not_expr>(right_simplified)) {
        if (left_simplified->to_string() == right_not->operand()->to_string()) {
            return true_();
        }
    }

    if (auto left_not = std::dynamic_pointer_cast<not_expr>(left_simplified)) {
        if (right_simplified->to_string() == left_not->operand()->to_string()) {
            return true_();
        }
    }

    return or_(left_simplified, right_simplified);
}

inline expr_ptr xor_expr::simplify() const {
    auto left_simplified = left_->simplify();
    auto right_simplified = right_->simplify();

    // A ⊕ false → A
    if (auto right_const = std::dynamic_pointer_cast<constant>(right_simplified)) {
        if (!right_const->value()) return left_simplified;
        // A ⊕ true → ¬A
        return not_(left_simplified)->simplify();
    }

    if (auto left_const = std::dynamic_pointer_cast<constant>(left_simplified)) {
        if (!left_const->value()) return right_simplified;
        return not_(right_simplified)->simplify();
    }

    // A ⊕ A → false
    if (left_simplified->to_string() == right_simplified->to_string()) {
        return false_();
    }

    return xor_(left_simplified, right_simplified);
}

inline expr_ptr implies_expr::simplify() const {
    auto left_simplified = left_->simplify();
    auto right_simplified = right_->simplify();

    // false → B → true
    if (auto left_const = std::dynamic_pointer_cast<constant>(left_simplified)) {
        if (!left_const->value()) return true_();
        // true → B → B
        return right_simplified;
    }

    // A → true → true
    if (auto right_const = std::dynamic_pointer_cast<constant>(right_simplified)) {
        if (right_const->value()) return true_();
        // A → false → ¬A
        return not_(left_simplified)->simplify();
    }

    // A → A → true
    if (left_simplified->to_string() == right_simplified->to_string()) {
        return true_();
    }

    // Convert to ¬A ∨ B and simplify
    return or_(not_(left_simplified), right_simplified)->simplify();
}

// =============================================================================
// Normal form conversions
// =============================================================================

inline expr_ptr not_expr::to_nnf() const {
    // Push negation down using De Morgan's laws
    if (auto not_op = std::dynamic_pointer_cast<not_expr>(operand_)) {
        return not_op->operand()->to_nnf();
    }

    if (auto and_op = std::dynamic_pointer_cast<and_expr>(operand_)) {
        return or_(not_(and_op->left())->to_nnf(), not_(and_op->right())->to_nnf());
    }

    if (auto or_op = std::dynamic_pointer_cast<or_expr>(operand_)) {
        return and_(not_(or_op->left())->to_nnf(), not_(or_op->right())->to_nnf());
    }

    if (auto implies_op = std::dynamic_pointer_cast<implies_expr>(operand_)) {
        return and_(implies_op->left()->to_nnf(), not_(implies_op->right())->to_nnf());
    }

    if (auto xor_op = std::dynamic_pointer_cast<xor_expr>(operand_)) {
        return iff(xor_op->left(), xor_op->right())->to_nnf();
    }

    return not_(operand_->to_nnf());
}

inline expr_ptr and_expr::to_nnf() const {
    return and_(left_->to_nnf(), right_->to_nnf());
}

inline expr_ptr or_expr::to_nnf() const {
    return or_(left_->to_nnf(), right_->to_nnf());
}

inline expr_ptr xor_expr::to_nnf() const {
    // A ⊕ B → (A ∧ ¬B) ∨ (¬A ∧ B)
    return or_(and_(left_->to_nnf(), not_(right_)->to_nnf()),
               and_(not_(left_)->to_nnf(), right_->to_nnf()));
}

inline expr_ptr implies_expr::to_nnf() const {
    // A → B → ¬A ∨ B
    return or_(not_(left_)->to_nnf(), right_->to_nnf());
}

// CNF and DNF conversions would require more complex algorithms
// These are placeholder implementations
inline expr_ptr not_expr::to_cnf() const { return to_nnf(); }
inline expr_ptr and_expr::to_cnf() const { return and_(left_->to_cnf(), right_->to_cnf()); }
inline expr_ptr or_expr::to_cnf() const { return or_(left_->to_cnf(), right_->to_cnf()); }
inline expr_ptr xor_expr::to_cnf() const { return to_nnf()->to_cnf(); }
inline expr_ptr implies_expr::to_cnf() const { return to_nnf()->to_cnf(); }

inline expr_ptr not_expr::to_dnf() const { return to_nnf(); }
inline expr_ptr and_expr::to_dnf() const { return and_(left_->to_dnf(), right_->to_dnf()); }
inline expr_ptr or_expr::to_dnf() const { return or_(left_->to_dnf(), right_->to_dnf()); }
inline expr_ptr xor_expr::to_dnf() const { return to_nnf()->to_dnf(); }
inline expr_ptr implies_expr::to_dnf() const { return to_nnf()->to_dnf(); }

// =============================================================================
// SAT Solver using DPLL algorithm
// =============================================================================

class sat_solver {
public:
    struct assignment {
        std::unordered_map<std::string, bool> values;
        bool satisfiable;

        operator bool() const { return satisfiable; }
    };

private:
    expr_ptr formula_;
    std::vector<std::string> variables_;

    // Unit propagation
    std::optional<std::pair<std::string, bool>> find_unit_clause(
        const expr_ptr& formula,
        const std::unordered_map<std::string, bool>& assignment) {

        // Look for clauses with only one unassigned variable
        // This is a simplified version - full implementation would need CNF
        return std::nullopt;
    }

    // Pure literal elimination
    std::optional<std::pair<std::string, bool>> find_pure_literal(
        const expr_ptr& formula,
        const std::unordered_map<std::string, bool>& assignment) {

        // Look for variables that appear with only one polarity
        // This is a simplified version
        return std::nullopt;
    }

    // DPLL recursive solver
    bool dpll(std::unordered_map<std::string, bool>& assignment) {
        // Evaluate with current assignment
        try {
            bool result = formula_->evaluate(assignment);
            if (assignment.size() == variables_.size()) {
                return result;
            }
        } catch (...) {
            // Some variables not assigned yet
        }

        // Find next unassigned variable
        std::string next_var;
        for (const auto& var : variables_) {
            if (assignment.find(var) == assignment.end()) {
                next_var = var;
                break;
            }
        }

        if (next_var.empty()) {
            return false;  // No more variables to assign
        }

        // Try true assignment
        assignment[next_var] = true;
        if (dpll(assignment)) {
            return true;
        }

        // Try false assignment
        assignment[next_var] = false;
        if (dpll(assignment)) {
            return true;
        }

        // Backtrack
        assignment.erase(next_var);
        return false;
    }

public:
    explicit sat_solver(expr_ptr formula)
        : formula_(std::move(formula)) {
        variables_ = formula_->variables();
    }

    assignment solve() {
        assignment result;
        std::unordered_map<std::string, bool> values;

        result.satisfiable = dpll(values);
        result.values = std::move(values);

        return result;
    }

    // Check if formula is satisfiable
    bool is_satisfiable() {
        return solve().satisfiable;
    }

    // Check if formula is a tautology
    bool is_tautology() {
        auto negated = not_(formula_);
        sat_solver neg_solver(negated);
        return !neg_solver.is_satisfiable();
    }

    // Get all satisfying assignments
    std::vector<std::unordered_map<std::string, bool>> all_solutions() {
        std::vector<std::unordered_map<std::string, bool>> solutions;

        // Generate all possible assignments (2^n)
        size_t n = variables_.size();
        size_t total = 1ULL << n;

        for (size_t i = 0; i < total; ++i) {
            std::unordered_map<std::string, bool> assignment;
            for (size_t j = 0; j < n; ++j) {
                assignment[variables_[j]] = (i >> j) & 1;
            }

            if (formula_->evaluate(assignment)) {
                solutions.push_back(assignment);
            }
        }

        return solutions;
    }
};

// =============================================================================
// Truth table generation
// =============================================================================

class truth_table {
private:
    expr_ptr formula_;
    std::vector<std::string> variables_;
    std::vector<std::vector<bool>> table_;

public:
    explicit truth_table(expr_ptr formula)
        : formula_(std::move(formula)) {
        variables_ = formula_->variables();
        generate();
    }

    void generate() {
        size_t n = variables_.size();
        size_t rows = 1ULL << n;

        table_.clear();
        table_.reserve(rows);

        for (size_t i = 0; i < rows; ++i) {
            std::vector<bool> row;
            row.reserve(n + 1);

            std::unordered_map<std::string, bool> assignment;
            for (size_t j = 0; j < n; ++j) {
                bool value = (i >> (n - 1 - j)) & 1;
                row.push_back(value);
                assignment[variables_[j]] = value;
            }

            row.push_back(formula_->evaluate(assignment));
            table_.push_back(row);
        }
    }

    std::string to_string() const {
        std::ostringstream oss;

        // Header
        for (const auto& var : variables_) {
            oss << var << " ";
        }
        oss << "| Result\n";

        // Separator
        for (size_t i = 0; i < variables_.size(); ++i) {
            for (size_t j = 0; j < variables_[i].length(); ++j) {
                oss << "-";
            }
            oss << " ";
        }
        oss << "| ------\n";

        // Rows
        for (const auto& row : table_) {
            for (size_t i = 0; i < variables_.size(); ++i) {
                // Align with variable name width
                std::string val = row[i] ? "T" : "F";
                oss << val;
                for (size_t j = 1; j < variables_[i].length(); ++j) {
                    oss << " ";
                }
                oss << " ";
            }
            oss << "| " << (row.back() ? "T" : "F") << "\n";
        }

        return oss.str();
    }

    const std::vector<std::vector<bool>>& data() const { return table_; }
    const std::vector<std::string>& variable_names() const { return variables_; }
};

// =============================================================================
// Karnaugh map for minimization (2-4 variables)
// =============================================================================

class karnaugh_map {
private:
    expr_ptr formula_;
    std::vector<std::string> variables_;
    std::vector<std::vector<bool>> map_;

    // Gray code generation
    std::vector<std::string> gray_code(int n) {
        if (n == 0) return {""};
        if (n == 1) return {"0", "1"};

        auto prev = gray_code(n - 1);
        std::vector<std::string> result;

        for (const auto& s : prev) {
            result.push_back("0" + s);
        }
        for (auto it = prev.rbegin(); it != prev.rend(); ++it) {
            result.push_back("1" + *it);
        }

        return result;
    }

public:
    explicit karnaugh_map(expr_ptr formula)
        : formula_(std::move(formula)) {
        variables_ = formula_->variables();
        if (variables_.size() > 4) {
            throw std::runtime_error("Karnaugh map supports up to 4 variables");
        }
        generate();
    }

    void generate() {
        size_t n = variables_.size();

        if (n <= 2) {
            // 2x2 map
            map_.resize(2, std::vector<bool>(2));
            auto gray = gray_code(1);

            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    std::unordered_map<std::string, bool> assignment;
                    if (n >= 1) assignment[variables_[0]] = (gray[i][0] == '1');
                    if (n >= 2) assignment[variables_[1]] = (gray[j][0] == '1');

                    map_[i][j] = formula_->evaluate(assignment);
                }
            }
        } else if (n <= 4) {
            // 4x4 map
            map_.resize(4, std::vector<bool>(4));
            auto gray = gray_code(2);

            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    std::unordered_map<std::string, bool> assignment;
                    if (n >= 1) assignment[variables_[0]] = (gray[i][0] == '1');
                    if (n >= 2) assignment[variables_[1]] = (gray[i][1] == '1');
                    if (n >= 3) assignment[variables_[2]] = (gray[j][0] == '1');
                    if (n >= 4) assignment[variables_[3]] = (gray[j][1] == '1');

                    map_[i][j] = formula_->evaluate(assignment);
                }
            }
        }
    }

    std::string to_string() const {
        std::ostringstream oss;

        if (variables_.size() <= 2) {
            oss << "Karnaugh Map (2x2):\n";
            oss << "    ";
            for (int j = 0; j < 2; ++j) {
                oss << j << " ";
            }
            oss << "\n";

            for (int i = 0; i < 2; ++i) {
                oss << i << " | ";
                for (int j = 0; j < 2; ++j) {
                    oss << (map_[i][j] ? "1" : "0") << " ";
                }
                oss << "\n";
            }
        } else {
            oss << "Karnaugh Map (4x4):\n";
            // More complex formatting for 4x4
        }

        return oss.str();
    }

    // Find prime implicants and minimize
    expr_ptr minimize() {
        // Quine-McCluskey algorithm would go here
        // For now, return simplified version
        return formula_->simplify();
    }
};

// =============================================================================
// Binary Decision Diagrams (BDD) - Elegant compact representation
// =============================================================================

class bdd {
public:
    struct node {
        std::string var;          // Variable name (empty for terminal nodes)
        std::shared_ptr<node> low;  // Low (false) branch
        std::shared_ptr<node> high; // High (true) branch
        bool terminal_value;       // Value for terminal nodes
        size_t id;                 // Unique node ID for sharing

        bool is_terminal() const { return var.empty(); }

        node(bool value) : terminal_value(value), id(0) {}
        node(const std::string& v, std::shared_ptr<node> l, std::shared_ptr<node> h, size_t i)
            : var(v), low(l), high(h), terminal_value(false), id(i) {}
    };

    using node_ptr = std::shared_ptr<node>;

private:
    node_ptr root_;
    std::vector<std::string> var_order_;
    mutable std::unordered_map<std::string, node_ptr> unique_table_;
    mutable size_t next_id_ = 2; // 0 and 1 reserved for terminals

    // Terminal nodes (shared)
    static node_ptr true_node() {
        static auto n = std::make_shared<node>(true);
        n->id = 1;
        return n;
    }

    static node_ptr false_node() {
        static auto n = std::make_shared<node>(false);
        n->id = 0;
        return n;
    }

    // Make node with uniqueness checking
    node_ptr make_node(const std::string& var, node_ptr low, node_ptr high) const {
        // Reduction: if both branches are same, return one
        if (low == high) return low;

        // Check unique table
        std::string key = var + "_" + std::to_string(low->id) + "_" + std::to_string(high->id);
        auto it = unique_table_.find(key);
        if (it != unique_table_.end()) {
            return it->second;
        }

        // Create new node
        auto new_node = std::make_shared<node>(var, low, high, next_id_++);
        unique_table_[key] = new_node;
        return new_node;
    }

    // Apply operation on two BDDs
    node_ptr apply(node_ptr f, node_ptr g,
                   std::function<bool(bool, bool)> op,
                   std::unordered_map<std::string, node_ptr>& cache) {

        // Terminal cases
        if (f->is_terminal() && g->is_terminal()) {
            return op(f->terminal_value, g->terminal_value) ? true_node() : false_node();
        }

        // Check cache
        std::string key = std::to_string(f->id) + "_" + std::to_string(g->id);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }

        // Determine top variable
        std::string top_var;
        if (f->is_terminal()) {
            top_var = g->var;
        } else if (g->is_terminal()) {
            top_var = f->var;
        } else {
            // Use variable ordering
            auto f_pos = std::find(var_order_.begin(), var_order_.end(), f->var);
            auto g_pos = std::find(var_order_.begin(), var_order_.end(), g->var);
            top_var = (f_pos <= g_pos) ? f->var : g->var;
        }

        // Shannon expansion
        node_ptr f_low = (f->is_terminal() || f->var != top_var) ? f : f->low;
        node_ptr f_high = (f->is_terminal() || f->var != top_var) ? f : f->high;
        node_ptr g_low = (g->is_terminal() || g->var != top_var) ? g : g->low;
        node_ptr g_high = (g->is_terminal() || g->var != top_var) ? g : g->high;

        node_ptr low = apply(f_low, g_low, op, cache);
        node_ptr high = apply(f_high, g_high, op, cache);

        auto result = make_node(top_var, low, high);
        cache[key] = result;
        return result;
    }

    // Convert expression to BDD
    node_ptr from_expression(expr_ptr expr, int var_index = 0) {
        if (var_index >= var_order_.size()) {
            // All variables assigned, evaluate
            std::unordered_map<std::string, bool> assignment;
            for (int i = 0; i < var_order_.size(); ++i) {
                assignment[var_order_[i]] = false; // Default assignment
            }
            return expr->evaluate(assignment) ? true_node() : false_node();
        }

        // Shannon expansion on current variable
        std::string var = var_order_[var_index];

        // Build restricted expressions
        auto expr_with_false = restrict(expr, var, false);
        auto expr_with_true = restrict(expr, var, true);

        node_ptr low = from_expression(expr_with_false, var_index + 1);
        node_ptr high = from_expression(expr_with_true, var_index + 1);

        return make_node(var, low, high);
    }

    // Restrict expression by setting variable to value
    expr_ptr restrict(expr_ptr expr, const std::string& var, bool value) {
        // This would ideally walk the expression tree and substitute
        // For now, simplified implementation
        return expr; // Placeholder
    }

public:
    bdd() : root_(false_node()) {}

    explicit bdd(expr_ptr expr) {
        var_order_ = expr->variables();
        std::sort(var_order_.begin(), var_order_.end());
        root_ = from_expression(expr);
    }

    // Boolean operations
    bdd operator&(const bdd& other) const {
        bdd result;
        result.var_order_ = merge_orders(var_order_, other.var_order_);
        std::unordered_map<std::string, node_ptr> cache;
        result.root_ = const_cast<bdd*>(this)->apply(
            root_, other.root_,
            [](bool a, bool b) { return a && b; },
            cache
        );
        return result;
    }

    bdd operator|(const bdd& other) const {
        bdd result;
        result.var_order_ = merge_orders(var_order_, other.var_order_);
        std::unordered_map<std::string, node_ptr> cache;
        result.root_ = const_cast<bdd*>(this)->apply(
            root_, other.root_,
            [](bool a, bool b) { return a || b; },
            cache
        );
        return result;
    }

    bdd operator^(const bdd& other) const {
        bdd result;
        result.var_order_ = merge_orders(var_order_, other.var_order_);
        std::unordered_map<std::string, node_ptr> cache;
        result.root_ = const_cast<bdd*>(this)->apply(
            root_, other.root_,
            [](bool a, bool b) { return a != b; },
            cache
        );
        return result;
    }

    bdd operator!() const {
        bdd result = *this;
        result.root_ = negate(root_);
        return result;
    }

    // Evaluate with assignment
    bool evaluate(const std::unordered_map<std::string, bool>& assignment) const {
        node_ptr current = root_;

        while (!current->is_terminal()) {
            auto it = assignment.find(current->var);
            if (it == assignment.end()) {
                throw std::runtime_error("Variable not assigned: " + current->var);
            }
            current = it->second ? current->high : current->low;
        }

        return current->terminal_value;
    }

    // Count satisfying assignments
    size_t sat_count() const {
        return sat_count_recursive(root_, 0);
    }

    // Get one satisfying assignment
    std::optional<std::unordered_map<std::string, bool>> sat_one() const {
        if (root_ == false_node()) return std::nullopt;

        std::unordered_map<std::string, bool> assignment;
        node_ptr current = root_;

        while (!current->is_terminal()) {
            if (current->high != false_node()) {
                assignment[current->var] = true;
                current = current->high;
            } else {
                assignment[current->var] = false;
                current = current->low;
            }
        }

        return assignment;
    }

    // Get all satisfying assignments
    std::vector<std::unordered_map<std::string, bool>> all_sat() const {
        std::vector<std::unordered_map<std::string, bool>> results;
        std::unordered_map<std::string, bool> current;
        all_sat_recursive(root_, current, results);
        return results;
    }

    // Size of BDD (number of nodes)
    size_t size() const {
        std::unordered_set<size_t> visited;
        return count_nodes(root_, visited);
    }

    // Export to DOT format for visualization
    std::string to_dot() const {
        std::ostringstream oss;
        oss << "digraph BDD {\n";
        oss << "  rankdir=TB;\n";

        std::unordered_set<size_t> visited;
        to_dot_recursive(root_, oss, visited);

        oss << "}\n";
        return oss.str();
    }

private:
    static std::vector<std::string> merge_orders(const std::vector<std::string>& o1,
                                                  const std::vector<std::string>& o2) {
        std::vector<std::string> result = o1;
        for (const auto& v : o2) {
            if (std::find(result.begin(), result.end(), v) == result.end()) {
                result.push_back(v);
            }
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    node_ptr negate(node_ptr n) const {
        if (n->is_terminal()) {
            return n->terminal_value ? false_node() : true_node();
        }
        return make_node(n->var, negate(n->low), negate(n->high));
    }

    size_t sat_count_recursive(node_ptr n, size_t depth) const {
        if (n == false_node()) return 0;
        if (n == true_node()) {
            return 1ULL << (var_order_.size() - depth);
        }

        size_t var_skip = 0;
        auto it = std::find(var_order_.begin(), var_order_.end(), n->var);
        if (it != var_order_.end()) {
            var_skip = std::distance(var_order_.begin(), it) - depth;
        }

        size_t low_count = sat_count_recursive(n->low, depth + var_skip + 1);
        size_t high_count = sat_count_recursive(n->high, depth + var_skip + 1);

        return (low_count + high_count) * (1ULL << var_skip);
    }

    void all_sat_recursive(node_ptr n,
                          std::unordered_map<std::string, bool>& current,
                          std::vector<std::unordered_map<std::string, bool>>& results) const {
        if (n == false_node()) return;
        if (n == true_node()) {
            results.push_back(current);
            return;
        }

        current[n->var] = false;
        all_sat_recursive(n->low, current, results);

        current[n->var] = true;
        all_sat_recursive(n->high, current, results);

        current.erase(n->var);
    }

    size_t count_nodes(node_ptr n, std::unordered_set<size_t>& visited) const {
        if (!n || visited.count(n->id)) return 0;
        visited.insert(n->id);

        if (n->is_terminal()) return 1;
        return 1 + count_nodes(n->low, visited) + count_nodes(n->high, visited);
    }

    void to_dot_recursive(node_ptr n, std::ostringstream& oss,
                         std::unordered_set<size_t>& visited) const {
        if (!n || visited.count(n->id)) return;
        visited.insert(n->id);

        if (n->is_terminal()) {
            oss << "  n" << n->id << " [shape=box,label=\""
                << (n->terminal_value ? "1" : "0") << "\"];\n";
        } else {
            oss << "  n" << n->id << " [label=\"" << n->var << "\"];\n";
            oss << "  n" << n->id << " -> n" << n->low->id << " [style=dashed];\n";
            oss << "  n" << n->id << " -> n" << n->high->id << ";\n";

            to_dot_recursive(n->low, oss, visited);
            to_dot_recursive(n->high, oss, visited);
        }
    }
};

// =============================================================================
// Tseitin Transformation - Efficient CNF conversion
// =============================================================================

class tseitin_transformer {
private:
    int next_aux_var_ = 1;
    std::vector<std::vector<int>> clauses_;
    std::unordered_map<std::string, int> var_map_;
    std::unordered_map<expr_ptr, int> subformula_vars_;

    int get_var_id(const std::string& name) {
        auto it = var_map_.find(name);
        if (it == var_map_.end()) {
            int id = var_map_.size() + 1;
            var_map_[name] = id;
            return id;
        }
        return it->second;
    }

    int get_aux_var() {
        return -(next_aux_var_++); // Negative for auxiliary variables
    }

    int transform_recursive(expr_ptr expr) {
        // Check if already processed
        auto it = subformula_vars_.find(expr);
        if (it != subformula_vars_.end()) {
            return it->second;
        }

        // Handle different expression types
        if (auto var = std::dynamic_pointer_cast<variable>(expr)) {
            int var_id = get_var_id(var->name());
            subformula_vars_[expr] = var_id;
            return var_id;
        }

        if (auto const_expr = std::dynamic_pointer_cast<constant>(expr)) {
            int aux = get_aux_var();
            if (const_expr->value()) {
                clauses_.push_back({aux}); // Force aux to be true
            } else {
                clauses_.push_back({-aux}); // Force aux to be false
            }
            subformula_vars_[expr] = aux;
            return aux;
        }

        if (auto not_expr = std::dynamic_pointer_cast<class not_expr>(expr)) {
            int child = transform_recursive(not_expr->operand());
            int aux = get_aux_var();

            // aux ↔ ¬child
            // (aux ∨ child) ∧ (¬aux ∨ ¬child)
            clauses_.push_back({aux, child});
            clauses_.push_back({-aux, -child});

            subformula_vars_[expr] = aux;
            return aux;
        }

        if (auto and_expr = std::dynamic_pointer_cast<class and_expr>(expr)) {
            int left = transform_recursive(and_expr->left());
            int right = transform_recursive(and_expr->right());
            int aux = get_aux_var();

            // aux ↔ (left ∧ right)
            // (¬aux ∨ left) ∧ (¬aux ∨ right) ∧ (aux ∨ ¬left ∨ ¬right)
            clauses_.push_back({-aux, left});
            clauses_.push_back({-aux, right});
            clauses_.push_back({aux, -left, -right});

            subformula_vars_[expr] = aux;
            return aux;
        }

        if (auto or_expr = std::dynamic_pointer_cast<class or_expr>(expr)) {
            int left = transform_recursive(or_expr->left());
            int right = transform_recursive(or_expr->right());
            int aux = get_aux_var();

            // aux ↔ (left ∨ right)
            // (aux ∨ ¬left) ∧ (aux ∨ ¬right) ∧ (¬aux ∨ left ∨ right)
            clauses_.push_back({aux, -left});
            clauses_.push_back({aux, -right});
            clauses_.push_back({-aux, left, right});

            subformula_vars_[expr] = aux;
            return aux;
        }

        if (auto xor_expr = std::dynamic_pointer_cast<class xor_expr>(expr)) {
            int left = transform_recursive(xor_expr->left());
            int right = transform_recursive(xor_expr->right());
            int aux = get_aux_var();

            // aux ↔ (left ⊕ right)
            // Convert XOR to CNF clauses
            clauses_.push_back({-aux, -left, -right});
            clauses_.push_back({-aux, left, right});
            clauses_.push_back({aux, -left, right});
            clauses_.push_back({aux, left, -right});

            subformula_vars_[expr] = aux;
            return aux;
        }

        // Default: treat as variable
        return get_var_id("unknown");
    }

public:
    struct cnf_formula {
        std::vector<std::vector<int>> clauses;
        std::unordered_map<std::string, int> variable_mapping;
        int num_variables;

        std::string to_dimacs() const {
            std::ostringstream oss;
            oss << "p cnf " << num_variables << " " << clauses.size() << "\n";

            for (const auto& clause : clauses) {
                for (int lit : clause) {
                    oss << lit << " ";
                }
                oss << "0\n";
            }

            return oss.str();
        }
    };

    cnf_formula transform(expr_ptr expr) {
        clauses_.clear();
        var_map_.clear();
        subformula_vars_.clear();
        next_aux_var_ = 1;

        // Transform the formula
        int root_var = transform_recursive(expr);

        // Assert the root formula is true
        clauses_.push_back({root_var});

        // Count total variables (original + auxiliary)
        int total_vars = var_map_.size() + next_aux_var_ - 1;

        cnf_formula result;
        result.clauses = clauses_;
        result.variable_mapping = var_map_;
        result.num_variables = total_vars;

        return result;
    }
};

// =============================================================================
// CDCL SAT Solver - Conflict-Driven Clause Learning
// =============================================================================

class cdcl_solver {
public:
    enum class literal_value { UNASSIGNED, TRUE, FALSE };

    struct clause {
        std::vector<int> literals;
        bool is_learned;
        double activity;

        clause(std::vector<int> lits, bool learned = false)
            : literals(std::move(lits)), is_learned(learned), activity(0.0) {}

        bool is_unit(const std::vector<literal_value>& assignment) const {
            int unassigned_count = 0;
            for (int lit : literals) {
                int var = std::abs(lit);
                if (assignment[var] == literal_value::UNASSIGNED) {
                    unassigned_count++;
                    if (unassigned_count > 1) return false;
                }
            }
            return unassigned_count == 1;
        }

        bool is_satisfied(const std::vector<literal_value>& assignment) const {
            for (int lit : literals) {
                int var = std::abs(lit);
                bool positive = lit > 0;
                if ((positive && assignment[var] == literal_value::TRUE) ||
                    (!positive && assignment[var] == literal_value::FALSE)) {
                    return true;
                }
            }
            return false;
        }

        std::optional<int> get_unit_literal(const std::vector<literal_value>& assignment) const {
            for (int lit : literals) {
                int var = std::abs(lit);
                if (assignment[var] == literal_value::UNASSIGNED) {
                    return lit;
                }
            }
            return std::nullopt;
        }
    };

private:
    std::vector<clause> clauses_;
    std::vector<literal_value> assignment_;
    std::vector<int> decision_level_;
    std::vector<int> antecedent_clause_;
    std::vector<double> var_activity_;
    std::stack<int> trail_;
    std::stack<int> trail_lim_;
    int num_vars_;
    int current_level_;
    double decay_factor_;
    double var_inc_;

    // Variable selection heuristic (VSIDS)
    int select_variable() {
        int best_var = 0;
        double best_activity = -1.0;

        for (int v = 1; v <= num_vars_; ++v) {
            if (assignment_[v] == literal_value::UNASSIGNED &&
                var_activity_[v] > best_activity) {
                best_activity = var_activity_[v];
                best_var = v;
            }
        }

        return best_var;
    }

    // Unit propagation with conflict detection
    std::optional<int> unit_propagate() {
        while (true) {
            bool found_unit = false;

            for (size_t i = 0; i < clauses_.size(); ++i) {
                if (clauses_[i].is_satisfied(assignment_)) continue;

                if (clauses_[i].is_unit(assignment_)) {
                    auto unit_lit = clauses_[i].get_unit_literal(assignment_);
                    if (!unit_lit) continue;

                    found_unit = true;
                    int var = std::abs(*unit_lit);
                    bool value = *unit_lit > 0;

                    assignment_[var] = value ? literal_value::TRUE : literal_value::FALSE;
                    decision_level_[var] = current_level_;
                    antecedent_clause_[var] = i;
                    trail_.push(*unit_lit);

                    // Check for conflict
                    for (size_t j = 0; j < clauses_.size(); ++j) {
                        bool all_false = true;
                        for (int lit : clauses_[j].literals) {
                            int v = std::abs(lit);
                            bool pos = lit > 0;
                            if (assignment_[v] == literal_value::UNASSIGNED ||
                                (pos && assignment_[v] == literal_value::TRUE) ||
                                (!pos && assignment_[v] == literal_value::FALSE)) {
                                all_false = false;
                                break;
                            }
                        }
                        if (all_false) {
                            return j; // Return conflicting clause index
                        }
                    }
                }
            }

            if (!found_unit) break;
        }

        return std::nullopt; // No conflict
    }

    // Analyze conflict and learn clause
    std::pair<std::vector<int>, int> analyze_conflict(int conflict_clause) {
        std::set<int> seen;
        std::vector<int> learned_clause;
        int backtrack_level = 0;

        // Start with the conflicting clause
        for (int lit : clauses_[conflict_clause].literals) {
            int var = std::abs(lit);
            if (decision_level_[var] == current_level_) {
                seen.insert(var);
            } else if (decision_level_[var] > 0) {
                learned_clause.push_back(-lit);
                backtrack_level = std::max(backtrack_level, decision_level_[var]);
            }
        }

        // Walk back through implications
        while (!trail_.empty()) {
            int lit = trail_.top();
            int var = std::abs(lit);

            if (seen.count(var) > 0) {
                seen.erase(var);

                if (seen.empty()) {
                    // Found the first UIP (Unique Implication Point)
                    learned_clause.push_back(-lit);
                    break;
                }

                // Add antecedent clause literals
                if (antecedent_clause_[var] >= 0) {
                    for (int ante_lit : clauses_[antecedent_clause_[var]].literals) {
                        int ante_var = std::abs(ante_lit);
                        if (ante_var != var && decision_level_[ante_var] == current_level_) {
                            seen.insert(ante_var);
                        } else if (ante_var != var && decision_level_[ante_var] > 0) {
                            learned_clause.push_back(-ante_lit);
                            backtrack_level = std::max(backtrack_level, decision_level_[ante_var]);
                        }
                    }
                }
            }

            trail_.pop();
        }

        // Update variable activities (bump variables in conflict)
        for (int lit : learned_clause) {
            int var = std::abs(lit);
            var_activity_[var] += var_inc_;
        }

        // Decay activities
        var_inc_ *= (1.0 / decay_factor_);

        return {learned_clause, backtrack_level};
    }

    // Backtrack to given level
    void backtrack(int level) {
        while (current_level_ > level) {
            int lim = trail_lim_.top();
            trail_lim_.pop();

            while (trail_.size() > lim) {
                int lit = trail_.top();
                trail_.pop();
                int var = std::abs(lit);
                assignment_[var] = literal_value::UNASSIGNED;
                decision_level_[var] = -1;
                antecedent_clause_[var] = -1;
            }

            current_level_--;
        }
    }

public:
    explicit cdcl_solver(int num_vars)
        : num_vars_(num_vars),
          assignment_(num_vars + 1, literal_value::UNASSIGNED),
          decision_level_(num_vars + 1, -1),
          antecedent_clause_(num_vars + 1, -1),
          var_activity_(num_vars + 1, 0.0),
          current_level_(0),
          decay_factor_(0.95),
          var_inc_(1.0) {}

    void add_clause(const std::vector<int>& literals) {
        clauses_.emplace_back(literals);
    }

    bool solve() {
        while (true) {
            // Unit propagation
            auto conflict = unit_propagate();

            if (conflict) {
                // Conflict found
                if (current_level_ == 0) {
                    return false; // UNSAT
                }

                // Learn clause from conflict
                auto [learned_clause, backtrack_level] = analyze_conflict(*conflict);

                // Backtrack
                backtrack(backtrack_level);

                // Add learned clause
                clauses_.emplace_back(learned_clause, true);
            } else {
                // No conflict, try to make a decision
                int var = select_variable();

                if (var == 0) {
                    return true; // SAT - all variables assigned
                }

                // Make decision (try positive first)
                current_level_++;
                trail_lim_.push(trail_.size());
                assignment_[var] = literal_value::TRUE;
                decision_level_[var] = current_level_;
                antecedent_clause_[var] = -1; // Decision, not implication
                trail_.push(var);
            }
        }
    }

    std::unordered_map<std::string, bool> get_model(
        const std::unordered_map<std::string, int>& var_map) const {

        std::unordered_map<std::string, bool> model;

        for (const auto& [name, id] : var_map) {
            if (id > 0 && id <= num_vars_) {
                model[name] = (assignment_[id] == literal_value::TRUE);
            }
        }

        return model;
    }

    // Statistics
    size_t num_learned_clauses() const {
        return std::count_if(clauses_.begin(), clauses_.end(),
                            [](const clause& c) { return c.is_learned; });
    }

    size_t num_decisions() const {
        size_t count = 0;
        for (int i = 1; i <= num_vars_; ++i) {
            if (antecedent_clause_[i] == -1 && decision_level_[i] > 0) {
                count++;
            }
        }
        return count;
    }
};

// =============================================================================
// Boolean Function Synthesis from Examples
// =============================================================================

class function_synthesizer {
public:
    struct example {
        std::unordered_map<std::string, bool> inputs;
        bool output;
    };

private:
    std::vector<example> positive_examples_;
    std::vector<example> negative_examples_;
    std::vector<std::string> variables_;

    // Generate candidate terms for DNF
    std::vector<expr_ptr> generate_candidate_terms() {
        std::vector<expr_ptr> terms;

        // Generate all possible conjunctions up to a certain size
        int n = variables_.size();
        for (int mask = 1; mask < (1 << (2 * n)); ++mask) {
            expr_ptr term = nullptr;

            for (int i = 0; i < n; ++i) {
                int bits = (mask >> (2 * i)) & 3;
                if (bits == 0) continue; // Variable not in term

                expr_ptr var_expr = var(variables_[i]);
                if (bits == 2) { // Negated variable
                    var_expr = not_(var_expr);
                }

                if (term == nullptr) {
                    term = var_expr;
                } else {
                    term = and_(term, var_expr);
                }
            }

            if (term != nullptr) {
                terms.push_back(term);
            }
        }

        return terms;
    }

    // Check if term covers example
    bool covers(expr_ptr term, const example& ex) {
        return term->evaluate(ex.inputs) == ex.output;
    }

    // Find minimal DNF using greedy set cover
    expr_ptr find_minimal_dnf() {
        auto terms = generate_candidate_terms();
        std::vector<expr_ptr> selected_terms;

        // Track which positive examples are covered
        std::vector<bool> covered(positive_examples_.size(), false);

        while (std::any_of(covered.begin(), covered.end(),
                          [](bool c) { return !c; })) {

            expr_ptr best_term = nullptr;
            int best_score = -1;

            // Find term that covers most uncovered positive examples
            // without covering any negative examples
            for (const auto& term : terms) {
                bool covers_negative = false;
                for (const auto& neg : negative_examples_) {
                    if (term->evaluate(neg.inputs)) {
                        covers_negative = true;
                        break;
                    }
                }

                if (covers_negative) continue;

                int score = 0;
                for (size_t i = 0; i < positive_examples_.size(); ++i) {
                    if (!covered[i] && term->evaluate(positive_examples_[i].inputs)) {
                        score++;
                    }
                }

                if (score > best_score) {
                    best_score = score;
                    best_term = term;
                }
            }

            if (best_term == nullptr) {
                break; // No valid term found
            }

            selected_terms.push_back(best_term);

            // Mark covered examples
            for (size_t i = 0; i < positive_examples_.size(); ++i) {
                if (best_term->evaluate(positive_examples_[i].inputs)) {
                    covered[i] = true;
                }
            }
        }

        // Combine selected terms with OR
        if (selected_terms.empty()) {
            return false_();
        }

        expr_ptr result = selected_terms[0];
        for (size_t i = 1; i < selected_terms.size(); ++i) {
            result = or_(result, selected_terms[i]);
        }

        return result;
    }

    // Find using decision tree approach
    expr_ptr build_decision_tree(const std::vector<example>& examples,
                                 const std::set<std::string>& available_vars,
                                 int depth = 0) {

        if (depth > 10) { // Depth limit
            return false_();
        }

        // Check if all examples have same output
        if (!examples.empty()) {
            bool all_same = true;
            bool first_output = examples[0].output;
            for (const auto& ex : examples) {
                if (ex.output != first_output) {
                    all_same = false;
                    break;
                }
            }
            if (all_same) {
                return const_(first_output);
            }
        }

        if (available_vars.empty() || examples.empty()) {
            return false_();
        }

        // Choose best splitting variable (information gain)
        std::string best_var;
        double best_gain = -1.0;

        for (const auto& v : available_vars) {
            std::vector<example> true_branch, false_branch;

            for (const auto& ex : examples) {
                if (ex.inputs.at(v)) {
                    true_branch.push_back(ex);
                } else {
                    false_branch.push_back(ex);
                }
            }

            // Calculate information gain (simplified)
            double gain = static_cast<double>(true_branch.size() * false_branch.size()) /
                         (examples.size() * examples.size());

            if (gain > best_gain) {
                best_gain = gain;
                best_var = v;
            }
        }

        // Split on best variable
        std::vector<example> true_branch, false_branch;
        for (const auto& ex : examples) {
            if (ex.inputs.at(best_var)) {
                true_branch.push_back(ex);
            } else {
                false_branch.push_back(ex);
            }
        }

        auto new_vars = available_vars;
        new_vars.erase(best_var);

        auto true_subtree = build_decision_tree(true_branch, new_vars, depth + 1);
        auto false_subtree = build_decision_tree(false_branch, new_vars, depth + 1);

        // Build if-then-else expression
        return or_(and_(var(best_var), true_subtree),
                  and_(not_(var(best_var)), false_subtree));
    }

public:
    function_synthesizer() = default;

    void add_example(const std::unordered_map<std::string, bool>& inputs, bool output) {
        example ex{inputs, output};

        if (output) {
            positive_examples_.push_back(ex);
        } else {
            negative_examples_.push_back(ex);
        }

        // Update variables list
        for (const auto& [var_name, _] : inputs) {
            if (std::find(variables_.begin(), variables_.end(), var_name) ==
                variables_.end()) {
                variables_.push_back(var_name);
            }
        }
    }

    expr_ptr synthesize(const std::string& method = "dnf") {
        if (method == "dnf") {
            return find_minimal_dnf();
        } else if (method == "decision_tree") {
            std::vector<example> all_examples = positive_examples_;
            all_examples.insert(all_examples.end(),
                               negative_examples_.begin(),
                               negative_examples_.end());

            std::set<std::string> vars(variables_.begin(), variables_.end());
            return build_decision_tree(all_examples, vars)->simplify();
        } else if (method == "karnaugh") {
            // Use Karnaugh map for small problems
            if (variables_.size() <= 4) {
                // Build truth table expression
                expr_ptr result = false_();

                for (const auto& ex : positive_examples_) {
                    expr_ptr term = true_();
                    for (const auto& v : variables_) {
                        if (ex.inputs.at(v)) {
                            term = and_(term, var(v));
                        } else {
                            term = and_(term, not_(var(v)));
                        }
                    }
                    result = or_(result, term);
                }

                // Minimize using Karnaugh map
                karnaugh_map kmap(result);
                return kmap.minimize();
            }
        }

        return false_();
    }

    // Verify synthesized function against examples
    bool verify(expr_ptr function) {
        for (const auto& ex : positive_examples_) {
            if (!function->evaluate(ex.inputs)) {
                return false;
            }
        }

        for (const auto& ex : negative_examples_) {
            if (function->evaluate(ex.inputs)) {
                return false;
            }
        }

        return true;
    }

    // Generate counter-example if function is incorrect
    std::optional<example> find_counter_example(expr_ptr function) {
        for (const auto& ex : positive_examples_) {
            if (!function->evaluate(ex.inputs)) {
                return ex;
            }
        }

        for (const auto& ex : negative_examples_) {
            if (function->evaluate(ex.inputs)) {
                return ex;
            }
        }

        return std::nullopt;
    }
};

// =============================================================================
// Type erasure for boolean expressions
// =============================================================================

class any_boolean_expr {
private:
    expr_ptr expr_;

public:
    any_boolean_expr() = default;

    any_boolean_expr(expr_ptr e) : expr_(std::move(e)) {}

    template<typename T>
    any_boolean_expr(T&& e) : expr_(std::forward<T>(e)) {}

    bool evaluate(const std::unordered_map<std::string, bool>& vars) const {
        return expr_->evaluate(vars);
    }

    any_boolean_expr simplify() const {
        return expr_->simplify();
    }

    std::string to_string() const {
        return expr_ ? expr_->to_string() : "null";
    }

    int complexity() const {
        return expr_ ? expr_->complexity() : 0;
    }

    std::vector<std::string> variables() const {
        return expr_ ? expr_->variables() : std::vector<std::string>{};
    }

    // Conversion operators
    operator bool() const { return expr_ != nullptr; }

    expr_ptr get() const { return expr_; }
};

} // namespace stepanov::boolean