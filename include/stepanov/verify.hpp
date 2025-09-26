#ifndef STEPANOV_VERIFY_HPP
#define STEPANOV_VERIFY_HPP

#include <concepts>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <optional>
#include <variant>
#include <source_location>
#include <iostream>
#include <iomanip>

namespace stepanov::verify {

// Formal verification and property-based testing for Stepanov algorithms
// Inspired by Design by Contract, QuickCheck, and refinement types

// ============================================================================
// Design by Contract
// ============================================================================

// Contract violation exception
class contract_violation : public std::logic_error {
    std::source_location location;
    std::string expression;

public:
    contract_violation(const std::string& msg,
                      const std::string& expr,
                      const std::source_location& loc = std::source_location::current())
        : std::logic_error(msg), location(loc), expression(expr) {}

    std::string full_message() const {
        std::stringstream ss;
        ss << "Contract violation at " << location.file_name()
           << ":" << location.line() << ":" << location.column()
           << " in function '" << location.function_name() << "'\n"
           << "Expression: " << expression << "\n"
           << "Message: " << what();
        return ss.str();
    }
};

// Compile-time or runtime contract checking
#ifdef NDEBUG
    #define CONTRACT_MODE_RELEASE
#else
    #define CONTRACT_MODE_DEBUG
#endif

// Precondition: must be true at function entry
#ifdef CONTRACT_MODE_RELEASE
    #define REQUIRES(expr) ((void)0)
#else
    #define REQUIRES(expr) \
        do { \
            if (!(expr)) { \
                throw stepanov::verify::contract_violation( \
                    "Precondition failed", #expr); \
            } \
        } while (0)
#endif

// Postcondition: must be true at function exit
#ifdef CONTRACT_MODE_RELEASE
    #define ENSURES(expr) ((void)0)
#else
    #define ENSURES(expr) \
        do { \
            if (!(expr)) { \
                throw stepanov::verify::contract_violation( \
                    "Postcondition failed", #expr); \
            } \
        } while (0)
#endif

// Invariant: must be true throughout object lifetime
#ifdef CONTRACT_MODE_RELEASE
    #define INVARIANT(expr) ((void)0)
#else
    #define INVARIANT(expr) \
        do { \
            if (!(expr)) { \
                throw stepanov::verify::contract_violation( \
                    "Invariant violated", #expr); \
            } \
        } while (0)
#endif

// Assert with better error reporting
#ifdef CONTRACT_MODE_RELEASE
    #define ASSERT(expr) ((void)0)
#else
    #define ASSERT(expr) \
        do { \
            if (!(expr)) { \
                throw stepanov::verify::contract_violation( \
                    "Assertion failed", #expr); \
            } \
        } while (0)
#endif

// Contract class for RAII-based invariant checking
template<typename T>
class contract {
    T* object;
    std::function<bool(const T&)> invariant_check;
    bool checking_enabled;

public:
    contract(T* obj, std::function<bool(const T&)> inv)
        : object(obj), invariant_check(inv), checking_enabled(true) {
        check_invariant();
    }

    ~contract() {
        if (checking_enabled && !std::uncaught_exceptions()) {
            check_invariant();
        }
    }

    void disable() { checking_enabled = false; }
    void enable() { checking_enabled = true; }

private:
    void check_invariant() {
        if (checking_enabled && !invariant_check(*object)) {
            throw contract_violation("Class invariant violated", "invariant_check");
        }
    }
};

// Scope guard for ensuring postconditions
template<typename Func>
class scope_exit {
    Func func;
    bool active;

public:
    explicit scope_exit(Func f) : func(std::move(f)), active(true) {}

    ~scope_exit() {
        if (active) func();
    }

    void dismiss() { active = false; }
};

template<typename Func>
scope_exit(Func) -> scope_exit<Func>;

// ============================================================================
// Refinement Types
// ============================================================================

// Refined type: a type with a predicate
template<typename T, typename Predicate>
class refined {
    T value;
    Predicate pred;

    void check() const {
        if (!pred(value)) {
            throw contract_violation("Refinement predicate failed", "predicate(value)");
        }
    }

public:
    refined(T val, Predicate p = {}) : value(std::move(val)), pred(std::move(p)) {
        check();
    }

    refined& operator=(const T& val) {
        value = val;
        check();
        return *this;
    }

    const T& get() const { return value; }
    T& get() { return value; }

    operator const T&() const { return value; }

    // Preserve refinement through operations
    template<typename Op>
    auto map(Op op) const -> refined<decltype(op(value)), Predicate> {
        return refined<decltype(op(value)), Predicate>(op(value), pred);
    }
};

// Common refinement predicates
struct positive {
    template<typename T>
    bool operator()(const T& x) const { return x > T(0); }
};

struct non_negative {
    template<typename T>
    bool operator()(const T& x) const { return x >= T(0); }
};

struct bounded {
    double min, max;

    template<typename T>
    bool operator()(const T& x) const {
        return x >= T(min) && x <= T(max);
    }
};

struct non_empty {
    template<typename Container>
    bool operator()(const Container& c) const {
        return !c.empty();
    }
};

// Type aliases for common refined types
template<typename T>
using positive_t = refined<T, positive>;

template<typename T>
using non_negative_t = refined<T, non_negative>;

// Probability type with fixed bounds
template<typename T>
class probability_t : public refined<T, bounded> {
public:
    probability_t(T val) : refined<T, bounded>(val, bounded{0.0, 1.0}) {}
};

template<typename Container>
using non_empty_t = refined<Container, non_empty>;

// Dependent pair: second type depends on first value
template<typename T, typename DependentGen>
class dependent_pair {
    T first;
    using second_type = decltype(std::declval<DependentGen>()(std::declval<T>()));
    second_type second;

public:
    dependent_pair(T f, DependentGen gen)
        : first(std::move(f)), second(gen(first)) {}

    const T& get_first() const { return first; }
    const second_type& get_second() const { return second; }
};

// ============================================================================
// Property-Based Testing (QuickCheck-style)
// ============================================================================

// Random value generator
template<typename T>
class generator {
public:
    virtual ~generator() = default;
    virtual T generate(std::mt19937& rng) = 0;
    virtual T shrink(const T& value) { return value; }  // Minimal counterexample
};

// Integer generator
template<std::integral T>
class int_generator : public generator<T> {
    std::uniform_int_distribution<T> dist;

public:
    int_generator(T min = std::numeric_limits<T>::min(),
                 T max = std::numeric_limits<T>::max())
        : dist(min, max) {}

    T generate(std::mt19937& rng) override {
        return dist(rng);
    }

    T shrink(const T& value) override {
        // Shrink toward zero
        if (value > 0) return value / 2;
        if (value < 0) return value / 2;
        return 0;
    }
};

// Floating-point generator
template<std::floating_point T>
class float_generator : public generator<T> {
    std::uniform_real_distribution<T> dist;

public:
    float_generator(T min = T(0), T max = T(1))
        : dist(min, max) {}

    T generate(std::mt19937& rng) override {
        return dist(rng);
    }

    T shrink(const T& value) override {
        // Shrink toward zero
        return value * T(0.5);
    }
};

// Vector generator
template<typename T>
class vector_generator : public generator<std::vector<T>> {
    std::unique_ptr<generator<T>> element_gen;
    size_t min_size, max_size;

public:
    vector_generator(std::unique_ptr<generator<T>> gen,
                    size_t min = 0, size_t max = 100)
        : element_gen(std::move(gen)), min_size(min), max_size(max) {}

    std::vector<T> generate(std::mt19937& rng) override {
        std::uniform_int_distribution<size_t> size_dist(min_size, max_size);
        size_t size = size_dist(rng);

        std::vector<T> result;
        result.reserve(size);

        for (size_t i = 0; i < size; ++i) {
            result.push_back(element_gen->generate(rng));
        }

        return result;
    }

    std::vector<T> shrink(const std::vector<T>& value) override {
        if (value.size() <= min_size) return value;

        // Try removing elements
        std::vector<T> shrunk = value;
        shrunk.pop_back();
        return shrunk;
    }
};

// Property test result
struct test_result {
    bool passed;
    std::string property_name;
    std::optional<std::string> counterexample;
    size_t num_tests;
    std::chrono::milliseconds duration;
};

// Property-based test runner
template<typename... Args>
class property_test {
    std::string name;
    std::function<bool(Args...)> property;
    std::tuple<std::unique_ptr<generator<Args>>...> generators;
    size_t num_tests;
    std::mt19937 rng;

public:
    property_test(std::string n,
                 std::function<bool(Args...)> prop,
                 std::unique_ptr<generator<Args>>... gens,
                 size_t tests = 100)
        : name(std::move(n)),
          property(std::move(prop)),
          generators(std::move(gens)...),
          num_tests(tests),
          rng(std::random_device{}()) {}

    test_result run() {
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < num_tests; ++i) {
            auto values = generate_values();

            if (!std::apply(property, values)) {
                // Found counterexample - try to shrink it
                auto shrunk = shrink_values(values);

                auto end = std::chrono::steady_clock::now();
                return {
                    false,
                    name,
                    format_counterexample(shrunk),
                    i + 1,
                    std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                };
            }
        }

        auto end = std::chrono::steady_clock::now();
        return {
            true,
            name,
            std::nullopt,
            num_tests,
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        };
    }

private:
    std::tuple<Args...> generate_values() {
        return std::apply([this](auto&... gen) {
            return std::make_tuple(gen->generate(rng)...);
        }, generators);
    }

    std::tuple<Args...> shrink_values(const std::tuple<Args...>& values) {
        // Try to find minimal failing case
        auto current = values;

        while (true) {
            auto shrunk = std::apply([this](const auto&... vals) {
                return std::apply([&vals...](auto&... gen) {
                    return std::make_tuple(gen->shrink(vals)...);
                }, generators);
            }, current);

            if (shrunk == current) break;  // Can't shrink further

            if (!std::apply(property, shrunk)) {
                current = shrunk;  // Found smaller counterexample
            } else {
                break;  // Shrunk version passes, keep current
            }
        }

        return current;
    }

    std::string format_counterexample(const std::tuple<Args...>& values) {
        std::stringstream ss;
        ss << "Counterexample found:\n";
        size_t i = 0;
        std::apply([&ss, &i](const auto&... vals) {
            ((ss << "  arg" << i++ << ": " << vals << "\n"), ...);
        }, values);
        return ss.str();
    }
};

// Common property templates
namespace properties {
    // Commutativity: f(a, b) = f(b, a)
    template<typename T, typename Op>
    bool commutative(const T& a, const T& b, Op op) {
        return op(a, b) == op(b, a);
    }

    // Associativity: f(f(a, b), c) = f(a, f(b, c))
    template<typename T, typename Op>
    bool associative(const T& a, const T& b, const T& c, Op op) {
        return op(op(a, b), c) == op(a, op(b, c));
    }

    // Identity: f(a, id) = a
    template<typename T, typename Op>
    bool identity(const T& a, const T& id, Op op) {
        return op(a, id) == a && op(id, a) == a;
    }

    // Idempotence: f(f(a)) = f(a)
    template<typename T, typename Op>
    bool idempotent(const T& a, Op op) {
        return op(op(a)) == op(a);
    }

    // Invariant preservation
    template<typename T, typename Op, typename Inv>
    bool preserves_invariant(const T& input, Op operation, Inv invariant) {
        if (!invariant(input)) return true;  // Skip invalid inputs
        auto result = operation(input);
        return invariant(result);
    }

    // Round-trip: decode(encode(x)) = x
    template<typename T, typename Encode, typename Decode>
    bool round_trip(const T& value, Encode encode, Decode decode) {
        return decode(encode(value)) == value;
    }

    // Monotonicity: x <= y => f(x) <= f(y)
    template<typename T, typename Op>
    bool monotonic(const T& x, const T& y, Op op) {
        if (x <= y) {
            return op(x) <= op(y);
        }
        return true;  // Vacuously true if precondition not met
    }

    // Distributivity: a * (b + c) = (a * b) + (a * c)
    template<typename T, typename Mul, typename Add>
    bool distributive(const T& a, const T& b, const T& c, Mul mul, Add add) {
        return mul(a, add(b, c)) == add(mul(a, b), mul(a, c));
    }
}

// ============================================================================
// Stateful Property Testing
// ============================================================================

// Model-based testing for stateful systems
template<typename System, typename Model>
class stateful_test {
public:
    // Command that can be applied to both system and model
    struct command {
        virtual ~command() = default;
        virtual void apply_system(System& sys) = 0;
        virtual void apply_model(Model& model) = 0;
        virtual bool check_postcondition(const System& sys, const Model& model) = 0;
        virtual std::string name() const = 0;
    };

private:
    System system;
    Model model;
    std::vector<std::unique_ptr<command>> command_generators;
    std::mt19937 rng;

public:
    stateful_test(System sys, Model mod)
        : system(std::move(sys)), model(std::move(mod)), rng(std::random_device{}()) {}

    void add_command(std::unique_ptr<command> cmd) {
        command_generators.push_back(std::move(cmd));
    }

    test_result run(size_t num_commands) {
        auto start = std::chrono::steady_clock::now();
        std::vector<std::string> command_sequence;

        try {
            for (size_t i = 0; i < num_commands; ++i) {
                // Choose random command
                std::uniform_int_distribution<size_t> dist(0, command_generators.size() - 1);
                auto& cmd = command_generators[dist(rng)];

                command_sequence.push_back(cmd->name());

                // Apply to both system and model
                cmd->apply_system(system);
                cmd->apply_model(model);

                // Check postcondition
                if (!cmd->check_postcondition(system, model)) {
                    auto end = std::chrono::steady_clock::now();

                    std::stringstream ss;
                    ss << "State divergence after command sequence:\n";
                    for (const auto& name : command_sequence) {
                        ss << "  " << name << "\n";
                    }

                    return {
                        false,
                        "Stateful property",
                        ss.str(),
                        i + 1,
                        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                    };
                }
            }
        } catch (const std::exception& e) {
            auto end = std::chrono::steady_clock::now();

            std::stringstream ss;
            ss << "Exception thrown: " << e.what() << "\n";
            ss << "After command sequence:\n";
            for (const auto& name : command_sequence) {
                ss << "  " << name << "\n";
            }

            return {
                false,
                "Stateful property",
                ss.str(),
                command_sequence.size(),
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            };
        }

        auto end = std::chrono::steady_clock::now();
        return {
            true,
            "Stateful property",
            std::nullopt,
            num_commands,
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        };
    }
};

// ============================================================================
// Bounded Model Checking
// ============================================================================

// Symbolic execution path
template<typename State>
struct execution_path {
    std::vector<State> states;
    std::vector<std::string> transitions;
    bool is_complete;
    std::optional<std::string> violation;
};

// Simple bounded model checker
template<typename State>
class bounded_model_checker {
public:
    using transition_fn = std::function<std::vector<std::pair<State, std::string>>(const State&)>;
    using property_fn = std::function<bool(const State&)>;
    using invariant_fn = std::function<bool(const State&)>;

private:
    State initial_state;
    transition_fn get_transitions;
    std::vector<property_fn> safety_properties;
    std::vector<invariant_fn> invariants;
    size_t max_depth;

public:
    bounded_model_checker(State init, transition_fn trans, size_t depth = 10)
        : initial_state(std::move(init)),
          get_transitions(std::move(trans)),
          max_depth(depth) {}

    void add_safety_property(property_fn prop) {
        safety_properties.push_back(std::move(prop));
    }

    void add_invariant(invariant_fn inv) {
        invariants.push_back(std::move(inv));
    }

    std::vector<execution_path<State>> check() {
        std::vector<execution_path<State>> violations;
        std::vector<State> current_path;
        std::vector<std::string> transitions;

        check_recursive(initial_state, current_path, transitions, violations, 0);

        return violations;
    }

private:
    void check_recursive(const State& state,
                        std::vector<State>& path,
                        std::vector<std::string>& transitions,
                        std::vector<execution_path<State>>& violations,
                        size_t depth) {
        // Check invariants
        for (const auto& inv : invariants) {
            if (!inv(state)) {
                path.push_back(state);
                violations.push_back({
                    path,
                    transitions,
                    depth >= max_depth,
                    "Invariant violation"
                });
                path.pop_back();
                return;
            }
        }

        // Check safety properties
        for (const auto& prop : safety_properties) {
            if (!prop(state)) {
                path.push_back(state);
                violations.push_back({
                    path,
                    transitions,
                    depth >= max_depth,
                    "Safety property violation"
                });
                path.pop_back();
                return;
            }
        }

        // Check depth bound
        if (depth >= max_depth) {
            return;
        }

        // Explore transitions
        path.push_back(state);
        auto next_states = get_transitions(state);

        for (const auto& [next, transition] : next_states) {
            transitions.push_back(transition);
            check_recursive(next, path, transitions, violations, depth + 1);
            transitions.pop_back();
        }

        path.pop_back();
    }
};

// Loop invariant inference helper
template<typename State, typename LoopBody>
class loop_invariant_finder {
    State initial;
    LoopBody body;
    std::function<bool(const State&)> loop_condition;

public:
    loop_invariant_finder(State init, LoopBody b,
                         std::function<bool(const State&)> cond)
        : initial(std::move(init)), body(std::move(b)), loop_condition(std::move(cond)) {}

    // Try to find invariant that holds throughout loop
    template<typename Invariant>
    bool verify_invariant(Invariant inv) {
        State current = initial;

        // Check initial state
        if (!inv(current)) return false;

        // Simulate loop execution
        size_t max_iterations = 1000;
        size_t i = 0;

        while (loop_condition(current) && i < max_iterations) {
            State next = body(current);

            // Check invariant preservation
            if (!inv(next)) return false;

            current = next;
            i++;
        }

        return true;
    }

    // Generate candidate invariants
    std::vector<std::function<bool(const State&)>> suggest_invariants() {
        std::vector<std::function<bool(const State&)>> candidates;

        // This is a simplified version - real invariant inference
        // would use techniques like abstract interpretation

        // Try simple bounds
        State current = initial;
        State min_state = initial;
        State max_state = initial;

        size_t samples = 100;
        for (size_t i = 0; i < samples && loop_condition(current); ++i) {
            current = body(current);
            // Update bounds (assuming State has comparison operators)
            // min_state = std::min(min_state, current);
            // max_state = std::max(max_state, current);
        }

        // Generate bound-based invariants
        // candidates.push_back([min_state, max_state](const State& s) {
        //     return s >= min_state && s <= max_state;
        // });

        return candidates;
    }
};

// Test harness for running all verification tests
class test_suite {
    std::vector<std::function<test_result()>> tests;
    std::string name;

public:
    explicit test_suite(std::string n) : name(std::move(n)) {}

    template<typename TestFunc>
    void add_test(TestFunc test) {
        tests.push_back(std::move(test));
    }

    void run() {
        std::cout << "Running test suite: " << name << "\n";
        std::cout << std::string(50, '-') << "\n";

        size_t passed = 0;
        size_t failed = 0;

        for (const auto& test : tests) {
            auto result = test();

            if (result.passed) {
                std::cout << "[PASS] " << result.property_name;
                std::cout << " (" << result.num_tests << " tests in ";
                std::cout << result.duration.count() << "ms)\n";
                passed++;
            } else {
                std::cout << "[FAIL] " << result.property_name << "\n";
                if (result.counterexample) {
                    std::cout << *result.counterexample << "\n";
                }
                failed++;
            }
        }

        std::cout << std::string(50, '-') << "\n";
        std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    }
};

} // namespace stepanov::verify

#endif // STEPANOV_VERIFY_HPP