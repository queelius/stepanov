/**
 * Lazy Infinite Computation
 * ==========================
 *
 * This example demonstrates working with infinite data structures
 * as naturally as finite ones. We show that infinity is not a
 * special case but the general case of which finite is a subset.
 */

#include <stepanov/lazy.hpp>
#include <stepanov/algorithms.hpp>
#include <stepanov/math.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace stepanov::examples {

/**
 * Infinite Sequences
 */
namespace sequences {

    // Natural numbers: 0, 1, 2, 3, ...
    auto naturals() {
        return stepanov::lazy_sequence([n = 0]() mutable {
            return n++;
        });
    }

    // Fibonacci: 1, 1, 2, 3, 5, 8, ...
    auto fibonacci() {
        return stepanov::lazy_sequence([a = 0, b = 1]() mutable {
            auto current = a;
            a = b;
            b = current + b;
            return current;
        });
    }

    // Primes using the Sieve of Eratosthenes (infinite)
    auto primes() {
        return stepanov::lazy_sequence([sieve = std::vector<bool>(), n = 2]() mutable {
            while (true) {
                if (n >= sieve.size()) {
                    sieve.resize(n * 2, true);
                }

                bool is_prime = true;
                for (int i = 2; i * i <= n; ++i) {
                    if (n % i == 0) {
                        is_prime = false;
                        break;
                    }
                }

                if (is_prime) {
                    return n++;
                }
                n++;
            }
        });
    }

    // Collatz sequence from any starting point
    auto collatz(int start) {
        return stepanov::lazy_sequence([n = start]() mutable {
            auto current = n;
            if (n == 1) {
                // Stay at 1 forever
                return current;
            }
            n = (n % 2 == 0) ? n / 2 : 3 * n + 1;
            return current;
        });
    }

    // Pi digits using Machin's formula (infinite precision)
    auto pi_digits() {
        return stepanov::lazy_sequence([
            q = 1, r = 0, t = 1, k = 1, n = 3, l = 3
        ]() mutable {
            while (true) {
                if (4 * q + r - t < n * t) {
                    // Yield digit
                    auto digit = n;
                    auto nr = 10 * (r - n * t);
                    n = ((10 * (3 * q + r)) / t) - 10 * n;
                    q *= 10;
                    r = nr;
                    return digit;
                } else {
                    auto nr = (2 * q + r) * l;
                    auto nn = (q * (7 * k + 2) + r * l) / (t * l);
                    q *= k;
                    t *= l;
                    l += 2;
                    k++;
                    n = nn;
                    r = nr;
                }
            }
        });
    }
}

/**
 * Infinite Data Structures
 */
namespace structures {

    // Infinite binary tree where each node contains its path
    template<typename T>
    struct infinite_tree {
        T value;
        stepanov::lazy<infinite_tree> left;
        stepanov::lazy<infinite_tree> right;

        infinite_tree(T val, auto left_gen, auto right_gen)
            : value(val)
            , left([=]() { return left_gen(); })
            , right([=]() { return right_gen(); })
        {}

        // Create infinite complete binary tree
        static infinite_tree complete(T root = T{0}) {
            return infinite_tree(root,
                [v = 2 * root + 1]() { return complete(v); },
                [v = 2 * root + 2]() { return complete(v); }
            );
        }

        // Traverse to depth n
        void traverse_depth(int depth, auto visit) {
            if (depth <= 0) return;
            visit(value);
            if (depth > 1) {
                left.force().traverse_depth(depth - 1, visit);
                right.force().traverse_depth(depth - 1, visit);
            }
        }
    };

    // Infinite game tree for minimax
    struct game_tree {
        int position;
        int score;
        stepanov::lazy<std::vector<game_tree>> children;

        game_tree(int pos) : position(pos), score(evaluate(pos)) {
            children = stepanov::lazy([pos]() {
                return generate_moves(pos);
            });
        }

        static int evaluate(int pos) {
            // Simple evaluation function
            return pos % 7 - 3;
        }

        static std::vector<game_tree> generate_moves(int pos) {
            // Generate possible moves
            std::vector<game_tree> moves;
            moves.push_back(game_tree(pos * 2));
            moves.push_back(game_tree(pos * 2 + 1));
            moves.push_back(game_tree(pos + 1));
            return moves;
        }

        // Minimax with alpha-beta pruning on infinite tree
        int minimax(int depth, int alpha, int beta, bool maximizing) {
            if (depth == 0) return score;

            auto& moves = children.force();
            if (maximizing) {
                int max_eval = std::numeric_limits<int>::min();
                for (auto& child : moves) {
                    int eval = child.minimax(depth - 1, alpha, beta, false);
                    max_eval = std::max(max_eval, eval);
                    alpha = std::max(alpha, eval);
                    if (beta <= alpha) break;  // Pruning
                }
                return max_eval;
            } else {
                int min_eval = std::numeric_limits<int>::max();
                for (auto& child : moves) {
                    int eval = child.minimax(depth - 1, alpha, beta, true);
                    min_eval = std::min(min_eval, eval);
                    beta = std::min(beta, eval);
                    if (beta <= alpha) break;  // Pruning
                }
                return min_eval;
            }
        }
    };

    // Infinite stream with memoization
    template<typename T>
    class memoized_stream {
        mutable std::vector<T> cache;
        std::function<T(size_t)> generator;

    public:
        memoized_stream(std::function<T(size_t)> gen)
            : generator(gen) {}

        T operator[](size_t n) const {
            while (cache.size() <= n) {
                cache.push_back(generator(cache.size()));
            }
            return cache[n];
        }

        // Get range [start, end)
        std::vector<T> range(size_t start, size_t end) const {
            std::vector<T> result;
            for (size_t i = start; i < end; ++i) {
                result.push_back((*this)[i]);
            }
            return result;
        }
    };
}

/**
 * Demonstrations
 */
void demo_infinite_sequences() {
    std::cout << "=== Infinite Sequences ===\n\n";

    // First 20 Fibonacci numbers
    std::cout << "First 20 Fibonacci numbers:\n";
    for (auto fib : sequences::fibonacci() | stepanov::take(20)) {
        std::cout << fib << " ";
    }
    std::cout << "\n\n";

    // First 15 primes
    std::cout << "First 15 prime numbers:\n";
    for (auto prime : sequences::primes() | stepanov::take(15)) {
        std::cout << prime << " ";
    }
    std::cout << "\n\n";

    // Twin primes (primes where p+2 is also prime)
    std::cout << "First 10 twin prime pairs:\n";
    auto primes = sequences::primes();
    auto twin_primes = primes
        | stepanov::window(2)
        | stepanov::filter([](auto pair) {
            return pair[1] - pair[0] == 2;
        });

    for (auto pair : twin_primes | stepanov::take(10)) {
        std::cout << "(" << pair[0] << ", " << pair[1] << ") ";
    }
    std::cout << "\n\n";

    // Collatz conjecture demonstration
    std::cout << "Collatz sequence starting from 27:\n";
    int count = 0;
    for (auto n : sequences::collatz(27)) {
        std::cout << n << " ";
        if (n == 1) break;
        if (++count > 100) {
            std::cout << "...";
            break;
        }
    }
    std::cout << "\n\n";

    // Pi computation
    std::cout << "First 50 digits of Pi: 3.";
    int digit_count = 0;
    for (auto digit : sequences::pi_digits() | stepanov::take(50)) {
        if (digit_count++ > 0) std::cout << digit;
    }
    std::cout << "...\n\n";
}

void demo_infinite_trees() {
    std::cout << "=== Infinite Trees ===\n\n";

    // Infinite complete binary tree
    auto tree = structures::infinite_tree<int>::complete(1);

    std::cout << "First 3 levels of infinite binary tree:\n";
    std::cout << "Level 0: ";
    tree.traverse_depth(1, [](int v) { std::cout << v << " "; });
    std::cout << "\nLevel 1: ";
    tree.left.force().traverse_depth(1, [](int v) { std::cout << v << " "; });
    std::cout << " ";
    tree.right.force().traverse_depth(1, [](int v) { std::cout << v << " "; });
    std::cout << "\nLevel 2: ";
    tree.left.force().left.force().traverse_depth(1, [](int v) { std::cout << v << " "; });
    std::cout << " ";
    tree.left.force().right.force().traverse_depth(1, [](int v) { std::cout << v << " "; });
    std::cout << " ";
    tree.right.force().left.force().traverse_depth(1, [](int v) { std::cout << v << " "; });
    std::cout << " ";
    tree.right.force().right.force().traverse_depth(1, [](int v) { std::cout << v << " "; });
    std::cout << "\n\n";

    // Game tree with minimax
    std::cout << "Minimax on infinite game tree:\n";
    structures::game_tree game(1);
    int best_score = game.minimax(5, std::numeric_limits<int>::min(),
                                     std::numeric_limits<int>::max(), true);
    std::cout << "Best score from position 1 (depth 5): " << best_score << "\n\n";
}

void demo_lazy_computation() {
    std::cout << "=== Lazy Computation Examples ===\n\n";

    // Compute only what's needed
    auto expensive_computation = stepanov::lazy_sequence([n = 0]() mutable {
        // Simulate expensive computation
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return n++ * n;
    });

    std::cout << "Computing squares lazily (with delay):\n";
    auto start = std::chrono::steady_clock::now();

    // Only compute first 5
    for (auto val : expensive_computation | stepanov::take(5)) {
        std::cout << val << " ";
    }

    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "\nTime: " << ms << "ms (only computed what was needed)\n\n";

    // Infinite recursive definitions
    std::cout << "Hamming numbers (5-smooth numbers):\n";
    // Numbers of the form 2^a * 3^b * 5^c

    auto hamming = stepanov::lazy_sequence([]() {
        static structures::memoized_stream<int> h([](size_t n) -> int {
            if (n == 0) return 1;

            static size_t i2 = 0, i3 = 0, i5 = 0;
            static structures::memoized_stream<int>* self = nullptr;
            if (!self) self = &h;

            int h2 = (*self)[i2] * 2;
            int h3 = (*self)[i3] * 3;
            int h5 = (*self)[i5] * 5;

            int next = std::min({h2, h3, h5});

            if (next == h2) i2++;
            if (next == h3) i3++;
            if (next == h5) i5++;

            return next;
        });

        static size_t index = 0;
        return h[index++];
    });

    std::cout << "First 20 Hamming numbers: ";
    for (auto h : hamming | stepanov::take(20)) {
        std::cout << h << " ";
    }
    std::cout << "\n\n";
}

void demo_infinite_search() {
    std::cout << "=== Infinite Search Space ===\n\n";

    // Find first Pythagorean triple where a > 100
    std::cout << "Searching infinite space for Pythagorean triples:\n";

    auto pythagorean_triples = stepanov::lazy_sequence([]() {
        static int m = 2, n = 1;

        while (true) {
            if (m > n && (m - n) % 2 == 1 && std::gcd(m, n) == 1) {
                int a = m * m - n * n;
                int b = 2 * m * n;
                int c = m * m + n * n;

                n++;
                if (n >= m) {
                    m++;
                    n = 1;
                }

                return std::make_tuple(a, b, c);
            }

            n++;
            if (n >= m) {
                m++;
                n = 1;
            }
        }
    });

    std::cout << "First 10 primitive Pythagorean triples:\n";
    for (auto [a, b, c] : pythagorean_triples | stepanov::take(10)) {
        std::cout << "(" << a << ", " << b << ", " << c << ") ";
        std::cout << "Check: " << a << "² + " << b << "² = " << (a*a + b*b)
                  << " = " << c << "² = " << (c*c) << "\n";
    }
    std::cout << "\n";

    // Find perfect numbers
    std::cout << "Searching for perfect numbers (this might take a moment):\n";

    auto perfect_numbers = sequences::naturals()
        | stepanov::filter([](int n) {
            if (n <= 1) return false;
            int sum = 1;
            for (int i = 2; i * i <= n; ++i) {
                if (n % i == 0) {
                    sum += i;
                    if (i != n / i) sum += n / i;
                }
            }
            return sum == n;
        });

    std::cout << "First 4 perfect numbers: ";
    for (auto p : perfect_numbers | stepanov::take(4)) {
        std::cout << p << " ";
    }
    std::cout << "\n\n";
}

void demo_philosophical() {
    std::cout << "=== The Philosophy of Infinity ===\n\n";

    std::cout << "Traditional programming treats infinity as an error:\n";
    std::cout << "  while(true) { }  // Bug!\n";
    std::cout << "  for(i = 0; ; i++) { }  // Mistake!\n\n";

    std::cout << "But mathematics embraces infinity:\n";
    std::cout << "  ℕ = {0, 1, 2, 3, ...}\n";
    std::cout << "  π = 3.14159265358979...\n";
    std::cout << "  lim(n→∞) = ...\n\n";

    std::cout << "Stepanov brings mathematical infinity to programming:\n\n";

    // Define an infinite series: 1 + 1/2 + 1/4 + 1/8 + ...
    auto geometric_series = stepanov::lazy_sequence([n = 1.0]() mutable {
        auto term = 1.0 / n;
        n *= 2;
        return term;
    });

    // Compute partial sums
    double sum = 0;
    std::cout << "Geometric series 1 + 1/2 + 1/4 + 1/8 + ...:\n";
    for (auto term : geometric_series | stepanov::take(10)) {
        sum += term;
        std::cout << "  Sum of first " << std::setw(2)
                  << static_cast<int>(std::log2(1.0/term) + 1) << " terms: "
                  << sum << "\n";
    }
    std::cout << "  Limit as n→∞: 2\n\n";

    std::cout << "Key Insights:\n";
    std::cout << "1. Lazy evaluation makes infinity practical\n";
    std::cout << "2. We compute only what we observe\n";
    std::cout << "3. Infinite structures are first-class citizens\n";
    std::cout << "4. The finite is just a special case of the infinite\n\n";

    std::cout << "This is not a trick or optimization.\n";
    std::cout << "This is programming as it should be:\n";
    std::cout << "Mathematical. Elegant. Infinite.\n";
}

} // namespace stepanov::examples

int main() {
    using namespace stepanov::examples;

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Lazy Infinite Computation                        ║\n";
    std::cout << "║     Working with Infinity as Naturally as Finite          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    demo_infinite_sequences();
    demo_infinite_trees();
    demo_lazy_computation();
    demo_infinite_search();
    demo_philosophical();

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  'To iterate is human, to recurse divine.'                ║\n";
    std::cout << "║                        - L. Peter Deutsch                  ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║  'To recurse finitely is human,                           ║\n";
    std::cout << "║   to recurse infinitely is Stepanov.'                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    return 0;
}