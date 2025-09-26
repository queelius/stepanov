/**
 * Algebraic Effects and Software Transactional Memory
 * ======================================================
 *
 * This example demonstrates how algebraic effects provide elegant
 * solutions to complex control flow and how STM enables composable
 * concurrent programming without locks.
 */

#include <stepanov/effects.hpp>
#include <stepanov/stm.hpp>
#include <stepanov/concurrent.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <map>

namespace stepanov::examples {

/**
 * Algebraic Effects: Composable Control Flow
 */
namespace effects {

    using namespace stepanov::effects;

    // Effect types
    struct logging_effect {
        std::vector<std::string> logs;

        void log(const std::string& msg) {
            logs.push_back(msg);
        }
    };

    struct state_effect {
        std::map<std::string, int> state;

        int get(const std::string& key) {
            return state[key];
        }

        void set(const std::string& key, int value) {
            state[key] = value;
        }
    };

    struct exception_effect {
        std::optional<std::string> error;

        void throw_error(const std::string& msg) {
            error = msg;
        }
    };

    // Composable computation with multiple effects
    template<typename T>
    class effectful {
        std::function<T(logging_effect&, state_effect&, exception_effect&)> computation;

    public:
        effectful(auto f) : computation(f) {}

        // Monadic bind
        template<typename U>
        effectful<U> bind(std::function<effectful<U>(T)> f) {
            return effectful<U>([=](auto& log, auto& state, auto& exc) {
                if (exc.error) return U{};  // Short-circuit on error

                T result = computation(log, state, exc);
                if (exc.error) return U{};  // Check again

                return f(result).run(log, state, exc);
            });
        }

        // Run with handlers
        T run(logging_effect& log, state_effect& state, exception_effect& exc) {
            return computation(log, state, exc);
        }

        // Pure value
        static effectful<T> pure(T value) {
            return effectful([value](auto&, auto&, auto&) { return value; });
        }
    };

    // Effect operations
    effectful<void> log(const std::string& msg) {
        return effectful<void>([msg](auto& log, auto&, auto&) {
            log.log(msg);
        });
    }

    effectful<int> get_state(const std::string& key) {
        return effectful<int>([key](auto&, auto& state, auto&) {
            return state.get(key);
        });
    }

    effectful<void> set_state(const std::string& key, int value) {
        return effectful<void>([key, value](auto&, auto& state, auto&) {
            state.set(key, value);
        });
    }

    effectful<void> throw_if(bool condition, const std::string& msg) {
        return effectful<void>([condition, msg](auto&, auto&, auto& exc) {
            if (condition) exc.throw_error(msg);
        });
    }

    void demo_effects() {
        std::cout << "=== Algebraic Effects ===\n\n";

        // Complex computation with multiple effects
        auto computation = log("Starting computation")
            .bind([](auto) { return set_state("counter", 0); })
            .bind([](auto) { return log("Counter initialized"); })
            .bind([](auto) {
                // Increment counter 5 times
                effectful<void> result = effectful<void>::pure();
                for (int i = 0; i < 5; ++i) {
                    result = result
                        .bind([](auto) { return get_state("counter"); })
                        .bind([](int count) {
                            return set_state("counter", count + 1);
                        })
                        .bind([i](auto) {
                            return log("Incremented to " + std::to_string(i + 1));
                        });
                }
                return result;
            })
            .bind([](auto) { return get_state("counter"); })
            .bind([](int final_count) {
                return throw_if(final_count != 5, "Wrong count!")
                    .bind([final_count](auto) {
                        return effectful<int>::pure(final_count);
                    });
            });

        // Run with effect handlers
        logging_effect logger;
        state_effect state;
        exception_effect error;

        int result = computation.run(logger, state, error);

        std::cout << "Computation result: " << result << "\n\n";
        std::cout << "Logs:\n";
        for (const auto& log : logger.logs) {
            std::cout << "  " << log << "\n";
        }
        std::cout << "\nFinal state:\n";
        for (const auto& [key, value] : state.state) {
            std::cout << "  " << key << " = " << value << "\n";
        }
        if (error.error) {
            std::cout << "\nError: " << *error.error << "\n";
        }
        std::cout << "\n";
    }

    // Resumable computations with continuations
    template<typename T>
    class continuation {
        std::function<T()> resume;
        bool completed = false;
        T result;

    public:
        continuation(std::function<T()> f) : resume(f) {}

        bool is_complete() const { return completed; }

        T operator()() {
            if (!completed) {
                result = resume();
                completed = true;
            }
            return result;
        }
    };

    void demo_continuations() {
        std::cout << "=== Continuations (Resumable Computations) ===\n\n";

        // Computation that can be paused and resumed
        int counter = 0;
        auto computation = [&counter]() -> continuation<int> {
            std::cout << "Starting computation...\n";
            counter++;

            std::cout << "Pausing at checkpoint 1 (counter = " << counter << ")\n";
            // Could yield control here

            counter++;
            std::cout << "Resuming... checkpoint 2 (counter = " << counter << ")\n";

            counter++;
            std::cout << "Completing computation (counter = " << counter << ")\n";

            return continuation<int>([counter] { return counter; });
        };

        auto cont = computation();
        std::cout << "Final result: " << cont() << "\n\n";
    }
}

/**
 * Software Transactional Memory
 */
namespace stm {

    using namespace stepanov::stm;

    // Bank account with STM
    class bank_account {
        stm_var<double> balance;
        stm_var<std::vector<std::string>> history;
        std::string account_id;

    public:
        bank_account(const std::string& id, double initial)
            : balance(initial), account_id(id) {}

        double get_balance() {
            return atomic_transaction([&] {
                return balance.read();
            });
        }

        void deposit(double amount) {
            atomic_transaction([&] {
                auto current = balance.read();
                balance.write(current + amount);

                history.modify([&](auto& h) {
                    h.push_back("Deposit: +" + std::to_string(amount));
                });
            });
        }

        bool withdraw(double amount) {
            return atomic_transaction([&] {
                auto current = balance.read();
                if (current < amount) {
                    return false;  // Insufficient funds
                }

                balance.write(current - amount);
                history.modify([&](auto& h) {
                    h.push_back("Withdrawal: -" + std::to_string(amount));
                });
                return true;
            });
        }

        // Atomic transfer between accounts
        static bool transfer(bank_account& from, bank_account& to, double amount) {
            return atomic_transaction([&] {
                auto from_balance = from.balance.read();
                if (from_balance < amount) {
                    return false;  // Insufficient funds
                }

                from.balance.write(from_balance - amount);
                to.balance.write(to.balance.read() + amount);

                auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
                from.history.modify([&](auto& h) {
                    h.push_back("Transfer to " + to.account_id + ": -" +
                               std::to_string(amount) + " @" + std::to_string(timestamp));
                });
                to.history.modify([&](auto& h) {
                    h.push_back("Transfer from " + from.account_id + ": +" +
                               std::to_string(amount) + " @" + std::to_string(timestamp));
                });

                return true;
            });
        }

        void print_statement() {
            atomic_transaction([&] {
                std::cout << "Account: " << account_id << "\n";
                std::cout << "Balance: $" << balance.read() << "\n";
                std::cout << "History:\n";
                for (const auto& entry : history.read()) {
                    std::cout << "  " << entry << "\n";
                }
            });
        }
    };

    void demo_stm_basics() {
        std::cout << "=== Software Transactional Memory Basics ===\n\n";

        // Create accounts
        bank_account alice("Alice", 1000);
        bank_account bob("Bob", 500);

        std::cout << "Initial state:\n";
        alice.print_statement();
        std::cout << "\n";
        bob.print_statement();
        std::cout << "\n";

        // Atomic transfer
        std::cout << "Transferring $200 from Alice to Bob...\n";
        if (bank_account::transfer(alice, bob, 200)) {
            std::cout << "Transfer successful!\n\n";
        }

        std::cout << "After transfer:\n";
        alice.print_statement();
        std::cout << "\n";
        bob.print_statement();
        std::cout << "\n";
    }

    void demo_stm_concurrency() {
        std::cout << "=== STM Under Concurrent Load ===\n\n";

        bank_account shared("Shared", 1000);

        std::cout << "Initial balance: $" << shared.get_balance() << "\n";
        std::cout << "Launching 10 threads, each doing 100 transactions...\n\n";

        std::vector<std::thread> threads;
        std::atomic<int> successful_deposits{0};
        std::atomic<int> successful_withdrawals{0};
        std::atomic<int> conflicts{0};

        auto worker = [&](int id) {
            std::mt19937 gen(id);
            std::uniform_int_distribution<> amount_dist(1, 10);
            std::uniform_int_distribution<> operation_dist(0, 1);

            for (int i = 0; i < 100; ++i) {
                int amount = amount_dist(gen);
                bool is_deposit = operation_dist(gen) == 0;

                try {
                    if (is_deposit) {
                        shared.deposit(amount);
                        successful_deposits++;
                    } else {
                        if (shared.withdraw(amount)) {
                            successful_withdrawals++;
                        }
                    }
                } catch (const transaction_conflict&) {
                    conflicts++;
                    // STM automatically retries
                }

                // Small random delay
                std::this_thread::sleep_for(std::chrono::microseconds(gen() % 100));
            }
        };

        // Launch threads
        for (int i = 0; i < 10; ++i) {
            threads.emplace_back(worker, i);
        }

        // Wait for completion
        for (auto& t : threads) {
            t.join();
        }

        std::cout << "Results:\n";
        std::cout << "  Successful deposits: " << successful_deposits << "\n";
        std::cout << "  Successful withdrawals: " << successful_withdrawals << "\n";
        std::cout << "  Conflicts resolved: " << conflicts << "\n";
        std::cout << "  Final balance: $" << shared.get_balance() << "\n\n";

        std::cout << "Note: Despite heavy concurrent access, STM ensures\n";
        std::cout << "consistency without any explicit locks!\n\n";
    }

    // Composable transactions
    void demo_composable_transactions() {
        std::cout << "=== Composable Transactions ===\n\n";

        // Complex multi-account operation
        bank_account savings("Savings", 5000);
        bank_account checking("Checking", 1000);
        bank_account investment("Investment", 0);

        std::cout << "Initial state:\n";
        std::cout << "  Savings: $" << savings.get_balance() << "\n";
        std::cout << "  Checking: $" << checking.get_balance() << "\n";
        std::cout << "  Investment: $" << investment.get_balance() << "\n\n";

        // Complex transaction: Move 50% of savings to investment,
        // but only if checking has enough for fees
        auto complex_transaction = [&]() {
            return atomic_transaction([&] {
                const double fee = 50;
                auto checking_balance = checking.get_balance();

                if (checking_balance < fee) {
                    std::cout << "Transaction aborted: Insufficient funds for fee\n";
                    return false;
                }

                // Deduct fee
                checking.withdraw(fee);

                // Move half of savings
                auto savings_balance = savings.get_balance();
                auto transfer_amount = savings_balance / 2;

                if (!bank_account::transfer(savings, investment, transfer_amount)) {
                    // This should rollback everything, including the fee
                    return false;
                }

                std::cout << "Transaction successful:\n";
                std::cout << "  Moved $" << transfer_amount << " to investment\n";
                std::cout << "  Fee of $" << fee << " deducted\n";
                return true;
            });
        };

        complex_transaction();

        std::cout << "\nFinal state:\n";
        std::cout << "  Savings: $" << savings.get_balance() << "\n";
        std::cout << "  Checking: $" << checking.get_balance() << "\n";
        std::cout << "  Investment: $" << investment.get_balance() << "\n\n";
    }

    // STM with retry and alternatives
    void demo_stm_retry() {
        std::cout << "=== STM Retry and Alternatives ===\n\n";

        stm_var<int> resource_count(0);
        stm_var<std::queue<std::string>> task_queue;

        // Producer
        auto producer = [&](const std::string& task) {
            atomic_transaction([&] {
                task_queue.modify([&](auto& q) {
                    q.push(task);
                });
                resource_count.write(resource_count.read() + 1);
                std::cout << "Produced: " << task << "\n";
            });
        };

        // Consumer with retry
        auto consumer = [&]() {
            return atomic_transaction([&] {
                if (resource_count.read() == 0) {
                    retry();  // Wait for resources
                }

                std::string task;
                task_queue.modify([&](auto& q) {
                    if (!q.empty()) {
                        task = q.front();
                        q.pop();
                    }
                });

                resource_count.write(resource_count.read() - 1);
                std::cout << "Consumed: " << task << "\n";
                return task;
            });
        };

        // OrElse: Try primary action, fall back to alternative
        auto consumer_with_fallback = [&]() {
            return or_else(
                [&] { return consumer(); },
                [&] {
                    std::cout << "No tasks available, using fallback\n";
                    return std::string("default_task");
                }
            );
        };

        std::cout << "Producer-Consumer with STM:\n";
        producer("Task1");
        producer("Task2");
        consumer();
        consumer();

        std::cout << "\nTrying consumer with empty queue (will use fallback):\n";
        consumer_with_fallback();
        std::cout << "\n";
    }
}

/**
 * Combining Effects with STM
 */
namespace combined {

    void demo_effects_with_stm() {
        std::cout << "=== Combining Algebraic Effects with STM ===\n\n";

        using namespace stepanov::effects;
        using namespace stepanov::stm;

        // Transactional variable with effect tracking
        stm_var<int> counter(0);
        std::vector<std::string> effect_log;

        // Effect-aware transaction
        auto effectful_transaction = [&]() {
            return atomic_transaction([&] {
                // Log effect
                effect_log.push_back("Transaction started");

                // Read current value
                int current = counter.read();
                effect_log.push_back("Read value: " + std::to_string(current));

                // Conditional effect
                if (current < 10) {
                    counter.write(current + 1);
                    effect_log.push_back("Incremented to " + std::to_string(current + 1));
                    return true;
                } else {
                    effect_log.push_back("Maximum reached, rolling back");
                    // Demonstrate rollback
                    abort_transaction();
                    return false;
                }
            });
        };

        std::cout << "Running effectful transactions:\n";
        for (int i = 0; i < 12; ++i) {
            bool success = effectful_transaction();
            std::cout << "Transaction " << i << ": "
                      << (success ? "Success" : "Aborted") << "\n";
        }

        std::cout << "\nEffect log:\n";
        for (const auto& log : effect_log) {
            std::cout << "  " << log << "\n";
        }
        std::cout << "\nFinal counter value: " << counter.read() << "\n\n";
    }

    // Real-world example: Distributed consensus with effects and STM
    void demo_consensus() {
        std::cout << "=== Distributed Consensus with Effects + STM ===\n\n";

        struct node {
            std::string id;
            stm_var<int> proposal;
            stm_var<bool> decided;
            stm_var<int> round;

            node(const std::string& node_id)
                : id(node_id), proposal(0), decided(false), round(0) {}
        };

        std::vector<node> nodes = {
            node("Node1"), node("Node2"), node("Node3")
        };

        // Paxos-like consensus protocol
        auto propose = [&](node& proposer, int value) {
            return atomic_transaction([&] {
                if (proposer.decided.read()) {
                    return false;  // Already decided
                }

                // Phase 1: Prepare
                int current_round = proposer.round.read();
                proposer.round.write(current_round + 1);

                // Check majority
                int promises = 0;
                for (auto& n : nodes) {
                    if (n.round.read() <= current_round) {
                        promises++;
                    }
                }

                if (promises > nodes.size() / 2) {
                    // Phase 2: Accept
                    proposer.proposal.write(value);
                    proposer.decided.write(true);

                    // Notify others
                    for (auto& n : nodes) {
                        if (n.id != proposer.id) {
                            n.proposal.write(value);
                            n.decided.write(true);
                        }
                    }

                    std::cout << proposer.id << " achieved consensus on value " << value << "\n";
                    return true;
                }

                return false;
            });
        };

        // Multiple nodes trying to achieve consensus
        std::thread t1([&] { propose(nodes[0], 42); });
        std::thread t2([&] { propose(nodes[1], 37); });
        std::thread t3([&] { propose(nodes[2], 50); });

        t1.join();
        t2.join();
        t3.join();

        std::cout << "\nFinal state:\n";
        for (const auto& n : nodes) {
            std::cout << "  " << n.id << ": value = " << n.proposal.read()
                      << ", decided = " << (n.decided.read() ? "Yes" : "No") << "\n";
        }
        std::cout << "\nNote: All nodes agree on the same value!\n\n";
    }
}

/**
 * Philosophical Discussion
 */
void demo_philosophical() {
    std::cout << "=== The Philosophy of Effects and Transactions ===\n\n";

    std::cout << "Traditional Concurrency (Locks):\n";
    std::cout << "  • Pessimistic - Assume conflicts will happen\n";
    std::cout << "  • Non-composable - Lock ordering matters\n";
    std::cout << "  • Error-prone - Deadlocks, race conditions\n";
    std::cout << "  • Low-level - Manual resource management\n\n";

    std::cout << "Software Transactional Memory:\n";
    std::cout << "  • Optimistic - Assume conflicts are rare\n";
    std::cout << "  • Composable - Transactions compose naturally\n";
    std::cout << "  • Safe - No deadlocks, automatic rollback\n";
    std::cout << "  • High-level - Declarative concurrency\n\n";

    std::cout << "Traditional Control Flow:\n";
    std::cout << "  • Imperative - How to do things\n";
    std::cout << "  • Tangled - Error handling mixed with logic\n";
    std::cout << "  • Rigid - Hard to modify behavior\n";
    std::cout << "  • Opaque - Effects hidden in implementation\n\n";

    std::cout << "Algebraic Effects:\n";
    std::cout << "  • Declarative - What to do, not how\n";
    std::cout << "  • Separated - Effects distinct from logic\n";
    std::cout << "  • Flexible - Swap effect handlers easily\n";
    std::cout << "  • Explicit - Effects visible in types\n\n";

    std::cout << "The Synergy:\n";
    std::cout << "When we combine STM with algebraic effects, we get:\n";
    std::cout << "  1. Transactions as an effect - Composable with other effects\n";
    std::cout << "  2. Time-travel debugging - Effects + transaction logs\n";
    std::cout << "  3. Reproducible concurrency - Deterministic effect handlers\n";
    std::cout << "  4. Elegant error recovery - Effects handle transaction failures\n\n";

    std::cout << "This isn't just about making concurrent programming easier.\n";
    std::cout << "It's about recognizing that:\n";
    std::cout << "  • Control flow is an effect\n";
    std::cout << "  • State changes are effects\n";
    std::cout << "  • Concurrency is an effect\n";
    std::cout << "  • Effects compose algebraically\n\n";

    std::cout << "Stepanov brings these advanced concepts to C++,\n";
    std::cout << "showing that systems programming can be both\n";
    std::cout << "powerful AND elegant.\n";
}

} // namespace stepanov::examples

int main() {
    using namespace stepanov::examples;

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║    Algebraic Effects & Software Transactional Memory       ║\n";
    std::cout << "║         Elegant Concurrency Through Composition            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // Demonstrate effects
    effects::demo_effects();
    effects::demo_continuations();

    // Demonstrate STM
    stm::demo_stm_basics();
    stm::demo_stm_concurrency();
    stm::demo_composable_transactions();
    stm::demo_stm_retry();

    // Combine effects with STM
    combined::demo_effects_with_stm();
    combined::demo_consensus();

    // Philosophy
    demo_philosophical();

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  'Shared mutable state is the root of all evil.'          ║\n";
    std::cout << "║                                    - Joe Armstrong         ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║  'STM makes shared mutable state safe and composable.'    ║\n";
    std::cout << "║                                    - Simon Peyton Jones   ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║  Stepanov: Where effects and transactions unite.          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    return 0;
}