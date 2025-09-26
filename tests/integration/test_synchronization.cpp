#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <cassert>
#include <mutex>
#include "../include/stepanov/synchronization.hpp"

using namespace stepanov::synchronization;
using namespace std::chrono_literals;

// Shared counter for testing mutual exclusion
std::atomic<int> shared_counter{0};
std::atomic<int> critical_section_count{0};

template<typename Lock>
void test_mutual_exclusion(Lock& lock, int thread_id, int iterations) {
    thread_id::set(thread_id);

    for (int i = 0; i < iterations; ++i) {
        lock.lock();

        // Critical section
        int old_count = critical_section_count.fetch_add(1);
        assert(old_count == 0);  // Verify mutual exclusion

        shared_counter++;

        // Simulate some work
        std::this_thread::sleep_for(std::chrono::microseconds(10));

        critical_section_count.fetch_sub(1);

        lock.unlock();
    }
}

void test_peterson_lock() {
    std::cout << "Testing Peterson lock (2 threads)..." << std::endl;

    peterson_lock lock;
    shared_counter = 0;
    critical_section_count = 0;

    const int iterations = 100;

    std::thread t1(test_mutual_exclusion<peterson_lock>,
                   std::ref(lock), 0, iterations);
    std::thread t2(test_mutual_exclusion<peterson_lock>,
                   std::ref(lock), 1, iterations);

    t1.join();
    t2.join();

    assert(shared_counter == 2 * iterations);
    std::cout << "✓ Peterson lock passed (counter = " << shared_counter << ")\n" << std::endl;
}

void test_tournament_lock() {
    std::cout << "Testing Tournament lock (8 threads)..." << std::endl;

    tournament_lock<8> lock;
    shared_counter = 0;
    critical_section_count = 0;

    const int iterations = 50;
    const int num_threads = 8;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(test_mutual_exclusion<tournament_lock<8>>,
                            std::ref(lock), i, iterations);
    }

    for (auto& t : threads) {
        t.join();
    }

    assert(shared_counter == num_threads * iterations);
    std::cout << "✓ Tournament lock passed (counter = " << shared_counter << ")\n" << std::endl;
}

void test_reader_writer_lock() {
    std::cout << "Testing Reader-Writer Tournament lock..." << std::endl;

    tournament_rw_lock<4> rw_lock;
    std::atomic<int> readers{0};
    std::atomic<int> writers{0};
    std::atomic<int> read_count{0};
    std::atomic<int> write_count{0};
    std::atomic<bool> stop{false};

    // Reader function
    auto reader = [&](int id) {
        thread_id::set(id);
        while (!stop) {
            auto guard = rw_lock.make_read_guard();

            readers.fetch_add(1);
            assert(writers == 0);  // No writers during read

            // Simulate reading
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            read_count++;

            readers.fetch_sub(1);
        }
    };

    // Writer function
    auto writer = [&](int id) {
        thread_id::set(id);
        while (!stop) {
            auto guard = rw_lock.make_write_guard();

            writers.fetch_add(1);
            assert(readers == 0);  // No readers during write
            assert(writers == 1);  // Only one writer

            // Simulate writing
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            write_count++;

            writers.fetch_sub(1);
        }
    };

    // Start readers and writers
    std::vector<std::thread> threads;
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(reader, i);
    }
    for (int i = 3; i < 5; ++i) {
        threads.emplace_back(writer, i);
    }

    // Let them run for a while
    std::this_thread::sleep_for(100ms);
    stop = true;

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "✓ RW lock passed (reads: " << read_count
              << ", writes: " << write_count << ")\n" << std::endl;
}

void test_spin_lock() {
    std::cout << "Testing Spin lock..." << std::endl;

    spin_lock lock;
    shared_counter = 0;
    critical_section_count = 0;

    const int iterations = 100;
    const int num_threads = 4;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(test_mutual_exclusion<spin_lock>,
                            std::ref(lock), i, iterations);
    }

    for (auto& t : threads) {
        t.join();
    }

    assert(shared_counter == num_threads * iterations);
    std::cout << "✓ Spin lock passed (counter = " << shared_counter << ")\n" << std::endl;
}

void test_ticket_lock() {
    std::cout << "Testing Ticket lock..." << std::endl;

    ticket_lock lock;
    shared_counter = 0;
    critical_section_count = 0;

    const int iterations = 100;
    const int num_threads = 4;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(test_mutual_exclusion<ticket_lock>,
                            std::ref(lock), i, iterations);
    }

    for (auto& t : threads) {
        t.join();
    }

    assert(shared_counter == num_threads * iterations);
    std::cout << "✓ Ticket lock passed (counter = " << shared_counter << ")\n" << std::endl;
}

void test_mcs_lock() {
    std::cout << "Testing MCS lock..." << std::endl;

    mcs_lock lock;
    shared_counter = 0;
    critical_section_count = 0;

    const int iterations = 100;
    const int num_threads = 4;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(test_mutual_exclusion<mcs_lock>,
                            std::ref(lock), i, iterations);
    }

    for (auto& t : threads) {
        t.join();
    }

    assert(shared_counter == num_threads * iterations);
    std::cout << "✓ MCS lock passed (counter = " << shared_counter << ")\n" << std::endl;
}

void test_lock_free_queue() {
    std::cout << "Testing Lock-free queue..." << std::endl;

    lock_free_queue<int> queue;
    std::atomic<int> sum{0};
    const int items_per_thread = 1000;
    const int num_producers = 2;
    const int num_consumers = 2;

    // Producer function
    auto producer = [&](int id) {
        for (int i = 0; i < items_per_thread; ++i) {
            queue.enqueue(i);
        }
    };

    // Consumer function
    auto consumer = [&]() {
        int local_sum = 0;
        int consumed = 0;
        while (consumed < items_per_thread) {
            auto item = queue.dequeue();
            if (item) {
                local_sum += *item;
                consumed++;
            } else {
                std::this_thread::yield();
            }
        }
        sum.fetch_add(local_sum);
    };

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    for (int i = 0; i < num_producers; ++i) {
        producers.emplace_back(producer, i);
    }

    for (int i = 0; i < num_consumers; ++i) {
        consumers.emplace_back(consumer);
    }

    for (auto& t : producers) {
        t.join();
    }
    for (auto& t : consumers) {
        t.join();
    }

    int expected_sum = num_producers * items_per_thread * (items_per_thread - 1) / 2;
    assert(sum == expected_sum);
    std::cout << "✓ Lock-free queue passed (sum = " << sum << ")\n" << std::endl;
}

void test_barrier() {
    std::cout << "Testing Barrier..." << std::endl;

    const int num_threads = 4;
    barrier bar(num_threads);
    std::atomic<int> phase1_complete{0};
    std::atomic<int> phase2_complete{0};

    auto worker = [&]() {
        // Phase 1
        phase1_complete.fetch_add(1);
        bar.wait();

        // All threads should have completed phase 1
        assert(phase1_complete == num_threads);

        // Phase 2
        phase2_complete.fetch_add(1);
        bar.wait();

        // All threads should have completed phase 2
        assert(phase2_complete == num_threads);
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "✓ Barrier passed\n" << std::endl;
}

void test_type_erasure() {
    std::cout << "Testing type-erased lock..." << std::endl;

    auto lock = any_lock::make<spin_lock>();
    shared_counter = 0;

    const int iterations = 100;
    const int num_threads = 4;

    auto worker = [&](int id) {
        thread_id::set(id);
        for (int i = 0; i < iterations; ++i) {
            any_lock::guard g(lock);
            shared_counter++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    assert(shared_counter == num_threads * iterations);
    std::cout << "✓ Type-erased lock passed (counter = " << shared_counter << ")\n" << std::endl;
}

void test_try_lock() {
    std::cout << "Testing try_lock functionality..." << std::endl;

    peterson_lock lock;
    thread_id::set(0);

    assert(lock.try_lock());
    assert(!lock.try_lock());  // Should fail as already locked
    lock.unlock();
    assert(lock.try_lock());  // Should succeed after unlock
    lock.unlock();

    std::cout << "✓ try_lock passed\n" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing stepanov::synchronization module" << std::endl;
    std::cout << "========================================\n" << std::endl;

    try {
        test_peterson_lock();
        test_tournament_lock();
        test_reader_writer_lock();
        test_spin_lock();
        test_ticket_lock();
        test_mcs_lock();
        test_lock_free_queue();
        test_barrier();
        test_type_erasure();
        test_try_lock();

        std::cout << "========================================" << std::endl;
        std::cout << "All synchronization tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}