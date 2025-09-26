#pragma once

#include <atomic>
#include <thread>
#include <memory>
#include <vector>
#include <array>
#include <chrono>
#include <functional>
#include <queue>
#include <mutex>
#include <cmath>
#include <algorithm>

namespace stepanov::synchronization {

// =============================================================================
// Thread ID management
// =============================================================================

class thread_id {
private:
    static thread_local int id_;
    static std::atomic<int> next_id_;

public:
    static int get() {
        if (id_ == -1) {
            id_ = next_id_.fetch_add(1);
        }
        return id_;
    }

    static void set(int id) {
        id_ = id;
    }

    static void reset() {
        id_ = -1;
    }
};

thread_local int thread_id::id_ = -1;
std::atomic<int> thread_id::next_id_{0};

// =============================================================================
// Basic lock interface
// =============================================================================

template<typename Derived>
class basic_lock {
public:
    void lock() {
        static_cast<Derived*>(this)->lock_impl();
    }

    bool try_lock() {
        return static_cast<Derived*>(this)->try_lock_impl();
    }

    void unlock() {
        static_cast<Derived*>(this)->unlock_impl();
    }
};

// =============================================================================
// Peterson's algorithm for 2 threads
// =============================================================================

class peterson_lock : public basic_lock<peterson_lock> {
private:
    std::atomic<bool> flag_[2];
    std::atomic<int> victim_;
    int less_than_;  // For tournament tree usage

public:
    explicit peterson_lock(int less_than = 1)
        : victim_(0), less_than_(less_than) {
        flag_[0].store(false);
        flag_[1].store(false);
    }

    void lock_impl() {
        int me = (thread_id::get() < less_than_) ? 0 : 1;
        int other = 1 - me;

        flag_[me].store(true);
        victim_.store(me);

        // Spin while other thread wants to enter and I'm the victim
        while (flag_[other].load() && victim_.load() == me) {
            std::this_thread::yield();
        }
    }

    bool try_lock_impl() {
        int me = (thread_id::get() < less_than_) ? 0 : 1;
        int other = 1 - me;

        flag_[me].store(true);
        victim_.store(me);

        if (flag_[other].load() && victim_.load() == me) {
            flag_[me].store(false);
            return false;
        }
        return true;
    }

    void unlock_impl() {
        int me = (thread_id::get() < less_than_) ? 0 : 1;
        flag_[me].store(false);
    }
};

// =============================================================================
// Tournament tree lock for N threads
// =============================================================================

template<int N>
class tournament_lock : public basic_lock<tournament_lock<N>> {
private:
    static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of 2");

    std::vector<std::unique_ptr<peterson_lock>> locks_;
    static constexpr int tree_size_ = 2 * N - 1;

    // Get the leaf node index for current thread
    int get_leaf_index() const {
        return (N - 1) + thread_id::get();
    }

    // Get parent index in binary tree
    static int parent(int index) {
        return (index - 1) / 2;
    }

    // Initialize the tournament tree
    void initialize_tree() {
        locks_.reserve(tree_size_);

        // Create all internal nodes (Peterson locks)
        for (int i = 0; i < N - 1; ++i) {
            int level_size = 1 << (std::__lg(N - i));
            int less_than = calculate_less_than(i, level_size);
            locks_.emplace_back(std::make_unique<peterson_lock>(less_than));
        }
    }

    // Calculate the less_than value for Peterson lock at given position
    int calculate_less_than(int index, int level_size) const {
        int level = 0;
        int temp = index;
        while (temp >= (1 << level)) {
            temp -= (1 << level);
            level++;
        }

        int position_in_level = temp;
        int subtree_size = N / (1 << (level + 1));
        return (position_in_level + 1) * subtree_size;
    }

public:
    tournament_lock() {
        initialize_tree();
    }

    void lock_impl() {
        int tid = thread_id::get();
        if (tid >= N) {
            throw std::runtime_error("Thread ID exceeds tournament lock capacity");
        }

        // Path from leaf to root
        std::vector<int> path;
        int index = get_leaf_index();

        while (index > 0) {
            index = parent(index);
            path.push_back(index);
        }

        // Acquire locks from leaf to root
        for (int lock_index : path) {
            locks_[lock_index]->lock();
        }
    }

    bool try_lock_impl() {
        int tid = thread_id::get();
        if (tid >= N) {
            return false;
        }

        std::vector<int> path;
        std::vector<int> acquired;
        int index = get_leaf_index();

        while (index > 0) {
            index = parent(index);
            path.push_back(index);
        }

        // Try to acquire all locks
        for (int lock_index : path) {
            if (locks_[lock_index]->try_lock()) {
                acquired.push_back(lock_index);
            } else {
                // Release all acquired locks
                for (auto it = acquired.rbegin(); it != acquired.rend(); ++it) {
                    locks_[*it]->unlock();
                }
                return false;
            }
        }

        return true;
    }

    void unlock_impl() {
        int tid = thread_id::get();
        if (tid >= N) {
            return;
        }

        // Path from root to leaf (reverse order)
        std::vector<int> path;
        int index = get_leaf_index();

        while (index > 0) {
            index = parent(index);
            path.push_back(index);
        }

        // Release locks from root to leaf
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            locks_[*it]->unlock();
        }
    }
};

// =============================================================================
// Reader-Writer Tournament Lock
// =============================================================================

template<int N>
class tournament_rw_lock {
private:
    tournament_lock<N> writer_lock_;
    std::atomic<int> reader_count_{0};
    std::atomic<bool> writer_waiting_{false};
    peterson_lock reader_entry_lock_;

public:
    class read_guard {
    private:
        tournament_rw_lock* lock_;

    public:
        explicit read_guard(tournament_rw_lock* lock) : lock_(lock) {
            lock_->read_lock();
        }

        ~read_guard() {
            if (lock_) {
                lock_->read_unlock();
            }
        }

        // Move semantics
        read_guard(read_guard&& other) noexcept : lock_(other.lock_) {
            other.lock_ = nullptr;
        }

        read_guard& operator=(read_guard&& other) noexcept {
            if (this != &other) {
                if (lock_) {
                    lock_->read_unlock();
                }
                lock_ = other.lock_;
                other.lock_ = nullptr;
            }
            return *this;
        }

        // Delete copy operations
        read_guard(const read_guard&) = delete;
        read_guard& operator=(const read_guard&) = delete;
    };

    class write_guard {
    private:
        tournament_rw_lock* lock_;

    public:
        explicit write_guard(tournament_rw_lock* lock) : lock_(lock) {
            lock_->write_lock();
        }

        ~write_guard() {
            if (lock_) {
                lock_->write_unlock();
            }
        }

        // Move semantics
        write_guard(write_guard&& other) noexcept : lock_(other.lock_) {
            other.lock_ = nullptr;
        }

        write_guard& operator=(write_guard&& other) noexcept {
            if (this != &other) {
                if (lock_) {
                    lock_->write_unlock();
                }
                lock_ = other.lock_;
                other.lock_ = nullptr;
            }
            return *this;
        }

        // Delete copy operations
        write_guard(const write_guard&) = delete;
        write_guard& operator=(const write_guard&) = delete;
    };

    void read_lock() {
        // Prevent new readers if writer is waiting
        reader_entry_lock_.lock();
        while (writer_waiting_.load()) {
            reader_entry_lock_.unlock();
            std::this_thread::yield();
            reader_entry_lock_.lock();
        }

        // First reader acquires the write lock
        if (reader_count_.fetch_add(1) == 0) {
            writer_lock_.lock();
        }
        reader_entry_lock_.unlock();
    }

    void read_unlock() {
        // Last reader releases the write lock
        if (reader_count_.fetch_sub(1) == 1) {
            writer_lock_.unlock();
        }
    }

    void write_lock() {
        writer_waiting_.store(true);
        writer_lock_.lock();
        writer_waiting_.store(false);

        // Wait for all readers to finish
        while (reader_count_.load() > 0) {
            std::this_thread::yield();
        }
    }

    void write_unlock() {
        writer_lock_.unlock();
    }

    read_guard make_read_guard() {
        return read_guard(this);
    }

    write_guard make_write_guard() {
        return write_guard(this);
    }
};

// =============================================================================
// Spin lock for comparison
// =============================================================================

class spin_lock : public basic_lock<spin_lock> {
private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

public:
    void lock_impl() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    bool try_lock_impl() {
        return !flag_.test_and_set(std::memory_order_acquire);
    }

    void unlock_impl() {
        flag_.clear(std::memory_order_release);
    }
};

// =============================================================================
// Ticket lock for fairness
// =============================================================================

class ticket_lock : public basic_lock<ticket_lock> {
private:
    std::atomic<uint64_t> ticket_{0};
    std::atomic<uint64_t> serving_{0};

public:
    void lock_impl() {
        uint64_t my_ticket = ticket_.fetch_add(1);
        while (serving_.load() != my_ticket) {
            std::this_thread::yield();
        }
    }

    bool try_lock_impl() {
        uint64_t current = serving_.load();
        uint64_t next = ticket_.load();

        if (current != next) {
            return false;
        }

        return ticket_.compare_exchange_strong(next, next + 1);
    }

    void unlock_impl() {
        serving_.fetch_add(1);
    }
};

// =============================================================================
// MCS (Mellor-Crummey and Scott) lock
// =============================================================================

class mcs_lock : public basic_lock<mcs_lock> {
private:
    struct node {
        std::atomic<bool> locked{false};
        std::atomic<node*> next{nullptr};
    };

    std::atomic<node*> tail_{nullptr};
    static thread_local node my_node_;

public:
    void lock_impl() {
        node* me = &my_node_;
        me->locked.store(true);
        me->next.store(nullptr);

        node* prev = tail_.exchange(me);
        if (prev != nullptr) {
            prev->next.store(me);
            while (me->locked.load()) {
                std::this_thread::yield();
            }
        }
    }

    bool try_lock_impl() {
        node* me = &my_node_;
        me->locked.store(false);
        me->next.store(nullptr);

        node* expected = nullptr;
        return tail_.compare_exchange_strong(expected, me);
    }

    void unlock_impl() {
        node* me = &my_node_;
        node* successor = me->next.load();

        if (successor == nullptr) {
            node* expected = me;
            if (tail_.compare_exchange_strong(expected, nullptr)) {
                return;
            }

            // Wait for successor to link itself
            while ((successor = me->next.load()) == nullptr) {
                std::this_thread::yield();
            }
        }

        successor->locked.store(false);
    }
};

thread_local mcs_lock::node mcs_lock::my_node_;

// =============================================================================
// Type-erased lock wrapper
// =============================================================================

class any_lock {
private:
    struct concept_t {
        virtual ~concept_t() = default;
        virtual void lock() = 0;
        virtual bool try_lock() = 0;
        virtual void unlock() = 0;
        virtual std::unique_ptr<concept_t> clone() const = 0;
    };

    template<typename Lock>
    struct model : concept_t {
        Lock lock_;

        template<typename... Args>
        explicit model(Args&&... args) : lock_(std::forward<Args>(args)...) {}

        void lock() override {
            lock_.lock();
        }

        bool try_lock() override {
            return lock_.try_lock();
        }

        void unlock() override {
            lock_.unlock();
        }

        std::unique_ptr<concept_t> clone() const override {
            // Locks generally can't be copied, so we return nullptr
            // or could throw an exception
            return nullptr;
        }
    };

    std::unique_ptr<concept_t> pimpl_;

public:
    template<typename Lock, typename... Args>
    static any_lock make(Args&&... args) {
        any_lock result;
        result.pimpl_ = std::make_unique<model<Lock>>(std::forward<Args>(args)...);
        return result;
    }

    void lock() {
        if (pimpl_) {
            pimpl_->lock();
        }
    }

    bool try_lock() {
        return pimpl_ && pimpl_->try_lock();
    }

    void unlock() {
        if (pimpl_) {
            pimpl_->unlock();
        }
    }

    // RAII guard
    class guard {
    private:
        any_lock* lock_;

    public:
        explicit guard(any_lock& lock) : lock_(&lock) {
            lock_->lock();
        }

        ~guard() {
            if (lock_) {
                lock_->unlock();
            }
        }

        guard(guard&& other) noexcept : lock_(other.lock_) {
            other.lock_ = nullptr;
        }

        guard& operator=(guard&& other) noexcept {
            if (this != &other) {
                if (lock_) {
                    lock_->unlock();
                }
                lock_ = other.lock_;
                other.lock_ = nullptr;
            }
            return *this;
        }

        guard(const guard&) = delete;
        guard& operator=(const guard&) = delete;
    };
};

// =============================================================================
// Lock-free utilities
// =============================================================================

template<typename T>
class lock_free_queue {
private:
    struct node {
        std::atomic<T*> data;
        std::atomic<node*> next;

        node() : data(nullptr), next(nullptr) {}
    };

    std::atomic<node*> head_;
    std::atomic<node*> tail_;

public:
    lock_free_queue() {
        node* dummy = new node;
        head_.store(dummy);
        tail_.store(dummy);
    }

    ~lock_free_queue() {
        while (node* old_head = head_.load()) {
            head_.store(old_head->next);
            delete old_head->data.load();
            delete old_head;
        }
    }

    void enqueue(T item) {
        node* new_node = new node;
        T* data = new T(std::move(item));
        new_node->data.store(data);

        node* prev_tail = tail_.exchange(new_node);
        prev_tail->next.store(new_node);
    }

    std::optional<T> dequeue() {
        node* head = head_.load();
        node* next = head->next.load();

        if (next == nullptr) {
            return std::nullopt;
        }

        T* data = next->data.exchange(nullptr);
        head_.store(next);
        delete head;

        if (data) {
            T result = std::move(*data);
            delete data;
            return result;
        }

        return std::nullopt;
    }
};

// =============================================================================
// Barrier for synchronization
// =============================================================================

class barrier {
private:
    const size_t threshold_;
    std::atomic<size_t> count_;
    std::atomic<size_t> generation_;

public:
    explicit barrier(size_t count)
        : threshold_(count), count_(count), generation_(0) {}

    void wait() {
        size_t gen = generation_.load();

        if (count_.fetch_sub(1) == 1) {
            // Last thread to arrive
            count_.store(threshold_);
            generation_.fetch_add(1);
        } else {
            // Wait for generation to change
            while (generation_.load() == gen) {
                std::this_thread::yield();
            }
        }
    }
};

// =============================================================================
// Hierarchical NUMA-Aware Lock
// =============================================================================

template<int NumNodes, int ThreadsPerNode>
class numa_aware_lock {
private:
    static constexpr int TotalThreads = NumNodes * ThreadsPerNode;

    // Two-level hierarchy: node-level and global-level locks
    std::array<tournament_lock<ThreadsPerNode>, NumNodes> node_locks_;
    tournament_lock<NumNodes> global_lock_;
    std::atomic<int> node_owners_[NumNodes];
    std::atomic<bool> node_has_waiters_[NumNodes];

    // Get NUMA node for current thread
    int get_numa_node() const {
        int tid = thread_id::get();
        return tid / ThreadsPerNode;
    }

    // Get local thread ID within NUMA node
    int get_local_tid() const {
        int tid = thread_id::get();
        return tid % ThreadsPerNode;
    }

public:
    numa_aware_lock() {
        for (int i = 0; i < NumNodes; ++i) {
            node_owners_[i].store(-1);
            node_has_waiters_[i].store(false);
        }
    }

    void lock() {
        int node = get_numa_node();
        int local_tid = get_local_tid();

        // First, acquire local node lock
        thread_id::set(local_tid);  // Set thread ID for local lock
        node_locks_[node].lock();

        // Check if we're the first from our node
        int expected = -1;
        if (node_owners_[node].compare_exchange_strong(expected, thread_id::get())) {
            // We're the node representative, acquire global lock
            thread_id::set(node);  // Set node ID for global lock
            global_lock_.lock();
        } else {
            // Another thread from our node holds the global lock
            node_has_waiters_[node].store(true);
        }

        // Restore original thread ID
        thread_id::set(node * ThreadsPerNode + local_tid);
    }

    void unlock() {
        int node = get_numa_node();
        int local_tid = get_local_tid();
        int tid = thread_id::get();

        // Check if we hold the global lock
        if (node_owners_[node].load() == tid) {
            // Check if there are waiters from our node
            if (node_has_waiters_[node].load()) {
                // Pass ownership to another thread in our node
                node_has_waiters_[node].store(false);
                // Don't release global lock - it's passed to next waiter
            } else {
                // Release global lock
                thread_id::set(node);
                global_lock_.unlock();
                node_owners_[node].store(-1);
            }
        }

        // Release local node lock
        thread_id::set(local_tid);
        node_locks_[node].unlock();
        thread_id::set(tid);  // Restore thread ID
    }
};

// =============================================================================
// Adaptive Spinning Lock - Switches between spinning and blocking
// =============================================================================

class adaptive_lock {
private:
    std::atomic<bool> locked_{false};
    std::atomic<int> spin_count_{0};
    std::atomic<double> success_rate_{0.5};
    static constexpr int initial_spins_ = 1000;
    static constexpr int max_spins_ = 10000;
    static constexpr int min_spins_ = 100;
    static constexpr double alpha_ = 0.1;  // Learning rate

    int calculate_spin_count() const {
        double rate = success_rate_.load();
        int spins = static_cast<int>(initial_spins_ * (0.5 + rate));
        return std::min(max_spins_, std::max(min_spins_, spins));
    }

    void update_success_rate(bool success) {
        double current = success_rate_.load();
        double new_rate = (1 - alpha_) * current + alpha_ * (success ? 1.0 : 0.0);
        success_rate_.store(new_rate);
    }

public:
    void lock() {
        int spins = calculate_spin_count();
        int spin_attempts = 0;
        bool acquired_by_spinning = false;

        // Adaptive spinning phase
        while (spin_attempts < spins) {
            bool expected = false;
            if (locked_.compare_exchange_weak(expected, true,
                                             std::memory_order_acquire,
                                             std::memory_order_relaxed)) {
                acquired_by_spinning = true;
                break;
            }

            // Exponential backoff
            for (int i = 0; i < std::min(spin_attempts, 16); ++i) {
                std::this_thread::yield();
            }

            spin_attempts++;
        }

        // Update statistics
        if (acquired_by_spinning) {
            update_success_rate(true);
            return;
        }

        // Fall back to blocking
        update_success_rate(false);

        while (true) {
            bool expected = false;
            if (locked_.compare_exchange_weak(expected, true,
                                             std::memory_order_acquire,
                                             std::memory_order_relaxed)) {
                break;
            }

            // Sleep for increasing durations
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    bool try_lock() {
        bool expected = false;
        return locked_.compare_exchange_strong(expected, true,
                                              std::memory_order_acquire,
                                              std::memory_order_relaxed);
    }

    void unlock() {
        locked_.store(false, std::memory_order_release);
    }
};

// =============================================================================
// Priority Lock with Inheritance
// =============================================================================

class priority_lock {
private:
    struct waiter {
        int priority;
        int thread_id;
        std::atomic<bool> ready{false};

        waiter(int p, int tid) : priority(p), thread_id(tid) {}
    };

    std::atomic<bool> locked_{false};
    std::atomic<int> owner_tid_{-1};
    std::atomic<int> owner_priority_{0};
    std::vector<std::unique_ptr<waiter>> waiters_;
    std::mutex waiters_mutex_;

    void boost_priority(int new_priority) {
        int current = owner_priority_.load();
        while (current < new_priority) {
            if (owner_priority_.compare_exchange_weak(current, new_priority)) {
                break;
            }
        }
    }

public:
    void lock(int priority = 0) {
        int tid = thread_id::get();

        // Fast path - try to acquire immediately
        bool expected = false;
        if (locked_.compare_exchange_strong(expected, true,
                                           std::memory_order_acquire)) {
            owner_tid_.store(tid);
            owner_priority_.store(priority);
            return;
        }

        // Slow path - add to waiters queue
        auto my_waiter = std::make_unique<waiter>(priority, tid);
        waiter* waiter_ptr = my_waiter.get();

        {
            std::lock_guard<std::mutex> guard(waiters_mutex_);
            waiters_.push_back(std::move(my_waiter));

            // Priority inheritance - boost owner's priority
            if (priority > owner_priority_.load()) {
                boost_priority(priority);
            }
        }

        // Wait until signaled
        while (!waiter_ptr->ready.load()) {
            std::this_thread::yield();
        }

        // We've been selected - acquire the lock
        locked_.store(true, std::memory_order_acquire);
        owner_tid_.store(tid);
        owner_priority_.store(priority);
    }

    void unlock() {
        int tid = thread_id::get();
        if (owner_tid_.load() != tid) {
            return;  // Not the owner
        }

        std::lock_guard<std::mutex> guard(waiters_mutex_);

        if (waiters_.empty()) {
            // No waiters - just release
            owner_tid_.store(-1);
            owner_priority_.store(0);
            locked_.store(false, std::memory_order_release);
            return;
        }

        // Find highest priority waiter
        auto best = waiters_.begin();
        for (auto it = waiters_.begin(); it != waiters_.end(); ++it) {
            if ((*it)->priority > (*best)->priority) {
                best = it;
            }
        }

        // Signal the selected waiter
        (*best)->ready.store(true);

        // Remove from waiters list
        waiters_.erase(best);

        // Reset our priority
        owner_tid_.store(-1);
        owner_priority_.store(0);
    }

    bool try_lock(int priority = 0) {
        int tid = thread_id::get();
        bool expected = false;

        if (locked_.compare_exchange_strong(expected, true,
                                           std::memory_order_acquire)) {
            owner_tid_.store(tid);
            owner_priority_.store(priority);
            return true;
        }

        return false;
    }
};

// =============================================================================
// Delegation Lock - Threads delegate critical section execution
// =============================================================================

template<typename T>
class delegation_lock {
public:
    using operation_t = std::function<T()>;

private:
    struct request {
        operation_t operation;
        std::atomic<bool> done{false};
        T result;
        int thread_id;

        request(operation_t op, int tid)
            : operation(std::move(op)), thread_id(tid) {}
    };

    std::atomic<bool> locked_{false};
    std::queue<std::unique_ptr<request>> requests_;
    std::mutex queue_mutex_;
    std::atomic<int> combiner_tid_{-1};

    void process_requests() {
        std::vector<std::unique_ptr<request>> batch;

        {
            std::lock_guard<std::mutex> guard(queue_mutex_);
            while (!requests_.empty()) {
                batch.push_back(std::move(requests_.front()));
                requests_.pop();
            }
        }

        // Execute all operations in batch
        for (auto& req : batch) {
            req->result = req->operation();
            req->done.store(true);
        }
    }

public:
    T execute(operation_t operation) {
        int tid = thread_id::get();

        // Try to become the combiner
        bool expected = false;
        if (locked_.compare_exchange_strong(expected, true,
                                           std::memory_order_acquire)) {
            combiner_tid_.store(tid);

            // Execute our operation
            T result = operation();

            // Process pending requests
            process_requests();

            // Release combiner role
            combiner_tid_.store(-1);
            locked_.store(false, std::memory_order_release);

            return result;
        }

        // Add request to queue and wait
        auto req = std::make_unique<request>(std::move(operation), tid);
        request* req_ptr = req.get();

        {
            std::lock_guard<std::mutex> guard(queue_mutex_);
            requests_.push(std::move(req));
        }

        // Wait for combiner to execute our operation
        while (!req_ptr->done.load()) {
            // Check if we should become combiner (previous one might have crashed)
            expected = false;
            if (locked_.compare_exchange_strong(expected, true,
                                               std::memory_order_acquire)) {
                combiner_tid_.store(tid);

                // Process all requests including ours
                process_requests();

                combiner_tid_.store(-1);
                locked_.store(false, std::memory_order_release);
                break;
            }

            std::this_thread::yield();
        }

        return req_ptr->result;
    }
};

// =============================================================================
// Combining Lock - Multiple operations combined into one
// =============================================================================

template<typename T>
class combining_tree_lock {
private:
    static constexpr int MAX_THREADS = 64;

    struct node {
        enum class status { IDLE, FIRST, SECOND, RESULT, ROOT };

        std::atomic<status> state{status::IDLE};
        std::atomic<bool> locked{false};
        T first_value;
        T second_value;
        T result;
        std::function<T(T, T)> combine_fn;
        int id;

        node(int node_id) : id(node_id) {}
    };

    std::array<node, 2 * MAX_THREADS - 1> tree_;
    std::function<T(T, T)> combine_fn_;
    std::atomic<bool> root_lock_{false};

    int get_leaf_id(int tid) const {
        return MAX_THREADS - 1 + tid;
    }

    int parent(int id) const {
        return (id - 1) / 2;
    }

public:
    explicit combining_tree_lock(std::function<T(T, T)> fn)
        : combine_fn_(fn) {
        for (int i = 0; i < tree_.size(); ++i) {
            tree_[i] = node(i);
            tree_[i].combine_fn = fn;
        }
    }

    T combine_and_execute(T value) {
        int tid = thread_id::get();
        int node_id = get_leaf_id(tid);
        node* current = &tree_[node_id];

        // Traverse up the tree
        while (node_id > 0) {
            int parent_id = parent(node_id);
            node* parent_node = &tree_[parent_id];

            // Try to be first
            auto expected = node::status::IDLE;
            if (parent_node->state.compare_exchange_strong(expected,
                                                          node::status::FIRST)) {
                parent_node->first_value = value;

                // Wait for second
                while (parent_node->state.load() != node::status::SECOND &&
                       parent_node->state.load() != node::status::ROOT) {
                    std::this_thread::yield();
                }

                if (parent_node->state.load() == node::status::ROOT) {
                    // We're at root, return combined value
                    return parent_node->result;
                }

                // Combine and continue up
                value = combine_fn_(parent_node->first_value,
                                  parent_node->second_value);
                parent_node->state.store(node::status::IDLE);
                node_id = parent_id;

            } else if (expected == node::status::FIRST) {
                // Be second
                parent_node->second_value = value;
                parent_node->state.store(node::status::SECOND);

                // Wait for result
                while (parent_node->state.load() != node::status::RESULT) {
                    std::this_thread::yield();
                }

                T result = parent_node->result;
                parent_node->state.store(node::status::IDLE);
                return result;
            }
        }

        // At root
        bool expected_lock = false;
        while (!root_lock_.compare_exchange_weak(expected_lock, true)) {
            expected_lock = false;
            std::this_thread::yield();
        }

        // Execute at root
        T result = value;  // Or execute some operation

        // Propagate result down
        tree_[0].result = result;
        tree_[0].state.store(node::status::ROOT);

        root_lock_.store(false);
        return result;
    }
};

} // namespace stepanov::synchronization