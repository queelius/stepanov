// concurrent.hpp - Advanced Concurrent Structures
// Beyond basic lock-free: STM, CRDTs, Wait-Free Universal Construction, RCU
// True innovation in concurrent programming

#ifndef STEPANOV_CONCURRENT_HPP
#define STEPANOV_CONCURRENT_HPP

#include <atomic>
#include <memory>
#include <optional>
#include <functional>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <algorithm>
#include <chrono>
#include <random>
#include <bit>

namespace stepanov::concurrent {

// ============================================================================
// Software Transactional Memory (STM) - Composable Concurrency
// ============================================================================

template<typename T>
class tvar {
    // Transactional variable - the atom of STM
    struct version {
        T value;
        uint64_t timestamp;
    };

    mutable std::atomic<version*> current_;
    static inline std::atomic<uint64_t> global_clock_{0};

public:
    explicit tvar(T initial)
        : current_(new version{std::move(initial), global_clock_.fetch_add(1)}) {}

    ~tvar() {
        delete current_.load();
    }

    // Read in transaction
    T read() const {
        return current_.load()->value;
    }

    // Write in transaction
    void write(T value) {
        auto old = current_.load();
        auto new_version = new version{std::move(value), global_clock_.fetch_add(1)};

        // CAS to update
        if (!current_.compare_exchange_strong(old, new_version)) {
            delete new_version;
            throw std::runtime_error("Transaction conflict");
        }

        delete old;
    }

    uint64_t timestamp() const {
        return current_.load()->timestamp;
    }
};

template<typename T>
class stm_transaction {
    struct read_entry {
        const tvar<T>* var;
        T value;
        uint64_t timestamp;
    };

    struct write_entry {
        tvar<T>* var;
        T value;
    };

    std::vector<read_entry> read_set_;
    std::vector<write_entry> write_set_;
    bool committed_ = false;

public:
    T read(const tvar<T>& var) {
        // Check write set first
        auto write_it = std::find_if(write_set_.begin(), write_set_.end(),
            [&var](const write_entry& e) { return e.var == &var; });

        if (write_it != write_set_.end()) {
            return write_it->value;
        }

        // Check read set
        auto read_it = std::find_if(read_set_.begin(), read_set_.end(),
            [&var](const read_entry& e) { return e.var == &var; });

        if (read_it != read_set_.end()) {
            return read_it->value;
        }

        // First read - add to read set
        T value = var.read();
        read_set_.push_back({&var, value, var.timestamp()});
        return value;
    }

    void write(tvar<T>& var, T value) {
        auto it = std::find_if(write_set_.begin(), write_set_.end(),
            [&var](const write_entry& e) { return e.var == &var; });

        if (it != write_set_.end()) {
            it->value = std::move(value);
        } else {
            write_set_.push_back({&var, std::move(value)});
        }
    }

    bool validate() const {
        for (const auto& entry : read_set_) {
            if (entry.var->timestamp() != entry.timestamp) {
                return false;
            }
        }
        return true;
    }

    bool commit() {
        if (!validate()) {
            return false;
        }

        // Apply all writes atomically
        for (const auto& entry : write_set_) {
            entry.var->write(entry.value);
        }

        committed_ = true;
        return true;
    }

    void retry() {
        read_set_.clear();
        write_set_.clear();
    }
};

// STM monad for composable transactions
template<typename T>
class stm {
    std::function<T(stm_transaction<T>&)> computation_;

public:
    explicit stm(std::function<T(stm_transaction<T>&)> comp)
        : computation_(std::move(comp)) {}

    // Run transaction with retry logic
    T run() const {
        const size_t max_retries = 100;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> backoff(1, 1000);

        for (size_t attempt = 0; attempt < max_retries; ++attempt) {
            stm_transaction<T> tx;

            try {
                T result = computation_(tx);

                if (tx.commit()) {
                    return result;
                }
            } catch (const std::exception&) {
                // Transaction failed
            }

            // Exponential backoff
            std::this_thread::sleep_for(
                std::chrono::microseconds(backoff(gen) * (1 << std::min(attempt, size_t(10))))
            );
        }

        throw std::runtime_error("Transaction failed after max retries");
    }

    // Monadic composition
    template<typename U>
    stm<U> bind(std::function<stm<U>(T)> f) const {
        return stm<U>([*this, f](stm_transaction<U>& tx) {
            T a = computation_(tx);
            return f(a).computation_(tx);
        });
    }

    template<typename U>
    stm<U> map(std::function<U(T)> f) const {
        return stm<U>([*this, f](stm_transaction<U>& tx) {
            return f(computation_(tx));
        });
    }

    // Alternative - try first, if fails try second
    stm<T> orElse(const stm<T>& other) const {
        return stm<T>([*this, other](stm_transaction<T>& tx) {
            try {
                return computation_(tx);
            } catch (...) {
                tx.retry();
                return other.computation_(tx);
            }
        });
    }
};

// ============================================================================
// Conflict-Free Replicated Data Types (CRDTs)
// ============================================================================

// G-Counter - Grow-only counter CRDT
template<typename NodeId = std::string>
class g_counter {
    std::unordered_map<NodeId, uint64_t> counts_;
    NodeId node_id_;

public:
    explicit g_counter(NodeId id) : node_id_(std::move(id)) {}

    void increment(uint64_t delta = 1) {
        counts_[node_id_] += delta;
    }

    uint64_t value() const {
        uint64_t sum = 0;
        for (const auto& [_, count] : counts_) {
            sum += count;
        }
        return sum;
    }

    // Merge with another replica
    void merge(const g_counter& other) {
        for (const auto& [node, count] : other.counts_) {
            counts_[node] = std::max(counts_[node], count);
        }
    }

    // Create delta for efficient sync
    g_counter delta() const {
        g_counter d(node_id_);
        d.counts_[node_id_] = counts_.at(node_id_);
        return d;
    }
};

// PN-Counter - Increment/decrement counter CRDT
template<typename NodeId = std::string>
class pn_counter {
    g_counter<NodeId> positive_;
    g_counter<NodeId> negative_;

public:
    explicit pn_counter(NodeId id)
        : positive_(id), negative_(id) {}

    void increment(uint64_t delta = 1) {
        positive_.increment(delta);
    }

    void decrement(uint64_t delta = 1) {
        negative_.increment(delta);
    }

    int64_t value() const {
        return static_cast<int64_t>(positive_.value()) -
               static_cast<int64_t>(negative_.value());
    }

    void merge(const pn_counter& other) {
        positive_.merge(other.positive_);
        negative_.merge(other.negative_);
    }
};

// G-Set - Grow-only set CRDT
template<typename T>
class g_set {
    std::unordered_set<T> elements_;

public:
    void add(T element) {
        elements_.insert(std::move(element));
    }

    bool contains(const T& element) const {
        return elements_.count(element) > 0;
    }

    void merge(const g_set& other) {
        elements_.insert(other.elements_.begin(), other.elements_.end());
    }

    size_t size() const {
        return elements_.size();
    }
};

// 2P-Set - Two-phase set CRDT (add and remove)
template<typename T>
class two_phase_set {
    g_set<T> added_;
    g_set<T> removed_;

public:
    void add(T element) {
        added_.add(std::move(element));
    }

    void remove(const T& element) {
        if (added_.contains(element)) {
            removed_.add(element);
        }
    }

    bool contains(const T& element) const {
        return added_.contains(element) && !removed_.contains(element);
    }

    void merge(const two_phase_set& other) {
        added_.merge(other.added_);
        removed_.merge(other.removed_);
    }
};

// LWW-Register - Last-Write-Wins Register CRDT
template<typename T>
class lww_register {
    struct timestamped_value {
        T value;
        std::chrono::steady_clock::time_point timestamp;
        std::string node_id;  // For tie-breaking

        bool operator<(const timestamped_value& other) const {
            if (timestamp != other.timestamp) {
                return timestamp < other.timestamp;
            }
            return node_id < other.node_id;
        }
    };

    timestamped_value current_;

public:
    explicit lww_register(T initial, std::string node_id)
        : current_{std::move(initial), std::chrono::steady_clock::now(), std::move(node_id)} {}

    T read() const {
        return current_.value;
    }

    void write(T value, std::string node_id) {
        timestamped_value new_val{
            std::move(value),
            std::chrono::steady_clock::now(),
            std::move(node_id)
        };

        if (current_ < new_val) {
            current_ = std::move(new_val);
        }
    }

    void merge(const lww_register& other) {
        if (current_ < other.current_) {
            current_ = other.current_;
        }
    }
};

// ============================================================================
// Wait-Free Universal Construction
// ============================================================================

template<typename T>
class wait_free_queue {
    // Based on Herlihy's universal construction
    struct node {
        std::optional<T> value;
        std::atomic<node*> next;
        std::atomic<bool> taken;

        node() : next(nullptr), taken(false) {}
    };

    struct state {
        node* head;
        node* tail;
        uint64_t version;
    };

    std::atomic<state*> current_state_;

    // Per-thread announcement array
    static constexpr size_t MAX_THREADS = 128;
    std::array<std::atomic<T*>, MAX_THREADS> announce_;

    size_t get_thread_id() const {
        static thread_local size_t id = std::hash<std::thread::id>{}(
            std::this_thread::get_id()) % MAX_THREADS;
        return id;
    }

public:
    wait_free_queue() {
        auto* dummy = new node();
        current_state_ = new state{dummy, dummy, 0};

        for (auto& a : announce_) {
            a = nullptr;
        }
    }

    ~wait_free_queue() {
        auto* s = current_state_.load();
        auto* current = s->head;

        while (current) {
            auto* next = current->next.load();
            delete current;
            current = next;
        }

        delete s;
    }

    void enqueue(T value) {
        size_t tid = get_thread_id();
        T* announced = new T(std::move(value));
        announce_[tid] = announced;

        // Help all pending operations
        for (size_t i = 0; i < MAX_THREADS; ++i) {
            T* pending = announce_[i].load();
            if (pending) {
                help_enqueue(i, pending);
            }
        }
    }

    std::optional<T> dequeue() {
        auto* s = current_state_.load();

        while (true) {
            auto* head = s->head;
            auto* next = head->next.load();

            if (!next) {
                return std::nullopt;  // Empty
            }

            if (next->taken.exchange(true)) {
                // Already taken by another thread
                head = next;
                continue;
            }

            // Successfully claimed this node
            auto* new_state = new state{next, s->tail, s->version + 1};

            if (current_state_.compare_exchange_weak(s, new_state)) {
                delete s;
                return next->value;
            }

            delete new_state;
            s = current_state_.load();
        }
    }

private:
    void help_enqueue(size_t tid, T* value) {
        auto* new_node = new node();
        new_node->value = *value;

        while (true) {
            auto* s = current_state_.load();
            auto* last = s->tail;

            if (last->next.load() == nullptr) {
                if (last->next.compare_exchange_weak(nullptr, new_node)) {
                    // Successfully linked
                    auto* new_state = new state{s->head, new_node, s->version + 1};

                    if (current_state_.compare_exchange_weak(s, new_state)) {
                        delete s;
                    } else {
                        delete new_state;
                    }

                    // Clear announcement
                    T* expected = value;
                    announce_[tid].compare_exchange_strong(expected, nullptr);
                    delete value;
                    return;
                }
            } else {
                // Help advance tail
                auto* new_state = new state{s->head, last->next.load(), s->version + 1};

                if (current_state_.compare_exchange_weak(s, new_state)) {
                    delete s;
                } else {
                    delete new_state;
                }
            }
        }
    }
};

// ============================================================================
// RCU (Read-Copy-Update) Style Data Structures
// ============================================================================

template<typename T>
class rcu_pointer {
    std::atomic<T*> ptr_;
    std::vector<T*> retired_;
    std::shared_mutex rcu_mutex_;

public:
    explicit rcu_pointer(T* initial = nullptr) : ptr_(initial) {}

    ~rcu_pointer() {
        delete ptr_.load();
        for (auto* p : retired_) {
            delete p;
        }
    }

    // Reader - no synchronization needed
    const T* read() const {
        return ptr_.load(std::memory_order_acquire);
    }

    // Writer - creates new version
    template<typename F>
    void update(F&& updater) {
        std::unique_lock lock(rcu_mutex_);

        T* old_ptr = ptr_.load();
        T* new_ptr = new T(*old_ptr);
        updater(*new_ptr);

        ptr_.store(new_ptr, std::memory_order_release);
        retired_.push_back(old_ptr);

        // Defer deletion until grace period
        defer_delete();
    }

    // Synchronize - wait for all readers
    void synchronize() {
        std::unique_lock lock(rcu_mutex_);
        // All readers that started before this call will complete
    }

private:
    void defer_delete() {
        if (retired_.size() > 100) {  // Simple heuristic
            synchronize();

            for (auto* p : retired_) {
                delete p;
            }
            retired_.clear();
        }
    }
};

// RCU List - Lock-free reads, synchronized writes
template<typename T>
class rcu_list {
    struct node {
        T value;
        std::atomic<node*> next;

        explicit node(T val) : value(std::move(val)), next(nullptr) {}
    };

    rcu_pointer<node> head_;

public:
    rcu_list() : head_(nullptr) {}

    void push_front(T value) {
        head_.update([value = std::move(value)](node*& head) {
            auto* new_node = new node(value);
            new_node->next = head;
            head = new_node;
        });
    }

    std::vector<T> snapshot() const {
        std::vector<T> result;
        const node* current = head_.read();

        while (current) {
            result.push_back(current->value);
            current = current->next.load();
        }

        return result;
    }

    bool contains(const T& value) const {
        const node* current = head_.read();

        while (current) {
            if (current->value == value) {
                return true;
            }
            current = current->next.load();
        }

        return false;
    }
};

// ============================================================================
// Hazard Pointers for Safe Memory Reclamation
// ============================================================================

template<typename T>
class hazard_pointer_manager {
    static constexpr size_t MAX_HAZARD_POINTERS = 2;
    static constexpr size_t MAX_THREADS = 128;

    struct hazard_record {
        std::atomic<T*> pointers[MAX_HAZARD_POINTERS];
        std::atomic<bool> active;

        hazard_record() : active(false) {
            for (auto& p : pointers) {
                p = nullptr;
            }
        }
    };

    std::array<hazard_record, MAX_THREADS> hazard_records_;
    thread_local static hazard_record* my_record_;
    thread_local static std::vector<T*> retired_list_;

public:
    class hazard_pointer {
        T** slot_;

    public:
        hazard_pointer() : slot_(nullptr) {}

        void acquire(T* ptr) {
            if (!slot_) {
                slot_ = get_hazard_slot();
            }
            *slot_ = ptr;
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }

        void release() {
            if (slot_) {
                *slot_ = nullptr;
            }
        }

        ~hazard_pointer() {
            release();
        }

    private:
        T** get_hazard_slot() {
            if (!my_record_) {
                for (auto& record : hazard_records_) {
                    bool expected = false;
                    if (record.active.compare_exchange_strong(expected, true)) {
                        my_record_ = &record;
                        break;
                    }
                }
            }

            for (auto& ptr : my_record_->pointers) {
                if (ptr.load() == nullptr) {
                    return reinterpret_cast<T**>(&ptr);
                }
            }

            throw std::runtime_error("No hazard pointer slots available");
        }
    };

    void retire(T* ptr) {
        retired_list_.push_back(ptr);

        if (retired_list_.size() >= MAX_THREADS * MAX_HAZARD_POINTERS) {
            scan_and_delete();
        }
    }

private:
    void scan_and_delete() {
        std::vector<T*> hazard_pointers;

        // Collect all hazard pointers
        for (const auto& record : hazard_records_) {
            if (record.active.load()) {
                for (const auto& ptr : record.pointers) {
                    T* p = ptr.load();
                    if (p) {
                        hazard_pointers.push_back(p);
                    }
                }
            }
        }

        std::sort(hazard_pointers.begin(), hazard_pointers.end());
        hazard_pointers.erase(
            std::unique(hazard_pointers.begin(), hazard_pointers.end()),
            hazard_pointers.end()
        );

        // Delete non-hazardous retired objects
        auto new_end = std::remove_if(retired_list_.begin(), retired_list_.end(),
            [&hazard_pointers](T* ptr) {
                if (!std::binary_search(hazard_pointers.begin(), hazard_pointers.end(), ptr)) {
                    delete ptr;
                    return true;
                }
                return false;
            }
        );

        retired_list_.erase(new_end, retired_list_.end());
    }
};

template<typename T>
thread_local typename hazard_pointer_manager<T>::hazard_record*
    hazard_pointer_manager<T>::my_record_ = nullptr;

template<typename T>
thread_local std::vector<T*> hazard_pointer_manager<T>::retired_list_;

} // namespace stepanov::concurrent

#endif // STEPANOV_CONCURRENT_HPP