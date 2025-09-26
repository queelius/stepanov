#pragma once

#include <concepts>
#include <type_traits>
#include <iterator>
#include <utility>
#include <cstddef>
#include <memory>
#include <functional>
#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include "iterators.hpp"
#include "concepts.hpp"

namespace stepanov {

// ================ Iterator Adaptors ================

// Reverse iterator - iterate backwards through bidirectional iterators
template<iterator_bidirectional I>
class reverse_iterator {
    I current_;

public:
    using iterator_type = I;
    using value_type = std::iter_value_t<I>;
    using difference_type = std::iter_difference_t<I>;
    using reference = std::iter_reference_t<I>;
    using pointer = typename std::iterator_traits<I>::pointer;
    using iterator_category = typename std::iterator_traits<I>::iterator_category;

    constexpr reverse_iterator() = default;
    constexpr explicit reverse_iterator(I i) : current_(i) {}

    template<typename U>
        requires std::convertible_to<U, I>
    constexpr reverse_iterator(const reverse_iterator<U>& other)
        : current_(other.base()) {}

    constexpr I base() const { return current_; }

    constexpr reference operator*() const {
        I tmp = current_;
        return *--tmp;
    }

    constexpr pointer operator->() const {
        I tmp = current_;
        --tmp;
        return std::to_address(tmp);
    }

    constexpr reverse_iterator& operator++() {
        --current_;
        return *this;
    }

    constexpr reverse_iterator operator++(int) {
        reverse_iterator tmp = *this;
        --current_;
        return tmp;
    }

    constexpr reverse_iterator& operator--() {
        ++current_;
        return *this;
    }

    constexpr reverse_iterator operator--(int) {
        reverse_iterator tmp = *this;
        ++current_;
        return tmp;
    }

    constexpr reverse_iterator& operator+=(difference_type n)
        requires iterator_random_access<I>
    {
        current_ -= n;
        return *this;
    }

    constexpr reverse_iterator& operator-=(difference_type n)
        requires iterator_random_access<I>
    {
        current_ += n;
        return *this;
    }

    constexpr reference operator[](difference_type n) const
        requires iterator_random_access<I>
    {
        return *(*this + n);
    }

    friend constexpr bool operator==(const reverse_iterator& x, const reverse_iterator& y) {
        return x.current_ == y.current_;
    }

    friend constexpr bool operator!=(const reverse_iterator& x, const reverse_iterator& y) {
        return !(x == y);
    }

    friend constexpr bool operator<(const reverse_iterator& x, const reverse_iterator& y)
        requires iterator_random_access<I>
    {
        return y.current_ < x.current_;
    }

    friend constexpr bool operator>(const reverse_iterator& x, const reverse_iterator& y)
        requires iterator_random_access<I>
    {
        return y < x;
    }

    friend constexpr bool operator<=(const reverse_iterator& x, const reverse_iterator& y)
        requires iterator_random_access<I>
    {
        return !(y < x);
    }

    friend constexpr bool operator>=(const reverse_iterator& x, const reverse_iterator& y)
        requires iterator_random_access<I>
    {
        return !(x < y);
    }

    friend constexpr reverse_iterator operator+(const reverse_iterator& i, difference_type n)
        requires iterator_random_access<I>
    {
        return reverse_iterator(i.current_ - n);
    }

    friend constexpr reverse_iterator operator+(difference_type n, const reverse_iterator& i)
        requires iterator_random_access<I>
    {
        return i + n;
    }

    friend constexpr reverse_iterator operator-(const reverse_iterator& i, difference_type n)
        requires iterator_random_access<I>
    {
        return reverse_iterator(i.current_ + n);
    }

    friend constexpr difference_type operator-(const reverse_iterator& x, const reverse_iterator& y)
        requires iterator_random_access<I>
    {
        return y.current_ - x.current_;
    }
};

template<typename I>
reverse_iterator(I) -> reverse_iterator<I>;

// Move iterator - moves elements instead of copying
template<input_iterator I>
class move_iterator {
    I current_;

public:
    using iterator_type = I;
    using value_type = std::iter_value_t<I>;
    using difference_type = std::iter_difference_t<I>;
    using reference = std::iter_rvalue_reference_t<I>;
    using pointer = I;
    using iterator_category = typename std::iterator_traits<I>::iterator_category;

    constexpr move_iterator() = default;
    constexpr explicit move_iterator(I i) : current_(i) {}

    template<typename U>
        requires std::convertible_to<U, I>
    constexpr move_iterator(const move_iterator<U>& other)
        : current_(other.base()) {}

    constexpr I base() const { return current_; }

    constexpr reference operator*() const {
        return std::move(*current_);
    }

    constexpr pointer operator->() const {
        return current_;
    }

    constexpr move_iterator& operator++() {
        ++current_;
        return *this;
    }

    constexpr move_iterator operator++(int) {
        move_iterator tmp = *this;
        ++current_;
        return tmp;
    }

    constexpr move_iterator& operator--()
        requires iterator_bidirectional<I>
    {
        --current_;
        return *this;
    }

    constexpr move_iterator operator--(int)
        requires iterator_bidirectional<I>
    {
        move_iterator tmp = *this;
        --current_;
        return tmp;
    }

    constexpr move_iterator& operator+=(difference_type n)
        requires iterator_random_access<I>
    {
        current_ += n;
        return *this;
    }

    constexpr move_iterator& operator-=(difference_type n)
        requires iterator_random_access<I>
    {
        current_ -= n;
        return *this;
    }

    constexpr reference operator[](difference_type n) const
        requires iterator_random_access<I>
    {
        return std::move(current_[n]);
    }

    friend constexpr bool operator==(const move_iterator& x, const move_iterator& y) {
        return x.current_ == y.current_;
    }

    friend constexpr bool operator!=(const move_iterator& x, const move_iterator& y) {
        return !(x == y);
    }

    friend constexpr bool operator<(const move_iterator& x, const move_iterator& y)
        requires iterator_random_access<I>
    {
        return x.current_ < y.current_;
    }

    friend constexpr bool operator>(const move_iterator& x, const move_iterator& y)
        requires iterator_random_access<I>
    {
        return y < x;
    }

    friend constexpr bool operator<=(const move_iterator& x, const move_iterator& y)
        requires iterator_random_access<I>
    {
        return !(y < x);
    }

    friend constexpr bool operator>=(const move_iterator& x, const move_iterator& y)
        requires iterator_random_access<I>
    {
        return !(x < y);
    }

    friend constexpr move_iterator operator+(const move_iterator& i, difference_type n)
        requires iterator_random_access<I>
    {
        return move_iterator(i.current_ + n);
    }

    friend constexpr move_iterator operator+(difference_type n, const move_iterator& i)
        requires iterator_random_access<I>
    {
        return i + n;
    }

    friend constexpr move_iterator operator-(const move_iterator& i, difference_type n)
        requires iterator_random_access<I>
    {
        return move_iterator(i.current_ - n);
    }

    friend constexpr difference_type operator-(const move_iterator& x, const move_iterator& y)
        requires iterator_random_access<I>
    {
        return x.current_ - y.current_;
    }
};

template<typename I>
move_iterator(I) -> move_iterator<I>;

// Insert iterator adaptors
template<typename Container>
class back_insert_iterator_stepanov {
protected:
    Container* container_;

public:
    using value_type = void;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = void;
    using iterator_category = std::output_iterator_tag;
    using container_type = Container;

    constexpr back_insert_iterator_stepanov() : container_(nullptr) {}
    constexpr explicit back_insert_iterator_stepanov(Container& c) : container_(&c) {}

    constexpr back_insert_iterator_stepanov& operator=(const typename Container::value_type& value) {
        container_->push_back(value);
        return *this;
    }

    constexpr back_insert_iterator_stepanov& operator=(typename Container::value_type&& value) {
        container_->push_back(std::move(value));
        return *this;
    }

    constexpr back_insert_iterator_stepanov& operator*() { return *this; }
    constexpr back_insert_iterator_stepanov& operator++() { return *this; }
    constexpr back_insert_iterator_stepanov operator++(int) { return *this; }
};

template<typename Container>
constexpr back_insert_iterator_stepanov<Container> back_inserter_stepanov(Container& c) {
    return back_insert_iterator_stepanov<Container>(c);
}

template<typename Container>
class front_insert_iterator_stepanov {
protected:
    Container* container_;

public:
    using value_type = void;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = void;
    using iterator_category = std::output_iterator_tag;
    using container_type = Container;

    constexpr front_insert_iterator_stepanov() : container_(nullptr) {}
    constexpr explicit front_insert_iterator_stepanov(Container& c) : container_(&c) {}

    constexpr front_insert_iterator_stepanov& operator=(const typename Container::value_type& value) {
        container_->push_front(value);
        return *this;
    }

    constexpr front_insert_iterator_stepanov& operator=(typename Container::value_type&& value) {
        container_->push_front(std::move(value));
        return *this;
    }

    constexpr front_insert_iterator_stepanov& operator*() { return *this; }
    constexpr front_insert_iterator_stepanov& operator++() { return *this; }
    constexpr front_insert_iterator_stepanov operator++(int) { return *this; }
};

template<typename Container>
constexpr front_insert_iterator_stepanov<Container> front_inserter_stepanov(Container& c) {
    return front_insert_iterator_stepanov<Container>(c);
}

template<typename Container>
class insert_iterator_stepanov {
protected:
    Container* container_;
    typename Container::iterator iter_;

public:
    using value_type = void;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = void;
    using iterator_category = std::output_iterator_tag;
    using container_type = Container;

    insert_iterator_stepanov() : container_(nullptr) {}
    insert_iterator_stepanov(Container& c, typename Container::iterator i)
        : container_(&c), iter_(i) {}

    insert_iterator_stepanov& operator=(const typename Container::value_type& value) {
        iter_ = container_->insert(iter_, value);
        ++iter_;
        return *this;
    }

    insert_iterator_stepanov& operator=(typename Container::value_type&& value) {
        iter_ = container_->insert(iter_, std::move(value));
        ++iter_;
        return *this;
    }

    insert_iterator_stepanov& operator*() { return *this; }
    insert_iterator_stepanov& operator++() { return *this; }
    insert_iterator_stepanov operator++(int) { return *this; }
};

template<typename Container>
insert_iterator_stepanov<Container> inserter_stepanov(Container& c, typename Container::iterator i) {
    return insert_iterator_stepanov<Container>(c, i);
}

// Function iterator - generates values from a function
template<typename F, typename State = std::size_t>
class function_iterator {
    F func_;
    State state_;

public:
    using value_type = std::decay_t<decltype(std::declval<F>()(std::declval<State>()))>;
    using reference = value_type;
    using pointer = value_type*;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;

    function_iterator() = default;
    function_iterator(F f, State s = State{}) : func_(f), state_(s) {}

    reference operator*() const { return func_(state_); }

    function_iterator& operator++() {
        ++state_;
        return *this;
    }

    function_iterator operator++(int) {
        function_iterator tmp = *this;
        ++*this;
        return tmp;
    }

    friend bool operator==(const function_iterator& x, const function_iterator& y) {
        return x.state_ == y.state_;
    }

    friend bool operator!=(const function_iterator& x, const function_iterator& y) {
        return !(x == y);
    }
};

template<typename F, typename State>
function_iterator(F, State) -> function_iterator<F, State>;

// Permutation iterator - permutes elements during iteration
template<iterator_random_access I, iterator_random_access IndexIterator>
class permutation_iterator {
    I base_;
    IndexIterator index_;

public:
    using value_type = std::iter_value_t<I>;
    using reference = std::iter_reference_t<I>;
    using pointer = typename std::iterator_traits<I>::pointer;
    using difference_type = std::iter_difference_t<IndexIterator>;
    using iterator_category = typename std::iterator_traits<IndexIterator>::iterator_category;

    permutation_iterator() = default;
    permutation_iterator(I base, IndexIterator index) : base_(base), index_(index) {}

    reference operator*() const { return base_[*index_]; }
    pointer operator->() const { return std::to_address(base_ + *index_); }

    permutation_iterator& operator++() {
        ++index_;
        return *this;
    }

    permutation_iterator operator++(int) {
        permutation_iterator tmp = *this;
        ++*this;
        return tmp;
    }

    permutation_iterator& operator--()
        requires iterator_bidirectional<IndexIterator>
    {
        --index_;
        return *this;
    }

    permutation_iterator operator--(int)
        requires iterator_bidirectional<IndexIterator>
    {
        permutation_iterator tmp = *this;
        --*this;
        return tmp;
    }

    permutation_iterator& operator+=(difference_type n) {
        index_ += n;
        return *this;
    }

    permutation_iterator& operator-=(difference_type n) {
        index_ -= n;
        return *this;
    }

    reference operator[](difference_type n) const {
        return base_[index_[n]];
    }

    friend bool operator==(const permutation_iterator& x, const permutation_iterator& y) {
        return x.index_ == y.index_;
    }

    friend bool operator!=(const permutation_iterator& x, const permutation_iterator& y) {
        return !(x == y);
    }

    friend bool operator<(const permutation_iterator& x, const permutation_iterator& y) {
        return x.index_ < y.index_;
    }

    friend bool operator>(const permutation_iterator& x, const permutation_iterator& y) {
        return y < x;
    }

    friend bool operator<=(const permutation_iterator& x, const permutation_iterator& y) {
        return !(y < x);
    }

    friend bool operator>=(const permutation_iterator& x, const permutation_iterator& y) {
        return !(x < y);
    }

    friend permutation_iterator operator+(const permutation_iterator& i, difference_type n) {
        return permutation_iterator(i.base_, i.index_ + n);
    }

    friend permutation_iterator operator+(difference_type n, const permutation_iterator& i) {
        return i + n;
    }

    friend permutation_iterator operator-(const permutation_iterator& i, difference_type n) {
        return permutation_iterator(i.base_, i.index_ - n);
    }

    friend difference_type operator-(const permutation_iterator& x, const permutation_iterator& y) {
        return x.index_ - y.index_;
    }
};

// Stride iterator - skips elements
template<iterator_forward I>
class stride_iterator {
    I current_;
    I end_;
    std::iter_difference_t<I> stride_;

    void advance_to_end() {
        if (stride_ > 0) {
            auto dist = iterator_distance(current_, end_);
            if (dist < stride_) {
                current_ = end_;
            } else {
                iterator_advance(current_, stride_);
            }
        }
    }

public:
    using value_type = std::iter_value_t<I>;
    using reference = std::iter_reference_t<I>;
    using pointer = typename std::iterator_traits<I>::pointer;
    using difference_type = std::iter_difference_t<I>;
    using iterator_category = typename std::iterator_traits<I>::iterator_category;

    stride_iterator() = default;
    stride_iterator(I current, I end, difference_type stride)
        : current_(current), end_(end), stride_(stride) {}

    reference operator*() const { return *current_; }
    pointer operator->() const { return std::to_address(current_); }

    stride_iterator& operator++() {
        advance_to_end();
        return *this;
    }

    stride_iterator operator++(int) {
        stride_iterator tmp = *this;
        ++*this;
        return tmp;
    }

    friend bool operator==(const stride_iterator& x, const stride_iterator& y) {
        return x.current_ == y.current_;
    }

    friend bool operator!=(const stride_iterator& x, const stride_iterator& y) {
        return !(x == y);
    }
};

// Circular iterator - wraps around at the end
template<iterator_forward I>
class circular_iterator {
    I current_;
    I begin_;
    I end_;
    bool complete_;

public:
    using value_type = std::iter_value_t<I>;
    using reference = std::iter_reference_t<I>;
    using pointer = typename std::iterator_traits<I>::pointer;
    using difference_type = std::iter_difference_t<I>;
    using iterator_category = std::forward_iterator_tag;

    circular_iterator() = default;
    circular_iterator(I current, I begin, I end, bool complete = false)
        : current_(current), begin_(begin), end_(end), complete_(complete) {}

    reference operator*() const { return *current_; }
    pointer operator->() const { return std::to_address(current_); }

    circular_iterator& operator++() {
        ++current_;
        if (current_ == end_) {
            current_ = begin_;
            complete_ = true;
        }
        return *this;
    }

    circular_iterator operator++(int) {
        circular_iterator tmp = *this;
        ++*this;
        return tmp;
    }

    friend bool operator==(const circular_iterator& x, const circular_iterator& y) {
        return x.current_ == y.current_ && x.complete_ == y.complete_;
    }

    friend bool operator!=(const circular_iterator& x, const circular_iterator& y) {
        return !(x == y);
    }
};

// I/O Stream iterators
template<typename T, typename CharT = char, typename Traits = std::char_traits<CharT>>
class istream_iterator {
    std::basic_istream<CharT, Traits>* stream_;
    T value_;
    bool end_of_stream_;

    void read() {
        if (stream_ && !(*stream_ >> value_)) {
            stream_ = nullptr;
            end_of_stream_ = true;
        }
    }

public:
    using value_type = T;
    using reference = const T&;
    using pointer = const T*;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;
    using char_type = CharT;
    using traits_type = Traits;
    using istream_type = std::basic_istream<CharT, Traits>;

    istream_iterator() : stream_(nullptr), end_of_stream_(true) {}
    istream_iterator(istream_type& stream)
        : stream_(&stream), end_of_stream_(false) {
        read();
    }

    reference operator*() const { return value_; }
    pointer operator->() const { return &value_; }

    istream_iterator& operator++() {
        read();
        return *this;
    }

    istream_iterator operator++(int) {
        istream_iterator tmp = *this;
        ++*this;
        return tmp;
    }

    friend bool operator==(const istream_iterator& x, const istream_iterator& y) {
        return x.stream_ == y.stream_;
    }

    friend bool operator!=(const istream_iterator& x, const istream_iterator& y) {
        return !(x == y);
    }
};

template<typename T, typename CharT = char, typename Traits = std::char_traits<CharT>>
class ostream_iterator {
    std::basic_ostream<CharT, Traits>* stream_;
    const CharT* delim_;

public:
    using value_type = void;
    using reference = void;
    using pointer = void;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::output_iterator_tag;
    using char_type = CharT;
    using traits_type = Traits;
    using ostream_type = std::basic_ostream<CharT, Traits>;

    ostream_iterator(ostream_type& stream, const CharT* delim = nullptr)
        : stream_(&stream), delim_(delim) {}

    ostream_iterator& operator=(const T& value) {
        *stream_ << value;
        if (delim_) *stream_ << delim_;
        return *this;
    }

    ostream_iterator& operator*() { return *this; }
    ostream_iterator& operator++() { return *this; }
    ostream_iterator operator++(int) { return *this; }
};

// ================ Iterator Utilities ================

// Optimized distance function using tag dispatch
namespace detail {
    template<input_iterator I, sentinel_for<I> S>
    constexpr auto distance_impl(I first, S last, std::input_iterator_tag) {
        difference_type<I> n = 0;
        while (first != last) {
            ++n;
            ++first;
        }
        return n;
    }

    template<input_iterator I, sentinel_for<I> S>
    constexpr auto distance_impl(I first, S last, std::random_access_iterator_tag) {
        return last - first;
    }
}

template<input_iterator I, sentinel_for<I> S>
constexpr auto distance(I first, S last) {
    if constexpr (sized_sentinel_for<S, I>) {
        return last - first;
    } else {
        using category = typename std::iterator_traits<I>::iterator_category;
        return detail::distance_impl(first, last, category{});
    }
}

// Optimized advance with bounds checking
template<input_iterator I>
constexpr void advance(I& i, difference_type<I> n) {
    iterator_advance(i, n);
}

template<input_iterator I, sentinel_for<I> S>
constexpr void advance(I& i, difference_type<I> n, S bound) {
    iterator_advance_to(i, n, bound);
}

// Next and prev with bounds checking
template<input_iterator I>
constexpr I next(I i, difference_type<I> n = 1) {
    iterator_advance(i, n);
    return i;
}

template<input_iterator I, sentinel_for<I> S>
constexpr I next(I i, S bound) {
    if (i != bound) ++i;
    return i;
}

template<input_iterator I, sentinel_for<I> S>
constexpr I next(I i, difference_type<I> n, S bound) {
    iterator_advance_to(i, n, bound);
    return i;
}

template<iterator_bidirectional I>
constexpr I prev(I i, difference_type<I> n = 1) {
    iterator_advance(i, -n);
    return i;
}

// Iterator swap
template<typename I1, typename I2>
    requires std::indirectly_swappable<I1, I2>
constexpr void iter_swap(I1 a, I2 b) {
    using std::swap;
    swap(*a, *b);
}

// Make move iterator helper
template<input_iterator I>
constexpr move_iterator<I> make_move_iterator(I i) {
    return move_iterator<I>(i);
}

// Make reverse iterator helper
template<iterator_bidirectional I>
constexpr reverse_iterator<I> make_reverse_iterator(I i) {
    return reverse_iterator<I>(i);
}

// Sentinel types for different termination conditions

// Null-terminated sentinel
template<typename T>
struct null_sentinel {
    bool operator==(const T* ptr) const {
        return *ptr == T{};
    }

    friend bool operator==(const T* ptr, null_sentinel s) {
        return s == ptr;
    }
};

// Predicate-based sentinel
template<typename Pred>
class predicate_sentinel {
    Pred pred_;

public:
    explicit predicate_sentinel(Pred p) : pred_(p) {}

    template<typename I>
        requires std::invocable<const Pred&, std::iter_reference_t<I>>
    bool operator==(const I& i) const {
        return pred_(*i);
    }

    template<typename I>
        requires std::invocable<const Pred&, std::iter_reference_t<I>>
    friend bool operator==(const I& i, const predicate_sentinel& s) {
        return s == i;
    }
};

template<typename Pred>
predicate_sentinel(Pred) -> predicate_sentinel<Pred>;

// Counted sentinel
struct counted_sentinel {
    std::ptrdiff_t count;

    template<typename I>
    friend bool operator==(const counted_iterator<I>& i, counted_sentinel s) {
        return i.count() == s.count;
    }

    template<typename I>
    friend bool operator==(counted_sentinel s, const counted_iterator<I>& i) {
        return i == s;
    }

    template<typename I>
    friend std::iter_difference_t<I> operator-(const counted_iterator<I>& i, counted_sentinel s) {
        return s.count - i.count();
    }

    template<typename I>
    friend std::iter_difference_t<I> operator-(counted_sentinel s, const counted_iterator<I>& i) {
        return i.count() - s.count;
    }
};

} // namespace stepanov