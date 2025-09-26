#pragma once

#include <concepts>
#include <type_traits>
#include <iterator>
#include <utility>
#include <cstddef>
#include "concepts.hpp"

namespace stepanov {

// Iterator category concepts following Stepanov's principles
// These refine the standard concepts with additional semantic guarantees

template<typename I>
concept readable_iterator = requires(I i) {
    typename std::iter_value_t<I>;
    typename std::iter_reference_t<I>;
    { *i } -> std::convertible_to<std::iter_reference_t<I>>;
};

template<typename I, typename T>
concept writable_iterator = requires(I i, T val) {
    { *i = val } -> std::same_as<std::iter_reference_t<I>>;
};

template<typename I>
concept incrementable = regular<I> && requires(I i) {
    { ++i } -> std::same_as<I&>;
    { i++ } -> std::convertible_to<I>;
};

template<typename I>
concept input_iterator = readable_iterator<I> && incrementable<I> && requires(I i, I j) {
    { i == j } -> std::convertible_to<bool>;
    { i != j } -> std::convertible_to<bool>;
};

template<typename I>
concept iterator_forward = input_iterator<I> &&
    std::constructible_from<I> &&
    std::is_reference_v<std::iter_reference_t<I>> &&
    std::same_as<std::remove_cvref_t<std::iter_reference_t<I>>, std::iter_value_t<I>> &&
    requires(I i) {
        { i++ } -> std::convertible_to<I>;
    };

template<typename I>
concept iterator_bidirectional = iterator_forward<I> && requires(I i) {
    { --i } -> std::same_as<I&>;
    { i-- } -> std::convertible_to<I>;
};

template<typename I>
concept iterator_random_access = iterator_bidirectional<I> &&
    std::totally_ordered<I> &&
    requires(I i, I j, std::iter_difference_t<I> n) {
        { i += n } -> std::same_as<I&>;
        { i -= n } -> std::same_as<I&>;
        { i + n } -> std::same_as<I>;
        { n + i } -> std::same_as<I>;
        { i - n } -> std::same_as<I>;
        { i - j } -> std::same_as<std::iter_difference_t<I>>;
        { i[n] } -> std::convertible_to<std::iter_reference_t<I>>;
    };

template<typename I>
concept contiguous_iterator = iterator_random_access<I> &&
    std::is_lvalue_reference_v<std::iter_reference_t<I>> &&
    std::same_as<std::iter_value_t<I>, std::remove_cvref_t<std::iter_reference_t<I>>> &&
    requires(I i) {
        { std::to_address(i) } -> std::same_as<std::add_pointer_t<std::iter_reference_t<I>>>;
    };

// Sentinel concepts for end markers
template<typename S, typename I>
concept sentinel_for = std::semiregular<S> && input_iterator<I> &&
    requires(const I& i, const S& s) {
        { i == s } -> std::convertible_to<bool>;
        { i != s } -> std::convertible_to<bool>;
    };

template<typename S, typename I>
concept sized_sentinel_for = sentinel_for<S, I> &&
    requires(const I& i, const S& s) {
        { s - i } -> std::same_as<std::iter_difference_t<I>>;
        { i - s } -> std::same_as<std::iter_difference_t<I>>;
    };

// Range concepts
template<typename R>
concept range = requires(R& r) {
    { std::ranges::begin(r) } -> input_iterator;
    { std::ranges::end(r) } -> sentinel_for<decltype(std::ranges::begin(r))>;
};

template<typename R>
concept sized_range = range<R> && requires(R& r) {
    { std::ranges::size(r) } -> std::convertible_to<std::size_t>;
};

template<typename R>
concept forward_range = range<R> && forward_iterator<std::ranges::iterator_t<R>>;

template<typename R>
concept bidirectional_range = forward_range<R> && bidirectional_iterator<std::ranges::iterator_t<R>>;

template<typename R>
concept random_access_range = bidirectional_range<R> && random_access_iterator<std::ranges::iterator_t<R>>;

template<typename R>
concept contiguous_range = random_access_range<R> && contiguous_iterator<std::ranges::iterator_t<R>>;

// Iterator traits and utilities
template<typename I>
using value_type = std::iter_value_t<I>;

template<typename I>
using difference_type = std::iter_difference_t<I>;

template<typename I>
using reference = std::iter_reference_t<I>;

template<typename I>
using pointer = std::add_pointer_t<reference<I>>;

// Distance operations following Stepanov's principles
template<input_iterator I, sentinel_for<I> S>
constexpr difference_type<I> iterator_distance(I first, S last) {
    if constexpr (sized_sentinel_for<S, I>) {
        return last - first;
    } else {
        difference_type<I> n = 0;
        while (first != last) {
            ++n;
            ++first;
        }
        return n;
    }
}

template<input_iterator I>
constexpr void iterator_advance(I& i, difference_type<I> n) {
    if constexpr (random_access_iterator<I>) {
        i += n;
    } else if constexpr (bidirectional_iterator<I>) {
        if (n >= 0) {
            while (n-- > 0) ++i;
        } else {
            while (n++ < 0) --i;
        }
    } else {
        while (n-- > 0) ++i;
    }
}

template<input_iterator I, sentinel_for<I> S>
constexpr void iterator_advance_to(I& i, difference_type<I> n, S bound) {
    if constexpr (sized_sentinel_for<S, I>) {
        auto dist = bound - i;
        if (n >= dist) {
            i = std::move(bound);
        } else {
            iterator_advance(i, n);
        }
    } else if constexpr (iterator_bidirectional<I> && std::same_as<I, S>) {
        if (n >= 0) {
            while (n-- > 0 && i != bound) ++i;
        } else {
            while (n++ < 0 && i != bound) --i;
        }
    } else {
        while (n-- > 0 && i != bound) ++i;
    }
}

template<input_iterator I>
constexpr I iterator_next(I i, difference_type<I> n = 1) {
    iterator_advance(i, n);
    return i;
}

template<bidirectional_iterator I>
constexpr I iterator_prev(I i, difference_type<I> n = 1) {
    iterator_advance(i, -n);
    return i;
}

// Iterator pair (half-open range)
template<typename I>
    requires input_iterator<I>
struct iterator_pair {
    using iterator = I;
    using value_type = std::iter_value_t<I>;
    using difference_type = std::iter_difference_t<I>;
    using reference = std::iter_reference_t<I>;

    I first;
    I last;

    constexpr iterator_pair() = default;
    constexpr iterator_pair(I f, I l) : first(f), last(l) {}

    constexpr I begin() const { return first; }
    constexpr I end() const { return last; }

    constexpr bool empty() const { return first == last; }

    constexpr difference_type size() const
        requires sized_sentinel_for<I, I>
    {
        return iterator_distance(first, last);
    }
};

template<typename I>
iterator_pair(I, I) -> iterator_pair<I>;

// Counted iterator - iterator with a count
template<input_iterator I>
class counted_iterator {
    I current;
    std::iter_difference_t<I> length;

public:
    using iterator_type = I;
    using value_type = std::iter_value_t<I>;
    using difference_type = std::iter_difference_t<I>;
    using reference = std::iter_reference_t<I>;
    using iterator_category = typename std::iterator_traits<I>::iterator_category;

    constexpr counted_iterator() = default;
    constexpr counted_iterator(I i, std::iter_difference_t<I> n) : current(i), length(n) {}

    constexpr I base() const { return current; }
    constexpr std::iter_difference_t<I> count() const { return length; }

    constexpr reference operator*() const { return *current; }
    constexpr auto operator->() const { return std::to_address(current); }

    constexpr counted_iterator& operator++() {
        ++current;
        --length;
        return *this;
    }

    constexpr counted_iterator operator++(int) {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    constexpr counted_iterator& operator--()
        requires bidirectional_iterator<I>
    {
        --current;
        ++length;
        return *this;
    }

    constexpr counted_iterator operator--(int)
        requires bidirectional_iterator<I>
    {
        auto tmp = *this;
        --*this;
        return tmp;
    }

    constexpr counted_iterator& operator+=(difference_type n)
        requires random_access_iterator<I>
    {
        current += n;
        length -= n;
        return *this;
    }

    constexpr counted_iterator& operator-=(difference_type n)
        requires random_access_iterator<I>
    {
        current -= n;
        length += n;
        return *this;
    }

    constexpr reference operator[](difference_type n) const
        requires random_access_iterator<I>
    {
        return current[n];
    }

    friend constexpr bool operator==(const counted_iterator& x, const counted_iterator& y) {
        return x.length == y.length;
    }

    friend constexpr bool operator!=(const counted_iterator& x, const counted_iterator& y) {
        return x.length != y.length;
    }

    friend constexpr bool operator<(const counted_iterator& x, const counted_iterator& y)
        requires random_access_iterator<I>
    {
        return y.length < x.length;
    }

    friend constexpr bool operator>(const counted_iterator& x, const counted_iterator& y)
        requires random_access_iterator<I>
    {
        return y < x;
    }

    friend constexpr bool operator<=(const counted_iterator& x, const counted_iterator& y)
        requires random_access_iterator<I>
    {
        return !(y < x);
    }

    friend constexpr bool operator>=(const counted_iterator& x, const counted_iterator& y)
        requires random_access_iterator<I>
    {
        return !(x < y);
    }

    friend constexpr counted_iterator operator+(const counted_iterator& i, difference_type n)
        requires random_access_iterator<I>
    {
        return counted_iterator(i.current + n, i.length - n);
    }

    friend constexpr counted_iterator operator+(difference_type n, const counted_iterator& i)
        requires random_access_iterator<I>
    {
        return i + n;
    }

    friend constexpr counted_iterator operator-(const counted_iterator& i, difference_type n)
        requires random_access_iterator<I>
    {
        return counted_iterator(i.current - n, i.length + n);
    }

    friend constexpr difference_type operator-(const counted_iterator& x, const counted_iterator& y)
        requires random_access_iterator<I>
    {
        return y.length - x.length;
    }
};

// Iterator facade - base class for creating custom iterators
template<typename Derived, typename ValueType, typename CategoryTag,
         typename Reference = ValueType&, typename Difference = std::ptrdiff_t>
class iterator_facade {
public:
    using value_type = ValueType;
    using reference = Reference;
    using pointer = std::add_pointer_t<Reference>;
    using difference_type = Difference;
    using iterator_category = CategoryTag;

protected:
    constexpr Derived& derived() { return static_cast<Derived&>(*this); }
    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }

public:
    constexpr reference operator*() const {
        return derived().dereference();
    }

    constexpr pointer operator->() const {
        return std::addressof(operator*());
    }

    constexpr Derived& operator++() {
        derived().increment();
        return derived();
    }

    constexpr Derived operator++(int) {
        Derived tmp = derived();
        ++*this;
        return tmp;
    }

    template<typename D = Derived>
        requires bidirectional_iterator<D>
    constexpr Derived& operator--() {
        derived().decrement();
        return derived();
    }

    template<typename D = Derived>
        requires bidirectional_iterator<D>
    constexpr Derived operator--(int) {
        Derived tmp = derived();
        --*this;
        return tmp;
    }

    template<typename D = Derived>
        requires random_access_iterator<D>
    constexpr Derived& operator+=(difference_type n) {
        derived().advance(n);
        return derived();
    }

    template<typename D = Derived>
        requires random_access_iterator<D>
    constexpr Derived& operator-=(difference_type n) {
        return *this += -n;
    }

    template<typename D = Derived>
        requires random_access_iterator<D>
    constexpr reference operator[](difference_type n) const {
        return *(derived() + n);
    }

    friend constexpr bool operator==(const iterator_facade& x, const iterator_facade& y) {
        return x.derived().equal(y.derived());
    }

    friend constexpr bool operator!=(const iterator_facade& x, const iterator_facade& y) {
        return !(x == y);
    }

    template<typename D = Derived>
        requires random_access_iterator<D>
    friend constexpr bool operator<(const iterator_facade& x, const iterator_facade& y) {
        return x.derived().distance_to(y.derived()) > 0;
    }

    template<typename D = Derived>
        requires random_access_iterator<D>
    friend constexpr bool operator>(const iterator_facade& x, const iterator_facade& y) {
        return y < x;
    }

    template<typename D = Derived>
        requires random_access_iterator<D>
    friend constexpr bool operator<=(const iterator_facade& x, const iterator_facade& y) {
        return !(y < x);
    }

    template<typename D = Derived>
        requires random_access_iterator<D>
    friend constexpr bool operator>=(const iterator_facade& x, const iterator_facade& y) {
        return !(x < y);
    }

    template<typename D = Derived>
        requires random_access_iterator<D>
    friend constexpr Derived operator+(const iterator_facade& i, difference_type n) {
        Derived tmp = i.derived();
        return tmp += n;
    }

    template<typename D = Derived>
        requires random_access_iterator<D>
    friend constexpr Derived operator+(difference_type n, const iterator_facade& i) {
        return i + n;
    }

    template<typename D = Derived>
        requires random_access_iterator<D>
    friend constexpr Derived operator-(const iterator_facade& i, difference_type n) {
        Derived tmp = i.derived();
        return tmp -= n;
    }

    template<typename D = Derived>
        requires random_access_iterator<D>
    friend constexpr difference_type operator-(const iterator_facade& x, const iterator_facade& y) {
        return y.derived().distance_to(x.derived());
    }
};

} // namespace stepanov