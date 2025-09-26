#pragma once

#include <concepts>
#include <utility>
#include <functional>
#include <optional>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <map>
#include <limits>
#include "iterators.hpp"
#include "iterators_enhanced.hpp"
#include "ranges.hpp"
#include "concepts.hpp"

namespace stepanov {

// ================ Advanced Range Adaptors ================

// Chunk view - non-overlapping windows of size n
template<forward_range R>
class chunk_view : public view_base<chunk_view<R>> {
    R base_;
    std::ptrdiff_t chunk_size_;

    template<bool IsConst>
    class iterator {
        using Base = std::conditional_t<IsConst, const R, R>;
        std::ranges::iterator_t<Base> current_;
        std::ranges::sentinel_t<Base> end_;
        std::ptrdiff_t chunk_size_;
        std::ptrdiff_t missing_ = 0;

    public:
        using value_type = take_view<iterator_pair<std::ranges::iterator_t<Base>>>;
        using reference = value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        iterator() = default;
        iterator(std::ranges::iterator_t<Base> current,
                std::ranges::sentinel_t<Base> end,
                difference_type chunk_size)
            : current_(current), end_(end), chunk_size_(chunk_size) {
            missing_ = chunk_size;
        }

        reference operator*() const {
            auto chunk_end = current_;
            iterator_advance_to(chunk_end, chunk_size_, end_);
            return take_view(iterator_pair(current_, chunk_end), chunk_size_);
        }

        iterator& operator++() {
            missing_ = chunk_size_;
            iterator_advance_to(current_, chunk_size_, end_);
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.current_ == y.current_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }

        friend bool operator==(const iterator& x, std::default_sentinel_t) {
            return x.current_ == x.end_;
        }

        friend bool operator==(std::default_sentinel_t, const iterator& x) {
            return x.current_ == x.end_;
        }
    };

public:
    chunk_view() = default;
    constexpr chunk_view(R base, std::ptrdiff_t chunk_size)
        : base_(std::move(base)), chunk_size_(chunk_size) {}

    constexpr auto begin() {
        return iterator<false>(std::ranges::begin(base_), std::ranges::end(base_), chunk_size_);
    }

    constexpr auto end() {
        return std::default_sentinel;
    }

    constexpr auto begin() const {
        return iterator<true>(std::ranges::begin(base_), std::ranges::end(base_), chunk_size_);
    }

    constexpr auto end() const {
        return std::default_sentinel;
    }
};

template<typename R>
chunk_view(R&&, std::ptrdiff_t)
    -> chunk_view<std::remove_cvref_t<R>>;

// Slide view - overlapping windows (sliding window)
template<forward_range R>
class slide_view : public view_base<slide_view<R>> {
    R base_;
    std::ptrdiff_t window_size_;

    template<bool IsConst>
    class iterator {
        using Base = std::conditional_t<IsConst, const R, R>;
        std::ranges::iterator_t<Base> current_;
        std::ranges::iterator_t<Base> last_;
        std::ptrdiff_t window_size_;

    public:
        using value_type = iterator_pair<std::ranges::iterator_t<Base>>;
        using reference = value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = typename std::iterator_traits<std::ranges::iterator_t<Base>>::iterator_category;

        iterator() = default;
        iterator(std::ranges::iterator_t<Base> current,
                std::ranges::iterator_t<Base> last,
                difference_type window_size)
            : current_(current), last_(last), window_size_(window_size) {}

        reference operator*() const {
            auto window_end = current_;
            iterator_advance(window_end, window_size_);
            return iterator_pair(current_, window_end);
        }

        iterator& operator++() {
            ++current_;
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        iterator& operator--()
            requires iterator_bidirectional<std::ranges::iterator_t<Base>>
        {
            --current_;
            return *this;
        }

        iterator operator--(int)
            requires iterator_bidirectional<std::ranges::iterator_t<Base>>
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.current_ == y.current_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }
    };

public:
    slide_view() = default;
    constexpr slide_view(R base, std::ptrdiff_t window_size)
        : base_(std::move(base)), window_size_(window_size) {}

    constexpr auto begin() {
        auto first = std::ranges::begin(base_);
        auto last = std::ranges::end(base_);
        auto window_last = first;
        iterator_advance_to(window_last, window_size_ - 1, last);
        return iterator<false>(first, window_last, window_size_);
    }

    constexpr auto end() {
        auto first = std::ranges::begin(base_);
        auto last = std::ranges::end(base_);
        auto window_last = first;
        iterator_advance_to(window_last, window_size_ - 1, last);
        if (window_last != last) {
            ++window_last;
        }
        return iterator<false>(window_last, window_last, window_size_);
    }
};

template<typename R>
slide_view(R&&, std::ptrdiff_t)
    -> slide_view<std::remove_cvref_t<R>>;

// Cycle view - repeat range infinitely
template<forward_range R>
class cycle_view : public view_base<cycle_view<R>> {
    R base_;
    mutable std::optional<std::ranges::iterator_t<R>> cached_begin_;

    class iterator {
        std::ranges::iterator_t<R> current_;
        std::ranges::iterator_t<R> begin_;
        std::ranges::iterator_t<R> end_;
        std::size_t cycle_count_ = 0;

    public:
        using value_type = std::ranges::range_value_t<R>;
        using reference = std::ranges::range_reference_t<R>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(std::ranges::iterator_t<R> current,
                std::ranges::iterator_t<R> begin,
                std::ranges::iterator_t<R> end)
            : current_(current), begin_(begin), end_(end) {}

        reference operator*() const { return *current_; }

        iterator& operator++() {
            ++current_;
            if (current_ == end_) {
                current_ = begin_;
                ++cycle_count_;
            }
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.current_ == y.current_ && x.cycle_count_ == y.cycle_count_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }
    };

public:
    cycle_view() = default;
    constexpr explicit cycle_view(R base) : base_(std::move(base)) {}

    constexpr auto begin() {
        if (!cached_begin_) {
            cached_begin_ = std::ranges::begin(base_);
        }
        return iterator(*cached_begin_, *cached_begin_, std::ranges::end(base_));
    }

    constexpr std::unreachable_sentinel_t end() const noexcept {
        return std::unreachable_sentinel;
    }
};

template<typename R>
cycle_view(R&&) -> cycle_view<std::remove_cvref_t<R>>;

// Cartesian product view
template<forward_range... Rs>
    requires (sizeof...(Rs) > 0)
class cartesian_product_view : public view_base<cartesian_product_view<Rs...>> {
    std::tuple<Rs...> bases_;

    template<bool IsConst>
    class iterator {
        using Bases = std::conditional_t<IsConst, std::tuple<const Rs...>, std::tuple<Rs...>>;
        std::tuple<std::ranges::iterator_t<Rs>...> currents_;
        Bases* bases_ = nullptr;

        template<std::size_t N>
        void next_impl() {
            if constexpr (N > 0) {
                ++std::get<N>(currents_);
                if (std::get<N>(currents_) == std::ranges::end(std::get<N>(*bases_))) {
                    std::get<N>(currents_) = std::ranges::begin(std::get<N>(*bases_));
                    next_impl<N-1>();
                }
            } else {
                ++std::get<0>(currents_);
            }
        }

    public:
        using value_type = std::tuple<std::ranges::range_value_t<Rs>...>;
        using reference = std::tuple<std::ranges::range_reference_t<Rs>...>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(std::tuple<std::ranges::iterator_t<Rs>...> currents, Bases* bases)
            : currents_(std::move(currents)), bases_(bases) {}

        reference operator*() const {
            return std::apply([](auto&... its) {
                return reference(*its...);
            }, currents_);
        }

        iterator& operator++() {
            next_impl<sizeof...(Rs) - 1>();
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.currents_ == y.currents_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }
    };

public:
    cartesian_product_view() = default;
    constexpr cartesian_product_view(Rs... bases) : bases_(std::move(bases)...) {}

    constexpr auto begin() {
        return std::apply([this](auto&... bases) {
            return iterator<false>(std::make_tuple(std::ranges::begin(bases)...), &bases_);
        }, bases_);
    }

    constexpr auto end() {
        return std::apply([this](auto&... bases) {
            auto ends = std::make_tuple(std::ranges::end(bases)...);
            auto begins = std::make_tuple(std::ranges::begin(bases)...);
            std::get<0>(begins) = std::get<0>(ends);
            return iterator<false>(begins, &bases_);
        }, bases_);
    }
};

template<typename... Rs>
cartesian_product_view(Rs&&...) -> cartesian_product_view<std::remove_cvref_t<Rs>...>;

// Join view - flattens nested ranges
template<forward_range R>
    requires forward_range<std::ranges::range_value_t<R>>
class join_view : public view_base<join_view<R>> {
    R base_;
    mutable std::optional<std::ranges::iterator_t<R>> outer_current_;

    class iterator {
        using InnerRange = std::ranges::range_value_t<R>;
        std::ranges::iterator_t<R> outer_current_;
        std::ranges::sentinel_t<R> outer_end_;
        std::optional<std::ranges::iterator_t<InnerRange>> inner_current_;
        std::optional<std::ranges::sentinel_t<InnerRange>> inner_end_;

        void find_next() {
            while (outer_current_ != outer_end_) {
                if (!inner_current_ || *inner_current_ == *inner_end_) {
                    inner_current_ = std::ranges::begin(*outer_current_);
                    inner_end_ = std::ranges::end(*outer_current_);
                    if (*inner_current_ != *inner_end_) {
                        return;
                    }
                    ++outer_current_;
                } else {
                    return;
                }
            }
        }

    public:
        using value_type = std::ranges::range_value_t<InnerRange>;
        using reference = std::ranges::range_reference_t<InnerRange>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(std::ranges::iterator_t<R> outer_current, std::ranges::sentinel_t<R> outer_end)
            : outer_current_(outer_current), outer_end_(outer_end) {
            find_next();
        }

        reference operator*() const {
            return **inner_current_;
        }

        iterator& operator++() {
            ++*inner_current_;
            find_next();
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.outer_current_ == y.outer_current_ &&
                   (!x.inner_current_ || !y.inner_current_ ||
                    *x.inner_current_ == *y.inner_current_);
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }
    };

public:
    join_view() = default;
    constexpr explicit join_view(R base) : base_(std::move(base)) {}

    constexpr auto begin() {
        return iterator(std::ranges::begin(base_), std::ranges::end(base_));
    }

    constexpr auto end() {
        return iterator(std::ranges::end(base_), std::ranges::end(base_));
    }
};

template<typename R>
join_view(R&&) -> join_view<std::remove_cvref_t<R>>;

// Split view - splits range by delimiter
template<forward_range R, typename Delimiter>
class split_view : public view_base<split_view<R, Delimiter>> {
    R base_;
    Delimiter delim_;

    class iterator {
        std::ranges::iterator_t<R> current_;
        std::ranges::sentinel_t<R> end_;
        Delimiter delim_;
        std::optional<std::ranges::iterator_t<R>> next_;

        void find_next() {
            next_ = std::find(current_, end_, delim_);
        }

    public:
        using value_type = iterator_pair<std::ranges::iterator_t<R>>;
        using reference = value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(std::ranges::iterator_t<R> current, std::ranges::sentinel_t<R> end, Delimiter delim)
            : current_(current), end_(end), delim_(delim) {
            find_next();
        }

        reference operator*() const {
            return iterator_pair(current_, *next_);
        }

        iterator& operator++() {
            current_ = *next_;
            if (current_ != end_) {
                ++current_;
            }
            find_next();
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.current_ == y.current_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }
    };

public:
    split_view() = default;
    constexpr split_view(R base, Delimiter delim)
        : base_(std::move(base)), delim_(std::move(delim)) {}

    constexpr auto begin() {
        return iterator(std::ranges::begin(base_), std::ranges::end(base_), delim_);
    }

    constexpr auto end() {
        return iterator(std::ranges::end(base_), std::ranges::end(base_), delim_);
    }
};

template<typename R, typename Delimiter>
split_view(R&&, Delimiter) -> split_view<std::remove_cvref_t<R>, Delimiter>;

// Unique view - removes consecutive duplicates
template<forward_range R, typename Pred = std::equal_to<>>
class unique_view : public view_base<unique_view<R, Pred>> {
    R base_;
    Pred pred_;

    class iterator {
        std::ranges::iterator_t<R> current_;
        std::ranges::sentinel_t<R> end_;
        Pred* pred_;

        void find_next() {
            if (current_ != end_) {
                auto prev = current_;
                ++current_;
                while (current_ != end_ && (*pred_)(*prev, *current_)) {
                    ++current_;
                }
            }
        }

    public:
        using value_type = std::ranges::range_value_t<R>;
        using reference = std::ranges::range_reference_t<R>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(std::ranges::iterator_t<R> current, std::ranges::sentinel_t<R> end, Pred* pred)
            : current_(current), end_(end), pred_(pred) {}

        reference operator*() const { return *current_; }

        iterator& operator++() {
            find_next();
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.current_ == y.current_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }
    };

public:
    unique_view() = default;
    constexpr unique_view(R base, Pred pred = Pred{})
        : base_(std::move(base)), pred_(std::move(pred)) {}

    constexpr auto begin() {
        return iterator(std::ranges::begin(base_), std::ranges::end(base_), &pred_);
    }

    constexpr auto end() {
        return iterator(std::ranges::end(base_), std::ranges::end(base_), &pred_);
    }
};

template<typename R, typename Pred = std::equal_to<>>
unique_view(R&&, Pred = Pred{}) -> unique_view<std::remove_cvref_t<R>, Pred>;

// Group by view - groups consecutive elements by predicate
template<forward_range R, typename Pred = std::equal_to<>>
class group_by_view : public view_base<group_by_view<R, Pred>> {
    R base_;
    Pred pred_;

    class iterator {
        std::ranges::iterator_t<R> current_;
        std::ranges::iterator_t<R> next_;
        std::ranges::sentinel_t<R> end_;
        Pred* pred_;

        void find_next() {
            if (current_ != end_) {
                next_ = current_;
                auto prev = next_;
                ++next_;
                while (next_ != end_ && (*pred_)(*prev, *next_)) {
                    prev = next_;
                    ++next_;
                }
            } else {
                next_ = end_;
            }
        }

    public:
        using value_type = iterator_pair<std::ranges::iterator_t<R>>;
        using reference = value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(std::ranges::iterator_t<R> current, std::ranges::sentinel_t<R> end, Pred* pred)
            : current_(current), end_(end), pred_(pred) {
            find_next();
        }

        reference operator*() const {
            return iterator_pair(current_, next_);
        }

        iterator& operator++() {
            current_ = next_;
            find_next();
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.current_ == y.current_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }
    };

public:
    group_by_view() = default;
    constexpr group_by_view(R base, Pred pred = Pred{})
        : base_(std::move(base)), pred_(std::move(pred)) {}

    constexpr auto begin() {
        return iterator(std::ranges::begin(base_), std::ranges::end(base_), &pred_);
    }

    constexpr auto end() {
        return iterator(std::ranges::end(base_), std::ranges::end(base_), &pred_);
    }
};

template<typename R, typename Pred = std::equal_to<>>
group_by_view(R&&, Pred = Pred{}) -> group_by_view<std::remove_cvref_t<R>, Pred>;

// Reverse view - reverses iteration order
template<bidirectional_range R>
class reverse_view : public view_base<reverse_view<R>> {
    R base_;

public:
    reverse_view() = default;
    constexpr explicit reverse_view(R base) : base_(std::move(base)) {}

    constexpr auto begin() {
        return make_reverse_iterator(std::ranges::end(base_));
    }

    constexpr auto end() {
        return make_reverse_iterator(std::ranges::begin(base_));
    }

    constexpr auto begin() const {
        return make_reverse_iterator(std::ranges::end(base_));
    }

    constexpr auto end() const {
        return make_reverse_iterator(std::ranges::begin(base_));
    }

    constexpr auto size() const
        requires sized_range<R>
    {
        return std::ranges::size(base_);
    }
};

template<typename R>
reverse_view(R&&) -> reverse_view<std::remove_cvref_t<R>>;

// Cache view - caches computed values
template<forward_range R>
class cache_view : public view_base<cache_view<R>> {
    R base_;
    mutable std::vector<std::ranges::range_value_t<R>> cache_;
    mutable std::optional<std::ranges::iterator_t<R>> current_pos_;

    class iterator {
        cache_view* parent_;
        std::size_t index_;

    public:
        using value_type = std::ranges::range_value_t<R>;
        using reference = const value_type&;
        using pointer = const value_type*;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;

        iterator() = default;
        iterator(cache_view* parent, std::size_t index)
            : parent_(parent), index_(index) {}

        reference operator*() const {
            parent_->ensure_cached(index_);
            return parent_->cache_[index_];
        }

        pointer operator->() const {
            return &**this;
        }

        iterator& operator++() {
            ++index_;
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        iterator& operator--() {
            --index_;
            return *this;
        }

        iterator operator--(int) {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        iterator& operator+=(difference_type n) {
            index_ += n;
            return *this;
        }

        iterator& operator-=(difference_type n) {
            index_ -= n;
            return *this;
        }

        reference operator[](difference_type n) const {
            return *(*this + n);
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.index_ == y.index_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }

        friend bool operator<(const iterator& x, const iterator& y) {
            return x.index_ < y.index_;
        }

        friend bool operator>(const iterator& x, const iterator& y) {
            return y < x;
        }

        friend bool operator<=(const iterator& x, const iterator& y) {
            return !(y < x);
        }

        friend bool operator>=(const iterator& x, const iterator& y) {
            return !(x < y);
        }

        friend iterator operator+(const iterator& i, difference_type n) {
            return iterator(i.parent_, i.index_ + n);
        }

        friend iterator operator+(difference_type n, const iterator& i) {
            return i + n;
        }

        friend iterator operator-(const iterator& i, difference_type n) {
            return iterator(i.parent_, i.index_ - n);
        }

        friend difference_type operator-(const iterator& x, const iterator& y) {
            return x.index_ - y.index_;
        }
    };

    void ensure_cached(std::size_t index) const {
        if (!current_pos_) {
            current_pos_ = std::ranges::begin(base_);
        }

        while (cache_.size() <= index && *current_pos_ != std::ranges::end(base_)) {
            cache_.push_back(**current_pos_);
            ++*current_pos_;
        }
    }

public:
    cache_view() = default;
    constexpr explicit cache_view(R base) : base_(std::move(base)) {}

    auto begin() {
        return iterator(this, 0);
    }

    auto end() {
        if constexpr (sized_range<R>) {
            return iterator(this, std::ranges::size(base_));
        } else {
            // For non-sized ranges, we need to cache everything
            auto it = std::ranges::begin(base_);
            auto e = std::ranges::end(base_);
            while (it != e) {
                cache_.push_back(*it);
                ++it;
            }
            return iterator(this, cache_.size());
        }
    }
};

template<typename R>
cache_view(R&&) -> cache_view<std::remove_cvref_t<R>>;

// ================ Lazy Generation and Infinite Ranges ================

// Generate view - infinite generation from function
template<typename F>
class generate_view : public view_base<generate_view<F>> {
    F func_;

    class iterator {
        F* func_;
        std::size_t index_ = 0;

    public:
        using value_type = std::decay_t<decltype(std::declval<F>()())>;
        using reference = value_type;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        explicit iterator(F* func, std::size_t index = 0)
            : func_(func), index_(index) {}

        reference operator*() const { return (*func_)(); }

        iterator& operator++() {
            ++index_;
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.index_ == y.index_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }
    };

public:
    generate_view() = default;
    constexpr explicit generate_view(F func) : func_(std::move(func)) {}

    constexpr auto begin() {
        return iterator(&func_);
    }

    constexpr std::unreachable_sentinel_t end() const noexcept {
        return std::unreachable_sentinel;
    }
};

template<typename F>
generate_view(F) -> generate_view<F>;

// Iterate view - infinite iteration f(x), f(f(x)), ...
template<typename T, typename F>
class iterate_view : public view_base<iterate_view<T, F>> {
    T initial_;
    F func_;

    class iterator {
        T value_;
        F* func_;

    public:
        using value_type = T;
        using reference = const T&;
        using pointer = const T*;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(T value, F* func) : value_(std::move(value)), func_(func) {}

        reference operator*() const { return value_; }
        pointer operator->() const { return &value_; }

        iterator& operator++() {
            value_ = (*func_)(value_);
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.value_ == y.value_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }
    };

public:
    iterate_view() = default;
    constexpr iterate_view(T initial, F func)
        : initial_(std::move(initial)), func_(std::move(func)) {}

    constexpr auto begin() {
        return iterator(initial_, &func_);
    }

    constexpr std::unreachable_sentinel_t end() const noexcept {
        return std::unreachable_sentinel;
    }
};

template<typename T, typename F>
iterate_view(T, F) -> iterate_view<T, F>;

// Repeat view - repeat value infinitely
template<typename T>
class repeat_view : public view_base<repeat_view<T>> {
    T value_;

    class iterator {
        const T* value_;
        std::size_t index_ = 0;

    public:
        using value_type = T;
        using reference = const T&;
        using pointer = const T*;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;

        iterator() = default;
        explicit iterator(const T* value, std::size_t index = 0)
            : value_(value), index_(index) {}

        reference operator*() const { return *value_; }
        pointer operator->() const { return value_; }

        iterator& operator++() {
            ++index_;
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        iterator& operator--() {
            --index_;
            return *this;
        }

        iterator operator--(int) {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        iterator& operator+=(difference_type n) {
            index_ += n;
            return *this;
        }

        iterator& operator-=(difference_type n) {
            index_ -= n;
            return *this;
        }

        reference operator[](difference_type) const { return *value_; }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.index_ == y.index_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }

        friend bool operator<(const iterator& x, const iterator& y) {
            return x.index_ < y.index_;
        }

        friend bool operator>(const iterator& x, const iterator& y) {
            return y < x;
        }

        friend bool operator<=(const iterator& x, const iterator& y) {
            return !(y < x);
        }

        friend bool operator>=(const iterator& x, const iterator& y) {
            return !(x < y);
        }

        friend iterator operator+(const iterator& i, difference_type n) {
            return iterator(i.value_, i.index_ + n);
        }

        friend iterator operator+(difference_type n, const iterator& i) {
            return i + n;
        }

        friend iterator operator-(const iterator& i, difference_type n) {
            return iterator(i.value_, i.index_ - n);
        }

        friend difference_type operator-(const iterator& x, const iterator& y) {
            return x.index_ - y.index_;
        }
    };

public:
    repeat_view() = default;
    constexpr explicit repeat_view(T value) : value_(std::move(value)) {}

    constexpr auto begin() const {
        return iterator(&value_);
    }

    constexpr std::unreachable_sentinel_t end() const noexcept {
        return std::unreachable_sentinel;
    }
};

template<typename T>
repeat_view(T) -> repeat_view<T>;

// Iota view - numeric sequence
template<typename T>
class iota_view : public view_base<iota_view<T>> {
    T start_;
    T bound_;

    class iterator {
        T value_;

    public:
        using value_type = T;
        using reference = const T&;
        using pointer = const T*;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;

        iterator() = default;
        explicit iterator(T value) : value_(value) {}

        reference operator*() const { return value_; }
        pointer operator->() const { return &value_; }

        iterator& operator++() {
            ++value_;
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        iterator& operator--() {
            --value_;
            return *this;
        }

        iterator operator--(int) {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        iterator& operator+=(difference_type n) {
            value_ += n;
            return *this;
        }

        iterator& operator-=(difference_type n) {
            value_ -= n;
            return *this;
        }

        T operator[](difference_type n) const { return value_ + n; }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.value_ == y.value_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }

        friend bool operator<(const iterator& x, const iterator& y) {
            return x.value_ < y.value_;
        }

        friend bool operator>(const iterator& x, const iterator& y) {
            return y < x;
        }

        friend bool operator<=(const iterator& x, const iterator& y) {
            return !(y < x);
        }

        friend bool operator>=(const iterator& x, const iterator& y) {
            return !(x < y);
        }

        friend iterator operator+(const iterator& i, difference_type n) {
            return iterator(i.value_ + n);
        }

        friend iterator operator+(difference_type n, const iterator& i) {
            return i + n;
        }

        friend iterator operator-(const iterator& i, difference_type n) {
            return iterator(i.value_ - n);
        }

        friend difference_type operator-(const iterator& x, const iterator& y) {
            return x.value_ - y.value_;
        }
    };

public:
    iota_view() = default;
    constexpr explicit iota_view(T start)
        : start_(start), bound_(std::numeric_limits<T>::max()) {}
    constexpr iota_view(T start, T bound)
        : start_(start), bound_(bound) {}

    constexpr auto begin() const {
        return iterator(start_);
    }

    constexpr auto end() const {
        return iterator(bound_);
    }

    constexpr auto size() const {
        return bound_ - start_;
    }
};

template<typename T>
iota_view(T) -> iota_view<T>;

template<typename T>
iota_view(T, T) -> iota_view<T>;

// Extended namespace for view adaptors
namespace views {
    inline constexpr auto chunk = []<typename N>(N n) {
        return [n]<typename R>(R&& r) {
            return chunk_view(std::forward<R>(r), n);
        };
    };

    inline constexpr auto slide = []<typename N>(N n) {
        return [n]<typename R>(R&& r) {
            return slide_view(std::forward<R>(r), n);
        };
    };

    inline constexpr auto cycle = []<typename R>(R&& r) {
        return cycle_view(std::forward<R>(r));
    };

    inline constexpr auto join = []<typename R>(R&& r) {
        return join_view(std::forward<R>(r));
    };

    inline constexpr auto split = []<typename Delimiter>(Delimiter delim) {
        return [delim]<typename R>(R&& r) {
            return split_view(std::forward<R>(r), delim);
        };
    };

    inline constexpr auto unique = []<typename R>(R&& r) {
        return unique_view(std::forward<R>(r));
    };

    inline constexpr auto group_by = []<typename Pred>(Pred pred) {
        return [pred]<typename R>(R&& r) {
            return group_by_view(std::forward<R>(r), pred);
        };
    };

    inline constexpr auto reverse = []<typename R>(R&& r) {
        return reverse_view(std::forward<R>(r));
    };

    inline constexpr auto cache = []<typename R>(R&& r) {
        return cache_view(std::forward<R>(r));
    };

    inline constexpr auto generate = []<typename F>(F f) {
        return generate_view(f);
    };

    inline constexpr auto iterate = []<typename T, typename F>(T initial, F f) {
        return iterate_view(initial, f);
    };

    inline constexpr auto repeat = []<typename T>(T value) {
        return repeat_view(value);
    };

    inline constexpr auto iota = []<typename T>(T start) {
        return iota_view(start);
    };

    inline constexpr auto iota_n = []<typename T>(T start, T bound) {
        return iota_view(start, bound);
    };
}

} // namespace stepanov