#pragma once

#include <concepts>
#include <utility>
#include <functional>
#include <optional>
#include "iterators.hpp"
#include "concepts.hpp"

namespace stepanov {

// Range view base class
template<typename Derived>
class view_base {
protected:
    constexpr Derived& derived() { return static_cast<Derived&>(*this); }
    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

// Filter view - lazily filters elements based on predicate
template<forward_range R, typename Pred>
    requires std::predicate<Pred, std::ranges::range_reference_t<R>>
class filter_view : public view_base<filter_view<R, Pred>> {
    R base_;
    Pred pred_;

    class iterator {
        std::ranges::iterator_t<R> current_;
        std::ranges::sentinel_t<R> end_;
        Pred* pred_;

        void find_next() {
            while (current_ != end_ && !(*pred_)(*current_)) {
                ++current_;
            }
        }

    public:
        using value_type = std::ranges::range_value_t<R>;
        using reference = std::ranges::range_reference_t<R>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(std::ranges::iterator_t<R> curr, std::ranges::sentinel_t<R> end, Pred* p)
            : current_(curr), end_(end), pred_(p) {
            find_next();
        }

        reference operator*() const { return *current_; }
        auto operator->() const { return std::to_address(current_); }

        iterator& operator++() {
            ++current_;
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
    filter_view() = default;
    constexpr filter_view(R base, Pred pred)
        : base_(std::move(base)), pred_(std::move(pred)) {}

    constexpr auto begin() {
        return iterator(std::ranges::begin(base_), std::ranges::end(base_), &pred_);
    }

    constexpr auto end() {
        return iterator(std::ranges::end(base_), std::ranges::end(base_), &pred_);
    }
};

template<typename R, typename Pred>
filter_view(R&&, Pred) -> filter_view<std::remove_cvref_t<R>, Pred>;

// Transform view - lazily applies function to elements
template<forward_range R, typename F>
    requires std::invocable<F, std::ranges::range_reference_t<R>>
class transform_view : public view_base<transform_view<R, F>> {
    R base_;
    F func_;

    class iterator {
        std::ranges::iterator_t<R> current_;
        F* func_;

    public:
        using value_type = std::remove_cvref_t<std::invoke_result_t<F, std::ranges::range_reference_t<R>>>;
        using reference = std::invoke_result_t<F, std::ranges::range_reference_t<R>>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = typename std::iterator_traits<std::ranges::iterator_t<R>>::iterator_category;

        iterator() = default;
        iterator(std::ranges::iterator_t<R> curr, F* f)
            : current_(curr), func_(f) {}

        reference operator*() const { return (*func_)(*current_); }

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
            requires iterator_bidirectional<std::ranges::iterator_t<R>>
        {
            --current_;
            return *this;
        }

        iterator operator--(int)
            requires iterator_bidirectional<std::ranges::iterator_t<R>>
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        iterator& operator+=(difference_type n)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            current_ += n;
            return *this;
        }

        iterator& operator-=(difference_type n)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            current_ -= n;
            return *this;
        }

        reference operator[](difference_type n) const
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return (*func_)(current_[n]);
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.current_ == y.current_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }

        friend bool operator<(const iterator& x, const iterator& y)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return x.current_ < y.current_;
        }

        friend bool operator>(const iterator& x, const iterator& y)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return y < x;
        }

        friend bool operator<=(const iterator& x, const iterator& y)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return !(y < x);
        }

        friend bool operator>=(const iterator& x, const iterator& y)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return !(x < y);
        }

        friend iterator operator+(const iterator& i, difference_type n)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return iterator(i.current_ + n, i.func_);
        }

        friend iterator operator+(difference_type n, const iterator& i)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return i + n;
        }

        friend iterator operator-(const iterator& i, difference_type n)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return iterator(i.current_ - n, i.func_);
        }

        friend difference_type operator-(const iterator& x, const iterator& y)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return x.current_ - y.current_;
        }
    };

public:
    transform_view() = default;
    constexpr transform_view(R base, F func)
        : base_(std::move(base)), func_(std::move(func)) {}

    constexpr auto begin() {
        return iterator(std::ranges::begin(base_), &func_);
    }

    constexpr auto end() {
        return iterator(std::ranges::end(base_), &func_);
    }

    constexpr auto size() const
        requires sized_range<R>
    {
        return std::ranges::size(base_);
    }
};

template<typename R, typename F>
transform_view(R&&, F) -> transform_view<std::remove_cvref_t<R>, F>;

// Take view - takes first n elements
template<typename R>
    requires range<R>
class take_view : public view_base<take_view<R>> {
    R base_;
    std::ptrdiff_t count_;

    template<bool IsConst>
    class sentinel {
        using Base = std::conditional_t<IsConst, const R, R>;
        std::ranges::sentinel_t<Base> end_;
        std::ptrdiff_t count_;

    public:
        sentinel() = default;
        sentinel(std::ranges::sentinel_t<Base> end, std::ptrdiff_t count)
            : end_(end), count_(count) {}

        friend bool operator==(const counted_iterator<std::ranges::iterator_t<Base>>& x, const sentinel& y) {
            return x.count() == 0 || x.base() == y.end_;
        }

        friend bool operator==(const sentinel& x, const counted_iterator<std::ranges::iterator_t<Base>>& y) {
            return y == x;
        }

        friend bool operator!=(const counted_iterator<std::ranges::iterator_t<Base>>& x, const sentinel& y) {
            return !(x == y);
        }

        friend bool operator!=(const sentinel& x, const counted_iterator<std::ranges::iterator_t<Base>>& y) {
            return !(y == x);
        }
    };

public:
    take_view() = default;
    constexpr take_view(R base, std::ptrdiff_t count)
        : base_(std::move(base)), count_(count) {}

    constexpr auto begin() {
        return counted_iterator(std::ranges::begin(base_), count_);
    }

    constexpr auto end() {
        return sentinel<false>(std::ranges::end(base_), count_);
    }

    constexpr auto begin() const {
        return counted_iterator(std::ranges::begin(base_), count_);
    }

    constexpr auto end() const {
        return sentinel<true>(std::ranges::end(base_), count_);
    }

    constexpr auto size() const
        requires sized_range<R>
    {
        auto s = std::ranges::size(base_);
        return std::min(s, static_cast<decltype(s)>(count_));
    }
};

template<typename R>
take_view(R&&, std::ptrdiff_t) -> take_view<std::remove_cvref_t<R>>;

// Drop view - drops first n elements
template<forward_range R>
class drop_view : public view_base<drop_view<R>> {
    R base_;
    std::ptrdiff_t count_;
    mutable std::optional<std::ranges::iterator_t<R>> cached_begin_;

public:
    drop_view() = default;
    constexpr drop_view(R base, std::ptrdiff_t count)
        : base_(std::move(base)), count_(count) {}

    constexpr auto begin() {
        if (!cached_begin_) {
            auto it = std::ranges::begin(base_);
            iterator_advance_to(it, count_, std::ranges::end(base_));
            cached_begin_ = it;
        }
        return *cached_begin_;
    }

    constexpr auto begin() const
        requires std::ranges::random_access_range<const R>
    {
        return std::ranges::next(std::ranges::begin(base_), count_, std::ranges::end(base_));
    }

    constexpr auto end() const {
        return std::ranges::end(base_);
    }

    constexpr auto size() const
        requires sized_range<R>
    {
        auto s = std::ranges::size(base_);
        return s > static_cast<decltype(s)>(count_) ? s - count_ : 0;
    }
};

template<typename R>
drop_view(R&&, std::ptrdiff_t) -> drop_view<std::remove_cvref_t<R>>;

// Zip view - combines multiple ranges element-wise
template<forward_range... Rs>
    requires (sizeof...(Rs) > 0)
class zip_view : public view_base<zip_view<Rs...>> {
    std::tuple<Rs...> bases_;

    template<bool IsConst>
    class iterator {
        using Bases = std::conditional_t<IsConst,
            std::tuple<const Rs...>,
            std::tuple<Rs...>>;

    public:
        std::tuple<std::ranges::iterator_t<Rs>...> currents_;
        using value_type = std::tuple<std::ranges::range_value_t<Rs>...>;
        using reference = std::tuple<std::ranges::range_reference_t<Rs>...>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(std::tuple<std::ranges::iterator_t<Rs>...> currents)
            : currents_(std::move(currents)) {}

        reference operator*() const {
            return std::apply([](auto&... its) {
                return reference(*its...);
            }, currents_);
        }

        iterator& operator++() {
            std::apply([](auto&... its) {
                (++its, ...);
            }, currents_);
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

    template<bool IsConst>
    class sentinel {
        using Bases = std::conditional_t<IsConst,
            std::tuple<const Rs...>,
            std::tuple<Rs...>>;

    public:
        std::tuple<std::ranges::sentinel_t<Rs>...> ends_;
        sentinel() = default;
        sentinel(std::tuple<std::ranges::sentinel_t<Rs>...> ends)
            : ends_(std::move(ends)) {}

        friend bool operator==(const iterator<IsConst>& x, const sentinel& y) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return ((std::get<Is>(x.currents_) == std::get<Is>(y.ends_)) || ...);
            }(std::index_sequence_for<Rs...>{});
        }

        friend bool operator==(const sentinel& x, const iterator<IsConst>& y) {
            return y == x;
        }

        friend bool operator!=(const iterator<IsConst>& x, const sentinel& y) {
            return !(x == y);
        }

        friend bool operator!=(const sentinel& x, const iterator<IsConst>& y) {
            return !(y == x);
        }
    };

public:
    zip_view() = default;
    constexpr zip_view(Rs... bases)
        : bases_(std::move(bases)...) {}

    constexpr auto begin() {
        return std::apply([](auto&... bases) {
            return iterator<false>(std::make_tuple(std::ranges::begin(bases)...));
        }, bases_);
    }

    constexpr auto end() {
        return std::apply([](auto&... bases) {
            return sentinel<false>(std::make_tuple(std::ranges::end(bases)...));
        }, bases_);
    }

    constexpr auto begin() const {
        return std::apply([](const auto&... bases) {
            return iterator<true>(std::make_tuple(std::ranges::begin(bases)...));
        }, bases_);
    }

    constexpr auto end() const {
        return std::apply([](const auto&... bases) {
            return sentinel<true>(std::make_tuple(std::ranges::end(bases)...));
        }, bases_);
    }

    constexpr auto size() const
        requires (sized_range<Rs> && ...)
    {
        return std::apply([](const auto&... bases) {
            return std::min({std::ranges::size(bases)...});
        }, bases_);
    }
};

template<typename... Rs>
zip_view(Rs&&...) -> zip_view<std::remove_cvref_t<Rs>...>;

// Enumerate view - adds indices to elements
template<forward_range R>
class enumerate_view : public view_base<enumerate_view<R>> {
    R base_;

    class iterator {
        std::ranges::iterator_t<R> current_;
        std::ranges::range_difference_t<R> index_;

    public:
        using value_type = std::pair<std::ranges::range_difference_t<R>, std::ranges::range_value_t<R>>;
        using reference = std::pair<std::ranges::range_difference_t<R>, std::ranges::range_reference_t<R>>;
        using difference_type = std::ptrdiff_t;
        using iterator_category = typename std::iterator_traits<std::ranges::iterator_t<R>>::iterator_category;

        iterator() = default;
        iterator(std::ranges::iterator_t<R> curr, difference_type idx = 0)
            : current_(curr), index_(idx) {}

        reference operator*() const {
            return {index_, *current_};
        }

        iterator& operator++() {
            ++current_;
            ++index_;
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        iterator& operator--()
            requires iterator_bidirectional<std::ranges::iterator_t<R>>
        {
            --current_;
            --index_;
            return *this;
        }

        iterator operator--(int)
            requires iterator_bidirectional<std::ranges::iterator_t<R>>
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        iterator& operator+=(difference_type n)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            current_ += n;
            index_ += n;
            return *this;
        }

        iterator& operator-=(difference_type n)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            current_ -= n;
            index_ -= n;
            return *this;
        }

        reference operator[](difference_type n) const
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return {index_ + n, current_[n]};
        }

        friend bool operator==(const iterator& x, const iterator& y) {
            return x.current_ == y.current_;
        }

        friend bool operator!=(const iterator& x, const iterator& y) {
            return !(x == y);
        }

        friend bool operator<(const iterator& x, const iterator& y)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return x.current_ < y.current_;
        }

        friend bool operator>(const iterator& x, const iterator& y)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return y < x;
        }

        friend bool operator<=(const iterator& x, const iterator& y)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return !(y < x);
        }

        friend bool operator>=(const iterator& x, const iterator& y)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return !(x < y);
        }

        friend iterator operator+(const iterator& i, difference_type n)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return iterator(i.current_ + n, i.index_ + n);
        }

        friend iterator operator+(difference_type n, const iterator& i)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return i + n;
        }

        friend iterator operator-(const iterator& i, difference_type n)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return iterator(i.current_ - n, i.index_ - n);
        }

        friend difference_type operator-(const iterator& x, const iterator& y)
            requires iterator_random_access<std::ranges::iterator_t<R>>
        {
            return x.current_ - y.current_;
        }
    };

public:
    enumerate_view() = default;
    constexpr enumerate_view(R base)
        : base_(std::move(base)) {}

    constexpr auto begin() {
        return iterator(std::ranges::begin(base_), 0);
    }

    constexpr auto end() {
        if constexpr (sized_range<R>) {
            return iterator(std::ranges::end(base_), std::ranges::size(base_));
        } else {
            return iterator(std::ranges::end(base_));
        }
    }

    constexpr auto size() const
        requires sized_range<R>
    {
        return std::ranges::size(base_);
    }
};

template<typename R>
enumerate_view(R&&) -> enumerate_view<std::remove_cvref_t<R>>;

// Range adaptor objects for pipe syntax
namespace views {
    inline constexpr auto filter = []<typename Pred>(Pred pred) {
        return [pred = std::move(pred)]<typename R>(R&& r) {
            return filter_view(std::forward<R>(r), pred);
        };
    };

    inline constexpr auto transform = []<typename F>(F func) {
        return [func = std::move(func)]<typename R>(R&& r) {
            return transform_view(std::forward<R>(r), func);
        };
    };

    inline constexpr auto take = []<typename N>(N n) {
        return [n]<typename R>(R&& r) {
            return take_view(std::forward<R>(r), n);
        };
    };

    inline constexpr auto drop = []<typename N>(N n) {
        return [n]<typename R>(R&& r) {
            return drop_view(std::forward<R>(r), n);
        };
    };

    inline constexpr auto enumerate = []<typename R>(R&& r) {
        return enumerate_view(std::forward<R>(r));
    };
}

// Pipe operator for composing range adaptors
template<typename R, typename F>
    requires std::invocable<F, R>
auto operator|(R&& r, F&& f) -> decltype(std::forward<F>(f)(std::forward<R>(r))) {
    return std::forward<F>(f)(std::forward<R>(r));
}

} // namespace stepanov