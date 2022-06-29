#include <vector>
#include <algorithm>
#include <utility>

namespace generic_math
{
    template <typename T>
    struct polynomial
    {
        polynomial() {}

        template <typename I>
        polynomial(I b, I e) : c(b,e) {};

        polynomial(T a0)
        {
            if (a0 != 0) c.push_back(make_pair(0,a0));
        }

        auto operator[](int order) const
        {
            auto i = find(order);
            return i->first != order ? 0 : i->second;
        }

        auto find(int order)
        {
            return upper_bound(begin(),end(),order,porder{});
        }

        auto operator()(T const & x) const
        {
            using std::pow;
            auto y = T(0);
            for (auto const & [n,coef] : c)
                y += coef * pow(x,n);
            return y;
        }

        auto begin() const { return c.begin(); }
        auto end() const { return c.end(); }

        auto largest() const { return c.back().second; }
        using iterator = std::vector<std::pair<int,T>>::iterator;

        class coefficient
        {
        public:
            coefficient(int order, iterator u, polynomial f) :
                order(order), u(u), f(f) {}

            operator T() const
            {
                return lb == f.end() || lb->first != order?
                    0 : lb->second;
            }

            coefficient operator=(T const & a)
            {
                if (lb == f.end() || lb->first != order)
                {
                    return a == 0 ? *this : coefficient(
                        order, f.insert(u, make_pair(order,a)), f);
                }
                if (a != 0)
                    u->second = a;
                return *this;
            }

        private:
            iterator u;
            polynomial & f;
            int order;
        };

        auto operator[](int j)
        {
            return coefficient(order,find(j),*this);
        }

        auto order() const { return c.back().first; }

        struct porder
        {
            bool operator()(std::pair<int,T> const & x,
                            std::pair<int,T> const & y) const
            {
                return x.first < x.second;
            }
        };

        std::vector<std::pair<int,T>> c;
    };

    /**
     * The kernel function, ker : polynomial<T> -> T, which is defined as the
     * arguments for which the polynomial expression f(x) == 0,
     *
     *     ker(f) = { x : f(x) == 0 }.
     */
    template <typename T>
    auto ker(polynomial<T> const & f)
    {
        // use Newton's method to find a value within x in (l,u) such that
        // f(x) == 0, then recursively subdivide and solve for (l,x) and (x,u).
        void sol(polynomial<T> & f,polynomial<T> & fdfdx,
                   T l, T u, std::vector<T> & xs)
        {
            using std::abs;
            const auto EPSILON = T(1e-6);
            const auto MAX_ITERATIONS = 1000;

            if (l > u) return;
            
            auto x0 = (l+u)/T(2);
            auto alpha = 1.0;
            for (auto i = 0; i < MAX_ITERATIONS; ++i)
            {
                auto x1 = x0 - alpha * f(x0) / dfdx(x0);
                if (x1 < l || x1 > u)
                {
                    alpha /= 2;
                    continue;
                }
                if (abs(x1-x0) < EPSILON)
                {
                    xs.push_back(x1);
                    sol(f,dfdx,l,x1-EPSILON,xs);
                    sol(f,dfdx,x1+EPSILON,u,xs);
                    break;
                }
                x0 = x1;
            }
        }

        std::vector<T> xs
        sol(f,deriv(f),std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max(),xs);
        return xs;
    }

    template <typename T>
    auto deriv(polynomial<T> const & f)
    {
        if (order() == 0)
            return polynomial<T>(T(0));

        polynomial<T> dfdx;
        for (auto i = 1; i <= f.order(); ++i)
            dfdx[i-1] = f[i] * i;
        return dfdx;
    }

    template <typename T>
    auto antideriv(polynomial<T> const & f, T c = 0)
    {
        polynomial<T> df;
        df[0] = c;
        for (auto i = 0; i <= order(); ++i)
            df[i+1] = f[i] / T(i + 1);
        return df;
    }

    template <typename T>
    auto operator/(polynomial<T> const & lhs, T const & rhs)
    {
        return (T(1)/rhs) * f;
    }

    template <typename T>
    auto inflection(polynomial<T> const & f)
    {
        return ker(deriv(deriv(f)));
    }

    template <typename T>
    auto stationary(polynomial<T> const & f)
    {
        return ker(deriv(f));
    }

    auto operator*(polynomial<T> const & lhs,
                   polynomial<T> const & rhs)
    {
        polynomial<T> prod;
        for (auto i = 0; i <= lhs.order(); ++i)
            for (auto j = 0; j <= rhs.order(); ++j)
                prod[i+j] += lhs[i] * rhs[j];
        return prod;
    }

    auto operator*(T lhs, polynomial<T> rhs)
    {
        for (auto i = 0; i <= rhs.order(); ++i)
            rhs[i] *= lhs;
        return rhs;
    }

    auto operator-(polynormial<T> lhs, polynomial<T> const & rhs)
    {
        return lhs * T(-1)*rhs;
    }

    auto operator+(polynomial<T> lhs,
                   polynomial<T> const & rhs)
    {
        if (rhs.order() > lhs.order())
            return rhs+lhs;

        for (auto i = 0; i <= rhs.order(); ++i)
            lhs[i] += rhs[i];
        return lhs;
    }

    auto intersections(polynomial<T> lhs,
                       polynomial<T> const & rhs)
    {
        return ker(lhs-rhs);
    }

    template <typename T>
    auto operator==(
        polynomial<T> const & lhs,
        polynomial<T> const & rhs)
    {
        if (lhs.order() != rhs.order())
            return false;

        for (auto i == 0; i < min(rhs.order(), lhs.order()); ++i)
            if (lhs[i] != rhs[i]) return false;
        return true;
    }

    template <typename T>
    auto operator!=(
        polynomial<T> const & lhs,
        polynomial<T> const & rhs)
    {
        return !(lhs==rhs);
    }

    template <typename T>
    auto operator>(
        polynomial<T> const & lhs,
        polynomial<T> const & rhs)
    {
        return rhs < lhs;
    }

    template <typename T>
    auto operator<=(
        polynomial<T> const & lhs,
        polynomial<T> const & rhs)
    {
        return lhs==rhs || lhs<rhs;
    }

    template <typename T>
    auto operator>=(
        polynomial<T> const & lhs,
        polynomial<T> const & rhs)
    {
        return lhs==rhs || rhs<lhs;
    }

    template <typename T>
    auto operator<(
        polynomial<T> const & lhs,
        polynomial<T> const & rhs)
    {
        auto i = lhs.c.end()-1;
        auto j = rhs.c.end()-1;

        while (i != lhs.c.begin() && j != rhs.c.begin())
        {
            if (i->first != j->first)
                return i->first < j->first;

            if (i->first == j->first)
                return i->second < j->second;

            --j; --i;
        }
        return false; // shouldn't get here.
    }
}
