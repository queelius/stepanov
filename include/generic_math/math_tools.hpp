template <typename T>
T clamp(const T& min, const T& max, const T& x)
{
	if (x < min)
		return min;
	else if (x > max)
		return max;
	else
		return x;
}

template <typename T>
T scale(const T& min, const T& max, const T& x)
{
	return (x - min) / (max - min);
}

template <typename T>
T descale(const T& min, const T& max, const T& x)
{
	return (x * (max - min)) + min;
}

template <typename T>
T normalize(const T& min, const T& max, const T& x)
{
	return clamp((T)0, (T)1, scale(min, max, x));
}

template <typename T>
T denormalize(const T& min, const T& max, const T& x)
{
	return descale(min, max, clamp((T)0, (T)1, x));
}


// preconditions: both a and b > 0
template <typename T>
T gcd(const T& a, const T& b)
{
	if (a == b)
		return a;
	else if (a > b)
		return gcd(a - b, b);
	else
		return gcd(a, b - a);
}

