#pragma once

#include <utility>
#include <string>

#include <cuda_runtime.h>

using namespace std::string_literals;

namespace cross {

using dsize_t = unsigned int;
using ddiff_t = int;
using dpos_t = unsigned int;

constexpr size_t saturation_multiplier = 4000;

// Represents a two-dimensional vector, used to pass positions and sizes around.
// TODO: Copyright or something, copied from original thesis from src/common.hpp
template<typename T>
struct vec2
{
	T x;
	T y;


	__host__ __device__ vec2()
		:vec2<T>(0,0)
	{

	}

	__host__ __device__ vec2(T x, T y)
		:x(std::move(x)), y(std::move(y))
	{

	}

	template<typename U>
	__host__ __device__ vec2(const vec2<U>& v)
		:x(v.x), y(v.y)
	{

	}

	template<typename U>
	__host__ __device__ vec2(vec2<U>&& v)
		:x(std::move(v.x)), y(std::move(v.y))
	{

	}

	__host__ __device__ T area() const { return x * y; }

	__host__ __device__ vec2<T> operator+(const vec2<T>& rhs) const
	{
		return { x + rhs.x, y + rhs.y };
	}
	template<typename U>
	__host__ __device__ vec2<T> operator+(const U& rhs) const
	{
		return { x + rhs, y + rhs };
	}
	template<typename U>
	friend __host__ __device__ vec2<T> operator+(const U& lhs, const vec2<T>& rhs)
	{
		return { lhs + rhs.x, lhs + rhs.y };
	}

	__host__ __device__ vec2<T> operator-() const
	{
		return { -x, -y };
	}
	__host__ __device__ vec2<T> operator-(const vec2<T>& rhs) const
	{
		return { x - rhs.x, y - rhs.y };
	}
	template<typename U>
	__host__ __device__ vec2<T> operator-(const U& rhs) const
	{
		return { x - rhs, y - rhs };
	}

	__host__ __device__ vec2<T> operator*(const vec2<T>& rhs) const
	{
		return { x * rhs.x, y * rhs.y };
	}

	template<typename U>
	friend __host__ __device__ vec2<T> operator*(const U& lhs, const vec2<T>& rhs)
	{
		return { lhs * rhs.x, lhs * rhs.y };
	}
	template<typename U>
	__host__ __device__ vec2<T> operator*(const U& rhs) const
	{
		return { x * rhs, y * rhs };
	}

	__host__ __device__ vec2<T> operator*(const dim3& rhs) const
	{
		return { x * rhs.x, y * rhs.y };
	}


	__host__ __device__ vec2<T> operator/(const vec2<T>& rhs) const
	{
		return { x / rhs.x, y / rhs.y };
	}
	template<typename U>
	__host__ __device__ vec2<T> operator/(const U& rhs) const
	{
		return { x / rhs, y / rhs };
	}

	template<typename U>
	__host__ __device__ vec2<T> operator%(const U& rhs) const
	{
		return { x % rhs, y % rhs };
	}
	__host__ __device__ vec2<T> operator%(const vec2<T>& rhs) const
	{
		return { x % rhs.x, y % rhs.y };
	}

	/**
	 *  Row major linear index
	 */
	__host__ __device__ __inline__ dsize_t linear_idx(dsize_t cols) const { return y * cols + x; }

	/**
	 * Position from row major linear index
	 */
	static __host__ __device__ __inline__ vec2<T> from_linear_idx(T idx, dsize_t cols) {
        return { idx % cols, idx / cols };
    }
};

template<typename T>
inline bool operator==(const vec2<T> & lhs, const vec2<T> & rhs)
{
	return lhs.x == rhs.x && lhs.y == rhs.y;
}

template<typename T>
inline bool operator!=(const vec2<T> & lhs, const vec2<T> & rhs)
{
	return !(lhs == rhs);
}

template<typename T>
inline std::string to_string(const vec2<T>& v) {
	return "["s + std::to_string(v.x) + ","s + std::to_string(v.y) + "]"s;
}

using dsize2_t = vec2<unsigned int>;
using ddiff2_t = vec2<int>;
using dpos2_t = vec2<unsigned int>;

}