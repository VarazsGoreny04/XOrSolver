#pragma once

#undef NDEBUG
#include <cassert>
#include <iostream>
#include <functional>
#include <stdlib.h>

/// <summary>
/// A mathematical vector.
/// </summary>
/// <typeparam name="T">The type of the vector elements.</typeparam>
template <typename T>
class Vector
{
private:
	/// <summary>
	/// The length of the vector.
	/// </summary>
	size_t length;

	/// <summary>
	/// The elements of the vector.
	/// </summary>
	T* vector;

	/// <summary>
	/// Sets the last index of the vector to the given element.
	/// </summary>
	/// <param name="index">The the last index of the vector.</param>
	/// <param name="value">The element.</param>
	void maker(size_t index, T value) {
		assert(index + 1 == length);

		vector[index] = value;
	}

	/// <summary>
	/// Fills the vector with the given elements from the given index.
	/// </summary>
	/// <typeparam name="Args">The type of the collection of the elements.</typeparam>
	/// <param name="index">The first index to fill the elements from.</param>
	/// <param name="value">One element.</param>
	/// <param name="values">The collection of the elements.</param>
	template <typename... Args>
	void maker(size_t index, T value, Args... values) {
		assert(index < length);

		vector[index] = value;

		maker(index + 1, values...);
	}

public:
	/// <summary>
	/// Creates an empty vector.
	/// </summary>
	Vector() {
		this->length = 0;
		this->vector = nullptr;
	}

	/// <summary>
	/// Creates a vector with the given size.
	/// </summary>
	/// <param name="length">The size of the vector.</param>
	Vector(const size_t length) {
		this->length = length;
		this->vector = this->length > 0 ? new T[this->length]{} : nullptr;
	}

	/// <summary>
	/// Creates a vector identical to the given vector.
	/// </summary>
	/// <param name="other">The vector.</param>
	Vector(const Vector& other)
	{
		this->length = other.length;
		this->vector = new T[length];

		for (size_t i = 0; i < this->length; ++i)
			this->vector[i] = other.vector[i];
	}

	/// <summary>
	/// Deconstructs this vector.
	/// </summary>
	~Vector() {
		delete[] vector;
	}

	/// <returns>The length of this vector.</returns>
	const size_t getLength() const {
		return length;
	}

	/// <summary>
	/// Fills the vector with the given elements.
	/// </summary>
	/// <typeparam name="Args">The type of the collection of the elements.</typeparam>
	/// <param name="values">The collection of the elements.</param>
	/// <returns>This vector.</returns>
	template <typename... Args>
	Vector<T>& make(Args... values) {
		maker(0, values...);

		return *this;
	}

	/// <summary>
	/// Gets one element of the vector by the given index.
	/// </summary>
	/// <param name="index">The index.</param>
	/// <returns>A copy of the element.</returns>
	T at(const size_t index) const {
		assert(index < length);

		return vector[index];
	}

	/// <summary>
	/// Sets one element of the vector to the given value.
	/// </summary>
	/// <param name="index">The index of the element.</param>
	/// <param name="value">The value.</param>
	/// <returns>This vector.</returns>
	Vector<T>& put(const size_t index, const T value) {
		assert(index < length);

		vector[index] = value;

		return *this;
	}

	/// <summary>
	/// Adds two vectors.
	/// </summary>
	/// <param name="a">The first vector.</param>
	/// <param name="b">The second vector.</param>
	/// <returns>The created vector.</returns>
	static Vector<T> add(const Vector<T>& a, const Vector<T>& b) {
		assert(a.length == b.length);

		Vector<T> result(a.length);

		for (size_t i = 0; i < result.length; ++i)
			result.vector[i] = a.vector[i] + b.vector[i];

		return result;
	}

	/// <summary>
	/// Subtracts two vectors.
	/// </summary>
	/// <param name="a">The first vector.</param>
	/// <param name="b">The second vector.</param>
	/// <returns>The created vector.</returns>
	static Vector<T> subtract(const Vector<T>& a, const Vector<T>& b) {
		assert(a.length == b.length);

		Vector<T> result(a.length);

		for (size_t i = 0; i < result.length; ++i)
			result.vector[i] = a.vector[i] - b.vector[i];

		return result;
	}

	/// <summary>
	/// Calculates the product of two vectors.
	/// </summary>
	/// <param name="a">The first vector.</param>
	/// <param name="b">The second vector.</param>
	/// <returns>The result value.</returns>
	static T product(const Vector<T>& a, const Vector<T>& b) {
		assert(a.length == b.length);

		T result{};

		for (size_t i = 0; i < a.length; ++i)
			result = result + a.vector[i] * b.vector[i];

		return result;
	}

	/// <summary>
	/// Scales the elements of a vector by a value.
	/// </summary>
	/// <param name="a">The vector.</param>
	/// <param name="b">The value.</param>
	/// <returns>The created vector.</returns>
	static Vector<T> scale(const Vector<T>& a, const T b) {
		Vector<T> result(a.length);

		for (size_t i = 0; i < a.length; ++i)
			result.vector[i] = a.vector[i] * b;

		return result;
	}

	/// <summary>
	/// Scales the elements of a vector by the corresponding elements of another vector.
	/// </summary>
	/// <param name="a">The vector to scale.</param>
	/// <param name="b">The vector to scale by.</param>
	/// <returns>The created vector.</returns>
	static Vector<T> scale(const Vector<T>& a, const Vector<T>& b) {
		assert(a.length == b.length);

		Vector<T> result(a.length);

		for (size_t i = 0; i < a.length; ++i)
			result.vector[i] = a.vector[i] * b.vector[i];

		return result;
	}

	/// <summary>
	/// Applies a function to each element of the given vector.
	/// </summary>
	/// <param name="a">The vector.</param>
	/// <param name="predicate">The function.</param>
	/// <returns>The created vector.</returns>
	static Vector<T> apply(const Vector<T>& a, std::function<T(const T)> predicate) {
		Vector<T> result(a.length);

		for (size_t i = 0; i < a.length; ++i)
			result.vector[i] = predicate(a.vector[i]);

		return result;
	}

	Vector<T>& operator=(const Vector<T>& other) {
		if (this == &other)
			return *this;

		T* newVector = nullptr;

		if (other.length > 0) {
			newVector = new T[other.length];

			for (size_t i = 0; i < other.length; ++i)
				newVector[i] = other.vector[i];
		}

		delete[] vector;

		length = other.length;
		vector = newVector;

		return *this;
	}

	Vector<T> operator+(const Vector<T>& b) const {
		return Vector<T>::add(*this, b);
	}

	Vector<T> operator-(const Vector<T>& b) const {
		return Vector<T>::subtract(*this, b);
	}

	T operator*(const Vector<T>& b) const {
		return Vector<T>::product(*this, b);
	}
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const Vector<T>& a) {
	out << "[ ";

	if (a.getLength() > 0)
		out << a.at(0);

	for (size_t i = 1; i < a.getLength(); ++i)
		out << ", " << a.at(i);

	out << " ]";

	return out;
}