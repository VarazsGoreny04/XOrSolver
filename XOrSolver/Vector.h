#pragma once

#undef NDEBUG
#include <cassert>
#include <iostream>
#include <functional>
#include <stdlib.h>

template <typename T>
class Vector
{
private:
	size_t length;
	T* vector;

	void maker(size_t index, T value) {
		assert(index + 1 == length);

		vector[index] = value;
	}

	template <typename... Args>
	void maker(size_t index, T value, Args... values) {
		assert(index < length);

		vector[index] = value;

		maker(index + 1, values...);
	}

public:
	Vector() {
		this->length = 0;
		this->vector = nullptr;
	}

	Vector(const size_t length) {
		this->length = length;

		if (this->length < 1) {
			this->vector = nullptr;
			return;
		}

		this->vector = new T[this->length]{};
	}

	Vector(const Vector& other)
	{
		this->length = other.length;
		this->vector = new T[length];

		for (size_t i = 0; i < this->length; ++i)
			this->vector[i] = other.vector[i];
	}

	~Vector() {
		delete[] vector;
	}

	const size_t getLength() const {
		return length;
	}

	template <typename... Args>
	Vector<T>& make(Args... values) {
		maker(0, values...);

		return *this;
	}

	T at(const size_t i) const {
		assert(i < length);

		return vector[i];
	}

	Vector<T>& put(const size_t i, const T value) {
		assert(i < length);

		vector[i] = value;

		return *this;
	}

	static Vector<T> add(const Vector<T>& a, const Vector<T>& b) {
		assert(a.length == b.length);

		Vector<T> result(a.length);

		for (size_t i = 0; i < result.length; ++i)
			result.vector[i] = a.vector[i] + b.vector[i];

		return result;
	}

	static Vector<T> subtract(const Vector<T>& a, const Vector<T>& b) {
		assert(a.length == b.length);

		Vector<T> result(a.length);

		for (size_t i = 0; i < result.length; ++i)
			result.vector[i] = a.vector[i] - b.vector[i];

		return result;
	}

	static T product(const Vector<T>& a, const Vector<T>& b) {
		assert(a.length == b.length);

		T result{};

		for (size_t i = 0; i < a.length; ++i)
			result = result + a.vector[i] * b.vector[i];

		return result;
	}

	static Vector<T> scale(const Vector<T>& a, const T b) {
		Vector<T> result(a.length);

		for (size_t i = 0; i < a.length; ++i)
			result.vector[i] = a.vector[i] * b;

		return result;
	}

	static Vector<T> scale(const Vector<T>& a, const Vector<T>& b) {
		assert(a.length == b.length);

		Vector<T> result(a.length);

		for (size_t i = 0; i < a.length; ++i)
			result.vector[i] = a.vector[i] * b.vector[i];

		return result;
	}

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