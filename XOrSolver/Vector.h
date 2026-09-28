#pragma once

#include <stdlib.h>
#include <iostream>

template <typename T>
class Vector
{
private:
	T* vector;

public:
	size_t length;

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

		this->vector = new T[this->length];

		short* eraser = (short*)vector;
		size_t eraserLength = this->length * (sizeof(T) / sizeof(short));

		for (size_t i = 0; i < eraserLength; ++i) {
			eraser[i] = 0;
		}
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

	T at(const size_t i) const {
		if (i >= length)
			exit(1);

		return vector[i];
	}

	Vector<T>& put(const size_t i, const T value) {
		if (i >= length)
			exit(1);

		vector[i] = value;

		return *this;
	}

	static Vector<T> add(const Vector<T>& a, const Vector<T>& b) {
		if (a.length != b.length)
			exit(1);

		Vector<T> result = Vector<T>(a.length);

		for (size_t i = 0; i < result.length; ++i)
			result.vector[i] = a.vector[i] + b.vector[i];

		return result;
	}

	static Vector<T> subtract(const Vector<T>& a, const Vector<T>& b) {
		if (a.length != b.length)
			exit(1);

		Vector<T> result = Vector<T>(a.length);

		for (size_t i = 0; i < result.length; ++i)
			result.vector[i] = a.vector[i] - b.vector[i];

		return result;
	}

	static T product(const Vector<T>& a, const Vector<T>& b) {
		if (a.length != b.length)
			exit(1);

		T result = a.vector[0] * b.vector[0];

		for (size_t i = 1; i < a.length; ++i)
			result = result + a.vector[i] * b.vector[i];

		return result;
	}

	static Vector<T> scale(const Vector<T>& a, const T b) {
		Vector<T> result = Vector<T>(a.length);

		for (size_t i = 0; i < a.length; ++i)
			result.vector[i] = a.vector[i] * b;

		return result;
	}

	static Vector<T> scale(const Vector<T>& a, const Vector<T>& b) {
		if (a.length != b.length)
			exit(1);

		Vector<T> result = Vector<T>(a.length);

		for (size_t i = 0; i < a.length; ++i)
			result.vector[i] = a.vector[i] * b.vector[i];

		return result;
	}

	static Vector<T> apply(const Vector<T>& a, T (*func)(const T)) {
		Vector<T> result = Vector<T>(a.length);

		for (size_t i = 0; i < a.length; ++i)
			result.vector[i] = func(a.vector[i]);

		return result;
	}

	Vector<T>& operator=(const Vector<T>& other) {
		if (this == &other)
			return *this;

		delete[] vector;

		length = other.length;

		if (length < 1)
			vector = nullptr;
		else {
			vector = new T[length];

			for (size_t i = 0; i < length; ++i)
				vector[i] = other.vector[i];
		}

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
	out << std::endl << "[" << a.at(0);

	for (size_t i = 1; i < a.length; ++i)
		out << ", " << a.at(i);

	out << "]" << std::endl;

	return out;
}