#pragma once

#include <cassert>
#include <iostream>
#include <functional>
#include <stdlib.h>
#include "Vector.h"

template <typename T>
class Matrix
{
private:
	size_t lengthX;
	size_t lengthY;
	size_t length;
	T* matrix;

	void maker(int index, T value) {
		assert(index + 1 == length);

		matrix[index] = value;
	}

	template <typename... Args>
	void maker(int index, T value, Args... values) {
		assert(index < length);

		matrix[index] = value;

		maker(index + 1, values...);
	}

public:

	Matrix() {
		this->lengthX = 0;
		this->lengthY = 0;
		this->length = 0;
		this->matrix = nullptr;
	}

	Matrix(const size_t lengthX, const size_t lengthY) {
		this->lengthX = lengthX;
		this->lengthY = lengthY;
		this->length = this->lengthX * this->lengthY;

		assert(this->lengthX > 0 && this->lengthY > 0);

		if (this->length < 1) {
			this->matrix = nullptr;
			return;
		}

		this->matrix = new T[this->length]{};
	}

	Matrix(const Matrix& other)
	{
		this->lengthX = other.lengthX;
		this->lengthY = other.lengthY;
		this->length = other.length;
		this->matrix = new T[length];

		for (size_t i = 0; i < this->length; ++i)
			this->matrix[i] = other.matrix[i];
	}

	~Matrix() {
		delete[] matrix;
	}

	const size_t getLengthX() const {
		return lengthX;
	}

	const size_t getLengthY() const {
		return lengthY;
	}

	const size_t getLength() const {
		return length;
	}

	template <typename... Args>
	Matrix<T>& make(Args... values) {
		maker(0, values...);

		return *this;
	}

	T at(const size_t x, const size_t y) const {
		assert(x < lengthX && y < lengthY);

		return matrix[lengthX * y + x];
	}

	Matrix<T>& put(const size_t x, const size_t y, const T value) {
		assert(x < lengthX && y < lengthY);

		matrix[lengthX * y + x] = value;

		return *this;
	}

	static Matrix<T> transpose(const Matrix<T>& A) {
		Matrix<T> result(A.lengthY, A.lengthX);

		for (size_t y = 0; y < A.lengthY; ++y) {
			for (size_t x = 0; x < A.lengthX; ++x)
				result.matrix[A.lengthY * x + y] = A.matrix[A.lengthX * y + x];
		}

		return result;
	}

	static Matrix<T> add(const Matrix<T>& A, const Matrix<T>& B) {
		assert(A.lengthX == B.lengthX);
		assert(A.lengthY == B.lengthY);

		Matrix<T> result(A.lengthX, A.lengthY);

		for (size_t i = 0; i < result.length; ++i)
			result.matrix[i] = A.matrix[i] + B.matrix[i];

		return result;
	}

	static Matrix<T> subtract(const Matrix<T>& A, const Matrix<T>& B) {
		assert(A.lengthX == B.lengthX);
		assert(A.lengthY == B.lengthY);

		Matrix<T> result(A.lengthX, A.lengthY);

		for (size_t i = 0; i < result.length; ++i)
			result.matrix[i] = A.matrix[i] - B.matrix[i];

		return result;
	}

	static Vector<T> product(const Matrix<T>& A, const Vector<T>& b) {
		assert(A.lengthX == b.getLength());

		Vector<T> result(A.lengthY);

		for (size_t y = 0; y < A.lengthY; ++y) {
			T temp = 0;

			for (size_t x = 0; x < A.lengthX; ++x)
				temp += A.matrix[A.lengthX * y + x] * b.at(x);

			result.put(y, temp);
		}

		return result;
	}

	static Matrix<T> product(const Matrix<T>& A, const Matrix<T>& B) {
		assert(A.lengthX == B.lengthY);

		Matrix<T> result(B.lengthX, A.lengthY);

		for (size_t i = 0; i < A.lengthY; ++i) {
			for (size_t j = 0; j < B.lengthX; ++j) {
				T temp = 0;

				for (size_t k = 0; k < A.lengthX; ++k)
					temp += A.matrix[A.lengthX * i + k] * B.matrix[B.lengthX * k + j];

				result.matrix[B.lengthX * i + j] = temp;
			}
		}

		return result;
	}

	static Matrix<T> scale(const Matrix<T>& A, const T b) {
		Matrix<T> result(A.lengthX, A.lengthY);

		for (size_t i = 0; i < result.length; ++i)
			result.matrix[i] = A.matrix[i] * b;

		return result;
	}

	static Matrix<T> scale(const Matrix<T>& A, const Matrix<T>& B) {
		assert(A.lengthX == B.lengthX);
		assert(A.lengthY == B.lengthY);

		Matrix<T> result(A.lengthX, A.lengthY);

		for (size_t i = 0; i < A.length; ++i)
			result.matrix[i] = A.matrix[i] * B.matrix[i];

		return result;
	}

	static Matrix<T> gradient(const Vector<T>& a, const Vector<T>& b) {
		Matrix<T> result(b.getLength(), a.getLength());

		for (size_t y = 0; y < a.getLength(); ++y) {
			T aY = a.at(y);

			for (size_t x = 0; x < b.getLength(); ++x)
				result.matrix[b.getLength() * y + x] = aY * b.at(x);
		}

		return result;
	}

	static Matrix<T> apply(const Matrix<T>& A, std::function<T(const T)> predicate) {
		Matrix<T> result(A.lengthX, A.lengthY);

		for (size_t i = 0; i < A.length; ++i)
			result.matrix[i] = predicate(A.matrix[i]);

		return result;
	}

	Matrix<T>& operator=(const Matrix<T>& other) {
		if (this == &other)
			return *this;

		T* newMatrix = nullptr;

		if (other.length > 0) {
			newMatrix = new T[other.length];

			for (size_t i = 0; i < other.length; ++i)
				newMatrix[i] = other.matrix[i];
		}

		delete[] matrix;

		lengthX = other.lengthX;
		lengthY = other.lengthY;
		length = other.length;
		matrix = newMatrix;

		return *this;
	}

	Matrix<T> operator+(const Matrix<T>& B) const {
		return Matrix<T>::add(*this, B);
	}

	Matrix<T> operator-(const Matrix<T>& B) const {
		return Matrix<T>::subtract(*this, B);
	}

	Matrix<T> operator*(const Matrix<T>& B) const {
		return Matrix<T>::product(*this, B);
	}

	Vector<T> operator*(const Vector<T>& b) const {
		return Matrix<T>::product(*this, b);
	}
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& A) {
	out << "[ ";

	for (size_t y = 0; y < A.getLengthY(); ++y) {
		out << A.at(0, y);

		for (size_t x = 1; x < A.getLengthX(); ++x)
			out << ", " << A.at(x, y);

		out << "; ";
	}

	out << "]";

	return out;
}