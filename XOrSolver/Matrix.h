#pragma once

#include <stdlib.h>
#include <iostream>
#include <functional>
#include "Vector.h"

template <typename T>
class Matrix
{
private:
	size_t length;
	T* matrix;

public:
	size_t lengthX;
	size_t lengthY;

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

		if (this->length < 1 && (this->lengthX > 0 || this->lengthY > 0))
			exit(1);

		if (this->length < 1) {
			this->matrix = nullptr;
			return;
		}

		this->matrix = new T[this->length];

		short* eraser = (short*)matrix;
		size_t eraserLength = this->length * (sizeof(T) / sizeof(short));

		for (size_t i = 0; i < eraserLength; ++i) {
			eraser[i] = 0;
		}
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

	T at(const size_t x, const size_t y) const {
		if (x >= lengthX || y >= lengthY)
			exit(1);

		return matrix[lengthX * y + x];
	}

	Matrix<T>& put(const size_t x, const size_t y, const T value) {
		if (x >= lengthX || y >= lengthY)
			exit(1);

		matrix[lengthX * y + x] = value;

		return *this;
	}

	static Matrix<T> transpose(const Matrix<T>& A) {
		Matrix<T> result = Matrix<T>(A.lengthY, A.lengthX);

		for (size_t y = 0; y < A.lengthY; ++y) {
			for (size_t x = 0; x < A.lengthX; ++x)
				result.matrix[A.lengthY * y + x] = A.matrix[A.lengthY * x + y];
		}

		return result;
	}

	static Matrix<T> add(const Matrix<T>& A, const Matrix<T>& B) {
		if (A.lengthX != B.lengthX || A.lengthY != B.lengthY)
			exit(1);

		Matrix<T> result = Matrix<T>(A.lengthX, A.lengthY);

		for (size_t i = 0; i < result.length; ++i)
			result.matrix[i] = A.matrix[i] + B.matrix[i];

		return result;
	}

	static Matrix<T> subtract(const Matrix<T>& A, const Matrix<T>& B) {
		if (A.lengthX != B.lengthX || A.lengthY != B.lengthY)
			exit(1);

		Matrix<T> result = Matrix<T>(A.length, B.length);

		for (size_t i = 0; i < result.length; ++i)
			result.matrix[i] = A.matrix[i] - B.matrix[i];

		return result;
	}

	static Vector<T> product(const Matrix<T>& A, const Vector<T>& b) {
		if (A.lengthX != b.length)
			exit(1);

		Vector<T> result = Vector<T>(A.lengthY);

		for (size_t y = 0; y < A.lengthY; ++y) {
			float temp = 0;

			for (size_t x = 0; x < A.lengthX; ++x)
				temp += A.matrix[A.lengthX * y + x] * b.at(x);

			result.put(y, temp);
		}

		return result;
	}

	static Matrix<T> product(const Matrix<T>& A, const Matrix<T>& B) {
		if (A.lengthX != B.lengthY)
			exit(1);

		Matrix<T> result = Matrix<T>(B.lengthX, A.lengthY);

		for (size_t i = 0; i < A.lengthY; ++i) {
			for (size_t j = 0; j < B.lengthX; ++j) {
				for (size_t k = 0; k < A.lengthX; ++k)
					result.matrix[A.lengthX * i + j] += A.matrix[A.lengthX * i + k] * B.matrix[A.lengthY * k + j];
			}
		}

		return result;
	}

	static Matrix<T> scale(const Matrix<T>& A, const T b) {
		Matrix<T> result = Matrix<T>(A.lengthX, A.lengthY);

		for (size_t i = 0; i < result.length; ++i)
			result.matrix[i] = A.matrix[i] * b;

		return result;
	}

	static Matrix<T> scale(const Matrix<T>& A, const Matrix<T>& B) {
		if (A.lengthX != B.lengthX || A.lengthY != B.lengthY)
			exit(1);

		Matrix<T> result = Matrix<T>(A.lengthX, A.lengthY);

		for (size_t i = 0; i < A.length; ++i)
			result.matrix[i] = A.matrix[i] * B.matrix[i];

		return result;
	}

	static Matrix<T> gradiant(const Vector<T>& a, const Vector<T>& b) {
		Matrix<T> result = Matrix<T>(b.length, a.length);

		for (size_t y = 0; y < a.length; ++y) {
			T aY = a.at(y);

			for (size_t x = 0; x < b.length; ++x)
				result.matrix[a.length * y + x] = aY * b.at(x);
		}

		return result;
	}

	static Matrix<T> apply(const Matrix<T>& A, std::function<T(const T)> predicate) {
		Matrix<T> result = Matrix<T>(A.lengthX, A.lengthY);

		for (size_t i = 0; i < A.length; ++i)
			result.matrix[i] = predicate(A.matrix[i]);

		return result;
	}

	Matrix<T>& operator=(const Matrix<T>& other) {
		if (this == &other)
			return *this;

		delete[] matrix;

		lengthX = other.lengthX;
		lengthY = other.lengthY;
		length = other.length;

		if (length < 1)
			matrix = nullptr;
		else {
			matrix = new T[length];

			for (size_t i = 0; i < length; ++i)
				matrix[i] = other.matrix[i];
		}

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

	for (size_t y = 0; y < A.lengthY; ++y) {
		out << A.at(0, y);

		for (size_t x = 1; x < A.lengthX; ++x)
			out << ", " << A.at(x, y);

		out << "; ";
	}

	out << "]";

	return out;
}