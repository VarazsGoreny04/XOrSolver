#pragma once

#include "Vector.h"

template <typename T>
struct Matrix
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

	Matrix(size_t lengthX, size_t lengthY) {
		this->lengthX = lengthX;
		this->lengthY = lengthY;

		if (this->lengthX < 1 || this->lengthY < 1)
			exit(1);

		this->length = this->lengthX * this->lengthY;
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

	T at(size_t x, size_t y) {
		if (x >= lengthX || y >= lengthY)
			exit(1);

		return matrix[lengthX * y + x];
	}

	Matrix<T>& put(size_t x, size_t y, T value) {
		if (x >= lengthX || y >= lengthY)
			exit(1);

		matrix[lengthX * y + x] = value;

		return *this;
	}

	static Matrix<T> add(Matrix<T> A, Matrix<T> B) {
		if (A.lengthX != B.lengthX || A.lengthY != B.lengthY)
			exit(1);

		Matrix<T> result = Matrix<T>(A.length, B.length);

		for (size_t i = 0; i < result.length; ++i)
			result.matrix[i] = A.matrix[i] + B.matrix[i];

		return result;
	}

	static Vector<T> product(Matrix<T>& A, Vector<T>& b) {
		if (A.lengthX != b.length)
			exit(1);

		Vector<T> result = Vector<T>(A.lengthY);

		for (size_t i = 0; i < A.lengthY; ++i) {
			float temp = 0;

			for (size_t j = 0; j < A.lengthX; ++j)
				temp += A.matrix[A.lengthX * i + j] * b.at(j);

			result.put(i, temp);
		}

		return result;
	}

	static Matrix<T> product(Matrix<T>& A, Matrix<T>& B) {
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

	Matrix<T> operator+(Matrix<T> B) {
		return Matrix<T>::add(*this, B);
	}

	Matrix<T> operator*(Matrix<T> B) {
		return Matrix<T>::product(*this, B);
	}

	Vector<T> operator*(Vector<T> b) {
		return Matrix<T>::product(*this, b);
	}
};

template <typename T>
std::ostream& operator<<(std::ostream& out, Matrix<T> A) {
	out << "[" << std::endl;

	for (size_t y = 0; y < A.lengthY; ++y) {
		out << "  " << A.at(0, y);

		for (size_t x = 1; x < A.lengthX; ++x)
			out << ", " << A.at(x, y);

		out << ";" << std::endl;
	}

	out << "]" << std::endl << std::endl;

	return out;
}