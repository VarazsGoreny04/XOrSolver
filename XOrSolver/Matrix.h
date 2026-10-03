#pragma once

#undef NDEBUG
#include <cassert>
#include <iostream>
#include <functional>
#include <stdlib.h>
#include "Vector.h"

/// <summary>
/// A mathematical matrix.
/// </summary>
/// <typeparam name="T">The type of the matrix elements.</typeparam>
template <typename T>
class Matrix
{
private:
	/// <summary>
	/// The number of columns in the matrix.
	/// </summary>
	size_t lengthX;

	/// <summary>
	/// The number of rows in the matrix.
	/// </summary>
	size_t lengthY;

	/// <summary>
	/// The number of elements in the matrix.
	/// </summary>
	size_t length;

	/// <summary>
	/// The number of elements in the matrix.
	/// </summary>
	T* matrix;

	/// <summary>
	/// Sets the last index of the matrix to the given element.
	/// </summary>
	/// <param name="index">The the last index of the matrix.</param>
	/// <param name="value">The element.</param>
	void maker(int index, T value) {
		assert(index + 1 == length);

		matrix[index] = value;
	}

	/// <summary>
	/// Fills the matrix with the given elements from the given index.
	/// </summary>
	/// <typeparam name="Args">The type of the collection of the elements.</typeparam>
	/// <param name="index">The first index to fill the elements from.</param>
	/// <param name="value">One element.</param>
	/// <param name="values">The collection of the elements.</param>
	template <typename... Args>
	void maker(int index, T value, Args... values) {
		assert(index < length);

		matrix[index] = value;

		maker(index + 1, values...);
	}

public:
	/// <summary>
	/// Creates an empty matrix.
	/// </summary>
	Matrix() {
		this->lengthX = 0;
		this->lengthY = 0;
		this->length = 0;
		this->matrix = nullptr;
	}

	/// <summary>
	/// Creates a vector with the given size.
	/// </summary>
	/// <param name="lengthX">The number of columns in the matrix.</param>
	/// <param name="lengthY">The number of rows in the matrix.</param>
	Matrix(const size_t lengthX, const size_t lengthY) {
		this->lengthX = lengthX;
		this->lengthY = lengthY;
		this->length = this->lengthX * this->lengthY;

		assert(this->lengthX > 0 && this->lengthY > 0);

		this->matrix = this->length > 0 ? new T[this->length]{} : nullptr;
	}

	/// <summary>
	/// Creates a matrix identical to the given matrix.
	/// </summary>
	/// <param name="other">The matrix.</param>
	Matrix(const Matrix& other)
	{
		this->lengthX = other.lengthX;
		this->lengthY = other.lengthY;
		this->length = other.length;
		this->matrix = new T[length];

		for (size_t i = 0; i < this->length; ++i)
			this->matrix[i] = other.matrix[i];
	}

	/// <summary>
	/// Deconstructs this matrix.
	/// </summary>
	~Matrix() {
		delete[] matrix;
	}

	/// <returns>The number of columns in the matrix.</returns>
	const size_t getLengthX() const {
		return lengthX;
	}

	/// <returns>The number of rows in the matrix.</returns>
	const size_t getLengthY() const {
		return lengthY;
	}

	/// <returns>The number of elements in the matrix.</returns>
	const size_t getLength() const {
		return length;
	}

	/// <summary>
	/// Fills the matrix with the given elements.
	/// </summary>
	/// <typeparam name="Args">The type of the collection of the elements.</typeparam>
	/// <param name="values">The collection of the elements.</param>
	/// <returns>This matrix.</returns>
	template <typename... Args>
	Matrix<T>& make(Args... values) {
		maker(0, values...);

		return *this;
	}

	/// <summary>
	/// Gets one element of the matrix by the given indexes.
	/// </summary>
	/// <param name="x">The column index of the element.</param>
	/// <param name="y">The row index of the element.</param>
	/// <returns>A copy of the element.</returns>
	T at(const size_t x, const size_t y) const {
		assert(x < lengthX && y < lengthY);

		return matrix[lengthX * y + x];
	}

	/// <summary>
	/// Sets one element of the matrix to the given value.
	/// </summary>
	/// <param name="x">The column index of the element.</param>
	/// <param name="y">The row index of the element.</param>
	/// <param name="value">The value.</param>
	/// <returns>This vector.</returns>
	Matrix<T>& put(const size_t x, const size_t y, const T value) {
		assert(x < lengthX && y < lengthY);

		matrix[lengthX * y + x] = value;

		return *this;
	}

	/// <summary>
	/// Transposes the given matrix.
	/// </summary>
	/// <param name="A">The matrix.</param>
	/// <returns>The created matrix.</returns>
	static Matrix<T> transpose(const Matrix<T>& A) {
		Matrix<T> result(A.lengthY, A.lengthX);

		for (size_t y = 0; y < A.lengthY; ++y) {
			for (size_t x = 0; x < A.lengthX; ++x)
				result.matrix[A.lengthY * x + y] = A.matrix[A.lengthX * y + x];
		}

		return result;
	}

	/// <summary>
	/// Adds two matrices.
	/// </summary>
	/// <param name="A">The first matrix.</param>
	/// <param name="B">The second matrix.</param>
	/// <returns>The created matrix.</returns>
	static Matrix<T> add(const Matrix<T>& A, const Matrix<T>& B) {
		assert(A.lengthX == B.lengthX);
		assert(A.lengthY == B.lengthY);

		Matrix<T> result(A.lengthX, A.lengthY);

		for (size_t i = 0; i < result.length; ++i)
			result.matrix[i] = A.matrix[i] + B.matrix[i];

		return result;
	}

	/// <summary>
	/// Subtracts two matrices.
	/// </summary>
	/// <param name="A">The first matrix.</param>
	/// <param name="B">The second matrix.</param>
	/// <returns>The created matrix.</returns>
	static Matrix<T> subtract(const Matrix<T>& A, const Matrix<T>& B) {
		assert(A.lengthX == B.lengthX);
		assert(A.lengthY == B.lengthY);

		Matrix<T> result(A.lengthX, A.lengthY);

		for (size_t i = 0; i < result.length; ++i)
			result.matrix[i] = A.matrix[i] - B.matrix[i];

		return result;
	}

	/// <summary>
	/// Calculates the product of a matrix and a vector.
	/// </summary>
	/// <param name="A">The matrix.</param>
	/// <param name="b">The vector.</param>
	/// <returns>The created vector.</returns>
	static Vector<T> product(const Matrix<T>& A, const Vector<T>& b) {
		assert(A.lengthX == b.getLength());

		Vector<T> result(A.lengthY);

		for (size_t y = 0; y < A.lengthY; ++y) {
			T temp = 0;

			for (size_t x = 0; x < A.lengthX; ++x)
				temp = temp + A.matrix[A.lengthX * y + x] * b.at(x);

			result.put(y, temp);
		}

		return result;
	}

	/// <summary>
	/// Calculates the product of two matrices.
	/// </summary>
	/// <param name="A">The first matrix.</param>
	/// <param name="B">The second matrix.</param>
	/// <returns>The created matrix.</returns>
	static Matrix<T> product(const Matrix<T>& A, const Matrix<T>& B) {
		assert(A.lengthX == B.lengthY);

		Matrix<T> result(B.lengthX, A.lengthY);

		for (size_t i = 0; i < A.lengthY; ++i) {
			for (size_t j = 0; j < B.lengthX; ++j) {
				T temp = 0;

				for (size_t k = 0; k < A.lengthX; ++k)
					temp = temp + A.matrix[A.lengthX * i + k] * B.matrix[B.lengthX * k + j];

				result.matrix[B.lengthX * i + j] = temp;
			}
		}

		return result;
	}

	/// <summary>
	/// Scales the elements of a matrix by a value.
	/// </summary>
	/// <param name="A">The matrix.</param>
	/// <param name="b">The value.</param>
	/// <returns>The created matrix.</returns>
	static Matrix<T> scale(const Matrix<T>& A, const T b) {
		Matrix<T> result(A.lengthX, A.lengthY);

		for (size_t i = 0; i < result.length; ++i)
			result.matrix[i] = A.matrix[i] * b;

		return result;
	}

	/// <summary>
	/// Scales the elements of a matrix by the corresponding elements of another matrix.
	/// </summary>
	/// <param name="a">The matrix to scale.</param>
	/// <param name="b">The matrix to scale by.</param>
	/// <returns>The created matrix.</returns>
	static Matrix<T> scale(const Matrix<T>& A, const Matrix<T>& B) {
		assert(A.lengthX == B.lengthX);
		assert(A.lengthY == B.lengthY);

		Matrix<T> result(A.lengthX, A.lengthY);

		for (size_t i = 0; i < A.length; ++i)
			result.matrix[i] = A.matrix[i] * B.matrix[i];

		return result;
	}

	/// <summary>
	/// Calculates the gradient of two vectors.
	/// </summary>
	/// <param name="a">The first vector.</param>
	/// <param name="b">The second vector.</param>
	/// <returns>The created matrix.</returns>
	static Matrix<T> gradient(const Vector<T>& a, const Vector<T>& b) {
		Matrix<T> result(b.getLength(), a.getLength());

		for (size_t y = 0; y < a.getLength(); ++y) {
			T aY = a.at(y);

			for (size_t x = 0; x < b.getLength(); ++x)
				result.matrix[b.getLength() * y + x] = aY * b.at(x);
		}

		return result;
	}

	/// <summary>
	/// Applies a function to each element of the given matrix.
	/// </summary>
	/// <param name="A">The matrix.</param>
	/// <param name="predicate">The function.</param>
	/// <returns>The created matrix.</returns>
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