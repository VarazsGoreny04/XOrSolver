#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include "Vector.h"
#include "Matrix.h"

struct NeuralNetwork
{
private:
	size_t layers;
	Vector<Matrix<float>> weights;
	Vector<Vector<float>> biases;

	static float activation(float x) {
		return 1.f / (1.f + expf(-x));
	}

	static float randE() {
		const int precision = 1000;

		double random = (double)(rand() % (precision * 2 + 1) - precision) / precision;
		double normal = random * random * M_E;

		return (float)normal;
	}

public:
	NeuralNetwork(Vector<size_t> layers) {
		this->layers = layers.length;

		if (this->layers < 2)
			exit(1);

		this->weights = Vector<Matrix<float>>(this->layers - 1);
		this->biases = Vector<Vector<float>>(this->layers - 1);

		size_t prev = layers.at(0);
		size_t next;

		for (size_t i = 1; i < this->layers; ++i) {
			next = layers.at(i);

			Matrix<float> currentWeights = Matrix<float>(prev, next);
			Vector<float> currentBiases = Vector<float>(next);

			for (size_t y = 0; y < next; ++y) {
				for (size_t x = 0; x < prev; ++x)
					currentWeights.put(x, y, randE());

				currentBiases.put(y, randE());
			}

			std::cout << currentWeights;
			std::cout << currentBiases;

			this->weights.put(i - 1, currentWeights);
			this->biases.put(i - 1, currentBiases);

			prev = next;
		}
	}

	~NeuralNetwork() {}

	Vector<float> calculate(Vector<float>& inputs) const {
		Vector<float> result = Vector<float>(inputs);

		for (size_t i = 0; i < layers - 1; ++i) {
			Matrix<float> weight = weights.at(i);
			Vector<float> bias = biases.at(i);

			result = weight * result + bias;

			for (size_t i = 0; i < result.length; ++i)
				result.put(i, activation(result.at(i)));

		}

		std::cout << "-----------------\n" << result;

		return result;
	}
};