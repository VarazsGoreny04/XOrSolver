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

	static float randE() {
		const int precision = 1000;

		double random = (double)(rand() % (precision * 2 + 1) - precision) / precision;
		double normal = random * random * M_E;

		return (float)normal;
	}

	static float activation(const float x) {
		return 1.f / (1.f + expf(-x));
	}

	static float activationDerivative(const float x) {
		float temp = activation(x);

		return temp * (1.f - temp);
	}

	static Vector<float> loss(const Vector<float> expected, const  Vector<float> predicted) {
		Vector<float> loss = predicted - expected;

		return Vector<float>::scale(loss, loss);
	}

	static Vector<float> lossDerivative(const Vector<float> expected, const Vector<float> predicted) {
		Vector<float> loss = predicted - expected;

		return Vector<float>::scale(loss, -2.f);
	}

public:
	NeuralNetwork(const Vector<size_t>& layers) {
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

	Vector<float> forward(const Vector<float>& inputs) const {
		Vector<float> result = Vector<float>(inputs);

		for (size_t i = 0; i < layers - 1; ++i) {
			Matrix<float> weight = weights.at(i);
			Vector<float> bias = biases.at(i);

			result = Vector<float>::apply(weight * result + bias, activation);
		}

		return result;
	}

	/*float totalLoss(Matrix<float>& dataset) const {
		if (dataset.lengthX != 3)
			exit(1);

		float totalLoss = 0.f;

		for (size_t i = 0; i < dataset.lengthY; ++i) {
			Vector<float> temp = loss(dataset.at(2, i), forward(dataset.at(0, i), dataset.at(1, i)));

			for (size_t i = 0; i < dataset.lengthY; ++i) {
				totalLoss += ;
		}

		return totalLoss / dataset.lengthY;
	}

	float backward(const Matrix<float>& dataset) {
		if (dataset.lengthX != 3)
			exit(1);

		const float epsilon = 1e-2f;
		const float rate = 1e+1f;

		float weightChange1 = 0.f;
		float weightChange2 = 0.f;
		float biasChange = 0.f;

		for (size_t i = 0; i < dataset.lengthY; ++i) {
			float x1 = dataset.at(0, i);
			float x2 = dataset.at(1, i);
			float y = dataset.at(2, i);

			float ad = activationDerivative(forward(x1, x2));
			float ld = lossDerivative(y, forward(x1, x2));
			float ldXAd = ld * ad;

			weightChange1 += ldXAd * x1;
			weightChange2 += ldXAd * x2;
			biasChange += ldXAd;
		}

		this->weight1 += (weightChange1 / dataset.lengthY) * rate;
		this->weight2 += (weightChange2 / dataset.lengthY) * rate;
		this->bias += (biasChange / dataset.lengthY) * rate;

		return totalLoss(dataset);
	}*/

	static void run() {
		Vector<size_t> layers = Vector<size_t>(3)
			.put(0, 2)
			.put(1, 2)
			.put(2, 1);

		NeuralNetwork nn = NeuralNetwork(layers);

		Vector<float> input = Vector<float>(layers.at(0))
			.put(0, 1.f)
			.put(1, 2.f);

		Vector<float> result = nn.forward(input);
	}
};