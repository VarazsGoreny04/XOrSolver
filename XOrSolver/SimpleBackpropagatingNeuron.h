#pragma once

#include <cassert>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include "Matrix.h"

class SimpleBackpropagatingNeuron
{
private:
	float weight;
	float bias;

	static float randE() {
		const int precision = 1000;

		double random = (double)(rand() % (precision * 2 + 1) - precision) / precision;
		double normal = random * random * M_E;

		return (float)normal;
	}

	static float loss(const float expected, const float predicted) {
		float loss = predicted - expected;

		return loss * loss;
	}

	static float lossDerivative(const float expected, const float predicted) {
		float loss = predicted - expected;

		return 2.f * loss;
	}

public:
	SimpleBackpropagatingNeuron() {
		this->weight = randE();
		this->bias = 0.f;
	}

	~SimpleBackpropagatingNeuron() {}

	float forward(const float input) const {
		return weight * input + bias;
	}

	float totalLoss(const Matrix<float>& dataset) const {
		assert(dataset.getLengthX() == 2);

		float result = 0.f;

		for (size_t i = 0; i < dataset.getLengthY(); ++i)
			result += loss(dataset.at(1, i), forward(dataset.at(0, i)));

		return result / dataset.getLengthY();
	}

	float backward(const Matrix<float>& dataset) {
		assert(dataset.getLengthX() == 2);

		const float rate = 1e-1f;

		float weightChange = 0.f;
		float biasChange = 0.f;

		for (size_t i = 0; i < dataset.getLengthY(); ++i) {
			float x1 = dataset.at(0, i);
			float y = dataset.at(1, i);

			float temp = lossDerivative(y, forward(x1));

			weightChange += temp * x1;
			biasChange += temp;
		}

		this->weight -= (weightChange / dataset.getLengthY()) * rate;
		this->bias -= (biasChange / dataset.getLengthY()) * rate;

		return totalLoss(dataset);
	}

	static void run(const Matrix<float>& dataset) {
		SimpleBackpropagatingNeuron n;

		std::cout << "Initial weight: " << n.weight << std::endl
			<< "Initial bias: " << n.bias << std::endl << std::endl;

		float loss = 1.f;
		for (size_t i = 0; i < 1e+4 && loss > 1e-6; ++i) {
			loss = n.backward(dataset);
			
			//std::cout << "Current loss: " << loss << "  Current weight: " << n.weight << "  Current bias: " << n.bias << std::endl;
		}

		std::cout << "----------------------------" << std::endl << "Final loss: " << n.totalLoss(dataset) << std::endl;

		for (size_t i = 0; i < dataset.getLengthY(); ++i)
			std::cout << dataset.at(0, i) << " - " << dataset.at(1, i) << " : " << n.forward(dataset.at(0, i)) << std::endl;

		std::cout << std::endl << "Final weight: " << n.weight << std::endl
			<< "Final bias: " << n.bias << std::endl << std::endl;
	}
};