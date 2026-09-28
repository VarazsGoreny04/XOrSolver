#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "Matrix.h"

class SimpleBackpropagatingNeuron
{
private:
	static float randE() {
		const int precision = 1000;

		double random = (double)(rand() % (precision * 2 + 1) - precision) / precision;
		double normal = random * random * M_E;

		return (float)normal;
	}

public:
	float weight;
	float bias;

	SimpleBackpropagatingNeuron() {
		this->weight = randE();
		this->bias = 0.f;
	}

	~SimpleBackpropagatingNeuron() {}

	float forward(float input) const {
		return weight * input + bias;
	}

	static float loss(float expected, float predicted) {
		float loss = predicted - expected;

		return loss * loss;
	}

	static float lossDerivative(float expected, float predicted) {
		float loss = predicted - expected;

		return -2.f * loss;
	}

	float totalLoss(Matrix<float>& dataset) const {
		if (dataset.lengthX != 2)
			exit(1);

		float totalLoss = 0.f;

		for (size_t i = 0; i < dataset.lengthY; ++i)
			totalLoss += loss(dataset.at(1, i), forward(dataset.at(0, i)));

		return totalLoss / dataset.lengthY;
	}

	float backward(Matrix<float>& dataset) {
		if (dataset.lengthX != 2)
			exit(1);

		const float epsilon = 1e-5f;
		const float rate = 1e-1f;

		float weightChange = 0.f;
		float biasChange = 0.f;

		for (size_t i = 0; i < dataset.lengthY; ++i) {
			float x1 = dataset.at(0, i);
			float y = dataset.at(1, i);

			float temp = lossDerivative(y, forward(x1));

			weightChange += temp * x1;
			biasChange += temp;
		}

		this->weight += (weightChange / dataset.lengthY) * rate;
		this->bias += (biasChange / dataset.lengthY) * rate;

		return totalLoss(dataset);
	}
};

static void runSimpleBackpropagatingNeuron(Matrix<float>& dataset) {
	SimpleBackpropagatingNeuron n = SimpleBackpropagatingNeuron();

	std::cout << "Initial weight: " << n.weight << std::endl
		<< "Initial bias: " << n.bias << std::endl << std::endl;

	float loss = 1.f;
	for (size_t i = 0; i < 1e+4 && loss > 1e-6; ++i) {
		loss = n.backward(dataset);
		std::cout << "Current loss: " << loss << "  Current weight: " << n.weight << "  Current bias: " << n.bias << std::endl;
	}

	std::cout << "----------------------------" << std::endl << "Final loss: " << n.totalLoss(dataset) << std::endl;

	for (size_t i = 0; i < dataset.lengthY; ++i)
		std::cout << dataset.at(0, i) << " - " << dataset.at(1, i) << " : " << n.forward(dataset.at(0, i)) << std::endl;

	std::cout << std::endl << "Final weight: " << n.weight << std::endl
		<< "Final bias: " << n.bias << std::endl << std::endl;
}