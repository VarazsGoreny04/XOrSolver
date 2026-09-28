#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "Matrix.h"

class ComplexBackpropagatingNeuron
{
private:
	static float activation(float x) {
		return 1.f / (1.f + expf(-x));
	}

	static float activationDerivative(float x) {
		float temp = activation(x);

		return temp * (1.f - temp);
	}

	static float randE() {
		const int precision = 1000;

		double random = (double)(rand() % (precision * 2 + 1) - precision) / precision;
		double normal = random * random * M_E;

		return (float)normal;
	}

public:
	float weight1;
	float weight2;
	float bias;

	ComplexBackpropagatingNeuron() {
		this->weight1 = randE();
		this->weight2 = randE();
		this->bias = 0.f;
	}

	~ComplexBackpropagatingNeuron() {}

	float forward(float input1, float input2) const {
		return activation(weight1 * input1 + weight2 * input2 + bias);
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
		if (dataset.lengthX != 3)
			exit(1);

		size_t length = dataset.lengthY;
		float totalLoss = 0.f;

		for (size_t i = 0; i < length; ++i)
			totalLoss += loss(dataset.at(2, i), forward(dataset.at(0, i), dataset.at(1, i)));

		return totalLoss / length;
	}

	float backward(Matrix<float>& dataset) {
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
	}
};

static void runComplexBackpropagatingNeuron(Matrix<float>& dataset) {
	ComplexBackpropagatingNeuron n = ComplexBackpropagatingNeuron();

	std::cout << "Initial weight1: " << n.weight1 << std::endl
		<< "Initial weight1: " << n.weight2 << std::endl
		<< "Initial bias: " << n.bias << std::endl << std::endl;

	float loss = 1.f;
	for (size_t i = 0; i < 1e+4 && loss > 1e-6; ++i) {
		loss = n.backward(dataset);

		std::cout << "Current loss: " << loss <<
			"  Current weight1: " << n.weight1 <<
			"  Current weight2: " << n.weight2 <<
			"  Current bias: " << n.bias << std::endl;
	}

	std::cout << "----------------------------" << std::endl << "Final loss: " << n.totalLoss(dataset) << std::endl;

	for (size_t i = 0; i < dataset.lengthY; ++i) {
		std::cout << dataset.at(0, i) << " - " << dataset.at(1, i) << " - " << dataset.at(2, i)
			<< " : " << n.forward(dataset.at(0, i), dataset.at(1, i)) << std::endl;
	}

	std::cout << std::endl << "Final weight1: " << n.weight1 << std::endl
		<< "Final weight2: " << n.weight2 << std::endl
		<< "Final bias: " << n.bias << std::endl << std::endl;
}