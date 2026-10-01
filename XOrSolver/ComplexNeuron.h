#pragma once

#include <cassert>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include "Matrix.h"

class ComplexNeuron
{
private:
	float weight1;
	float weight2;
	float bias;

	static float randE() {
		const int precision = 1000;

		double random = (double)(rand() % (precision * 2 + 1) - precision) / precision;
		double normal = random * random * M_E;

		return (float)normal;
	}

	static float activation(const float x) {
		return 1.f / (1.f + expf(-x));
	}

	static float loss(const float expected, const float predicted) {
		float loss = predicted - expected;

		return loss * loss;
	}

public:
	ComplexNeuron() {
		this->weight1 = randE();
		this->weight2 = randE();
		this->bias = 0.f;
	}

	~ComplexNeuron() {}

	float forward(const float input1, const float input2) const {
		return activation(weight1 * input1 + weight2 * input2 + bias);
	}

	float totalLoss(const Matrix<float>& dataset) const {
		assert(dataset.lengthX == 3);

		float result = 0.f;

		for (size_t i = 0; i < dataset.lengthY; ++i)
			result += loss(dataset.at(2, i), forward(dataset.at(0, i), dataset.at(1, i)));

		return result / dataset.lengthY;
	}

	float backward(const Matrix<float>& dataset) {
		const float epsilon = 1e-2f;
		const float rate = 1e+2f;

		float initialWeight1 = weight1;
		float initialWeight2 = weight2;
		float initialBias = bias;

		float currentLoss = totalLoss(dataset);

		this->weight1 += epsilon;

		float weightAddLoss1 = totalLoss(dataset);

		this->weight1 = initialWeight1;
		this->weight2 += epsilon;

		float weightAddLoss2 = totalLoss(dataset);

		this->weight2 = initialWeight2;
		this->bias += epsilon;

		float biasAddLoss = totalLoss(dataset);

		this->weight1 = initialWeight1 + (currentLoss - weightAddLoss1) * rate;
		this->weight2 = initialWeight2 + (currentLoss - weightAddLoss2) * rate;
		this->bias = initialBias + (currentLoss - biasAddLoss) * rate;

		return totalLoss(dataset);
	}

	static void run(const Matrix<float>& dataset) {
		ComplexNeuron n = ComplexNeuron();

		std::cout << "Initial weight1: " << n.weight1 << std::endl
			<< "Initial weight1: " << n.weight2 << std::endl
			<< "Initial bias: " << n.bias << std::endl << std::endl;

		float loss = 1.f;
		for (size_t i = 0; i < 1e+4 && loss > 1e-6; ++i) {
			loss = n.backward(dataset);

			/*std::cout << "Current loss: " << loss <<
				"  Current weight1: " << n.weight1 <<
				"  Current weight2: " << n.weight2 <<
				"  Current bias: " << n.bias << std::endl;*/
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
};