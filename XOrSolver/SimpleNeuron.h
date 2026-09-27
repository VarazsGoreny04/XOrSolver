#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "Vector.h"
#include "Matrix.h"

class SimpleNeuron
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

	SimpleNeuron() {
		this->weight = randE();
		this->bias = 0.f;
	}

	~SimpleNeuron() { }

	float forward(float input) const {
		return weight * input + bias;
	}

	static float loss(float expected, float predicted) {
		float loss = predicted - expected;

		return loss * loss;
	}

	float totalLoss(Matrix<float>& dataset) const {
		if (dataset.lengthX != 2)
			exit(1);

		size_t length = dataset.lengthY;
		float totalLoss = 0.f;

		for (size_t i = 0; i < length; ++i)
			totalLoss += loss(dataset.at(1, i), forward(dataset.at(0, i)));

		return totalLoss / length;
	}

	float backward(Matrix<float>& dataset) {
		const float epsilon = 1e-5f;
		const float rate = 100;

		float initialWeight = weight;
		float initialBias = bias;

		float currentLoss = totalLoss(dataset);

		this->weight += epsilon;

		float weightAddLoss = totalLoss(dataset);

		this->weight = initialWeight;
		this->bias += epsilon;

		float biasAddLoss = totalLoss(dataset);

		this->weight = initialWeight + (currentLoss - weightAddLoss) * rate;
		this->bias = initialBias + (currentLoss - biasAddLoss) * rate;

		return currentLoss;
	}
};

static void runNeuron(Matrix<float>& dataset) {
	SimpleNeuron n = SimpleNeuron();

	std::cout << "Initial weight: " << n.weight << std::endl
			  << "Initial bias: " << n.bias << std::endl << std::endl;

	float loss = 1.f;
	for (size_t i = 0; i < 10000 && loss > 1e-6; ++i) {
		loss = n.backward(dataset);
		std::cout << "Current loss: " << loss << "  Current weight: " << n.weight << "  Current bias: " << n.bias << std::endl;
	}

	std::cout << "----------------------------" << std::endl << "Final loss: " << n.totalLoss(dataset) << std::endl;

	for (size_t i = 0; i < dataset.lengthY; ++i)
		std::cout << dataset.at(0, i) << " - " << dataset.at(1, i) << " : " << n.loss(dataset.at(1, i), n.forward(dataset.at(0, i))) << std::endl;

	std::cout << std::endl << "Final weight: " << n.weight << std::endl
		<< "Final bias: " << n.bias << std::endl << std::endl;
}