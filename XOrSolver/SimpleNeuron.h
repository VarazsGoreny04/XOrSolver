#pragma once

#undef NDEBUG
#include <cassert>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include "Matrix.h"

class SimpleNeuron
{
private:
	float weight;
	float bias;

	float epsilon;
	float learningRate;

	static float randE() {
		double random = 2 * ((double)rand() / RAND_MAX) - 1;
		double cubedRandom = random * random * random;

		return (float)(cubedRandom * M_E);
	}

	static float loss(const float expected, const float predicted) {
		float loss = predicted - expected;

		return loss * loss;
	}

public:
	SimpleNeuron(float epsilon, float learningRate) {
		this->weight = randE();
		this->bias = 0.f;

		this->epsilon = epsilon;
		this->learningRate = learningRate;
	}

	~SimpleNeuron() { }

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
		float initialWeight = weight;
		float initialBias = bias;

		float currentLoss = totalLoss(dataset);

		this->weight += epsilon;

		float weightAddLoss = totalLoss(dataset);

		this->weight = initialWeight;
		this->bias += epsilon;

		float biasAddLoss = totalLoss(dataset);

		this->weight = initialWeight + (currentLoss - weightAddLoss) * learningRate;
		this->bias = initialBias + (currentLoss - biasAddLoss) * learningRate;

		return totalLoss(dataset);
	}

	static void run(float epsilon, float learningRate, const Matrix<float>& dataset) {
		SimpleNeuron n(epsilon, learningRate);

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