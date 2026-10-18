#pragma once

#undef NDEBUG
#include <cassert>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include "Matrix.h"

/// <summary>
/// A neuron with two inputs and one output.
/// </summary>
struct ComplexNeuron
{
private:
	/// <summary>
	/// The weight of the first input.
	/// </summary>
	float weight1;

	/// <summary>
	/// The weight of the second input.
	/// </summary>
	float weight2;

	/// <summary>
	/// The bias of this neuron.
	/// </summary>
	float bias;

	/// <summary>
	/// The epsilon value used in the finite derivative calculation.
	/// </summary>
	float epsilon;

	/// <summary>
	/// The learning rate of the backpropagation algorithm.
	/// </summary>
	float learningRate;

	/// <summary>
	/// Creates a random value between -e and e.
	/// </summary>
	/// <returns>The created value.</returns>
	static float randE() {
		double random = 2 * ((double)rand() / RAND_MAX) - 1;
		double cubedRandom = random * random * random;

		return (float)(cubedRandom * M_E);
	}

	/// <summary>
	/// The activation function of the backpropagation.
	/// </summary>
	/// <param name="x">The input value.</param>
	/// <returns>The crated value.</returns>
	static float activation(const float x) {
		return 1.f / (1.f + expf(-x));
	}

	/// <summary>
	/// The loss/cost function.
	/// </summary>
	/// <param name="expected">The expected value.</param>
	/// <param name="predicted">The predicted value.</param>
	/// <returns>The squared difference between the expected and predicted values.</returns>
	static float loss(const float expected, const float predicted) {
		float loss = predicted - expected;

		return loss * loss;
	}

public:
	/// <summary>
	/// Creates a ComplexNeuron by the given values.
	/// </summary>
	/// <param name="epsilon">The epsilon value used in the finite derivative calculation.</param>
	/// <param name="learningRate">The learning rate of the backpropagation algorithm.</param>
	ComplexNeuron(float epsilon, float learningRate) {
		this->weight1 = randE();
		this->weight2 = randE();
		this->bias = 0.f;

		this->epsilon = epsilon;
		this->learningRate = learningRate;
	}

	/// <summary>
	/// Deconstructs this ComplexNeuron.
	/// </summary>
	~ComplexNeuron() {}

	/// <summary>
	/// Forward-propagates the network.
	/// </summary>
	/// <param name="input1">The first input of the network.</param>
	/// <param name="input2">The second input of the network.</param>
	/// <returns>The computed output.</returns>
	float forward(const float input1, const float input2) const {
		return activation(weight1 * input1 + weight2 * input2 + bias);
	}

	/// <summary>
	/// Calculates the loss function for all of the inputs in the dataset.
	/// </summary>
	/// <param name="dataset">The set of input and expected output values in a matrix.</param>
	/// <returns>The value of the total loss.</returns>
	float totalLoss(const Matrix<float>& dataset) const {
		assert(dataset.getLengthX() == 3);

		float result = 0.f;

		for (size_t i = 0; i < dataset.getLengthY(); ++i)
			result += loss(dataset.at(2, i), forward(dataset.at(0, i), dataset.at(1, i)));

		return result / dataset.getLengthY();
	}

	/// <summary>
	/// Back-propagates the network.
	/// </summary>
	/// <param name="dataset">The set of input and expected output values in a matrix.</param>
	/// <returns>The new total loss.</returns>
	float backward(const Matrix<float>& dataset) {
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

		this->weight1 = initialWeight1 + (currentLoss - weightAddLoss1) * learningRate;
		this->weight2 = initialWeight2 + (currentLoss - weightAddLoss2) * learningRate;
		this->bias = initialBias + (currentLoss - biasAddLoss) * learningRate;

		return totalLoss(dataset);
	}
	
	/// <summary>
	/// Builds a network and trains it on the given dataset.
	/// </summary>
	/// <param name="epsilon">The epsilon value used in the finite derivative.</param>
	/// <param name="learningRate">The learning rate of the backpropagation.</param>
	/// <param name="dataset">The set of input and expected output values in a matrix.</param>
	static void run(float epsilon, float learningRate, const Matrix<float>& dataset) {
		ComplexNeuron n(epsilon, learningRate);

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

		for (size_t i = 0; i < dataset.getLengthY(); ++i) {
			std::cout << dataset.at(0, i) << " - " << dataset.at(1, i) << " - " << dataset.at(2, i)
				<< " : " << n.forward(dataset.at(0, i), dataset.at(1, i)) << std::endl;
		}

		std::cout << std::endl << "Final weight1: " << n.weight1 << std::endl
			<< "Final weight2: " << n.weight2 << std::endl
			<< "Final bias: " << n.bias << std::endl << std::endl;
	}
};