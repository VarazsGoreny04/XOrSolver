#pragma once

#undef NDEBUG
#include <cassert>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include "Matrix.h"

/// <summary>
/// A very simple neuron with just one input and one output.
/// </summary>
struct SimpleNeuron
{
private:
	/// <summary>
	/// The weight of the input.
	/// </summary>
	float weight;

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
	/// Creates a SimpleNeuron by the given values.
	/// </summary>
	/// <param name="epsilon">The epsilon value used in the finite derivative calculation.</param>
	/// <param name="learningRate">The learning rate of the backpropagation algorithm.</param>
	SimpleNeuron(float epsilon, float learningRate) {
		this->weight = randE();
		this->bias = 0.f;

		this->epsilon = epsilon;
		this->learningRate = learningRate;
	}

	/// <summary>
	/// Deconstructs this SimpleNeuron.
	/// </summary>
	~SimpleNeuron() { }

	/// <summary>
	/// Forward-propagates the network.
	/// </summary>
	/// <param name="input">The input of the network.</param>
	/// <returns>The computed output.</returns>
	float forward(const float input) const {
		return weight * input + bias;
	}

	/// <summary>
	/// Calculates the loss function for all of the inputs in the dataset.
	/// </summary>
	/// <param name="dataset">The set of input and expected output values in a matrix.</param>
	/// <returns>The value of the total loss.</returns>
	float totalLoss(const Matrix<float>& dataset) const {
		assert(dataset.getLengthX() == 2);

		float result = 0.f;

		for (size_t i = 0; i < dataset.getLengthY(); ++i)
			result += loss(dataset.at(1, i), forward(dataset.at(0, i)));

		return result / dataset.getLengthY();
	}

	/// <summary>
	/// Back-propagates the network.
	/// </summary>
	/// <param name="dataset">The set of input and expected output values in a matrix.</param>
	/// <returns>The new total loss.</returns>
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

	/// <summary>
	/// Builds a network and trains it on the given dataset.
	/// </summary>
	/// <param name="epsilon">The epsilon value used in the finite derivative.</param>
	/// <param name="learningRate">The learning rate of the backpropagation.</param>
	/// <param name="dataset">The set of input and expected output values in a matrix.</param>
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