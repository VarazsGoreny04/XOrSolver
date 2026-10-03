#pragma once

#include <cassert>
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include "Vector.h"
#include "Matrix.h"

struct NeuralNetwork
{
private:
	struct LayerResults
	{
	public:
		Vector<float> weightedInputs;
		Vector<float> activationValues;
		// The derivatives of the weighted inputs are the previous activation values
		Vector<float> dActivationValues;

		LayerResults() {
			this->weightedInputs = Vector<float>();
			this->activationValues = Vector<float>();
			this->dActivationValues = Vector<float>();
		}

		LayerResults(const Vector<float>& weightedInputs, const Vector<float>& activationValues, const Vector<float>& dActivationValues) {
			this->weightedInputs = weightedInputs;
			this->activationValues = activationValues;
			this->dActivationValues = dActivationValues;
		}

		~LayerResults() {}
	};

	size_t layers; // Not including the input layer!
	size_t inputs;
	size_t outputs;
	Vector<Matrix<float>> weights;
	Vector<Vector<float>> biases;

	static float randE() {
		return (float)((2. * ((double)rand() / RAND_MAX) - 1.) * M_E);
	}

	static Vector<float> activation(const Vector<float> xs) {
		return Vector<float>::apply(xs, [](float x) -> float { return 1.f / (1.f + expf(-x)); });
	}

	static Vector<float> activationDerivative(const Vector<float> xs) {
		Vector<float> activated = activation(xs);

		return Vector<float>::apply(activated, [](float x) -> float { return  x * (1.f - x); });
	}

	static Vector<float> loss(const Vector<float> expected, const  Vector<float> predicted) {
		Vector<float> loss = predicted - expected;

		return Vector<float>::scale(loss, loss);
	}

	static Vector<float> lossDerivative(const Vector<float> expected, const Vector<float> predicted) {
		Vector<float> loss = predicted - expected;

		return Vector<float>::scale(loss, 2.f);
	}

	std::vector<LayerResults> getPartialResults(const Vector<float>& input) const {
		assert(input.length == inputs);

		Vector<float> activationValues = Vector<float>(input);

		std::vector<LayerResults> result = std::vector<LayerResults>();
		{
			LayerResults first = LayerResults();
			first.activationValues = input;
			result.push_back(first);
		}

		for (size_t i = 0; i < layers; ++i) {
			Matrix<float> weights = this->weights.at(i);
			Vector<float> biases = this->biases.at(i);

			Vector<float> weightedInputs = weights * activationValues + biases;
			activationValues = activation(weightedInputs);
			Vector<float> dActivationValues = activationDerivative(weightedInputs);

			result.push_back(LayerResults(weightedInputs, activationValues, dActivationValues));
		}

		return result;
	}

	void backwardOneData(const Vector<float>& xs, const Vector<float>& ys, Vector<Matrix<float>>& weightChanges, Vector<Vector<float>>& biasChanges) const {
		assert(weightChanges.length == layers);
		assert(biasChanges.length == layers);
		assert(xs.length == inputs);
		assert(ys.length == outputs);

		std::vector<LayerResults> partialResults = getPartialResults(xs);
		Vector<float> deltaVector;

		assert(partialResults.size() == layers + 1);

		std::vector<LayerResults>::iterator layerResult = --(partialResults.end());

		{
			deltaVector = Vector<float>::scale(lossDerivative(ys, (*layerResult).activationValues), (*layerResult).dActivationValues);

			weightChanges.put(layers - 1, Matrix<float>::gradient(deltaVector, (*(--layerResult)).activationValues));
			biasChanges.put(layers - 1, deltaVector);
		}

		for (size_t layer = layers - 1; layer-- > 0;) {
			deltaVector = Vector<float>::scale(Matrix<float>::transpose(weights.at(layer + 1)) * deltaVector, (*layerResult).dActivationValues);

			weightChanges.put(layer, Matrix<float>::gradient(deltaVector, (*(--layerResult)).activationValues));
			biasChanges.put(layer, deltaVector);
		}
	}

public:
	NeuralNetwork(const std::vector<size_t>& layers) {
		this->layers = layers.size() - 1;
		this->inputs = layers[0];
		this->outputs = layers[this->layers];

		assert(this->layers > 0);

		this->weights = Vector<Matrix<float>>(this->layers);
		this->biases = Vector<Vector<float>>(this->layers);

		size_t prev = layers.at(0);
		size_t next;

		for (size_t i = 1; i <= this->layers; ++i) {
			next = layers[i];

			Matrix<float> currentWeights = Matrix<float>(prev, next);
			Vector<float> currentBiases = Vector<float>(next);

			for (size_t y = 0; y < next; ++y) {
				for (size_t x = 0; x < prev; ++x)
					currentWeights.put(x, y, randE());
			}

			this->weights.put(i - 1, currentWeights);
			this->biases.put(i - 1, currentBiases);

			prev = next;
		}
	}

	~NeuralNetwork() {}

	Vector<float> forward(const Vector<float>& input) const {
		assert(input.length == inputs);

		Vector<float> result = Vector<float>(input);

		for (size_t i = 0; i < layers; ++i) {
			Matrix<float> weight = weights.at(i);
			Vector<float> bias = biases.at(i);

			result = activation(weight * result + bias);
		}

		return result;
	}

	float totalLoss(const Matrix<Vector<float>>& dataset) const {
		assert(dataset.lengthX == 2);

		float totalLoss = 0.f;

		for (size_t i = 0; i < dataset.lengthY; ++i) {
			Vector<float> temp = loss(dataset.at(1, i), forward(dataset.at(0, i)));

			for (size_t i = 0; i < temp.length; ++i)
				totalLoss += temp.at(i);
		}

		return totalLoss / dataset.lengthY;
	}

	float backward(const Matrix<Vector<float>>& dataset) {
		assert(dataset.lengthX == 2);

		const float rate = 1e+1f;

		Vector<Matrix<float>> allWeightChanges = Vector<Matrix<float>>(weights.length);
		Vector<Vector<float>> allBiasChanges = Vector<Vector<float>>(biases.length);

		if (dataset.lengthY > 0)
			backwardOneData(dataset.at(0, 0), dataset.at(1, 0), allWeightChanges, allBiasChanges);

		for (size_t i = 1; i < dataset.lengthY; ++i) {
			Vector<Matrix<float>> weightChanges = Vector<Matrix<float>>(weights.length);
			Vector<Vector<float>> biasChanges = Vector<Vector<float>>(biases.length);

			backwardOneData(dataset.at(0, i), dataset.at(1, i), weightChanges, biasChanges);

			allWeightChanges = allWeightChanges + weightChanges;
			allBiasChanges = allBiasChanges + biasChanges;
		}

		float scalar = rate / dataset.lengthY;

		weights = weights - Vector<Matrix<float>>::apply(allWeightChanges,
			[scalar](const Matrix<float>& A) -> Matrix<float> { return Matrix<float>::scale(A, scalar); });
		biases = biases - Vector<Vector<float>>::apply(allBiasChanges,
			[scalar](const Vector<float>& a) -> Vector<float> { return Vector<float>::scale(a, scalar); });

		return totalLoss(dataset);
	}

	static void run(const Matrix<Vector<float>>& dataset) {
		std::vector<size_t> layers = { 2, 2, 1 };

		NeuralNetwork n = NeuralNetwork(layers);

		std::cout << "Initial weights: " << n.weights << std::endl
			<< "Initial biases: " << n.biases << std::endl << std::endl;

		float loss = 1.f;
		for (size_t i = 0; i < 1e+3 && loss > 1e-6; ++i) {
			loss = n.backward(dataset);

			std::cout << "Current loss: " << loss << "  Current weights: " << n.weights << "  Current biases: " << n.biases << std::endl << std::endl;
		}

		std::cout << "----------------------------" << std::endl << "Final loss: " << n.totalLoss(dataset) << std::endl;

		for (size_t i = 0; i < dataset.lengthY; ++i)
			std::cout << dataset.at(0, i) << " - " << dataset.at(1, i) << " : " << n.forward(dataset.at(0, i)) << std::endl;

		std::cout << std::endl << "Final weights: " << n.weights << std::endl
			<< "Final biases: " << n.biases << std::endl << std::endl;
	}
};