#pragma once

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

	std::vector<LayerResults> getPartialResults(const Vector<float>& input) const {
		if (input.length != inputs)
			exit(1);

		Vector<float> partialResult = Vector<float>(input);

		std::vector<LayerResults> result = std::vector<LayerResults>();
		{
			LayerResults first = LayerResults();
			first.activationValues = input;
			result.push_back(first);
		}

		for (size_t i = 0; i < layers; ++i) {
			Matrix<float> weights = this->weights.at(i);
			Vector<float> biases = this->biases.at(i);

			Vector<float> weightedInputs = weights * partialResult + biases;
			Vector<float> activationValues = Vector<float>::apply(weightedInputs, activation);
			Vector<float> dActivationValues = Vector<float>::apply(weightedInputs, activationDerivative);

			result.push_back(LayerResults(weightedInputs, activationValues, dActivationValues));
		}

		return result;
	}

	void backwardOneData(const Vector<float>& xs, const Vector<float>& ys, Vector<Matrix<float>>& weightChanges, Vector<Vector<float>>& biasChanges) const {
		if (weightChanges.length != layers || biasChanges.length != layers || xs.length != inputs || ys.length != outputs) // csak debuging
			exit(1);

		std::vector<LayerResults> partialResults = getPartialResults(xs);

		/*if (partialResults.size() != layers + 1) // csak debuging
			exit(1);*/

		std::vector<LayerResults>::iterator layerResult = --(partialResults.end());

		{
			Vector<float> cWrtATimesAWrtZ = Vector<float>::scale(lossDerivative(ys, (*layerResult).activationValues), (*layerResult).dActivationValues);

			--layerResult;

			weightChanges.put(layers - 1, Matrix<float>::gradiant(cWrtATimesAWrtZ, (*layerResult).activationValues));
			biasChanges.put(layers - 1, cWrtATimesAWrtZ);
		}

		for (long long layer = layers - 2; layer >= 0; --layer) {
			Matrix<float> endAndWeights = Matrix<float>::scale(weightChanges.at(layer + 1), weights.at(layer + 1));

			Vector<float> endAndWeightsSummed = Vector<float>(endAndWeights.lengthX);
			for (size_t x = 0; x < endAndWeights.lengthX; ++x) {
				float sum = 0.f;

				for (size_t y = 0; y < endAndWeights.lengthY; ++y)
					sum += endAndWeights.at(x, y);

				endAndWeightsSummed.put(x, sum);
			}

			Vector<float> endAndWeightsTimesA = Vector<float>::scale(endAndWeightsSummed, (*layerResult).dActivationValues);
			
			--layerResult;

			weightChanges.put(layer, Matrix<float>::gradiant(endAndWeightsTimesA, (*layerResult).activationValues));
			biasChanges.put(layer, endAndWeightsTimesA);
		}
	}

public:
	NeuralNetwork(const std::vector<size_t>& layers) {
		this->layers = layers.size() - 1;
		this->inputs = layers[0];
		this->outputs = layers[this->layers];

		if (this->layers < 1)
			exit(1);

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

				currentBiases.put(y, 0);
			}

			this->weights.put(i - 1, currentWeights);
			this->biases.put(i - 1, currentBiases);

			prev = next;
		}
	}

	~NeuralNetwork() {}

	Vector<float> forward(const Vector<float>& input) const {
		if (input.length != inputs)
			exit(1);

		Vector<float> result = Vector<float>(input);

		for (size_t i = 0; i < layers; ++i) {
			Matrix<float> weight = weights.at(i);
			Vector<float> bias = biases.at(i);

			result = Vector<float>::apply(weight * result + bias, activation);
		}

		return result;
	}

	float totalLoss(const Matrix<Vector<float>>& dataset) const {
		if (dataset.lengthX != 2)
			exit(1);

		float totalLoss = 0.f;

		for (size_t i = 0; i < dataset.lengthY; ++i) {
			Vector<float> temp = loss(dataset.at(1, i), forward(dataset.at(0, i)));

			for (size_t i = 0; i < temp.length; ++i)
				totalLoss += temp.at(i);
		}

		return totalLoss / dataset.lengthY;
	}

	float backward(const Matrix<Vector<float>>& dataset) {
		if (dataset.lengthX != 2)
			exit(1);

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

		weights = weights + Vector<Matrix<float>>::apply(allWeightChanges,
			[scalar](const Matrix<float>& A) -> Matrix<float> { return Matrix<float>::scale(A, scalar); });
		biases = biases + Vector<Vector<float>>::apply(allBiasChanges,
			[scalar](const Vector<float>& a) -> Vector<float> { return Vector<float>::scale(a, scalar); });

		return totalLoss(dataset);
	}

	static void run(const Matrix<Vector<float>>& dataset) {
		std::vector<size_t> layers = {2, 2, 1};

		NeuralNetwork n = NeuralNetwork(layers);

		std::cout << "Initial weights: " << n.weights << std::endl
			<< "Initial biases: " << n.biases << std::endl << std::endl;

		float loss = 1.f;
		for (size_t i = 0; i < 1e+3 && loss > 1e-6; ++i) {
			loss = n.backward(dataset);
			//std::cout << "Current loss: " << loss << "  Current weights: " << n.weights << "  Current biases: " << n.biases << std::endl;
		}

		std::cout << "----------------------------" << std::endl << "Final loss: " << n.totalLoss(dataset) << std::endl;

		for (size_t i = 0; i < dataset.lengthY; ++i)
			std::cout << dataset.at(0, i) << " - " << dataset.at(1, i) << " : " << n.forward(dataset.at(0, i)) << std::endl;

		std::cout << std::endl << "Final weights: " << n.weights << std::endl
			<< "Final biases: " << n.biases << std::endl << std::endl;
	}
};