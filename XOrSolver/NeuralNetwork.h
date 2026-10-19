#pragma once

#undef NDEBUG
#include <cassert>
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include "Vector.h"
#include "Matrix.h"

/// <summary>
/// A neural network.
/// </summary>
struct NeuralNetwork
{
private:
	/// <summary>
	/// A partial result of the forward-propagation between two layers.
	/// </summary>
	struct LayerResults
	{
	public:
		/// <summary>
		/// The weighted inputs.
		/// </summary>
		Vector<float> weightedInputs;

		/// <summary>
		/// The activation values.
		/// </summary>
		Vector<float> activationValues;

		// The derivatives of the weighted inputs are the previous activation values

		/// <summary>
		/// The derivative of the activation values.
		/// </summary>
		Vector<float> dActivationValues;

		/// <summary>
		/// Creates a LayerResults instance.
		/// </summary>
		LayerResults() {
			this->weightedInputs = Vector<float>();
			this->activationValues = Vector<float>();
			this->dActivationValues = Vector<float>();
		}

		/// <summary>
		/// Creates a LayerResults instance by the activation values.
		/// </summary>
		/// <param name="activationValues">The activation values.</param>
		LayerResults(const Vector<float>& activationValues) {
			this->weightedInputs = Vector<float>();
			this->activationValues = activationValues;
			this->dActivationValues = Vector<float>();
		}

		/// <summary>
		/// Creates a LayerResults instance by the given values.
		/// </summary>
		/// <param name="weightedInputs">The weighted inputs.</param>
		/// <param name="activationValues">The activation values.</param>
		/// <param name="dActivationValues">The derivative of the activation values.</param>
		LayerResults(const Vector<float>& weightedInputs, const Vector<float>& activationValues, const Vector<float>& dActivationValues) {
			this->weightedInputs = weightedInputs;
			this->activationValues = activationValues;
			this->dActivationValues = dActivationValues;
		}

		/// <summary>
		/// Deconstructs this LayerResults.
		/// </summary>
		~LayerResults() {}
	};

	/// <summary>
	/// The number of layers in the network, NOT including the input layer.
	/// </summary>
	size_t layers;

	/// <summary>
	/// The number of neurons in the input layer.
	/// </summary>
	size_t inputs;

	/// <summary>
	/// The number of neurons in the output layer.
	/// </summary>
	size_t outputs;

	/// <summary>
	/// The weights of the network in matrix forms.
	/// The weights of a certain layer can be accessed by the index of the layer to get a matrix of weights out of the vector.
	/// The rows of the matrix contain the weights going out from the neuron of the previous layer.
	/// The columns of the matrix contain the weights arriving to the neuron of the next layer.
	/// </summary>
	Vector<Matrix<float>> weights;

	/// <summary>
	/// The biases of the network in vector forms.
	/// The biases of a certain layer can be accessed by the index of the layer to get a vector of biases out of the outer vector.
	/// </summary>
	Vector<Vector<float>> biases;

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
	/// <param name="xs">The input values.</param>
	/// <returns>The crated values.</returns>
	static Vector<float> activation(const Vector<float> xs) {
		return Vector<float>::apply(xs, [](float x) -> float { return 1.f / (1.f + expf(-x)); });
	}

	/// <summary>
	/// The the derivative of the activation function of the backpropagation.
	/// </summary>
	/// <param name="xs">The input values.</param>
	/// <returns>The crated values.</returns>
	static Vector<float> activationDerivative(const Vector<float> xs) {
		Vector<float> activated = activation(xs);

		return Vector<float>::apply(activated, [](float x) -> float { return  x * (1.f - x); });
	}

	/// <summary>
	/// The loss/cost function.
	/// </summary>
	/// <param name="expected">The expected values.</param>
	/// <param name="predicted">The predicted values.</param>
	/// <returns>The squared difference between the expected and predicted values.</returns>
	static Vector<float> loss(const Vector<float> expected, const  Vector<float> predicted) {
		Vector<float> loss = predicted - expected;

		return Vector<float>::scale(loss, loss);
	}

	/// <summary>
	/// The derivative of the loss/cost function.
	/// </summary>
	/// <param name="expected">The expected values.</param>
	/// <param name="predicted">The predicted values.</param>
	/// <returns>The derivative of the squared difference between the expected and predicted values.</returns>
	static Vector<float> lossDerivative(const Vector<float> expected, const Vector<float> predicted) {
		Vector<float> loss = predicted - expected;

		return Vector<float>::scale(loss, 2.f);
	}

	/// <summary>
	/// Forward-propagates the network and saves every partial result to a structure.
	/// </summary>
	/// <param name="input">The inputs of the network.</param>
	/// <returns>The saved structure.</returns>
	std::vector<LayerResults> getPartialResults(const Vector<float>& input) const {
		assert(input.getLength() == inputs);

		Vector<float> activationValues(input);

		std::vector<LayerResults> result{ LayerResults(activationValues) };

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

	/// <summary>
	/// Back-propagates the network. Calculates the weight and bias changes but only using one data.
	/// </summary>
	/// <param name="xs">The input.</param>
	/// <param name="ys">The expected output.</param>
	/// <param name="weightChanges">The calculated weight changes.</param>
	/// <param name="biasChanges">The calculated bias changes.</param>
	void backwardOneData(const Vector<float>& xs, const Vector<float>& ys, Vector<Matrix<float>>& weightChanges, Vector<Vector<float>>& biasChanges) const {
		assert(xs.getLength() == inputs);
		assert(ys.getLength() == outputs);
		assert(weightChanges.getLength() == layers);
		assert(biasChanges.getLength() == layers);

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
	/// <summary>
	/// Creates a NeuralNetwork by the given values.
	/// </summary>
	/// <param name="layers">The number of neurons in each layer.</param>
	/// <param name="learningRate">The learning rate of the backpropagation algorithm.</param>
	NeuralNetwork(const std::vector<size_t>& layers, float learningRate) {
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

			Matrix<float> currentWeights(prev, next);
			Vector<float> currentBiases(next);

			for (size_t y = 0; y < next; ++y) {
				for (size_t x = 0; x < prev; ++x)
					currentWeights.put(x, y, randE());
			}

			this->weights.put(i - 1, currentWeights);
			this->biases.put(i - 1, currentBiases);

			prev = next;
		}

		this->learningRate = learningRate;
	}

	/// <summary>
	/// Deconstructs this NeuralNetwork.
	/// </summary>
	~NeuralNetwork() {}

	/// <summary>
	/// Forward-propagates the network.
	/// </summary>
	/// <param name="input">The inputs of the network.</param>
	/// <returns>The computed output.</returns>
	Vector<float> forward(const Vector<float>& input) const {
		assert(input.getLength() == inputs);

		Vector<float> result(input);

		for (size_t i = 0; i < layers; ++i) {
			Matrix<float> weight = Matrix<float>(weights.at(i));
			Vector<float> bias = Vector<float>(biases.at(i));

			result = activation(weight * result + bias);
		}

		return result;
	}

	/// <summary>
	/// Calculates the loss function for all of the inputs in the dataset.
	/// </summary>
	/// <param name="dataset">The set of input and expected output values in a matrix.</param>
	/// <returns>The value of the total loss.</returns>
	float totalLoss(const Matrix<Vector<float>>& dataset) const {
		assert(dataset.getLengthX() == 2);

		float totalLoss = 0.f;

		for (size_t i = 0; i < dataset.getLengthY(); ++i) {
			Vector<float> temp = loss(dataset.at(1, i), forward(dataset.at(0, i)));

			for (size_t i = 0; i < temp.getLength(); ++i)
				totalLoss += temp.at(i);
		}

		return totalLoss / dataset.getLengthY();
	}

	/// <summary>
	/// Back-propagates the network.
	/// </summary>
	/// <param name="dataset">The set of input and expected output values in a matrix.</param>
	/// <returns>The new total loss.</returns>
	float backward(const Matrix<Vector<float>>& dataset) {
		assert(dataset.getLengthX() == 2);

		Vector<Matrix<float>> allWeightChanges = Vector<Matrix<float>>(weights.getLength());
		Vector<Vector<float>> allBiasChanges = Vector<Vector<float>>(biases.getLength());

		if (dataset.getLengthY() > 0)
			backwardOneData(dataset.at(0, 0), dataset.at(1, 0), allWeightChanges, allBiasChanges);

		for (size_t i = 1; i < dataset.getLengthY(); ++i) {
			Vector<Matrix<float>> weightChanges = Vector<Matrix<float>>(weights.getLength());
			Vector<Vector<float>> biasChanges = Vector<Vector<float>>(biases.getLength());

			backwardOneData(dataset.at(0, i), dataset.at(1, i), weightChanges, biasChanges);

			allWeightChanges = allWeightChanges + weightChanges;
			allBiasChanges = allBiasChanges + biasChanges;
		}

		float scalar = learningRate / dataset.getLengthY();

		weights = weights - Vector<Matrix<float>>::apply(allWeightChanges,
			[scalar](const Matrix<float>& A) -> Matrix<float> { return Matrix<float>::scale(A, scalar); });
		biases = biases - Vector<Vector<float>>::apply(allBiasChanges,
			[scalar](const Vector<float>& a) -> Vector<float> { return Vector<float>::scale(a, scalar); });

		return totalLoss(dataset);
	}

	/// <summary>
	/// Builds a network and trains it on the given dataset.
	/// </summary>
	/// <param name="layers">The number of neurons in each layer.</param>
	/// <param name="learningRate">The learning rate of the backpropagation.</param>
	/// <param name="dataset">The set of input and expected output values in a matrix.</param>
	static void run(const std::vector<size_t>& layers, float learningRate, const Matrix<Vector<float>>& dataset) {
		NeuralNetwork n(layers, learningRate);

		std::cout << "Initial weights: " << n.weights << std::endl
			<< "Initial biases: " << n.biases << std::endl << std::endl;

		float loss = 1.f;
		for (size_t i = 0; i < 1e+3 && loss > 1e-6; ++i) {
			loss = n.backward(dataset);

			//std::cout << "Current loss: " << loss << "  Current weights: " << n.weights << "  Current biases: " << n.biases << std::endl << std::endl;
		}

		std::cout << "----------------------------" << std::endl << "Final loss: " << n.totalLoss(dataset) << std::endl;

		for (size_t i = 0; i < dataset.getLengthY(); ++i)
			std::cout << dataset.at(0, i) << " - " << dataset.at(1, i) << " : " << n.forward(dataset.at(0, i)) << std::endl;

		std::cout << std::endl << "Final weights: " << n.weights << std::endl
			<< "Final biases: " << n.biases << std::endl << std::endl;
	}
};