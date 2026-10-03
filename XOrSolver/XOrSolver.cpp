#include <iostream>
#include "SimpleNeuron.h"
#include "ComplexNeuron.h"
#include "SimpleBackpropagatingNeuron.h"
#include "ComplexBackpropagatingNeuron.h"
#include "NeuralNetwork.h"

Matrix<float> twiceStates = Matrix<float>(2, 5).make(
	0, 0,
	1, 2,
	2, 4,
	3, 6,
	4, 8
);

Matrix<float> twicePlusOneStates = Matrix<float>(2, 5).make(
	0, 1,
	1, 3,
	2, 5,
	3, 7,
	4, 9
);

Matrix<float> orStates = Matrix<float>(3, 4).make(
	0, 0, 0,
	1, 0, 1,
	0, 1, 1,
	1, 1, 1
);

Matrix<float> andStates = Matrix<float>(3, 4).make(
	0, 0, 0,
	1, 0, 0,
	0, 1, 0,
	1, 1, 1
);

Matrix<float> norStates = Matrix<float>(3, 4).make(
	0, 0, 1,
	1, 0, 0,
	0, 1, 0,
	1, 1, 0
);

Matrix<float> nandStates = Matrix<float>(3, 4).make(
	0, 0, 1,
	1, 0, 1,
	0, 1, 1,
	1, 1, 0
);

Matrix<float> xorStates = Matrix<float>(3, 4).make(
	0, 0, 0,
	1, 0, 1,
	0, 1, 1,
	1, 1, 0
);

int main()
{
	srand(static_cast<unsigned int>(time(0)));

	// SimpleNeuron::run(twicePlusOneStates);
	// ComplexNeuron::run(orStates);
	// SimpleBackpropagatingNeuron::run(twiceStates);
	// ComplexBackpropagatingNeuron::run(orStates);

	{
		Matrix<float> dataset = Matrix<float>(xorStates);
		Matrix<Vector<float>> datasetMV = Matrix<Vector<float>>(2, dataset.getLengthY());
		for (size_t y = 0; y < dataset.getLengthY(); ++y) {
			datasetMV.put(0, y, Vector<float>(2).put(0, dataset.at(0, y)).put(1, dataset.at(1, y)));
			datasetMV.put(1, y, Vector<float>(1).put(0, dataset.at(2, y)));
		}

		NeuralNetwork::run(datasetMV);
	}

	return 0;
}