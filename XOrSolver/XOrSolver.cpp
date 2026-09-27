#include <iostream>
#include "SimpleNeuron.h"
#include "NeuralNetwork.h"

Matrix<float> twiceStates = Matrix<float>(2, 5)
.put(0, 0, 0).put(1, 0, 0)
.put(0, 1, 1).put(1, 1, 2)
.put(0, 2, 2).put(1, 2, 4)
.put(0, 3, 3).put(1, 3, 6)
.put(0, 4, 4).put(1, 4, 8);

Matrix<float> twicePlusOneStates = Matrix<float>(2, 5)
.put(0, 0, 0).put(1, 0, 1)
.put(0, 1, 1).put(1, 1, 3)
.put(0, 2, 2).put(1, 2, 5)
.put(0, 3, 3).put(1, 3, 7)
.put(0, 4, 4).put(1, 4, 9);

Matrix<float> orStates = Matrix<float>(3, 4)
.put(0, 0, 0).put(1, 0, 0).put(2, 0, 0)
.put(0, 1, 1).put(1, 1, 0).put(2, 1, 1)
.put(0, 2, 0).put(1, 2, 1).put(2, 2, 1)
.put(0, 3, 1).put(1, 3, 1).put(2, 3, 1);

Matrix<float> andStates = Matrix<float>(3, 4)
.put(0, 0, 0).put(1, 0, 0).put(2, 0, 0)
.put(0, 1, 1).put(1, 1, 0).put(2, 1, 0)
.put(0, 2, 0).put(1, 2, 1).put(2, 2, 0)
.put(0, 3, 1).put(1, 3, 1).put(2, 3, 1);

Matrix<float> norStates = Matrix<float>(3, 4)
.put(0, 0, 0).put(1, 0, 0).put(2, 0, 1)
.put(0, 1, 1).put(1, 1, 0).put(2, 1, 0)
.put(0, 2, 0).put(1, 2, 1).put(2, 2, 0)
.put(0, 3, 1).put(1, 3, 1).put(2, 3, 0);

Matrix<float> nandStates = Matrix<float>(3, 4)
.put(0, 0, 0).put(1, 0, 0).put(2, 0, 1)
.put(0, 1, 1).put(1, 1, 0).put(2, 1, 1)
.put(0, 2, 0).put(1, 2, 1).put(2, 2, 1)
.put(0, 3, 1).put(1, 3, 1).put(2, 3, 0);

Matrix<float> xorStates = Matrix<float>(3, 4)
.put(0, 0, 0).put(1, 0, 0).put(2, 0, 0)
.put(0, 1, 1).put(1, 1, 0).put(2, 1, 1)
.put(0, 2, 0).put(1, 2, 1).put(2, 2, 1)
.put(0, 3, 1).put(1, 3, 1).put(2, 3, 0);

int main()
{
	/*srand(static_cast<unsigned int>(time(0)));

	Vector<int> layers = Vector<int>(3)
		.put(0, 2)
		.put(1, 3)
		.put(2, 2);

	std::cout << layers << std::endl;*/

	runNeuron(twicePlusOneStates);

	return 0;
}