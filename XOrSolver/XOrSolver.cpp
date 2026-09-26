#include <iostream>
#include "Vector.h"
#include "NeuralNetwork.h"

int main()
{
	//srand(static_cast<unsigned int>(time(0)));

	Vector<size_t> layers = Vector<size_t>(3)
		.put(0, 2)
		.put(1, 2)
		.put(2, 1);

	NeuralNetwork nn = NeuralNetwork(layers);

	Vector<float> input = Vector<float>(layers.at(0))
		.put(0, 1.f)
		.put(1, 2.f);

	Vector<float> result = nn.calculate(input);

	return 0;
}