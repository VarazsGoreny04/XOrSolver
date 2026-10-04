XOrSolver
=========

This project is about learning the basics of neural networks. And what better way exists than to do the learning in steps? Each getting increasingly harder as we go into the details, logic and mathematics of building a scalable system with proper backpropagation and training.

Steps:
------
- **SimpleNeuron:**
  - Simulates one neuron with one input and one output. That's is!
  - Uses a finite derivative algorithm to calculate the weight and bias changes necessary for learning.
  - Able to learn how to ***multiply by 2***... very impressive. :)
- **ComplexNeuron:**
  - Simulates one neuron, but with two inputs and one output.
  - Uses the same finite derivative algorithm to calculate the weight and bias changes.
  - Capable of modelling certain logic gates, like: ***OR, AND, NOR, NAND***
- **SimpleBackpropagatingNeuron:**
  - As the name shows, this class is the same **SimpleNeuron** as earlier, but with backpropagation.
- **ComplexBackpropagatingNeuron:**
  - The same is true for this class. It is the same as **ComplexNeuron**, but with backpropagation.
- **NeuralNetwork:**
  - This is the final step of the journey.
  - This class makes it possible, to build a custom size neural network with just one array of numbers. (Each number represents the number of neuron in a layer.)
  - The weights and biases are stored in vectors and matrices. Both of which I implemented, to have all the tools needed to perform the forward-pass and backpropagation operations.
  - A network of 3 neurons (if we are not counting the input layer), 2 in the first and 1 in the second layer, can already simulate an ***XOR*** logic gate.

Conclusion:
-----------
The **NeuralNetwork** class is capable of solving much harder tasks than just a simple ***XOR*** operation, but it is out of the scope of this project. Of course it is not gonna be the fastest or easiest to use library on the internet, but it's a very strong first try in my opinion.
