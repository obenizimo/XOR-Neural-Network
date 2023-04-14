using System;
using System.Collections.Generic;

namespace xor_neural_network
{
    internal class Program
    {
        public class Neuron
        {
            public List<double> Weights { get; set; }
            public double Bias { get; set; }
            public double Delta { get; set; }

            // Constructor
            public Neuron(int inputSize)
            {
                Weights = new List<double>(inputSize);

                // Initialize weights and bias with random values between -1 and 1
                Random rand = new Random();
                for (int i = 0; i < inputSize; ++i)
                {
                    Weights.Add((rand.NextDouble() * 2) - 1);
                }
                Bias = (rand.NextDouble() * 2) - 1;
            }

            // Computes the output of the neuron for given inputs
            public double ComputeOutput(List<double> inputs)
            {
                double weightedSum = Bias;
                for (int i = 0; i < inputs.Count; ++i)
                {
                    weightedSum += inputs[i] * Weights[i];
                }
                return Sigmoid(weightedSum);
            }

            // Sigmoid activation function
            private double Sigmoid(double x)
            {
                return 1 / (1 + Math.Exp(-x));
            }

            // Derivative of the sigmoid activation function
            public double SigmoidDerivative(double x)
            {
                double sigmoid = Sigmoid(x);
                return sigmoid * (1 - sigmoid);
            }
        }

        public class NeuralNetwork
        {
            private Neuron[] inputLayer;
            private Neuron[] hiddenLayer;
            private Neuron outputNeuron;
            private double learningRate;

            // Constructor
            public NeuralNetwork(double learningRate = 0.1)
            {
                this.learningRate = learningRate;

                // Create input layer with two neurons
                inputLayer = new Neuron[2];
                for (int i = 0; i < inputLayer.Length; i++)
                {
                    inputLayer[i] = new Neuron(2);
                }

                // Create hidden layer with eight neurons
                hiddenLayer = new Neuron[8];
                for (int i = 0; i < hiddenLayer.Length; i++)
                {
                    hiddenLayer[i] = new Neuron(inputLayer.Length);
                }

                // Create output neuron
                outputNeuron = new Neuron(hiddenLayer.Length);
            }

            // Computes the output of the neural network for given inputs
            public double ComputeOutput(double x, double y)
            {
                List<double> inputs = new List<double> { x, y };
                List<double> hiddenLayerOutputs = new List<double>();
                for (int i = 0; i < hiddenLayer.Length; i++)
                {
                    hiddenLayerOutputs.Add(hiddenLayer[i].ComputeOutput(inputs));
                }
                return outputNeuron.ComputeOutput(hiddenLayerOutputs);
            }

            // Trains the neural network using backpropagation
            public void Train(double x, double y, double targetOutput)
            {
                // Compute the output and store input and hidden layer outputs
                List<double> inputs = new List<double> { x, y };
                List<double> hiddenLayerOutputs = new List<double>();
                for (int i = 0; i < hiddenLayer.Length; i++)
                {
                    hiddenLayerOutputs.Add(hiddenLayer[i].ComputeOutput(inputs));
                }
                double output = outputNeuron.ComputeOutput(hiddenLayerOutputs);

                // Compute the output error
                double outputError = targetOutput - output;

                // Update the output neuron
                Backpropagation(inputs, hiddenLayerOutputs, outputError);
            }

            // Updates the weights and biases of the neurons using backpropagation
            private void Backpropagation(List<double> inputs, List<double> hiddenLayerOutputs, double outputError)
            {
                // Compute the output delta
                outputNeuron.Delta = outputError * outputNeuron.SigmoidDerivative(outputNeuron.ComputeOutput(hiddenLayerOutputs));

                // Update the output neuron weights
                for (int i = 0; i < outputNeuron.Weights.Count; i++)
                {
                    outputNeuron.Weights[i] += learningRate * outputNeuron.Delta * hiddenLayerOutputs[i];
                }
                outputNeuron.Bias += learningRate * outputNeuron.Delta;

                // Compute the hidden layer deltas
                for (int i = 0; i < hiddenLayer.Length; i++)
                {
                    hiddenLayer[i].Delta = outputNeuron.Delta * outputNeuron.Weights[i] * hiddenLayer[i].SigmoidDerivative(hiddenLayer[i].ComputeOutput(inputs));
                }

                // Update the hidden layer neuron weights
                for (int i = 0; i < hiddenLayer.Length; i++)
                {
                    for (int j = 0; j < hiddenLayer[i].Weights.Count; j++)
                    {
                        hiddenLayer[i].Weights[j] += learningRate * hiddenLayer[i].Delta * inputs[j];
                    }
                    hiddenLayer[i].Bias += learningRate * hiddenLayer[i].Delta;
                }
            }
        }


        static void Main(string[] args)
        {
            NeuralNetwork network = new NeuralNetwork(0.1);

            // XOR training data
            double[][] inputs =
            {
                new[] {0.0, 0.0},
                new[] {0.0, 1.0},
                new[] {1.0, 0.0},
                new[] {1.0, 1.0}
            };

            double[] outputs = { 0, 1, 1, 0 };

            // Train the neural network
            int epochs = 50000;
            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < inputs.Length; j++)
                {
                    network.Train(inputs[j][0], inputs[j][1], outputs[j]);
                }
            }

            // Test the neural network
            Console.WriteLine("XOR Problem");
            Console.WriteLine("-----------");
            Console.WriteLine("0 XOR 0 = " + Math.Round(network.ComputeOutput(0, 0)));
            Console.WriteLine("0 XOR 1 = " + Math.Round(network.ComputeOutput(0, 1)));
            Console.WriteLine("1 XOR 0 = " + Math.Round(network.ComputeOutput(1, 0)));
            Console.WriteLine("1 XOR 1 = " + Math.Round(network.ComputeOutput(1, 1)));
            Console.ReadLine();
        }
    }
}

            // Trains the neural network using back

