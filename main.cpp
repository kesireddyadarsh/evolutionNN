//
//  main.cpp
//  evNN_V2
//
//  Created by adarsh kesireddy on 3/31/16.
//  Copyright © 2016 adarsh kesireddy. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cmath> 
#include <cstdlib>
#include <cassert>

using namespace std;

struct connect{
    double weight;
    double changeWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    vector<connect> z_outputWeights;
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    unsigned z_myIndex;
    double z_outputVal;
    void setOutputVal(double val) { z_outputVal = val; }
    double getOutputVal(void) const { return z_outputVal; }
    void feedForward(const Layer &prevLayer);
    double transferFunction(double x);
};

//This creates connection with neurons.
Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        z_outputWeights.push_back(connect());
        z_outputWeights.back().weight = randomWeight();
    }
    
    z_myIndex = myIndex;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    
    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.
    
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
        prevLayer[n].z_outputWeights[z_myIndex].weight;
    }
    
    z_outputVal = Neuron::transferFunction(sum);
}

double Neuron::transferFunction(double x)
{
    return tanh(x);
}



class Net{
public:
    Net(vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    vector<Layer> z_layer;
};

Net::Net(vector<unsigned> &topology){
    
    for(int  numLayers = 1; numLayers<=topology.size(); numLayers++){
        unsigned numOutputs = numLayers == topology.size() - 1 ? 0 : topology[numLayers + 1];
        z_layer.push_back(Layer());
        for(int numNeurons = 0; numNeurons<=topology[numLayers-1]; numNeurons++){
            z_layer.back().push_back(Neuron(numOutputs, numNeurons));
        }
    }
    
}

void Net::feedForward(const vector<double> &inputVals){
    assert(inputVals.size() == z_layer[0].size()-1);
    
    for (unsigned i=0; i<inputVals.size(); ++i) {
        z_layer[0][i].setOutputVal(inputVals[i]);
    }
    for (unsigned layerNum = 1; layerNum < z_layer.size(); ++layerNum) {
        Layer &prevLayer = z_layer[layerNum - 1];
        for (unsigned n = 0; n < z_layer[layerNum].size() - 1; ++n) {
            z_layer[layerNum][n].feedForward(prevLayer);
        }
    }

}

class population{
public:
    population(int numNN,vector<unsigned> &topology);
    vector<Net> popVector;
};

// variables used: indiNet -- object to Net
population::population(int numNN,vector<unsigned> &topology){
    
    for (int populationNum = 1 ; populationNum<=numNN; populationNum++) {
        Net indiNet(topology);  
        popVector.push_back(indiNet);
        cout<<"\n This is new network\n"<<endl;
    }
    cout<<"\nThis is population::::"<<endl;
    
}


int main(int argc, const char * argv[]) {
    // insert code here...
    cout << "Hello, World!\n";
    int numNN=100;
    vector<unsigned> topology;
    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(1);
    population mypop(numNN,topology);
    return 0;
}
