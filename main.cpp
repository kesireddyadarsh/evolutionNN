//
//  main.cpp
//  evv_NN_V4
//
//  Created by adarsh kesireddy on 4/11/16.
//  Copyright © 2016 adarsh kesireddy. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <time.h>
#include <stdlib.h>

using namespace std;

struct connect{
    double weight;
};

static double random_global(double a) { return a* (rand() / double(RAND_MAX)); }

// This is for each Neuron

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
    void feedForward(const Layer prevLayer);
    double transferFunction(double x);
};

//This creates connection with neurons.
Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c) {
        z_outputWeights.push_back(connect());
        z_outputWeights.back().weight = randomWeight();
    }
    z_myIndex = myIndex;
}

double Neuron::transferFunction(double x){
    return tanh(x);
}


void Neuron::feedForward(const Layer prevLayer){
    double sum = 0.0;
    bool debug_sum_flag = false;
    
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        if(debug_sum_flag == true){
            cout<<prevLayer[n].getOutputVal()<<endl;
            cout<<&prevLayer[n].z_outputWeights[z_myIndex];
            cout<<prevLayer[n].z_outputWeights[z_myIndex].weight;
        }
        sum += prevLayer[n].getOutputVal() * prevLayer[n].z_outputWeights[z_myIndex].weight;
        //cout<<"This is sum value"<<sum<<endl;
    }
    
    z_outputVal = Neuron::transferFunction(sum);
}


//This is single neural network

class Net{
public:
    Net(vector<unsigned> &topology);
    void feedForward(const vector<double> inputVals);
    vector<Layer> z_layer;
    double backProp(const vector<long double> targetVals);
    double z_error;
    void mutate();
};

Net::Net(vector<unsigned> &topology){
    
    for(int  numLayers = 0; numLayers<topology.size(); numLayers++){
        //unsigned numOutputs = numLayers == topology.size() - 1 ? 0 : topology[numLayers + 1];
        
        unsigned numOutputs;
        if (numLayers == topology.size()-1) {
            numOutputs=0;
        }else{
            numOutputs= topology[numLayers+1];
        }
        
        if(numOutputs>200){
            cout<<"Stop it number outputs coming out"<<numOutputs<<endl;
            exit(10);
        }
        z_layer.push_back(Layer());
        for(int numNeurons = 0; numNeurons <= topology[numLayers]; numNeurons++){
            //cout<<"This is neuron number:"<<numNeurons<<endl;
            z_layer.back().push_back(Neuron(numOutputs, numNeurons));
        }
        
        z_layer.back().back().setOutputVal(1.0);
        
    }
}

void Net::mutate(){
    /*//popVector[temp].z_layer[temp][temp].z_outputWeights[temp].weight
     */
    for (int l =0 ; l < z_layer.size(); l++) {
        for (int n =0 ; n< z_layer[l].size(); n++) {
            for (int z=0 ; z< z_layer[l][n].z_outputWeights.size(); z++) {
                z_layer[l][n].z_outputWeights[z].weight += random_global(.1)-random_global(.1);
            }
        }
    }
}

void Net::feedForward(const vector<double> inputVals){
    
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

double Net::backProp(const vector<long double> targetVals){
    // Calculate overall net error (RMS of output neuron errors)
    
    Layer &outputLayer = z_layer.back();
    z_error = 0.0;
    
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        z_error += delta * delta;
    }
    z_error /= outputLayer.size() - 1; // get average error squared
    z_error = sqrt(z_error)*100; // RMS
    return z_error;
}


//This is for population of neural network
class population{
public:
    population(int numNN,vector<unsigned> topology);
    vector<Net> popVector;
    void runNetwork(vector<double> &inputVal, vector<long double> &targetVal,int numNN);
    void sortError();
    void mutation(int numNN);
    void newerrorvector();
    void findindex();
    int returnIndex(int numNN);
    void repop(int numNN);
    
};

// variables used: indiNet -- object to Net
population::population(int numNN,vector<unsigned> topology){
    
    for (int populationNum = 0 ; populationNum<numNN; populationNum++) {
        //cout<<"This is neural network:"<<populationNum<<endl;
        Net indiNet(topology);
        popVector.push_back(indiNet);
    }
    
}

//Return index of higher

int population::returnIndex(int numNN){
    int temp = numNN;
    int number_1 = (rand() % temp);
    int number_2 = (rand() % temp);
    while (number_1 == number_2) {
        number_2 = (rand() % temp);
    }
    
    if (popVector[number_1].z_error<popVector[number_2].z_error) {
        return number_2;
    }else if (popVector[number_1].z_error>popVector[number_2].z_error){
        return number_1;
    }else{
        return NULL;
    }
}
void population::repop(int numNN){
    for (int temp =0 ; temp<numNN/2; temp++) {
        int R = rand()% popVector.size();
        popVector.push_back(popVector.at(R));
        popVector.back().mutate();
    }
}



void population::runNetwork(vector<double> &inputVal, vector<long double> &targetVal,int numNN){
    
    bool runNetwork_flag = false;   // flag for print
    
    for (int temp=0 ; temp< numNN; temp++) {
        //Run neural network.
        popVector[temp].feedForward(inputVal);
        popVector[temp].backProp(targetVal);
        cout<<popVector[temp].z_error<<endl;
    }
    
    for (int temp = 0 ; temp < numNN/2; temp++) {
        int temp_index = returnIndex(popVector.size());
        popVector.erase(popVector.begin()+temp_index);
    }
    cout<<"This is size::"<<popVector.size()<<endl;
    repop(numNN);
}




//This is main function

int main(int argc, const char * argv[]) {
    // insert code here...
    //cout << "Hello, World!\n";
    srand(time(NULL));
    vector<double> inputVal;
    vector<double> outputVal;
    vector<double> resultVal;
    vector<long double> targetVal;
    
    int numNN=10;
    vector<unsigned> topology;
    topology.clear();
    topology.push_back(6);
    topology.push_back(12);
    topology.push_back(1);
    population mypop(numNN,topology);
    
    bool z_debugger_flag = false;
    
    /*
     mass = 2;
     damper_constant = 8;
     spring_constant = 16;
     force = 0;
     displacement_0 = 1;
     velocity_0 = -1;
     reponse_t = e^-2t(cos2t+1/2*sin2t);
     */
    
    inputVal.push_back(2);
    inputVal.push_back(8);
    inputVal.push_back(16);
    inputVal.push_back(0);
    inputVal.push_back(1);
    inputVal.push_back(-1);
    //targetVal.push_back(0.135253);
    
    int time = 0;
    long double value;
    float PI = 3.1415;
    vector<int> time_vector;
    vector<long double> value_vector;
    
    
    for (time ; time<10 ; time++) {
        int power = -2*time;
        double exp_power = exp(power);
        float cos_value = cos((2*time)*PI/180);
        float sin_value = (1/2)*sin((2*time)*PI/180);
        value = exp_power*(cos_value+sin_value);
        value_vector.push_back(value);
        time_vector.push_back(time);
        cout<<"This is value::"<<value<<"at time::"<<time<<endl;
    }
    
    for (int temp_targetval =1 ; temp_targetval<time; temp_targetval++) {
        targetVal.clear();
        targetVal.push_back(value_vector[temp_targetval]);
        cout<<"This is targetvalue::"<<targetVal[temp_targetval]<<endl;
        for (int temp =0 ; temp<200; temp++) {
            mypop.runNetwork(inputVal,targetVal, numNN);
            cout<<"This is end of run:::"<<temp<<"\n"<<endl;
        }
        cout<< "This is end of network\n\n\n"<<endl;
    }
    
    
    //
    //for (int i=0; i<=numNN; i++) {
    
    //}
    
    return 0;
}