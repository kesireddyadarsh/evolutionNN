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
#include <time.h>
#include <stdlib.h>

using namespace std;

struct connect{
    double weight;
};


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
Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        z_outputWeights.push_back(connect());
        z_outputWeights.back().weight = randomWeight();
    }
    
    z_myIndex = myIndex;
}

double Neuron::transferFunction(double x)
{
    return tanh(x);
}


void Neuron::feedForward(const Layer prevLayer)
{
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
    void feedForward(const vector<double> &inputVals);
    vector<Layer> z_layer;
    double backProp(const vector<double> &targetVals);
    double z_error;
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
        
        if(numOutputs>5){
            cout<<"Stop it number outputs coming out"<<numOutputs<<endl;
            exit(10);
        }
        z_layer.push_back(Layer());
        for(int numNeurons = 0; numNeurons <= topology[numLayers]; numNeurons++){
            //cout<<"This is neuron number:"<<numNeurons<<endl;
            z_layer.back().push_back(Neuron(numOutputs, numNeurons));
        }
    }
}

void Net::feedForward(const vector<double> &inputVals){
    //assert(inputVals.size() == z_layer[0].size()-1);
    
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

double Net::backProp(const vector<double> &targetVals){
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

//Error showed be class (use construct)

//This is for population of neural network


class population{
public:
    population(int numNN,vector<unsigned> &topology);
    vector<Net> popVector;
    void runNetwork(vector<double> &inputVal, vector<double> &targetVal,int numNN);
    vector<double> error;
    vector<vector<double>> overallError;
    vector<double> newerror;
    void sortError();
    void mutation(int numNN);
    void newerrorvector();
    void findindex();
    vector<int> errorindex;
    vector<int> badindex;
    vector<double> temp_error;
};

// variables used: indiNet -- object to Net
population::population(int numNN,vector<unsigned> &topology){
    
    for (int populationNum = 0 ; populationNum<numNN; populationNum++) {
        //cout<<"This is neural network:"<<populationNum<<endl;
        Net indiNet(topology);  
        popVector.push_back(indiNet);
    }
    
}

/*
    Find lost error and push_back into newerror vector
*/

void population::newerrorvector(){
    
    bool newpopulation_print_flag = true;
    int size = error.size()/4;
    for (int temp =0 ; temp< size; temp++) {
        newerror.push_back(error[temp]);
        error.erase(error.begin());
    }
    if(newpopulation_print_flag== true){
        cout<< "This is error size after removing::"<<error.size()<<endl;
        /*for (int temp =0 ; temp<error.size(); temp++) {
            cout<<error[temp]<<endl;
        }*/
    }
    int looprotate = error.size()/3;
    for (int temp = 0; temp<looprotate; temp++ ) {
        int temp_1 = error.size();
        int number_1 = (rand() % temp_1)+1;
        int number_2 = (rand() % temp_1)+1;
        if(error[number_1]<error[number_2]){
            newerror.push_back(error[number_1]);
            error.erase(error.begin()+(number_1-1));
        }else if (error[number_1]>error[number_2]){
            newerror.push_back(error[number_2]);
            error.erase(error.begin()+(number_2-1));
        }else{
            
        }
    }
    if (newpopulation_print_flag == true) {
        cout<<"This is new error vector::"<<newerror.size()<<endl;
        /*for (int temp =0 ; temp<newerror.size(); temp++) {
            cout<<newerror[temp]<<endl;
        }*/
    }
    
}

/*
 This function is used to find index of nerual network which has small error
*/
void population::findindex(){
    for (int temp = 0 ; temp<temp_error.size(); temp++) {
        for (int temp_1 =0 ; temp_1<newerror.size(); temp_1++) {
            if (temp_error[temp] == newerror[temp_1]) {
                errorindex.push_back(temp);
            }
        }
    }
    cout<<"This is size of temp_error::::"<<errorindex.size()<<endl;
    for (int temp = 0 ; temp < temp_error.size(); temp++) {
        if (temp != newerror[temp]) {
            badindex.push_back(temp);
        }
    }
    /*cout<<"This is the index"<<endl;
    for (int temp =0 ; temp<errorindex.size(); temp++) {
        cout<<errorindex[temp]<<endl;
    }*/
}

//errorindex has good neural network numbers which should be mutated
//badindex holds neural networks which should be removed.
void population::mutation(int numNN){
    
    for (int temp =0; temp<numNN; temp++) {
        //create weights for bad index in popvector
    }
    for (int temp =0; temp<badindex.size(); temp++) {
        //random generate weights for popvector
        //Find the mean of weights for good vector and use those weights to generate randomnumber
    }
    for (int temp = 0 ; temp < numNN; temp++) {
        
        cout<<"This is value looking for"<<popVector[temp].z_layer[temp][temp].z_outputWeights[temp].weight<<endl;
    }
}

void population::runNetwork(vector<double> &inputVal, vector<double> &targetVal,int numNN){
    
    bool runNetwork_flag = false;   // flag for print
    
    for (int temp=0 ; temp< numNN; temp++) {
        //Run neural network.
         popVector[temp].feedForward(inputVal);
        double temp_1 = popVector[temp].backProp(targetVal);
        error.push_back(temp_1);
        temp_error.push_back(temp_1);
    }
    
    //Print after errors
    if (runNetwork_flag == true) {
        cout<<"This are first errors"<<endl;
        for (int temp =0 ; temp<error.size(); temp++) {
            cout<<error[temp]<<endl;
        }
        cout<<"This are temp errors"<<endl;
        for (int temp =0 ; temp<temp_error.size(); temp++) {
            cout<<error[temp]<<endl;
        }
    }
    
    sort(error.begin(), error.end());
    
    //Print after sorting
    if(runNetwork_flag==true){
        cout<<"This are sort error"<<endl;
        for (int temp =0 ; temp<error.size(); temp++) {
            cout<<error[temp]<<endl;
        }
    }
    
    overallError.push_back(error); // This is for development purpose no actual use so far
    
    newerrorvector(); //Only required neural network errors are used
    
    findindex(); //find index of requried neural network
    
    mutation(numNN);

    error.clear(); // clears error vector
    cout<<"This is total error:"<<overallError.size()<<endl;
}




//This is main function

int main(int argc, const char * argv[]) {
    // insert code here...
    //cout << "Hello, World!\n";
    srand(time(NULL));
    vector<double> inputVal;
    vector<double> outputVal;
    vector<double> resultVal;
    vector<double> targetVal;
    
    int numNN=100;
    vector<unsigned> topology;
    topology.clear();
    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(1);
    population mypop(numNN,topology);
    
    bool z_debugger_flag = false;
    
    if(z_debugger_flag == true){
    for (int i=0; i<100; i++) {
        int number = (rand() % 4)+1;
        switch (number) {
            case 1:
                inputVal.push_back(1.0);
                inputVal.push_back(1.0);
                targetVal.push_back(1.0);
                break;
            
            case 2:
                inputVal.push_back(1.0);
                inputVal.push_back(0.0);
                targetVal.push_back(1.0);
                break;
            
            case 3:
                inputVal.push_back(0.0);
                inputVal.push_back(1.0);
                targetVal.push_back(1.0);
                break;
            
            case 4:
                inputVal.push_back(0.0);
                inputVal.push_back(0.0);
                targetVal.push_back(0.0);
                break;
            
            default:
                inputVal.push_back(0.0);
                inputVal.push_back(0.0);
                targetVal.push_back(0.0);
                break;
        }
        
        mypop.runNetwork(inputVal, targetVal, numNN);
    }
    }else{
        inputVal.push_back(1.0);
        inputVal.push_back(0.0);
        targetVal.push_back(1.0);
        mypop.runNetwork(inputVal, targetVal, numNN);
    }
    
    //
    //for (int i=0; i<=numNN; i++) {
        
    //}
    
    return 0;
}
