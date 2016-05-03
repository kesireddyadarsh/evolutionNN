//
//  main.cpp
//  vibrations_linear_v2
//
//  Created by adarsh kesireddy on 4/26/16.
//  Copyright © 2016 adarsh kesireddy. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <time.h>
#include <stdlib.h>
#include <algorithm>
#include <fstream>

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
    //cout<<"This is output passing to node::"<<sum<<endl;
    z_outputVal = Neuron::transferFunction(sum);
}

//This is single neural network

class Net{
public:
    Net(vector<unsigned> topology);
    void feedForward(vector< long double> inputVal, vector<long double> value_vector,double max_time, float interval);
    vector<Layer> z_layer;
    double backProp();
    double z_error;
    double z_error_temp;
    vector<long double> z_error_vector;
    vector<long double> z_net_output;
    void mutate();
    double scale(double val, int max_range, int min_range);
    vector<double> outputvalues;
};

Net::Net(vector<unsigned> topology){
    
    for(int  numLayers = 0; numLayers<topology.size(); numLayers++){
        //unsigned numOutputs = numLayers == topology.size() - 1 ? 0 : topology[numLayers + 1];
        
        unsigned numOutputs;
        if (numLayers == topology.size()-1) {
            numOutputs=0;
        }else{
            numOutputs= topology[numLayers+1];
        }
        
        if(numOutputs>11){
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
    /*
     //popVector[temp].z_layer[temp][temp].z_outputWeights[temp].weight
     */
    for (int l =0 ; l < z_layer.size(); l++) {
        for (int n =0 ; n< z_layer[l].size(); n++) {
            for (int z=0 ; z< z_layer[l][n].z_outputWeights.size(); z++) {
                z_layer[l][n].z_outputWeights[z].weight += random_global(.5)-random_global(.5);
            }
        }
    }
}

double Net::scale(double val, int max_range, int min_range){
    val = val *(max_range-min_range); // val = val * (max-min);
    val = val + min_range; // val =val + (min)
    //cout<<"This is var in scale for output:: "<<val<<endl;
    return val;
}

void Net::feedForward(vector< long double> inputVal, vector<long double> value_vector,double max_time, float interval){
    long double output = 0.0;
    long double velocity_new, displacement_new;
    //Output of this loop is storing all outputs from neural network
    for (double each_net =0.0 ; each_net<=max_time; each_net=each_net+interval) {
        assert(inputVal.size() == z_layer[0].size()-1);
        for (unsigned i=0; i<inputVal.size(); ++i) {
            z_layer[0][i].setOutputVal(inputVal[i]);
        }
        for (unsigned layerNum = 1; layerNum < z_layer.size(); ++layerNum) {
            Layer &prevLayer = z_layer[layerNum - 1];
            for (unsigned n = 0; n < z_layer[layerNum].size() - 1; ++n) {
                z_layer[layerNum][n].feedForward(prevLayer);
            }
        }
        Layer &outputLayer = z_layer.back();
        for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
            output = outputLayer[n].getOutputVal();
            //cout<<output<<"\t";
            z_net_output.push_back(output);
        }
        velocity_new = (inputVal.at(0)+ (output*interval));
        displacement_new = (inputVal.at(1)+ (velocity_new*interval));
        inputVal.clear();
        inputVal.push_back(velocity_new);
        inputVal.push_back(displacement_new);
        inputVal.push_back(0);
    }
    //Calculate error using this loop
    for (int temp =0 ; temp <value_vector.size(); temp++) {
        long double temp_error = -value_vector.at(temp)+z_net_output.at(temp);
        z_error_vector.push_back(temp_error>0?temp_error:-temp_error);
    }
    
}


double Net::backProp(){
    z_error = 0.0;
    for (int temp = 0; temp< z_error_vector.size(); temp++) {
        //cout<<z_error_vector[temp]<<"\t";
        z_error += z_error_vector[temp];
    }
    //cout<<z_error_vector.size()<<endl;
    z_error_vector.clear();
    return z_error;
}



//This is for population of neural network
class population{
public:
    population(int numNN,vector<unsigned> topology);
    vector<Net> popVector;
    void runNetwork(vector<long double> inputVal, vector<long double> value_vector, int numNN, double max_time, float interval);
    void sortError();
    void mutation(int numNN);
    void newerrorvector();
    void findindex();
    int returnIndex(int numNN);
    void repop(int numNN);
    vector<double> error_vector;
    void clearNet();
    
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
    //double temp_1 =popVector[number_1].z_error;
    //double temp_2 =popVector[number_2].z_error;
    //cout<<"This is error in comparision::"<<popVector[number_1].z_error<<endl;
    //cout<<"This is error in comparision::"<<popVector[number_2].z_error<<endl;
    
    if (popVector[number_1].z_error<popVector[number_2].z_error) {
        return number_2;
    }else if (popVector[number_1].z_error>popVector[number_2].z_error){
        return number_1;
    }else{
        return NULL;
    }
}

void population::clearNet(){
    for (int temp =0 ; temp< popVector.size(); temp++) {
        popVector.at(temp).z_error=0.0; //Clears z_error
        popVector.at(temp).z_error_vector.clear(); // Clears error
        popVector.at(temp).outputvalues.clear();
        popVector.at(temp).z_net_output.clear();
    }
}

void population::repop(int numNN){
    vector<unsigned> a;
    for (int temp =0 ; temp<numNN/2; temp++) {
        int R = rand()% popVector.size();
        //popVector.push_back(popVector.at(R));
        //popVector.back().mutate();
        Net N(a);
        N=popVector.at(R);
        N.mutate();
        popVector.push_back(N);
    }
    clearNet();
}


void population::runNetwork(vector<long double> inputVal, vector<long double> value_vector, int numNN, double max_time,float interval){
    //loop runs for numNN
    for (int temp=0 ; temp< popVector.size(); temp++) {
        //Run neural network.
        popVector[temp].feedForward(inputVal, value_vector, max_time, interval);
        popVector[temp].backProp();
        error_vector.push_back(popVector[temp].z_error);
        //cout<<popVector[temp].z_error<<"\t";
    }
    sort(error_vector.begin(), error_vector.end());
    cout<<"This are errors"<<endl;
    for (int temp = 0 ; temp <error_vector.size() ; temp++) {
        cout<<error_vector.at(temp)<<"\t";
    }
    for (int check_lowest_error = 0 ; check_lowest_error<error_vector.size(); check_lowest_error++) {
        if(error_vector.at(0)==popVector[check_lowest_error].z_error){
            //cout<<"\n This is smallest error:"<<error_vector.at(0)<<endl;
            //cout<<" this"<<popVector[check_lowest_error].z_layer[2][0].z_outputVal<<endl;
            //cout<<popVector[check_lowest_error].z_error<<endl;
            error_vector.clear();
            cout<<"\n This are outputs::"<<endl;
            for (int print_out =0 ; print_out<popVector.at(check_lowest_error).z_net_output.size();print_out++ ) {
                cout<<popVector.at(check_lowest_error).z_net_output.at(print_out)<<"\t";
            }
            cout<<"\n";
            
            //for (int temp_1 =0 ; temp_1<popVector[check_lowest_error].outputvalues.size(); temp_1++) {
                //cout<<popVector.at(check_lowest_error).outputvalues.at(temp_1)<<"\t";
            //}
            //cout<<"completed"<<endl;
            /*for (int temp_2 =0; temp_2<popVector[temp_2].outputvalues.size(); temp_2++) {
                popVector.at(temp_2).outputvalues.clear();
            }*/
        }
        
    }
    for (int temp = 0 ; temp < numNN/2; temp++) {
        int temp_index = returnIndex(popVector.size());
        popVector.erase(popVector.begin()+temp_index);
    }
    
    cout<<"This are after error"<<endl;
    for (int temp = 0 ; temp <popVector.size() ; temp++) {
        cout<<popVector.at(temp).z_error<<"\t";
        popVector.at(temp).z_error=0.0;
    }
    //cout<<"This is size::"<<popVector.size()<<endl;
    repop(numNN);
}

double scale_input(double var,int max_range, int min_range){
    var = (var - min_range)/(max_range-min_range); //value - min/max-min
    return var;
}



//This is main function

int main(int argc, const char * argv[]) {
    
    srand(time(NULL));
    vector<long double> inputVal_scaled;
    vector<long double> inputVal;
    
    
    int numNN=100;
    float interval = 0.1;
    float max_time = 10.0;
    
    vector<unsigned> topology;
    topology.clear();
    inputVal.clear();
    topology.push_back(3);
    topology.push_back(4);
    topology.push_back(1);
    population mypop(numNN,topology);
    
    /*
     
     Conditions/Requirements for Vibrataions:
     
     Underdamped Vibrations
     
     
     Equation 2d(x)^2+8d(x)+16x=0;
     mass = 2;
     damper_constant = 8;
     spring_constant = 16;
     force = 0;
     displacement_0 = 1;
     velocity_0 = -1;
     reponse_t = e^-2t(cos2t+1/2*sin2t);
     */
    
     long double value;
     float PI = 3.1415;
     vector<float> time_vector;
     vector<long double> value_vector;
     
     
     for (float time=0.0 ; time<=max_time ; time=time+interval) {
         int power = -2*time;
         double exp_power = exp(power);
         float cos_value = cos((2*time)*PI/180);
         float sin_value = (1/2)*sin((2*time)*PI/180);
         value = exp_power*(cos_value+sin_value);
         value_vector.push_back(value);
         time_vector.push_back(time);
         //cout<<value<<endl;
     }
    
    //cout<<"This is test"
    
    bool test_init = true;
    
    if (test_init==true) {
        inputVal.push_back(-1); //velocity
        inputVal.push_back(1); //displacement
        inputVal.push_back(0); // force
        /*
        cout<<"This are value for total response"<<endl;
        for (int temp = 0 ; temp<value_vector.size(); temp++) {
            cout<<value_vector.at(temp)<<"\t";
        }
        */
        for (int iterations = 0 ; iterations < 1000 ; iterations++) {
            cout<<"\n";
            mypop.runNetwork(inputVal,value_vector, numNN, max_time, interval);
        }
    }
    
    
    return 0;
}