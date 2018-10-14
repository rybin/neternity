#include <cnpy.h>
#include <complex>
#include "npy.h"

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
using namespace std;

double activation(double val){
    return 1 / (1 + exp(0 - val));
}


double activationDer(double val) {
    return activation(val) * (1 - activation(val));
}

vector<double> layerF(Npy<double> input, Npy<double> layer1, Npy<double> bias1) {
    std::vector<double> v;
    v.resize(80);
    for (int i = 0; i < layer1.shape[0]; ++i)
    {
        double sum = 0;
        for (int j = 0; j < layer1.shape[1]; ++j)
        {
            for (int k = 0; k < layer1.shape[2]; ++k)
            {
                sum += layer1.get(i, j, k) * input.get(j, k);
            }
        }
        sum += bias1.get(i);
        v.at(i) = activation(sum);
    }
    return v;
}

vector<double> layerS(vector<double> input, Npy<double> layer2, Npy<double> bias2) {
    std::vector<double> v;
    v.resize(52);
    for (int i = 0; i < layer2.shape[0]; ++i)
    {
        double sum = 0;
        for (int j = 0; j < layer2.shape[1]; ++j)
        {
            sum += layer2.get(i, j) * input.at(j);
        }
        sum += bias2.get(i);
        v[i] = activation(sum);
    }
    return v;
}


int main(int argc, char const *argv[])
{
    cnpy::NpyArray layer1_c =  cnpy::npy_load("f/layer1.npy");
    cnpy::NpyArray layer2_c =  cnpy::npy_load("f/layer2.npy");
    cnpy::NpyArray bias1_c =  cnpy::npy_load("f/bias1.npy");
    cnpy::NpyArray bias2_c =  cnpy::npy_load("f/bias2.npy");
    cnpy::NpyArray input_c =  cnpy::npy_load("c.npy");
    
    Npy<double> layer1 (layer1_c);
    Npy<double> layer2 (layer2_c);
    Npy<double> bias1 (bias1_c);
    Npy<double> bias2 (bias2_c);
    Npy<double> input (input_c);
    
    vector<double> mid = layerF(input, layer1, bias1);
    // cout << "LayerF" << endl;
    vector<double> end = layerS(mid, layer2, bias2);
    // cout << "layerS" << endl;

    // for (int i = 0; i < end.size(); ++i)
    // {
    //     cout << end.at(i) << " " << endl;
    // }
    cout << "Extra: " << end.at(26) << endl;
}