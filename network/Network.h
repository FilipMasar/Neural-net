//
// Created by Filip Ma on 15/05/2020.
//

#ifndef NEURAL_NET_NETWORK_H
#define NEURAL_NET_NETWORK_H

#include <iostream>
#include <fstream>
#include <vector>

#include "../layers/DenseRelu.h"
#include "../layers/Softmax.h"


class Network {
private:
    std::vector<DenseRelu> denseLayers;
    Softmax finalLayer;
    int input_dim;

    auto cross_entropy_loss(std::vector<double> &prediction, int golden);

public:
    Network(int input_dim) : input_dim(input_dim) {}

    void addDenseLayer(int n_neurons);
    void addFinalLayer(int n_neurons);

    void fit(int epochs, double learning_rate,
            std::vector<std::vector<double> > &train_data, std::vector<int> &train_labels,
            std::vector<std::vector<double> > &dev_data, std::vector<int> &dev_labels);
    int predict(std::vector<double> &data);
    std::pair<double,double> predict(std::vector<std::vector<double> > &data, std::vector<int> &labels);

    void save(std::string file_name);
    void load(std::string file_name);
};


#endif //NEURAL_NET_NETWORK_H
