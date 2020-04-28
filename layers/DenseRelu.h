//
// Created by Filip Ma on 27/04/2020.
//

#ifndef NEURAL_NET_DENSERELU_H
#define NEURAL_NET_DENSERELU_H

#include <vector>

class DenseRelu {
private:
    std::vector<double> input;

    std::vector<std::vector<double> > weights;
    std::vector<double> biases;

    std::vector<double> A;

    std::vector<double> delta;

    int n_prev;
    int n_cur;

public:
    DenseRelu(int n_neurons, int n_neurons_prev);
    std::vector<double>& forwardPropagation(const std::vector<double> in);
    std::vector<double>& backwardPropagation(const std::vector<double>& derivative, double learning_rate);
};


#endif //NEURAL_NET_DENSERELU_H
