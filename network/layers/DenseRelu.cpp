//
// Created by Filip Ma on 27/04/2020.
//

#include "DenseRelu.h"

using matrix = std::vector<std::vector<double> >;
using vec = std::vector<double>;

DenseRelu::DenseRelu(int n_neurons, int n_neurons_prev) {
    n_prev = n_neurons_prev;
    n_cur = n_neurons;

    input = {};
    weights.resize(n_cur, std::vector<double>(n_prev));
    biases.resize(n_cur);
    A.resize(n_cur);
    delta.resize(n_prev);

    // randomly initialize
    for(auto & a : weights) {
        for(auto & b : a) {
            b = ((double) rand() / RAND_MAX - 0.5);
        }
    }
    for(auto & a : biases) {
        a = ((double) rand() / RAND_MAX - 0.5);
    }
}

vec& DenseRelu::forwardPropagation(const vec in) {
    input = in;

    // weights @ input + biases
    for (int i = 0; i < n_cur; ++i) {
        double tmp = 0;
        for (int j = 0; j < n_prev; ++j) {
            tmp += weights[i][j] * in[j];
        }
        tmp += biases[i];
        A[i] = tmp;
    }

    // apply relu
    for (int i = 0; i < n_cur; ++i) {
        if (A[i] < 0) A[i] = 0;
    }

    return A;
}

vec& DenseRelu::backwardPropagation(const vec& derivative, double learning_rate) {
    // return next derivative
    for(int i = 0; i < n_prev; ++i) {
        double tmp = 0;
        for(int j = 0; j < n_cur; ++j) {
            double tmp2 = weights[j][i]*derivative[j];
            if(input[i] > 0) {
                tmp += weights[j][i]*derivative[j];
            } else if(input[i] == 0) {
                tmp += 0;
            } else {
                tmp += -weights[j][i]*derivative[j];
            }
        }
        delta[i] = tmp;
    }
    // update weights
    for(int i = 0; i < n_cur; ++i) {
        for(int j = 0; j < n_prev; ++j) {
            weights[i][j] -= learning_rate * derivative[i]* input[j];
        }
    }
    // update biases
    for(int i = 0; i < n_cur; ++i) {
        biases[i] -=  learning_rate * derivative[i];
    }
    return delta;
}

int DenseRelu::size() {
    return n_cur;
}

int DenseRelu::size_prev() {
    return n_prev;
}

std::vector<std::vector<double> >& DenseRelu::get_weights() {
    return weights;
}

std::vector<double>& DenseRelu::get_biases() {
    return biases;
}

