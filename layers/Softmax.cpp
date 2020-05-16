//
// Created by Filip Ma on 27/04/2020.
//

#include "Softmax.h"

using matrix = std::vector<std::vector<double> >;
using vec = std::vector<double>;

Softmax::Softmax(int n_neurons, int n_neurons_prev) {
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

vec& Softmax::forwardPropagation(const vec in) {
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

    // apply softmax
    double sum = 0;
    for (int i = 0; i < n_cur; ++i)
        sum += exp(A[i]);
    for (int i = 0; i < n_cur; ++i)
        A[i] = exp(A[i]) / sum;

    return A;
}

vec& Softmax::backwardPropagation(const vec& derivative, double learning_rate) {
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

int Softmax::size() {
    return n_cur;
}

int Softmax::size_prev() {
    return n_prev;
}

std::vector<std::vector<double> >& Softmax::get_weights() {
    return weights;
}

std::vector<double>& Softmax::get_biases() {
    return biases;
}
