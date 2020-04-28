//
// Created by Filip Ma on 27/04/2020.
//

#ifndef NEURAL_NET_CROSSENTROPY_H
#define NEURAL_NET_CROSSENTROPY_H

#include <vector>
#include <math.h>
#include <tuple>

using matrix = std::vector<std::vector<double> >;
using vec = std::vector<double>;

namespace Loss {

    // return (loss, correct, derivation)
    auto cross_entropy_loss(vec &prediction, int golden) {

        vec derivative = prediction;
        derivative[golden] -= 1.0;

        double max = 0.0;
        int max_idx = 0;
        for(int i = 0; i < prediction.size(); i++) {
            if (prediction[i] > max) {
                max = prediction[i];
                max_idx = i;
            }
        }

        int correct = 0;
        if (max_idx == golden) correct = 1;

        return std::make_tuple(-log(prediction[golden]), correct, derivative);
    }
}


#endif //NEURAL_NET_CROSSENTROPY_H
