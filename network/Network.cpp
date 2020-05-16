//
// Created by Filip Ma on 15/05/2020.
//

#include "Network.h"
#include "../layers/DenseRelu.h"
#include "../layers/Softmax.h"


void Network::addDenseLayer(int n_neurons) {
    int n_neurons_prev;
    if(denseLayers.size() == 0) {
        n_neurons_prev = input_dim;
    } else {
        n_neurons_prev = denseLayers[denseLayers.size()-1].size();
    }
    denseLayers.push_back(DenseRelu(n_neurons, n_neurons_prev));
}

void Network::addFinalLayer(int n_neurons) {
    int n_neurons_prev = denseLayers[denseLayers.size()-1].size();
    finalLayer = Softmax(n_neurons, n_neurons_prev);
}

// return (loss, correct, derivation)
auto Network::cross_entropy_loss(std::vector<double> &prediction, int golden) {

    std::vector<double> derivative = prediction;
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

void Network::fit(int epochs, double learning_rate,
        std::vector<std::vector<double> > &train_data, std::vector<int> &train_labels,
        std::vector<std::vector<double> > &dev_data, std::vector<int> &dev_labels) {

    std::cout << "Training ...\n";
    for (int i = 0; i < epochs; ++i) {
        std::cout << "Epoch " << i + 1 << "/" << epochs << std::endl;
        double totalLoss = 0;
        int totalCorrect = 0;

        for (int j = 0; j < train_data.size(); ++j) {
            if (j % 100 == 99) std::cout << "\r" << j + 1 << "/" << train_data.size() << std::flush;
            // forward
            auto hidden = train_data[j];
            for(int l = 0; l < denseLayers.size(); ++l) {
                hidden = denseLayers[l].forwardPropagation(hidden);
            }
            auto predictions = finalLayer.forwardPropagation(hidden);

            auto tuple = cross_entropy_loss(predictions, train_labels[j]);
            totalLoss += std::get<0>(tuple);
            totalCorrect += std::get<1>(tuple);

            // backward
            auto tmp = std::get<2>(tuple);
            tmp = finalLayer.backwardPropagation(tmp, learning_rate);
            for(int l = denseLayers.size()-1; l>=0; --l) {
                tmp = denseLayers[l].backwardPropagation(tmp, learning_rate);
            }
        }
        double accuracy = static_cast<double>(totalCorrect) / static_cast<double>(train_data.size());
        double loss = totalLoss / static_cast<double>(train_data.size());
        std::cout << " - train_loss: " << loss << " - train_acc: " << accuracy;

        auto dev_results = predict(dev_data, dev_labels);
        std::cout << " - dev_loss: " << dev_results.first << " - dev_acc: " << dev_results.second << std::endl;
    }
}

// predict on 1 example
int Network::predict(std::vector<double> &data) {
    auto hidden = data;
    for(int l = 0; l < denseLayers.size(); ++l) {
        hidden = denseLayers[l].forwardPropagation(hidden);
    }
    auto prediction = finalLayer.forwardPropagation(hidden);

    double max = 0.0;
    int max_idx = 0;
    for(int i = 0; i < prediction.size(); i++) {
        if (prediction[i] > max) {
            max = prediction[i];
            max_idx = i;
        }
    }

    return max_idx;
}

// predict on whole dataset with data and labels
std::pair<double,double> Network::predict(std::vector<std::vector<double> > &data, std::vector<int> &labels) {

    double totalLoss = 0;
    int totalCorrect = 0;
    for (int i = 0; i < data.size(); ++i) {
        // forward
        auto hidden = data[i];
        for(int l = 0; l < denseLayers.size(); ++l) {
            hidden = denseLayers[l].forwardPropagation(hidden);
        }
        auto predictions = finalLayer.forwardPropagation(hidden);

        auto tuple = cross_entropy_loss(predictions, labels[i]);
        totalLoss += std::get<0>(tuple);
        totalCorrect += std::get<1>(tuple);
    }

    double accuracy = static_cast<double>(totalCorrect) / static_cast<double>(data.size());
    double loss = totalLoss / static_cast<double>(data.size());

    return std::make_pair(loss, accuracy);
}


void Network::save(std::string file_name) {
    std::cout << "Saving model to " << file_name << std::endl;

    std::ofstream f;
    f.open (file_name);

    // save dense layers
    f << denseLayers.size() << std::endl;
    for(auto & denseL : denseLayers) {
        int n_cur = denseL.size();
        int n_prev = denseL.size_prev();

        f << n_cur << " " << n_prev << "\n";
        // save weights
        for(int i = 0; i < n_cur; ++i) {
            for(int j = 0; j < n_prev; ++j) {
                f << denseL.get_weights()[i][j] << " ";
            }
        }
        f << "\n";
        // save biases
        for(int i = 0; i < n_cur; ++i) {
            f << denseL.get_biases()[i] << " ";
        }
        f << "\n";
    }

    // save final layer
    int n_cur = finalLayer.size();
    int n_prev = finalLayer.size_prev();

    f << n_cur << " " << n_prev << "\n";
    // save weights
    for(int i = 0; i < n_cur; ++i) {
        for(int j = 0; j < n_prev; ++j) {
            f << finalLayer.get_weights()[i][j] << " ";
        }
    }
    f << "\n";
    // save biases
    for(int i = 0; i < n_cur; ++i) {
        f << finalLayer.get_biases()[i] << " ";
    }
    f << "\n";

    f.close();
}

void Network::load(std::string file_name) {
    std::cout << "Loading model from " << file_name << std::endl;

    std::ifstream f(file_name);
    if (f.is_open()) {
        try {
            // load dense layers
            int n_denseL;
            int n_cur, n_prev;
            double x;
            f >> n_denseL;
            for (int i = 0; i < n_denseL; ++i) {
                f >> n_cur >> n_prev;
                addDenseLayer(n_cur);
                // load weights
                for (int j = 0; j < n_cur; ++j) {
                    for (int k = 0; k < n_prev; ++k) {
                        f >> x;
                        denseLayers[i].get_weights()[j][k] = x;
                    }
                }
                // load biases
                for (int j = 0; j < n_cur; ++j) {
                    f >> x;
                    denseLayers[i].get_biases()[j] = x;
                }
            }
            // load final layer
            f >> n_cur >> n_prev;
            addFinalLayer(n_cur);
            // load weights
            for (int j = 0; j < n_cur; ++j) {
                for (int k = 0; k < n_prev; ++k) {
                    f >> x;
                    finalLayer.get_weights()[j][k] = x;
                }
            }
            // load biases
            for (int j = 0; j < n_cur; ++j) {
                f >> x;
                finalLayer.get_biases()[j] = x;
            }

            f.close();
            std::cout << "Model has been loaded!\n\n";
        } catch(std::exception const& e) {
            std::cout << "Error - Unable to load model\n" << e.what() << std::endl;
        }
    } else {
        std::cout << "Unable to open file " << file_name << std::endl;
    }
}