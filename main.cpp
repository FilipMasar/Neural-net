#include <iostream>
#include <vector>

#include "DataManage.h"
#include "layers/DenseRelu.h"
#include "layers/Softmax.h"
#include "loss/CrossEntropy.h"

using namespace std;


int main() {
    vector<vector<double> > train_images;
    vector<int> train_labels;
    vector<vector<double> > test_images;
    vector<int> test_labels;

    int N_train = 1000;
    int N_test = 500;

    cout << "Loading dataset ..." << endl;
    DataManage::load_data(train_images, train_labels, "mnist_train.csv", N_train, 785);
    DataManage::load_data(test_images, test_labels, "mnist_test.csv", N_test, 785);

    DataManage::normalize(train_images);
    DataManage::normalize(test_images);

    DenseRelu l1(1000, 784);
    Softmax softmax(10, 1000);

    int epochs = 10;
    double learning_rate = 0.1;

    cout << "Training ..." << endl;
    for (int i = 0; i < epochs; ++i) {
        cout << "Epoch " << i+1 << "/" << epochs << endl;
        double totalLoss = 0;
        int totalCorrect = 0;

        for (int j = 0; j < N_train; ++j) {
            if (j % 100 == 99) cout << "\r" << j+1 << "/" << N_train << flush;
            // forward
            auto hidden = l1.forwardPropagation(train_images[j]);
            auto predictions = softmax.forwardPropagation(hidden);

            auto tuple = Loss::cross_entropy_loss(predictions, train_labels[j]);
            totalLoss += get<0>(tuple);
            totalCorrect += get<1>(tuple);

            // backward
            auto tmp3 = softmax.backwardPropagation(get<2>(tuple), learning_rate);
            auto tmp4 = l1.backwardPropagation(tmp3, learning_rate);

        }

        double accuracy = static_cast<double>(totalCorrect) / static_cast<double>(N_train);
        cout << " - loss: " << totalLoss << " - accuracy: " << accuracy << endl;
    }

    // prediction on test
    cout << "\nPredicting on test set ..." << endl;

    double totalLoss = 0;
    int totalCorrect = 0;
    for (int i = 0; i < N_test; ++i) {
        // forward
        auto hidden = l1.forwardPropagation(test_images[i]);
        auto predictions = softmax.forwardPropagation(hidden);

        auto tuple = Loss::cross_entropy_loss(predictions, test_labels[i]);
        totalLoss += get<0>(tuple);
        totalCorrect += get<1>(tuple);
    }

    double accuracy = static_cast<double>(totalCorrect) / static_cast<double>(N_test);
    cout << "TEST LOSS: " << totalLoss << endl;
    cout << "TEST ACCURACY: " << accuracy << endl;

    return 0;
}