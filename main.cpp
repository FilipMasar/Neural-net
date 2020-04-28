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

    int N = 1000;

    cout << "Loading dataset ..." << endl;
    DataManage::load_data(train_images, train_labels, "mnist_train.csv", N, 785);
    DataManage::load_data(test_images, test_labels, "mnist_test.csv", 300, 785);

    cout << "Normalizing dataset ..." << endl;
    DataManage::normalize(train_images);
    DataManage::normalize(test_images);

    cout << "Creating layers ..." << endl;
    DenseRelu l1(1000, 784);
    Softmax softmax(10, 1000);

    int epochs = 10;
    double learning_rate = 0.1;

    for (int i = 0; i < epochs; ++i) {
        cout << "Epoch " << i+1 << "/" << epochs << endl;
        double totalLoss = 0;
        int totalCorrect = 0;

        for (int j = 0; j < N; ++j) {
            if (j % 100 == 0) cout << "\r" << j << "/" << N << flush;
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

        double accuracy = static_cast<double>(totalCorrect) / static_cast<double>(N);
        cout << " - loss: " << totalLoss << " - accuracy: " << accuracy << endl;
    }

    // prediction on test
    cout << "\nPredicting on test set ..." << endl;

    double totalLoss = 0;
    int totalCorrect = 0;
    for (int i = 0; i < test_images.size(); ++i) {
        // forward
        auto hidden = l1.forwardPropagation(test_images[i]);
        auto predictions = softmax.forwardPropagation(hidden);

        auto tuple = Loss::cross_entropy_loss(predictions, test_labels[i]);
        totalLoss += get<0>(tuple);
        totalCorrect += get<1>(tuple);
    }

    double accuracy = static_cast<double>(totalCorrect) / static_cast<double>(test_images.size());
    cout << "TEST LOSS: " << totalLoss << endl;
    cout << "TEST ACCURACY: " << accuracy << endl;

    return 0;
}