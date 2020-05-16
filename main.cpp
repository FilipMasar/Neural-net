#include <iostream>
#include <vector>

#include "DataManage.h"
#include "network/Network.h"

using namespace std;


int main() {
    vector<vector<double> > train_images;
    vector<int> train_labels;
    vector<vector<double> > dev_images;
    vector<int> dev_labels;
    vector<vector<double> > test_images;
    vector<int> test_labels;

    int N_train = 60000;
    int N_dev = 5000;
    int N_test = 5000;

    cout << "Loading dataset ..." << endl;
    DataManage::load_mnist_train(train_images, train_labels, N_train);
    DataManage::load_mnist_test(dev_images, dev_labels, test_images, test_labels, N_dev, N_test);

    DataManage::normalize(train_images);
    DataManage::normalize(dev_images);
    DataManage::normalize(test_images);


    // Build the network
    Network network(784);

    network.addDenseLayer(100);
    network.addDenseLayer(50);
    network.addFinalLayer(10);

    // train the network
    network.fit(3, 0.01, train_images, train_labels, dev_images, dev_labels);

    // predicting using network
    cout << "\nPredicting on test set ..." << endl;
    auto test_results = network.predict(test_images, test_labels);
    cout << " - test_loss: " << test_results.first << " - test_acc: " << test_results.second << endl << endl;

    // saving model
    network.save("mymodel.txt");

    // 1 example prediction
    int prediction = network.predict(test_images[0]);
    cout << "prediction " << prediction << " gold " << test_labels[0] << endl;

    return 0;
}