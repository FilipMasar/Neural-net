#ifndef NEURAL_NET_MNISTMANAGE_H
#define NEURAL_NET_MNISTMANAGE_H


#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cassert>



namespace Mnist {

    void load_mnist_train(std::vector<std::vector<double> > &images, std::vector<int> &labels, int n_row);
    void load_mnist_test(std::vector<std::vector<double> > &dev_images, std::vector<int> &dev_labels,
                         std::vector<std::vector<double> > &test_images, std::vector<int> &test_labels,
                         int n_row_dev, int n_row_test);
    void normalize(std::vector<std::vector<double> > &images);

    void load_mnist_png(std::vector<double> &data, std::string file_path);
}



#endif //NEURAL_NET_MNISTMANAGE_H
