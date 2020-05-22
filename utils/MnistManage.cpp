#include "MnistManage.h"

extern "C" {
    #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
}


void Mnist::load_mnist_train(std::vector<std::vector<double> > &images, std::vector<int> &labels, int n_row) {
    assert(((void)"There is not that much data in train_set! 60000 is maximum", (n_row <= 60000)));

    int n_col = 785;

    images.resize(n_row, std::vector<double>(n_col - 1));
    labels.resize(n_row);

    std::ifstream file("datasets/mnist_train.csv", std::ios::in);

    if (file.is_open()) {

        std::string line, word;
        for (int i = 0; i < n_row; ++i) {
            getline(file, line);

            std::stringstream ss(line);

            // get golden label
            getline(ss, word, ',');
            labels[i] = stoi(word);

            // get image data
            for (int j = 0; j < n_col - 1; ++j) {
                getline(ss, word, ',');
                images[i][j] = stod(word);
            }
        }
        file.close();
    } else {
        std::cout << "mnist_train.csv not found in datasets directory!\n";
    }
}


void Mnist::load_mnist_test(std::vector<std::vector<double> > &dev_images, std::vector<int> &dev_labels,
                                 std::vector<std::vector<double> > &test_images, std::vector<int> &test_labels,
                                 int n_row_dev, int n_row_test) {

    assert(((void)"There is not that much data in test_set! dev + test should be at most 10000", (n_row_dev + n_row_test <= 10000)));

    int n_col = 785;

    dev_images.resize(n_row_dev, std::vector<double>(n_col - 1));
    dev_labels.resize(n_row_dev);
    test_images.resize(n_row_test, std::vector<double>(n_col - 1));
    test_labels.resize(n_row_test);

    std::ifstream file("datasets/mnist_train.csv", std::ios::in);

    if (file.is_open()) {

        std::string line, word;
        for (int i = 0; i < n_row_dev; ++i) {
            getline(file, line);

            std::stringstream ss(line);

            // get golden label
            getline(ss, word, ',');
            dev_labels[i] = stoi(word);

            // get image data
            for (int j = 0; j < n_col - 1; ++j) {
                getline(ss, word, ',');
                dev_images[i][j] = stod(word);
            }
        }
        for (int i = 0; i < n_row_test; ++i) {
            getline(file, line);

            std::stringstream ss(line);

            // get golden label
            getline(ss, word, ',');
            test_labels[i] = stoi(word);

            // get image data
            for (int j = 0; j < n_col - 1; ++j) {
                getline(ss, word, ',');
                test_images[i][j] = stod(word);
            }
        }
        file.close();
    } else {
        std::cout << "mnist_test.csv not found in datasets directory!\n";
    }
}


void Mnist::normalize(std::vector<std::vector<double> > &images) {
    for (auto &a : images) {
        for (auto &b : a) {
            b /= 255;
        }
    }
}


void Mnist::load_mnist_png(std::vector<double> &data, std::string file_path) {
    data.resize(784);

    int x,y,n;
//    file_path = "../" + file_path;
    unsigned char *tmp = stbi_load(file_path.c_str(), &x, &y, &n, 1);

    if (tmp != nullptr) {
        for(int j = 0; j < 784; ++j) {
            data[j] = static_cast<double>(tmp[j]);
        }
    } else {
        std::cout << "Error loading png file!\n";
    }

    // normalize
    for (auto &a : data) {
        a /= 255;
    }

    stbi_image_free(tmp);
}
