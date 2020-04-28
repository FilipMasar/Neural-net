#include "DataManage.h"


void DataManage::load_data(std::vector<std::vector<double> > &images, std::vector<int> &labels, std::string file_name,
                           int n_row, int n_col) {
    images.resize(n_row, std::vector<double>(n_col - 1));
    labels.resize(n_row);

    std::ifstream file("datasets/" + file_name, std::ios::in);

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
        std::cout << file_name << " not found in datasets directory!\n";
    }
}

void DataManage::normalize(std::vector<std::vector<double> > &images) {
    for (auto &a : images) {
        for (auto &b : a) {
            b /= 255;
        }
    }
}
