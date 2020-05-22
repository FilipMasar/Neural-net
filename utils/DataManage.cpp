//
// Created by Filip Ma on 22/05/2020.
//

#include "DataManage.h"

void DataManage::load_data(std::vector<std::vector<double> > &data, std::vector<int> &labels, std::string file_path,
                           int n_row, int n_col) {

    data.resize(n_row, std::vector<double>(n_col - 1));
    labels.resize(n_row);

    std::ifstream file(file_path, std::ios::in);

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
                data[i][j] = stod(word);
            }
        }
        file.close();
    } else {
        std::cout << file_path << " not found!\n";
    }
}