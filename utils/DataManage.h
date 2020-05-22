//
// Created by Filip Ma on 22/05/2020.
//

#ifndef NEURAL_NET_DATAMANAGE_H
#define NEURAL_NET_DATAMANAGE_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cassert>

namespace DataManage {

    void load_data(std::vector<std::vector<double> > &data, std::vector<int> &labels, std::string file_path, int n_row, int n_col);

}


#endif //NEURAL_NET_DATAMANAGE_H
