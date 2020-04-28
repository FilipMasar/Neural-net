#ifndef NEURAL_NET_DATAMANAGE_H
#define NEURAL_NET_DATAMANAGE_H


#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>


namespace DataManage {

    void load_data(std::vector<std::vector<double> > &images, std::vector<int> &labels, std::string file_name, int n_row, int n_col);
    void normalize(std::vector<std::vector<double> > &images);

}



#endif //NEURAL_NET_DATAMANAGE_H
