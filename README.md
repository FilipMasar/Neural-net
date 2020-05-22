# Neural-net
Implementation of neural net using SGD. Correctness is show on training on mnist dataset.

- [Neural-net](#neural-net)
    + [Requirements](#requirements)
- [Usage](#usage)
- [Documentation](#documentation)
    + [Network architecture](#network-architecture)
  * [network directory](#network-directory)
    + [Network.h](#networkh)
    + [layers/DenselLayer.h](#layers-densellayerh)
    + [layers/Softmax.h](#layers-softmaxh)
  * [utils directory](#utils-directory)
    + [utils/MnistManage.h](#utils-mnistmanageh)
    + [utils/DataManage.h](#utils-datamanageh)
    + [utils/std_image.h](#utils-std-imageh)

### Requirements
To train neural net on mnist dataset you first need to download train and test set to datasets directory. Follow `datasets/README.md`


# Usage
`network/Network.h` is the main point of neural net. It should be called to build neural net, train it and predict using its functions.

The example on how to use Network.h is in `main.cpp`. Whole project could be compiled using `make`

```c++
// constructor takes an input dimensionality
Network network(784);

// multiple Dense layers with ReLU activation could be added
//  - it takes number of neurons as an argument
network.addDenseLayer(100);
network.addDenseLayer(50);

// exactly 1 final layer has to be added. It is Dense layer with Softmax activation
//  - it takes number of neurons as an argument
network.addFinalLayer(10);

// network training
//  - takes number of epochs, learning rate, trainset and devset as arguments
network.fit(3, 0.01, train_images, train_labels, dev_images, dev_labels);

// network prediction on entire dataset
//  - data and labels has to be passed
//  - returns loss and accuracy as std::pair
auto test_results = network.predict(test_images, test_labels);

// network prediction on 1 example
//  - takes exactly 1 data
//  - return its prediction
int prediction = network.predict(test_images[0]);

// save model to mymodel.txt
network.save("mymodel.txt");

// To load the model you have to first create the network object with models input dimensionality
// You can then add additional dense layers or fine-tune the network
Network pretrained_network(784);
pretrained_network.load("mymodel.txt");

```

`utils/MnistManage.h` provides functions to load mnist datasets from csv files

`utils/DataManage.h` could be used to load any dataset from csv file. It has to follow the rules, that each row is representing 1 example and that the first column is the golden integer labels, and other columns are features(data) of an example.

`utils/stb_image.h` is a helper library for loading png files. More can be find here https://github.com/nothings/stb

Mnist contains 60000 training example and 10000 test example.
`mymodel.txt` is example of pretrained model - it got 96,4% accuracy on testset containing 5000 data.
Its input dimensionality is 784

If during training you get loss `nan` it means that training did not converged. Try decreasing learning_rate!

# Documentation

### Network architecture
INPUT -> multiple fully connected layers with RELU -> Fully connected layer with SOFTMAX -> CROSS ENTROPY LOSS

Training is performed using SGD one example at a time

## network directory
Contains code of neural network

### Network.h
It creates model, perform training and prediction, and could save/load a model
* addDenseLayer - adds `layers/DenselLayer.h` to network architecture
* addFinalLayer - adds `layers/Softmax.h` to network architecture
* fit
  - args:
    + number of epochs
    + learning_rate
    + reference to `vector<vector<double> >` training data
    + reference to `vector<int>` training labels
    + reference to `vector<vector<double> >` development data
    + reference to `vector<int>` development labels
  - performs training of the network
* predict
  - if only 1 data is passed returns its prediction
  - otherwise perform forward propagation on whole test, compares it with golden labels and returns accuracy and categorical cross entropy loss
* save
  - save model to specified file
* load
  - load model from specified file

### layers/DenselLayer.h
* forwardPropagation - gets input from the previous layer and forward propagate it through fully connected layer with ReLU actiovation
* backwardPropagation - perform back propagation through that layer

### layers/Softmax.h
* forwardPropagation - gets input from the previous layer and forward propagate it through fully connected layer with softmax actiovation
* backwardPropagation - perform back propagation through that layer

## utils directory
Contains code for managing data - loading datasets and normalizing it

### utils/MnistManage.h
* load_mnist_train
  - args:
    + reference to `vector<vector<double> >` where features will be stored
    + reference to `vector<int>` where golden labels will be stored
    + number of rows - how many training data you want
  - it loads training data from `datasets/mnist_train.csv`
* load_mnist_test
  - args:
      + reference to `vector<vector<double> >` where features for development set will be stored
      + reference to `vector<int>` where golden labels of development set will be stored
      + reference to `vector<vector<double> >` where features for test set will be stored
      + reference to `vector<int>` where golden labels of test set will be stored
      + number of rows for devset
      + number of rows for testset
    - it loads dev,test data from `datasets/mnist_test.csv`
* normalize
  - it gets reference to data and normalize it (devide by 255)
* load_mnist_png
  - args:
    + reference to `vector<double>` where image will be stored
    + file_path - path to image
  - it loads **.png 28x28 8-bit** image and normalize it 

### utils/DataManage.h
* load_data
  - args:
    + reference to `vector<vector<double> >` where features will be stored
    + reference to `vector<int>` where golden labels will be stored
    + file_path - path to .csv data file
    + number of rows - how many training data you want
    + number of columns - how many columns does .csv have
  - it loads training data specified .csv file. **It has to follow the rules, that each row is representing 1 example and that the first column is the golden integer labels, and other columns are features(numerical data) of an example.**

### utils/std_image.h
helper library for loading .png images from https://github.com/nothings/stb
                                           
