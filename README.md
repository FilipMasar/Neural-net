# Neural-net
Implementation of neural net using SGD on mnist dataset

### Requirements
To train neural net on mnist dataset you first need to download train and test set to datasets directory. Follow `datasets/README.md`


## network/Network.h
`network/Network.h` is the main point of neural net. It should be called to build neural net, train it and predict using its functions.

The example on how to use Network.h is in `main.cpp`

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

## utils/MnistManage.h
provides functions to load mnist datasets


### Details
Mnist contains 60000 training example and 10000 test example.
`mymodel.txt` is example of pretrained model - it got 96,4% accuracy on testset containing 5000 data.
Its input dimensionality is 784

If during training you get loss `nan` it means that training did not converged. Try decreasing learning_rate!

### Network architecture
INPUT -> multiple fully connected layers with RELU -> Fully connected layer with SOFTMAX -> CROSS ENTROPY LOSS

Training is performed using SGD one example at a time
