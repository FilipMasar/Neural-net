# Neural-net
Implementation of neural net on mnist dataset

### Requirements
To train neural net on mnist dataset you first need to download train and test set to datasets directory. Follow README.md in datasets

### Details
Mnist contains 60000 training example and 10000 test example. Due to the computational difficulty, by default training is perform only on 1000 examples and testing o 500
Number of epochs is set to 10 and learning rate to 0.1

With default settings and on standard CPU the script should run for about 10 minutes.

### Network architecture
INPUT -> Fully connected layer with RELU -> Fully connected layer with SOFTMAX -> CROSS ENTROPY LOSS
