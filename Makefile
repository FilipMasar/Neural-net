all:
    g++ -o NeuralNet main.cpp utils/* network/Network.h network/Network.cpp network/layers/*
    