#!/bin/bash

# Create the datasets directory if it doesn't exist
mkdir -p datasets/mnist


# Download the MNIST dataset
echo "Downloading the MNIST dataset..."
curl -L -o datasets/mnist.zip\
  https://www.kaggle.com/api/v1/datasets/download/oddrationale/mnist-in-csv

echo "Unzipping the MNIST dataset..."
unzip datasets/mnist.zip -d datasets/mnist

# Remove the zip file
rm datasets/mnist.zip

echo "Done"
