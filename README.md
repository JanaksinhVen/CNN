**The question-answer format repository for CNN**

# CNN and AutoEncoders[CNN.ipynb]
Welcome to the Hello World of image classification - a CNN trained on MNIST
dataset. You can use Pytorch for this Task. You can load the MNIST dataset
using PyTorch’s torchvision.
## Data visualization and Preprocessing 
1. Draw a graph that shows the distribution of the various labels across the
entire dataset. You are allowed to use standard libraries like Matplotlib.
2. Visualize several samples (say 5) of images from each class.
3. Check for any class imbalance and report.
4. Partition the dataset into train, validation, and test sets.
5. Write a function to visualize the feature maps. Your code should be able
to visualize feature maps of a trained model for any layer of the given
image.
## Model Building 
1. Construct a CNN model for Image classification using pytorch.
2. Your network should include convolutional layers, pooling layers, dropout
layers, and fully connected layers.
3. Construct and train a baseline CNN using the following architecture: 2
convolutional layers each with ReLU activation and subsequent max pooling, followed by a dropout and a fully-connected layer with softmax activation, optimized using the Adam optimizer and trained with the crossentropy loss function.
4. Display feature maps after applying convolution and pooling layers for any
one class and provide a brief analysis.
5. Report the training and validation loss and accuracy at each epoch.
## Hyperparameter Tuning and Evaluation
1. Use W&B to facilitate hyperparameter tuning. Experiment with various
architectures and hyperparameters: learning rate, batch size, kernel sizes
(filter size), strides, number of epochs, and dropout rates.
2. Compare the effect of using and not using dropout layers.
3. Log training/validation loss and accuracy, confusion matrices, and classspecific metrics using W&B.
## Model Evaluation and Analysis
1. Evaluate your best model on the test set and report accuracy, per-class
accuracy, and classification report.
2. Provide a clear visualization of the model’s performance, e.g., confusion
matrix.
7
3. Identify a few instances where the model makes incorrect predictions and
analyze possible reasons behind these misclassifications.
## Train on Noisy Dataset
In the subsequent parts, you have to work with a noisy mnist dataset. Download
the [mnist-with-awgn.mat](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/khushi_agarwal_students_iiit_ac_in/EnXYjh3ZU7xAtuz_NsvtB_MBhJHSL7EQzSJdeyPu8j8Khg?e=c09dfv) from here. You can load .mat file using scipy.io.
1. Train your best model from the previous parts of Task 4 on the noisy
mnist dataset which contains noise in it (additive white gaussian noise,
don’t worry it’s just a fancy name).
2. Report validation losses, validation scores, training losses, training scores.
3. Evaluate your model on test data and print the classification report.
## AutoEncoders to Save the Day
1. Implement an Autoencoder class which will help you de-noise the noisy
mnist dataset from Part 4.5.
2. Visualise the classes and feature space before and after de-noising.
3. Now using the de-noised dataset, train your best model from the previous
parts.
4. Report validation losses, validation scores, training losses, training scores.
5. Evaluate your model on test data and print the classification report.
6. Analyse and compare the results/accuracy scores as obtained in previous Parts.

# Multi-digit Recognition on Multi-MNIST Dataset [CNN-vs-MLP.ipynb]
Download the DoubleMNIST dataset from [here](https://drive.google.com/file/d/1MqQCdLt9TVE3joAMw4FwJp_B8F-htrAo/view). The DoubleMNIST dataset
contains images with two handwritten digits, and the task is to correctly identify
and classify each digit within the image.
Build and train models that can simultaneously recognize and predict the
two digits from a single image.This is basically another version of a multilabel
classification.  

• Display several images from the filtered dataset to familiarize yourself with
the dual-digit nature of the images.  

• From the Multi-MNIST dataset, filter out and exclude images where the
same digit appears twice. Your model will be trained only on images that
have distinct digits  

• Ensure you split the datasets into training and validation sets to evaluate
the model’s performance during hyperparameter tuning.  

### MLP on Multi-MNIST
1. Implement and train an MLP model on the MultiMNIST dataset.
2. Hyperparameter Tuning: Adjust the number of hidden layers and the
number of neurons within each layer to optimize performance and find
the best model.
3. Report the accuracies on the train and validation set.
4. Evaluate your trained model on test set and report the accuracy.
### CNN on Multi-MNIST
1. Design and train a CNN model on the MultiMNIST dataset.
2. Hyperparameter Tuning: Experiment with different learning rates, kernel
sizes, and dropout rates to determine the optimal configuration.
3. Report the accuracies on the train and validation set.
4. Evaluate your trained model on test set and report the accuracy.
### Testing on Single digit MNIST (regular MNIST)
Evaluate your trained model on the regular MNIST dataset with single-digit
images. See how the model, initially trained on images with two digits, performs
on these single-digit images. Report the accuracies.
## Permuted MNIST 
The Permuted MNIST dataset is a variation of the original MNIST dataset
where the pixels of each image are randomly permuted. This permutation makes
the task significantly more challenging because it destroys the spatial structure
of the images. In Permuted MNIST, the goal is still to recognize the digits but
now without relying on the spatial relationships between pixels.
Ensure you split the datasets into training and validation sets to evaluate
the model’s performance during hyperparameter tuning.
9
### MLP on Permuted-MNIST
1. Implement and train an MLP model on the Permuted-MNIST dataset.
2. Hyperparameter Tuning: Adjust the number of hidden layers and the
number of neurons within each layer to optimize performance and find
the best model.
3. Report the accuracies on the train and validation set.
4. Evaluate your trained model on test set and report the accuracy.
### CNN on Permuted-MNIST
1. Design and train a CNN model on the Permuted-MNIST dataset.
2. Hyperparameter Tuning: Experiment with different learning rates, kernel
sizes, and dropout rates to determine the optimal configuration.
3. Report the accuracies on the train and validation set.
4. Evaluate your trained model on test set and report the accuracy.
### Analysis
1. Contrast the performances of MLP vs. CNN for both datasets.
2. Discuss the observed differences and any challenges faced during training
and evaluation.
3. Compare the potential for overfitting between a CNN and an MLP in the
context of datasets in Task 5. Use training vs. validation loss/accuracy
plots to support your observations
