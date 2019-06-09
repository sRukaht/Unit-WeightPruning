The neural network is made using keras. For training, I have used the MNIST-fashion dataset.


Definition of functions:

Sort():
The function takes one argument: the weights of the trained model. It returns an array of individual weights in a sorted order. This is used in weight pruning of the k% weights.

 Wt_prune():
This function takes two arguments: ‘Weights’ and ‘k’. This function is used to prune the trained neural network using weight pruning. The function returns a tuple containing the percent sparsity and the accuracy of the pruned neural network.

Unt_prune():
This function  takes two arguments: ‘weights’ and ‘k’. This function is used to prune the model using unit pruning. This function returns a tuple containing the percent sparsity and the accuracy of the pruned neural network.
For unit pruning, the l2 norm of all the weights corresponding to a particular neuron is found and stored in array ‘l2norm’. Afterwards, the neurons are pruned.

wt_graph():
This function plots a graph between the percent sparsity and the accuracy in a weight pruned neural network.

unt_graph():
This function plots a graph between the percent sparsity and the accuracy in a weight pruned neural network.




