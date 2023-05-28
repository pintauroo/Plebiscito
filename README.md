# Plebiscito 

Plebiscito is a distributed and synchronous consensus protocol. It allows to handle resesources allocation between a set of nodes in a distributed fashion.
Specifically it is designed to allocate High-Performance Machine Learning Task (HPMLT).
We assume that the ML tasks to be allocated on the set of nodes consists of a Neural Network (NN). 
Potentially to better allocate the available resources over the different nodes it might be necessary to split the NN. 
To do so we do characterize each NN with a percentage of CPU, GPU and BW needed for each layer.
In this way, when a HPMLT request is received, Plebiscito protocol is going to determine the most efficient node to execute that task via a bidding mechanism.
Multiple messages are exchanged between the different nodes till a consensus is reached so that all of the nodes are going to be alligned who won the bid for that specific HPMLT.

## Code

To run an experiment:

python main.py 'req_number' 'alpha' 'nodes_number' 'utility_function'

Example:

python main.py 10 1 10 alpha_BW_CPU

- The available utility functions used for the bidding are: "alpha_BW_CPU" "alpha_GPU_BW" "alpha_GPU_CPU"
- The alpha parameter is comprised between 0 and 1 and it is used as a weight in the utility function between the two competing resources.



