"""
Topology building module
"""

import threading
import numpy as np


class topo:
    def __init__(self, func_name, max_bandwidth, min_bandwidth, num_clients, num_edges, edge_to_add=0, probability=0):
        self.n = num_edges #adjacency matrix

        self.to = getattr(self, func_name)
        self.b = max_bandwidth
        self.edge_to_add = []
        self.probability = probability
        
        self.adjacency_matrix_lock = threading.Lock()
        if func_name == "complete_graph":
            self.adjacency_matrix = self.compute_complete_graph()
        elif func_name == "ring_graph":
            self.adjacency_matrix = self.compute_ring_graph()
        elif func_name == "probability_graph":
            self.adjacency_matrix = self.compute_probabilistic_graph()
        
        for _ in range(edge_to_add):
            while True:
                from_edge = np.random.randint(0, num_edges-1)
                to_edge = np.random.randint(0, num_edges-1)
                if from_edge != to_edge and abs(from_edge-to_edge) != 1 and set([from_edge, to_edge]) not in self.edge_to_add:
                    self.edge_to_add.append(set([from_edge, to_edge]))
                    break
            
        # self.b = np.random.uniform(min_bandwidth, max_bandwidth, size=(num_clients, num_edges)) #bandwidth matrix

    def disconnect_node(self, node):
        """
        This function disconnects a node from the network.
        """
        with self.adjacency_matrix_lock:
            self.adjacency_matrix[node,:] = 0
            self.adjacency_matrix[:,node] = 0
        
    def call_func(self):
        return self.to()
    
    def linear_topology(self):
        """
        This function returns the adjacency matrix for a linear topology of n nodes.
        """
        # Create an empty adjacency matrix of size n x n
        adjacency_matrix = np.zeros((self.n, self.n))
        
        # Add edges to the adjacency matrix
        for i in range(self.n):
            if i == 0:
                # Connect node 0 to node 1
                adjacency_matrix[i][i+1] = 1
            elif i == self.n-1:
                # Connect node n-1 to node n-2
                adjacency_matrix[i][i-1] = 1
            else:
                # Connect node i to nodes i-1 and i+1
                adjacency_matrix[i][i-1] = 1
                adjacency_matrix[i][i+1] = 1
        
        return adjacency_matrix
    
    def compute_probabilistic_graph(self):
        adjacency_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(0, i):
                value = np.random.choice([0, 1], p=[1-self.probability, self.probability])
                adjacency_matrix[i][j] = value
                adjacency_matrix[j][i] = value
        return adjacency_matrix
    
    def probability_graph(self):
        with self.adjacency_matrix_lock:
            return self.adjacency_matrix

    def compute_complete_graph(self):
        adjacency_matrix = np.ones((self.n, self.n)) - np.eye(self.n)
        return adjacency_matrix
    
    def complete_graph(self):
        with self.adjacency_matrix_lock:
            return self.adjacency_matrix

    def compute_ring_graph(self):
        adjacency_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            adjacency_matrix[i][(i-1)%self.n] = 1
            adjacency_matrix[i][(i+1)%self.n] = 1
        for e in self.edge_to_add:
            adjacency_matrix[list(e)[0], list(e)[1]] = 1
            adjacency_matrix[list(e)[1], list(e)[0]] = 1
        return adjacency_matrix
    
    def ring_graph(self):
        with self.adjacency_matrix_lock:
            return self.adjacency_matrix

    def star_graph(self):
        adjacency_matrix = np.zeros((self.n, self.n))
        adjacency_matrix[0,:] = 1
        adjacency_matrix[:,0] = 1
        adjacency_matrix[0,0] = 0
        return adjacency_matrix

    def grid_graph(self):
        adjacency_matrix = np.zeros((self.n*self.n, self.n*self.n))
        for i in range(self.n):
            for j in range(self.n):
                node = i*self.n + j
                if i > 0:
                    adjacency_matrix[node][node-self.n] = 1
                if i < self.n-1:
                    adjacency_matrix[node][node+self.n] = 1
                if j > 0:
                    adjacency_matrix[node][node-1] = 1
                if j < self.n-1:
                    adjacency_matrix[node][node+1] = 1
        return adjacency_matrix

