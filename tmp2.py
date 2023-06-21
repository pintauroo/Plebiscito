import random
import threading
import sys
import pygraphviz as pgv
import sys
from collections import deque

def dijkstra(adj_matrix, start_node, end_node):
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes  # Track visited nodes
    distances = [sys.maxsize] * num_nodes  # Initialize distances with a large value
    distances[start_node] = 0  # Set distance of the start node to 0

    # Find the shortest path for all nodes
    for _ in range(num_nodes):
        min_distance = sys.maxsize
        min_node = -1

        # Find the node with the smallest distance from the set of unvisited nodes
        for node in range(num_nodes):
            if not visited[node] and distances[node] < min_distance:
                min_distance = distances[node]
                min_node = node

        visited[min_node] = True

        # Update distances for the adjacent nodes
        for node in range(num_nodes):
            if (
                not visited[node]
                and adj_matrix[min_node][node] == 1
                and distances[min_node] + 1 < distances[node]
            ):
                distances[node] = distances[min_node] + 1

    # Backtrack to find the shortest path
    if distances[end_node] == sys.maxsize:
        # No path exists
        return None, None

    path = deque()
    current_node = end_node
    path.appendleft(current_node)

    while current_node != start_node:
        for node in range(num_nodes):
            if (
                adj_matrix[node][current_node] == 1
                and distances[node] + 1 == distances[current_node]
            ):
                current_node = node
                path.appendleft(current_node)
                break

    return list(path), distances[end_node]

class NetworkTopology:
    class Edge:
        def __init__(self, id, bw):
            self.__id = id
            self.__bw = bw
            
        def get_id(self):
            return self.__id
        
        def get_bw(self):
            return self.__bw
        
        def set_bw(self, bw):
            self.__bw = bw
        
        def __str__(self) -> str:
            return "Edge_" + str(self.__id)
    
    def __init__(self, n_nodes, min_bw, max_bw, group_number, seed=None):
        if seed is not None:
            random.seed(seed)
        
        self.group_number = group_number
        self.n_nodes = n_nodes
        self.min_bw = min_bw
        self.max_bw = max_bw
        self.__lock = threading.Lock()
        
        self.__generate_topology()
    
    def __generate_topology(self):
        if self.n_nodes < self.group_number:
            print("Number of nodes in the network topology must be >= than the number of groups. Exiting...")
            sys.exit(1)
                    
        node_group = []
        self.edges = {}
        id = 0
        for i in range(self.n_nodes):
            node_group.append(id)
            id = (id+1)%self.group_number
                    
        self.connected = [[] for _ in range(self.n_nodes+self.group_number)]
        for i in range(self.n_nodes+self.group_number):
            for j in range(self.n_nodes+self.group_number):
                self.connected[i].append([])
                self.connected[i][j] = 0
                
        self.edge_id = [[] for _ in range(self.n_nodes+self.group_number)]
        for i in range(self.n_nodes+self.group_number):
            for j in range(self.n_nodes+self.group_number):
                self.edge_id[i].append([])
              
        # edge dai nodi agli switch  
        index = 0
        for i in range(self.n_nodes):
            e = NetworkTopology.Edge(index, random.uniform(self.min_bw, self.max_bw))
            self.edges[e.get_id] = e
            index += 1
            self.edge_id[i][self.n_nodes + node_group[i]] = e.get_id()
            self.edge_id[self.n_nodes + node_group[i]][i] = e.get_id()
            self.connected[i][self.n_nodes + node_group[i]] = 1
            self.connected[self.n_nodes + node_group[i]][i] = 1
        
        # edge tra gli switch    
        for i in range(self.group_number):
            next_id = self.n_nodes + (i+1)%self.group_number
            prev_id = self.n_nodes + self.group_number-1 if i == 0 else self.n_nodes + i-1
            e1 = NetworkTopology.Edge(index, random.uniform(self.min_bw, self.max_bw))
            self.edges[e1.get_id] = e1
            index += 1
            e2 = NetworkTopology.Edge(index, random.uniform(self.min_bw, self.max_bw))
            self.edges[e2.get_id] = e2
            index += 1
            self.edge_id[self.n_nodes+i][prev_id] = e1.get_id()
            self.edge_id[self.n_nodes+i][next_id] = e2.get_id()
            self.edge_id[prev_id][self.n_nodes+i] = e1.get_id()
            self.edge_id[next_id][self.n_nodes+i] = e2.get_id()
            self.connected[self.n_nodes+i][prev_id] = 1
            self.connected[self.n_nodes+i][next_id] = 1
            self.connected[prev_id][self.n_nodes+i] = 1
            self.connected[next_id][self.n_nodes+i] = 1
            
        self.path = [[] for _ in range(self.n_nodes)]
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                self.path[i].append([])
            
        for i in range(self.n_nodes):
            for j in range(self.n_nodes-(self.n_nodes-i)):
                if i != j:
                    shortest_path, _ = dijkstra(self.connected, i, j)
                    print(f"From {i} to {j}: ", end='')
                    for k in range(len(shortest_path)-1):
                        print(f"Edge: {self.edge_id[shortest_path[k]][shortest_path[k+1]]} ", end='')
                        self.path[i][j].append(self.edge_id[shortest_path[k]][shortest_path[k+1]])
                        self.path[j][i].append(self.edge_id[shortest_path[k]][shortest_path[k+1]])
                    print()
        
        self.__export_as_dot()
        

    def consume_bandwidth_between_nodes(self, id1, id2, bw):
        with self.__lock:
            edges = self.path[id1][id2]
            for e_id in edges:
                pass
                                     
    def __print_topology(self):
        for i in range(self.n_nodes):
            print(f"Node {i}: ", end="")
            for j in range(self.n_nodes):
                if i != j:
                    print(f"--> Node {j} (", end="")
                    for e in self.adj[i][j]:
                        print(e, end=" ")
                    print(") ", end="")
            print()
            
    def __export_as_dot(self):
        G = pgv.AGraph(strict=False, directed=False)
        for i in range(self.n_nodes+self.group_number):
            for j in range(self.n_nodes+self.group_number - (self.n_nodes+self.group_number-i)):
                if self.connected[i][j] == 1:
                    if i < self.n_nodes and j >= self.n_nodes:
                        G.add_edge("Node " + str(i), "Sw " + str(j), label=self.edge_id[i][j])
                    if i < self.n_nodes and j < self.n_nodes:
                        G.add_edge("Node " + str(i), "Node " + str(j), label=self.edge_id[i][j])
                    if i >= self.n_nodes and j >= self.n_nodes:
                        G.add_edge("Sw " + str(i), "Sw " + str(j), label=self.edge_id[i][j])
                    if i >= self.n_nodes and j < self.n_nodes:
                        G.add_edge("Sw " + str(i), "Node " + str(j), label=self.edge_id[i][j])
                    
        G.write("file.dot")          
        
n = NetworkTopology(6, 1000, 1000, 4)
                