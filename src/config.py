'''
Configuration variables module
'''

from src.node import node
import numpy as np
import datetime
import sys




num_clients=int(sys.argv[2])
num_edges=int(sys.argv[1])
bid_requests=2

#NN Resources
layer_number = 6
cost = 10
max_cost = 70
min_cost = 10

#Network Model Resources
max_bandwidth = 1000
min_bandwidth = 1000
b = np.random.uniform(min_bandwidth, max_bandwidth, size=(num_clients, num_edges))

#Edge Server Resources
node_max_res=20


#Links between edge servers
proba_0 = 0
m=np.random.choice([0, 1], size=(num_edges,num_edges),  p=[proba_0, 1-proba_0])
np.fill_diagonal(m, 0)

# m = [[0, 1, 0, 0, 0, 0],
#      [1, 0, 1, 0, 0, 0],
#      [0, 1, 0, 1, 0, 0],
#      [0, 0, 1, 0, 1, 0],
#      [0, 0, 0, 1, 0, 1],
#      [0, 0, 0, 0, 1, 0]]

# m = [[0, 1, 1, 1, 1, 1],
#      [1, 0, 1, 1, 1, 1],
#      [1, 1, 0, 1, 1, 1],
#      [1, 1, 1, 0, 1, 1],
#      [1, 1, 1, 1, 0, 1],
#      [1, 1, 1, 1, 1, 0]]

# m = [[0, 1, 0, ],
#      [1, 0, 1, ],
#      [0, 1, 0, ]]


# print(m)

# nodes = [[node(str(row)+str(col)) for row in range(max_nodes)] for col in range(max_nodes)]
nodes = [node(row) for row in range(num_edges)] 


counter =0
# NN= [50, 45, 40, 30, 25, 20, 15, 10, 5, 3]
# NN = np.random.randint(low=min_cost, high=max_cost, size=layer_number)
# NN=sorted(NN)
# NN=NN[::-1]
# next_NN = NN

min_layer_number = 2
max_layer_number = layer_number/2
# max_layer_number = layer_number


#Multiplicative factor
a=0.8


# data['bid'].append({
#         "keya":0,
#         "keyb":"0",
#         "keyc":"a",
#     })

NN_resources = np.ones(layer_number) * cost
NN_data_size = np.ones(layer_number) * cost
# NN_resources = np.random.randint(low=min_cost, high=max_cost, size=layer_number)
# NN_data_size = np.random.randint(low=min_cost, high=max_cost, size=layer_number)

def message_data(req_id, id_node):
    

    # NN=sorted(NN)
    # NN=NN[::-1]

    
    data = {
        "req_id": int(),
        "client_id": int(),
        "edge_id":int(),
        "NN_resources": NN_resources,
        "NN_data_size": NN_data_size
        }


    data['edge_id']=None
    data['req_id']=req_id
    data['client_id']=id_node
    # data['timestamp'].append(timestamp) #datetime.datetime.now()

    return data



def message_data_test(req_id, id_node):
    
    NN = np.random.randint(low=min_cost, high=max_cost, size=layer_number)
    # NN=sorted(NN)
    # NN=NN[::-1]
    data = {
        "req_id": int(),
        "client_id": int(),
        "auction_id": list(),
        "bid": list(),
        "x": list(),
        "timestamp": list(),
        "NN_resources": NN
    }


    for _ in range(0, len(NN)):
        data['x'].append(float('-inf'))
        data['bid'].append(float('-inf'))
        data['auction_id'].append(float('-inf'))
        data['timestamp'].append(datetime.datetime.now())

    data['req_id']=req_id
    data['client_id']=id_node
    # data['timestamp'].append(timestamp) #datetime.datetime.now()

    return data


"""
NOTES

I -> Index set of agents {1,..., Na}; Na -> Max Agent Number
J -> Index set of tasks {1,..., Nt}; Nt -> tasks set
Lt -> max lenght of the bundle

b_i -> bundle for task i
tao_i -> vector of task execution time for agent i
s_i -> communication timestamps with other agents carried by agent i
y_i -> list of winning bids for all tasks carried by agent i
z_i -> list of winning agents for all tasks carried by agent i


Nothe that each task L_t
"""