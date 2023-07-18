'''
Configuration variables module
'''

import random
from src.node import node
import numpy as np
import datetime
import sys
from src.dataset import JobList
from src.network_topology import NetworkTopology, TopologyType
from src.topology import topo
import pandas as pd
from multiprocessing.managers import SyncManager
from src.dataset_builder import generate_dataset

class MyManager(SyncManager): pass

MyManager.register('NetworkTopology', NetworkTopology)

# Alibaba datacenter servers
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
#type   |cap_cpu    |cap_gpu    |count      |tot_cpu    |tot_gpu    |cpu/gpu_ratio	|relative_occurrence    |relative_occurrence_cpu    |relative_occurrence_gpu
#0	    |64	        |2	        |798        |51072	    |1596	    |32.0	        |42.066421	            |32.618026	                |23.672501
#1	    |96	        |8	        |519	    |49824	    |4152	    |12.0	        |27.358988	            |31.820969	                |61.584100
#2	    |96	        |2          |497	    |47712	    |994	    |48.0	        |26.199262	            |30.472103	                |14.743400
#3	    |96	        |0	        |83	        |7968	    |0	        |inf	        |4.375329	            |5.088903	                |0.000000
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

counter = 0 #Messages counter
req_number = int(sys.argv[1]) #Total number of requests
a = float(sys.argv[2]) #Multiplicative factor
num_edges = int(sys.argv[3]) #Nodes number 
filename = str(sys.argv[4])

seed = None
if len(sys.argv) == 6:
    seed = int(sys.argv[5]) + 1
    random.seed(seed)

enable_logging = False 
use_net_topology = False

# dataset='./df_dataset.csv'

dataset = generate_dataset(req_number)

#Data analisys
# job_list_instance = JobList(dataset, num_jobs_limit=req_number)
# job_list_instance.select_jobs()
# job_dict = {job['job_id']: job for job in job_list_instance.job_list} # to find jobs by id

#df_jobs = pd.read_csv(dataset)

#df_jobs = df_jobs.head(req_number)

#print('jobs number = ' + str(len(df_jobs)))

# print(job_list_instance.job_list[0])

# calculate total bw, cpu, and gpu needed
tot_gpu = 0 
tot_cpu = 0 
tot_bw = 0 
for index, d in dataset.iterrows():
    tot_gpu += d["num_gpu"] 
    tot_cpu += d["num_cpu"] 
    tot_bw += float(d['bw'])


node_cpu = dataset['num_cpu'].quantile(0.75) * req_number / num_edges *2
node_gpu = dataset['num_gpu'].quantile(0.75) * req_number / num_edges *2
node_bw =  dataset['bw'].quantile(0.75) * req_number / num_edges *40000

print('cpu: ' +str(tot_cpu))
print('gpu: ' +str(tot_gpu))
print('bw: ' +str(tot_bw))
if tot_gpu != 0:
    cpu_gpu_ratio = tot_cpu / tot_gpu
    print('cpu_gpu_ratio: ' +str(cpu_gpu_ratio))
else:
    print('cpu_gpu_ratio: <inf>')
    
# node_gpu=float(tot_gpu/num_edges)
# node_cpu=float(tot_cpu/num_edges)
# node_bw=float(tot_bw/num_edges)
# node_bw=float(tot_bw/(num_edges*layer_number/min_layer_number))

# node_gpu = 10000000000
# node_cpu = 10000000000
# node_bw = 10000000000

num_clients=3

manager = MyManager()
manager.start()

#Build Topolgy
t = topo(func_name='ring_graph', max_bandwidth=node_bw, min_bandwidth=node_bw/2,num_clients=num_clients, num_edges=num_edges)
# network_t = manager.NetworkTopology(num_edges, node_bw, node_bw, group_number=3, seed=4, topology_type=TopologyType.FAT_TREE)



nodes = [node(row, random.randint(1,1000), None, use_net_topology=use_net_topology) for row in range(num_edges)]
