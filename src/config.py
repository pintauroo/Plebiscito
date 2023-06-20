'''
Configuration variables module
'''

import random
from src.node import node
import numpy as np
import datetime
import sys
from src.dataset import JobList
from src.topology import topo
import pandas as pd

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

#NN model
layer_number = 6 
min_layer_number = 3 #Min number of layers per node
max_layer_number = 3 #Max number of layers per node


dataset='./df_dataset.csv'

#Data analisys
job_list_instance = JobList(dataset, num_jobs_limit=req_number, seed=random.randint(1, 1000))
job_list_instance.select_jobs()
job_dict = {job['job_id']: job for job in job_list_instance.job_list} # to find jobs by id

#df_jobs = pd.read_csv(dataset)

#df_jobs = df_jobs.head(req_number)

#print('jobs number = ' + str(len(df_jobs)))

# print(job_list_instance.job_list[0])

# calculate total bw, cpu, and gpu needed
tot_gpu = 0 
tot_cpu = 0 
tot_bw = 0 
for d in job_list_instance.job_list:
    tot_gpu += d["num_gpu"] 
    tot_cpu += d["num_cpu"] 
    tot_bw += float(d['read_count'])

print('cpu: ' +str(tot_cpu))
print('gpu: ' +str(tot_gpu))
print('bw: ' +str(tot_bw))
if tot_gpu != 0:
    cpu_gpu_ratio = tot_cpu / tot_gpu
    print('cpu_gpu_ratio: ' +str(cpu_gpu_ratio))
else:
    print('cpu_gpu_ratio: <inf>')
    
node_gpu=float(tot_gpu/num_edges)*0.5
node_cpu=float(tot_cpu/num_edges)*0.5
node_bw=float(tot_bw/num_edges)
# node_bw=float(tot_bw/(num_edges*layer_number/min_layer_number))

# node_gpu = 10000000000
# node_cpu = 10000000000
node_bw = 10000000000

num_clients=len(set(d["user"] for d in job_list_instance.job_list))

#Build Topolgy
t = topo(func_name='ring_graph', max_bandwidth=node_bw, min_bandwidth=node_bw/2,num_clients=num_clients, num_edges=num_edges)

nodes = [node(row, random.randint(1,1000)) for row in range(num_edges)]


def message_data(job_id, user, num_gpu, num_cpu, duration, job_name, submit_time, gpu_type, num_inst, size, bandwidth):
    
    gpu = round(num_gpu / layer_number, 2)
    cpu = round(num_cpu / layer_number, 2)
    bw = round(float(bandwidth) / 2, 2)
    # bw = round(float(bandwidth) / min_layer_number, 2)

    NN_gpu = np.ones(layer_number) * gpu
    NN_cpu = np.ones(layer_number) * cpu
    NN_data_size = np.ones(layer_number) * bw
    
    data = {
        "job_id": int(),
        "user": int(),
        "num_gpu": int(),
        "num_cpu": int(),
        "duration": int(),
        "job_name": int(),
        "submit_time": int(),
        "gpu_type": int(),
        "num_inst": int(),
        "size": int(),
        "edge_id":int(),
        "NN_gpu": NN_gpu,
        "NN_cpu": NN_cpu,
        "NN_data_size": NN_data_size
        }


    data['edge_id']=None
    data['job_id']=job_id
    data['user']=user
    data['num_gpu']=num_gpu
    data['num_cpu']=num_cpu
    data['duration']=duration
    data['job_name']=job_name
    data['submit_time']=submit_time
    data['gpu_type']=gpu_type
    data['num_inst']=num_inst
    data['size']=size
    data['job_id']=job_id

    return data
