import random
import sys
import time
import numpy as np
import pandas as pd
from src.config import SchedulingAlgorithm

def assign_job_start_time(dataset: pd.DataFrame, time_instant):
    dataset.replace(-1, time_instant, inplace=True)
    return dataset
        
def extract_completed_jobs(dataset: pd.DataFrame, time_instant):
    if len(dataset) == 0:
        return dataset, dataset
    
    condition = dataset.exec_time + dataset.duration < time_instant
    ret = dataset[condition]
    
    if len(ret) > 0:
        dataset = dataset[~condition]
    
    return ret, dataset

def select_jobs(dataset, time_instant):
    return dataset[dataset['submit_time'] == time_instant]

def create_job_batch(dataset, batch_size):
    ret = dataset.head(batch_size)
    dataset.drop(index=dataset.index[:batch_size], axis=0, inplace=True)
    return ret

def schedule_jobs(jobs: pd.DataFrame, scheduling_algorithm: SchedulingAlgorithm):
    if scheduling_algorithm == SchedulingAlgorithm.FIFO:
        return jobs.sort_values(by=["submit_time"])
    elif scheduling_algorithm == SchedulingAlgorithm.SDF:
        return jobs.sort_values(by=["duration"])

def dispatch_job(dataset: pd.DataFrame, queues, logical_topology, failure_nodes, use_net_topology=False, split=True):        
    if use_net_topology:
        timeout = 1 # don't change it
    else:
        timeout = 0.2

    count = 0
    failure_time = random.randint(1, len(dataset) - 2)
    # for f in failure_nodes:
    #     logical_topology.disconnect_node(f)
    failure = False    
    for _, job in dataset.iterrows():
        data = message_data(
                    job['job_id'],
                    job['user'],
                    job['num_gpu'],
                    job['num_cpu'],
                    job['duration'],
                    job['bw'],
                    job['gpu_type'],
                    deallocate=False,
                    split=split
                )
        
        for i, q in enumerate(queues):
            if i in failure_nodes and failure:
                continue
            q.put(data)
            
        time.sleep(timeout)

        if count == failure_time:
            for f in failure_nodes:
                logical_topology.disconnect_node(f)
            failure = True
        
        count += 1
        

def get_simulation_end_time_instant(dataset):
    return dataset['submit_time'].max() + dataset['duration'].max()

def message_data(job_id, user, num_gpu, num_cpu, duration, bandwidth, gpu_type, deallocate=False, split=True):

    # use numpy to create an array of random numbers with length equal to the number of layers. As a constraint, the sum of the array must be equal to the number of GPUs
    NN_gpu = [num_gpu]
    NN_cpu = [num_cpu]
    NN_data_size = [bandwidth]

    bundle_size = 2
    
    data = {
        "job_id": int(),
        "user": int(),
        "num_gpu": int(),
        "num_cpu": int(),
        "duration": int(),
        "N_layer": len(NN_gpu),
        "N_layer_min": 1, # Do not change!! This could be either 1 or = to N_layer_max
        "N_layer_max": 1,
        "N_layer_bundle": bundle_size, 
        "edge_id":int(),
        "NN_gpu": NN_gpu,
        "NN_cpu": NN_cpu,
        "NN_data_size": NN_data_size,
        "gpu_type": gpu_type,
        }

    data['edge_id']=None
    data['job_id']=job_id
    data['user']=user
    data['num_gpu']=num_gpu
    data['num_cpu']=num_cpu
    data['duration']=duration
    data['job_id']=job_id
    
    if deallocate:
        data["unallocate"] = True

    return data
