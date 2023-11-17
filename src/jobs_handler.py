import random
import sys
import time
import numpy as np
import pandas as pd
from src.config import SchedulingAlgorithm

def assign_job_start_time(dataset: pd.DataFrame, time_instant):
    dataset.replace(-1, time_instant, inplace=True)
    return dataset
        
def extract_completed_jobs(dataset, time_instant):
    if len(dataset) == 0:
        return dataset, dataset
    ret = dataset[dataset["exec_time"] + dataset["duration"] < time_instant]

    dataset = dataset.drop(dataset[dataset["exec_time"] + dataset["duration"] < time_instant].index)
    
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

def dispatch_job(dataset, queues, use_net_topology=False):        
    if use_net_topology:
        timeout = 1 # don't change it
    else:
        timeout = 0.1

    for _, job in dataset.iterrows():
        time.sleep(timeout)
        data = message_data(
                    job['job_id'],
                    job['user'],
                    job['num_gpu'],
                    job['num_cpu'],
                    job['duration'],
                    job['bw'],
                    job['gpu_type']
                )
        
        for q in queues:
            q.put(data)

def get_simulation_end_time_instant(dataset):
    return dataset['submit_time'].max() + dataset['duration'].max()

def message_data(job_id, user, num_gpu, num_cpu, duration, bandwidth, gpu_type, deallocate=False):
    
    random.seed(job_id)
    np.random.seed(int(job_id))
    
    layer_number = random.choice([2, 4, 6, 8, 10])
    
    #print(bandwidth)
    
    # gpu = round(num_gpu / layer_number, 6)
    # cpu = round(num_cpu / layer_number, 6)
    # bw = round(float(bandwidth) / 2, 6)
    # bw = round(float(bandwidth) / min_layer_number, 2)

    # use numpy to create an array of random numbers with length equal to the number of layers. As a constraint, the sum of the array must be equal to the number of GPUs
    NN_gpu = np.random.dirichlet(np.ones(layer_number), size=1)[0] * num_gpu
    NN_cpu = np.random.dirichlet(np.ones(layer_number), size=1)[0] * num_cpu
    NN_data_size = np.random.dirichlet(np.ones(layer_number), size=1)[0] * bandwidth
    
    # NN_gpu = np.ones(layer_number) * gpu
    # NN_cpu = np.ones(layer_number) * cpu
    #NN_data_size = np.ones(layer_number) * bw

    max_layer_bid = random.choice([4, 6, 8, 10])
    bundle_size = 2
    
    data = {
        "job_id": int(),
        "user": int(),
        "num_gpu": int(),
        "num_cpu": int(),
        "duration": int(),
        "N_layer": len(NN_gpu),
        "N_layer_min": 1, # Do not change!! This could be either 1 or = to N_layer_max
        "N_layer_max": max_layer_bid,
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
