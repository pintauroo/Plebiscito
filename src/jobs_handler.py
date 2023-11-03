import random
import sys
import time
import src.config as c
import numpy as np

def assign_job_start_time(dataset, time_instant):
    for _, row in dataset.iterrows():
        row['exec_time'] = time_instant
        
def extract_completed_jobs(dataset, time_instant):
    if len(dataset) == 0:
        return dataset, dataset
    ret = dataset[dataset["exec_time"] + dataset["duration"] <= time_instant]
    # print("ret")
    # print(ret)

    dataset = dataset.drop(dataset[dataset["exec_time"] + dataset["duration"] <= time_instant].index)
    # print('DATASET')
    # print(dataset)
    
    return ret, dataset

def select_jobs(dataset, time_instant):
    return dataset[dataset['arrival_time'] == time_instant]

def create_job_batch(dataset, batch_size):
    ret = dataset.head(batch_size)
    #print(len(ret))
    dataset.drop(index=dataset.index[:batch_size], axis=0, inplace=True)
    return ret

def schedule_jobs(jobs):
    return jobs

def dispatch_job(dataset, queues):        
    if c.use_net_topology:
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
                    job['bw']
                )
        
        for q in queues:
            q.put(data)

def get_simulation_end_time_instant(dataset):
    return dataset['arrival_time'].max() + dataset['duration'].max()

def message_data(job_id, user, num_gpu, num_cpu, duration, bandwidth, deallocate=False):
    
    random.seed(job_id)
    np.random.seed(int(job_id))
    
    layer_number = random.choice([2, 4, 6, 8, 10])
    
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
        "NN_data_size": NN_data_size
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
