import pandas as pd
import numpy as np
import os
import random
import math

path = os.getcwd()
dataset = path + '/df_dataset.csv'

def generate_dataset(entries_num = 100):
    """
    Generate a new dataset with the specified number of entries.
    
    Args:
    - entries_num (int): The number of entries to generate.
    
    Returns:
    - pandas.DataFrame: A new dataset with the specified number of entries.
    """
    df = pd.read_csv(dataset)
    df = df[(df['num_gpu'] > 0)]
    
    random.seed(1) 
    
    new_dataset = []
    
    for i in range(entries_num):
        job_id = df["job_id"].iloc[i]
        user = df["user"].iloc[i]
        cpu = df["num_cpu"].iloc[i]/df["num_pod"].iloc[i]
        gpu = df["num_gpu"].iloc[i]/df["num_pod"].iloc[i]
        bw = df["write_count"].iloc[i]/df["num_pod"].iloc[i]
        duration = math.ceil(df["duration"].iloc[i]/10)
        #arrival_time = math.ceil(df["submit_time"].iloc[i]/10)
        
        # to align with the original paper from alibaba
        arrival_time = i // 10 + 1
        gpu_type = df["gpu_type"].iloc[i]
        
        new_dataset.append({'job_id': job_id, 'user': user, 'num_cpu': cpu, 'num_gpu': gpu, 'bw': bw, 'duration': duration, 'arrival_time': arrival_time, "exec_time": -1, 'deadline': arrival_time + math.ceil(duration * (1 + random.random())), 'priority': random.randint(1, 4), 'count': 1, "gpu_type": gpu_type})

    new_dataset = pd.DataFrame(new_dataset)

    return new_dataset

def generate_dataset_old(entries_num = 100):
    df = pd.read_csv(dataset)
    df = df[(df['num_cpu'] <= 10000) & (df['num_gpu'] <= 10000)]

    counts = df['count'].tolist()
    cpu_values = df['num_cpu'].tolist()
    gpu_values = df['num_gpu'].tolist()
    duration_median = df['duration_median'].tolist()
    bandwidth_median = df['bandwidth_median'].tolist()

    # Convert counts to a numpy array and normalize it
    counts = np.array(counts)
    probabilities = counts / np.sum(counts)

    # Generate a new dataset with entries proportional to the counts
    selected_entries = np.random.choice(range(len(counts)), size=entries_num, p=probabilities)

    new_dataset = []
    i = 0
    for entry_idx in selected_entries:
        cpu = cpu_values[entry_idx]
        gpu = gpu_values[entry_idx]
        bw = bandwidth_median[entry_idx]
        duration= duration_median[entry_idx]
        
        duration = random.randint(1, 15)
        arrival_time = random.randint(1, 30)
        
        # TODO: the arrival time should be modeled based on the dataset, instead of a random value
        new_dataset.append({'job_id': i, 'user': i, 'num_cpu': cpu, 'num_gpu': gpu, 'bw': bw, 'duration': duration, 'arrival_time': arrival_time, "exec_time": -1, 'deadline': arrival_time + duration + random.randint(1, 8), 'priority': random.randint(1, 4), 'count': 1})
        i+=1
        
    new_dataset = pd.DataFrame(new_dataset)

    return new_dataset




