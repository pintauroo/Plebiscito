import time
import src.config as c
import numpy as np

def dispatch_job(dataset, queues):
    job_ids=[]

    if c.use_net_topology:
        timeout = 3 # don't change it
    else:
        timeout = 0.01

    for index, job in dataset.iterrows():
        time.sleep(timeout)
        data = message_data(
                    job['job_id'],
                    job['user'],
                    job['num_gpu'],
                    job['num_cpu'],
                    job['duration'],
                    # job['job_name'],
                    # job['submit_time'],
                    # job['gpu_type'],
                    # job['num_inst'],
                    # job['size'],
                    job['bw']
                )
        #print(data)
        job_ids.append(job['job_id'])
        for q in queues:
            q.put(data)
    #time.sleep(0.1)
    return job_ids




# def message_data(job_id, user, num_gpu, num_cpu, duration, job_name, submit_time, gpu_type, num_inst, size, bandwidth):
def message_data(job_id, user, num_gpu, num_cpu, duration, bandwidth):
    
    gpu = round(num_gpu / c.layer_number, 2)
    cpu = round(num_cpu / c.layer_number, 2)
    bw = round(float(bandwidth) / 2, 2)
    # bw = round(float(bandwidth) / min_layer_number, 2)

    NN_gpu = np.ones(c.layer_number) * gpu
    NN_cpu = np.ones(c.layer_number) * cpu
    NN_data_size = np.ones(c.layer_number) * bw
    
    data = {
        "job_id": int(),
        "user": int(),
        "num_gpu": int(),
        "num_cpu": int(),
        "duration": int(),
        # "job_name": int(),
        # "submit_time": int(),
        # "gpu_type": int(),
        # "num_inst": int(),
        # "size": int(),
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
    # data['job_name']=job_name
    # data['submit_time']=submit_time
    # data['gpu_type']=gpu_type
    # data['num_inst']=num_inst
    # data['size']=size
    data['job_id']=job_id

    return data
