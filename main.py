from multiprocessing import Process, Event, Manager, JoinableQueue
import src.config as c
import src.utils as u
import src.jobs_handler as job
import time
import sys
import time
import logging
import signal
import os
import pandas as pd
from src.jobs_handler import *
import src.plot as plot

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

TRACE = 5
DEBUG = logging.DEBUG
INFO = logging.INFO

main_pid = ""

def sigterm_handler(signum, frame):
    # Perform cleanup actions here
    # ...    
    global main_pid
    if os.getpid() == main_pid:
        print("SIGINT received. Performing cleanup...")
        for t in nodes_thread:
            t.terminate()
            t.join()    
            
        # ...
        print("All processes have been gracefully teminated.")
        sys.exit(0)  # Exit gracefully


def setup_environment():
    # Register the SIGTERM signal handler
    signal.signal(signal.SIGINT, sigterm_handler)
    global main_pid
    main_pid = os.getpid()

    logging.addLevelName(TRACE, "TRACE")
    logging.basicConfig(filename='debug.log', level=TRACE, format='%(message)s', filemode='w')
    # logging.basicConfig(filename='debug.log', level=TRACE, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    logging.debug('Clients number: ' + str(c.num_clients))
    logging.debug('Edges number: ' + str(c.num_edges))
    logging.debug('Requests number: ' + str(c.req_number))
    
    
def setup_nodes(nodes_thread, terminate_processing_events, start_events, use_queue, manager, return_val, queues, progress_bid_events):
    for i in range(c.num_edges):
        q = JoinableQueue()
        e = Event() 
        
        queues.append(q)
        use_queue.append(e)
        
        e.set()

    #Generate threads for each node
    for i in range(c.num_edges):
        e = Event() 
        e2 = Event()
        e3 = Event()
        return_dict = manager.dict()
        
        c.nodes[i].set_queues(queues, use_queue)
        
        p = Process(target=c.nodes[i].work, args=(e, e2, e3, return_dict))
        nodes_thread.append(p)
        return_val.append(return_dict)
        terminate_processing_events.append(e)
        start_events.append(e2)
        e3.clear()
        progress_bid_events.append(e3)
        
        p.start()
        
    for e in start_events:
        e.wait()
        
    print("All the processes started.")

def print_final_results(start_time):
    logging.info('Tot messages: '+str(c.counter))
    print('Tot messages: '+str(c.counter))
    logging.info("Run time: %s" % (time.time() - start_time))
    print("Run time: %s" % (time.time() - start_time))

def collect_node_results(return_val, jobs, exec_time, time_instant):
    c.counter = 0
    c.job_count = {}
    
    if time_instant != 0:
        for v in return_val: 
            c.nodes[v["id"]].bids = v["bids"]
            for key in v["counter"]:
                if key not in c.job_count:
                    c.job_count[key] = 0
                c.job_count[key] += v["counter"][key]
                c.counter += v["counter"][key]
            c.nodes[v["id"]].updated_cpu = v["updated_cpu"]
            c.nodes[v["id"]].updated_gpu = v["updated_gpu"]
            c.nodes[v["id"]].updated_bw = v["updated_bw"]

        for j in job_ids:
            for i in range(c.num_edges):
                if j not in c.nodes[i].bids:
                    print('???????')
                    print(c.nodes[i].bids)
                    print(str(c.nodes[i].id) + ' ' +str(j))
                logging.info(
                    str(c.nodes[i].bids[j]['auction_id']) + 
                    ' id: ' + str(c.nodes[i].id) + 
                    ' complete: ' + str(c.nodes[i].bids[j]['complete']) +
                    ' complete_tiemstamp' + str(c.nodes[i].bids[j]['complete_timestamp'])+
                    ' count' + str(c.nodes[i].bids[j]['count'])+
                    ' used_tot_gpu: ' + str(c.nodes[i].initial_gpu)+' - ' +str(c.nodes[i].updated_gpu)  + ' = ' +str(c.nodes[i].initial_gpu - c.nodes[i].updated_gpu) + 
                    ' used_tot_cpu: ' + str(c.nodes[i].initial_cpu)+' - ' +str(c.nodes[i].updated_cpu)  + ' = ' +str(c.nodes[i].initial_cpu - c.nodes[i].updated_cpu) + 
                    ' used_tot_bw: '  + str(c.nodes[i].initial_bw)+' - '  +str(c.nodes[i].updated_bw) + ' = '  +str(c.nodes[i].initial_bw  - c.nodes[i].updated_bw))

    return u.calculate_utility(c.nodes, c.num_edges, c.counter, exec_time, c.req_number, jobs, c.a, time_instant)

def terminate_node_processing(nodes_thread, events):
    for e in events:
        e.set()
        
    # Block until all tasks are done.
    for nt in nodes_thread:
        nt.join()

if __name__ == "__main__":
    
    setup_environment()

    nodes_thread = []
    terminate_processing_events = []
    start_events = []
    progress_bid_events = []
    use_queue = []
    manager = Manager()
    return_val = []
    queues = []

    setup_nodes(nodes_thread, terminate_processing_events, start_events, use_queue, manager, return_val, queues, progress_bid_events)
    
    simulation_end = job.get_simulation_end_time_instant(c.dataset)

    job_ids=[]
    jobs = pd.DataFrame()
    running_jobs = pd.DataFrame()
    processed_jobs = pd.DataFrame()
    print(f"Total number of jobs: {len(c.dataset)}")
    
    start_time = time.time()
    collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, 0)
    
    time_instant = 1
    while True:
        start_time = time.time()
        
        print()
        print(f"Performing simulation at time {time_instant}.")
        new_jobs = job.select_jobs(c.dataset, time_instant)
        
        print(f"\tAdding {len(new_jobs)} to the job list for time instant {time_instant}.")
        jobs = pd.concat([jobs, new_jobs], sort=False)
        
        print(f"\tCurrent lenght of the job queue: {len(jobs)}.")
        jobs = job.schedule_jobs(jobs)
        jobs_to_submit = job.create_job_batch(jobs, 10)
        
        print(f"\tSubmitted jobs: {len(jobs_to_submit)}. \n\tJobs remaining in queue: {len(jobs)}.")
        
        if len(jobs_to_submit) > 0:                   
            job.dispatch_job(jobs_to_submit, queues)

            for e in progress_bid_events:
                e.wait()
                e.clear() 
            
        exec_time = time.time() - start_time
            
        assigned_jobs, unassigned_jobs = collect_node_results(return_val, jobs_to_submit, exec_time, time_instant)
                   
        job.assign_job_start_time(assigned_jobs, time_instant)
        
        print(f"\tAdding {len(unassigned_jobs)} unscheduled job(s) to the list.")
        jobs = pd.concat([jobs, unassigned_jobs], sort=False)  
        running_jobs = pd.concat([running_jobs, assigned_jobs], sort=False)
        processed_jobs = pd.concat([processed_jobs,assigned_jobs], sort=False)
        
        jobs_to_unallocate, running_jobs = job.extract_completed_jobs(running_jobs, time_instant)
        
        if len(jobs_to_unallocate) > 0:
            for _, j in jobs_to_unallocate.iterrows():
                data = message_data(
                        j['job_id'],
                        j['user'],
                        j['num_gpu'],
                        j['num_cpu'],
                        j['duration'],
                        j['bw'],
                        deallocate=True
                    )
                for q in queues:
                    q.put(data)

            for e in progress_bid_events:
                e.wait()
                e.clear()

        if len(unassigned_jobs) > 0:
            for _, j in unassigned_jobs.iterrows():
                data = message_data(
                        j['job_id'],
                        j['user'],
                        j['num_gpu'],
                        j['num_cpu'],
                        j['duration'],
                        j['bw'],
                        deallocate=True
                    )
                
                for q in queues:
                    q.put(data)
            
            for e in progress_bid_events:
                e.wait()
                e.clear()
            
        print(f"\tUnallocated {len(jobs_to_unallocate)} jobs.")
        
        time_instant += 1
        
        if len(processed_jobs) == len(c.dataset) and len(running_jobs) == 0 and len(jobs) == 0:
            break

    collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, time_instant)
    
    terminate_node_processing(nodes_thread, terminate_processing_events)
    
    processed_jobs.to_csv("jobs_report.csv")

    if c.use_net_topology:
        c.network_t.dump_to_file(c.filename, c.a)

    print_final_results(start_time)
    
    plot.plot_all(c.num_edges, c.filename, c.job_count, "plot")
