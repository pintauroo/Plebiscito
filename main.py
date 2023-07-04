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


TRACE = 5
DEBUG = logging.DEBUG
INFO = logging.INFO

def sigterm_handler(signum, frame):
    # Perform cleanup actions here
    # ...    
    if os.getpid() == main_pid:
        print("SIGINT received. Performing cleanup...")
        for t in nodes_thread:
            t.terminate()
            t.join()    
            
        # ...
        print("All processes have been gracefully teminated.")
        sys.exit(0)  # Exit gracefully


# Register the SIGTERM signal handler
signal.signal(signal.SIGINT, sigterm_handler)
main_pid = os.getpid()

logging.addLevelName(TRACE, "TRACE")
logging.basicConfig(filename='debug.log', level=TRACE, format='%(message)s', filemode='w')
# logging.basicConfig(filename='debug.log', level=TRACE, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

logging.debug('Clients number: ' + str(c.num_clients))
logging.debug('Edges number: ' + str(c.num_edges))
logging.debug('Requests number: ' + str(c.req_number))

nodes_thread = []
events = []
start_event = []
use_queue = []
manager = Manager()
return_val = []
queues = []


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
    return_dict = manager.dict()
    
    c.nodes[i].set_queues(queues, use_queue)
    
    p = Process(target=c.nodes[i].work, args=(e, e2, return_dict))
    nodes_thread.append(p)
    return_val.append(return_dict)
    events.append(e)
    start_event.append(e2)
    
    p.start()
    
for e in start_event:
    e.wait()
    
print("All the processes started.")

start_time = time.time()

job_ids=[]
print('request_number: ' +str(c.req_number))

job_ids = job.dispatch_job(c.dataset, queues)

    
for e in events:
    e.set()
    
# Block until all tasks are done.
for nt in nodes_thread:
   nt.join()

#Calculate stats
exec_time = time.time() - start_time


time.sleep(1) # Wait time nexessary to wait all threads to finish 

for v in return_val: 
    c.nodes[v["id"]].bids = v["bids"]
    c.counter += v["counter"]
    c.nodes[v["id"]].updated_cpu = v["updated_cpu"]
    c.nodes[v["id"]].updated_gpu = v["updated_gpu"]
    c.nodes[v["id"]].updated_bw = v["updated_bw"]

for j in job_ids:
    #print('\n')
    #print(j)
    logging.info("RESULTS req:" +str(j))
    for i in range(c.num_edges):
        if j not in c.nodes[i].bids:
            print('???????')
            print(c.nodes[i].bids)
            print(str(c.nodes[i].id) + ' ' +str(j))
        # print('ktm')
        logging.info(
            str(c.nodes[i].bids[j]['auction_id']) + 
            ' id: ' + str(c.nodes[i].id) + 
            ' used_tot_gpu: ' + str(c.nodes[i].initial_gpu)+' - ' +str(c.nodes[i].updated_gpu)  + ' = ' +str(c.nodes[i].initial_gpu - c.nodes[i].updated_gpu) + 
            ' used_tot_cpu: ' + str(c.nodes[i].initial_cpu)+' - ' +str(c.nodes[i].updated_cpu)  + ' = ' +str(c.nodes[i].initial_cpu - c.nodes[i].updated_cpu) + 
            ' used_tot_bw: '  + str(c.nodes[i].initial_bw)+' - '  +str(c.nodes[i].updated_bw) + ' = '  +str(c.nodes[i].initial_bw  - c.nodes[i].updated_bw))

u.calculate_utility(c.nodes, c.num_edges, c.counter, exec_time, c.req_number, job_ids, c.a)

if c.use_net_topology:
    c.network_t.dump_to_file(c.filename)

logging.info('Tot messages: '+str(c.counter))
print('Tot messages: '+str(c.counter))
logging.info("Run time: %s" % (time.time() - start_time))
print("Run time: %s" % (time.time() - start_time))
