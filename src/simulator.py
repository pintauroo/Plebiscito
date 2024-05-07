import math
from multiprocessing.managers import SyncManager
from multiprocessing import Process, Event, Manager, JoinableQueue
import random
import time
import pandas as pd
import signal
import logging
import os
import sys

from src.network_topology import NetworkTopology
from src.topology import topo as LogicalTopology
from src.network_topology import  TopologyType
from src.utils import generate_gpu_types
from src.node import node
from src.config import Utility, DebugLevel, SchedulingAlgorithm
import src.jobs_handler as job
import src.utils as utils
import src.plot as plot
from src.jobs_handler import message_data

class MyManager(SyncManager): pass

main_pid = ""
nodes_thread = []

def sigterm_handler(signum, frame):
    """Handles the SIGTERM signal by performing cleanup actions and gracefully terminating all processes."""
    # Perform cleanup actions here
    # ...    
    global main_pid
    if os.getpid() == main_pid:
        print("SIGINT received. Performing cleanup...")
        for t in nodes_thread:
            t.terminate()
            t.join()    
            
        print("All processes have been gracefully teminated.")
        sys.exit(0)  # Exit gracefully    

class Simulator_Plebiscito:
    def __init__(self, filename: str, n_nodes: int, node_bw: int, n_jobs: int, n_client: int, enable_logging: bool, use_net_topology: bool, progress_flag: bool, dataset: pd.DataFrame, alpha: float, utility: Utility, debug_level: DebugLevel, scheduling_algorithm: SchedulingAlgorithm, decrement_factor: float, split: bool, n_failures, edge_to_add, logical_topology_name, probability=0) -> None:   
        self.filename = filename + "_" + utility.name + "_" + scheduling_algorithm.name + "_" + str(decrement_factor)
        if split:
            self.filename = self.filename + "_split"
        else:
            self.filename = self.filename + "_nosplit"
            
        self.filename = self.filename + "_" + str(n_failures) + "_failures"
            
        self.n_nodes = n_nodes
        self.node_bw = node_bw
        self.n_jobs = n_jobs
        self.n_client = n_client
        self.enable_logging = enable_logging
        self.use_net_topology = use_net_topology
        self.progress_flag = progress_flag
        self.dataset = dataset
        self.debug_level = debug_level
        self.counter = 0
        self.alpha = alpha
        self.scheduling_algorithm = scheduling_algorithm
        self.decrement_factor = decrement_factor
        self.split = split
        self.n_failures = n_failures
        self.failure_nodes = random.sample(range(1, n_nodes), int(n_failures * n_nodes))
        self.edge_to_add = edge_to_add
        
        self.job_count = {}
        
        # create a suitable network topology for multiprocessing 
        MyManager.register('NetworkTopology', NetworkTopology)
        MyManager.register('LogicalTopology', LogicalTopology)
        self.physical_network_manager = MyManager()
        self.physical_network_manager.start()
        self.logical_network_manager = MyManager()
        self.logical_network_manager.start()
        
        #Build Topolgy
        self.t = self.logical_network_manager.LogicalTopology(func_name=logical_topology_name, max_bandwidth=node_bw, min_bandwidth=node_bw/2,num_clients=n_client, num_edges=n_nodes, edge_to_add=edge_to_add, probability=probability)
        self.network_t = self.physical_network_manager.NetworkTopology(n_nodes, node_bw, node_bw, group_number=4, seed=4, topology_type=TopologyType.FAT_TREE)
        
        self.nodes = []
        self.gpu_types = generate_gpu_types(n_nodes)

        for i in range(n_nodes):
            self.nodes.append(node(i, self.network_t, self.gpu_types[i], utility, alpha, enable_logging, self.t, n_nodes, progress_flag, use_net_topology=use_net_topology, decrement_factor=decrement_factor))
            
        # Set up the environment
        self.setup_environment()
            
    def setup_environment(self):
        """
        Set up the environment for the program.

        Registers the SIGTERM signal handler, sets the main process ID, and initializes logging.
        """
        
        signal.signal(signal.SIGINT, sigterm_handler)
        global main_pid
        main_pid = os.getpid()

        logging.addLevelName(DebugLevel.TRACE, "TRACE")
        logging.basicConfig(filename='debug.log', level=self.debug_level.value, format='%(message)s', filemode='w')

        logging.debug('Clients number: ' + str(self.n_client))
        logging.debug('Edges number: ' + str(self.n_nodes))
        logging.debug('Requests number: ' + str(self.n_jobs))
        
    def setup_nodes(self, terminate_processing_events, start_events, use_queue, manager, return_val, queues, progress_bid_events):
        """
        Sets up the nodes for processing. Generates threads for each node and starts them.
        
        Args:
        terminate_processing_events (list): A list of events to terminate processing for each node.
        start_events (list): A list of events to start processing for each node.
        use_queue (list): A list of events to indicate if a queue is being used by a node.
        manager (multiprocessing.Manager): A multiprocessing manager object.
        return_val (list): A list of return values for each node.
        queues (list): A list of queues for each node.
        progress_bid_events (list): A list of events to indicate progress of bid processing for each node.
        """
        global nodes_thread
        
        for i in range(self.n_nodes):
            q = JoinableQueue()
            e = Event() 
            
            queues.append(q)
            use_queue.append(e)
            
            e.set()

        #Generate threads for each node
        for i in range(self.n_nodes):
            e = Event() 
            e2 = Event()
            e3 = Event()
            return_dict = manager.dict()
            
            self.nodes[i].set_queues(queues, use_queue)
            
            p = Process(target=self.nodes[i].work, args=(e, e2, e3, return_dict))
            nodes_thread.append(p)
            return_val.append(return_dict)
            terminate_processing_events.append(e)
            start_events.append(e2)
            e3.clear()
            progress_bid_events.append(e3)
            
            p.start()
            
        for e in start_events:
            e.wait()
    
    def collect_node_results(self, return_val, jobs: pd.DataFrame, exec_time, time_instant, failure_nodes, save_on_file):
        """
        Collects the results from the nodes and updates the corresponding data structures.
        
        Args:
        - return_val: list of dictionaries containing the results from each node
        - jobs: list of job objects
        - exec_time: float representing the execution time of the jobs
        - time_instant: int representing the current time instant
        
        Returns:
        - float representing the utility value calculated based on the updated data structures
        """
        if time_instant != 0:
            for _, j in jobs.iterrows():
                self.job_count[j["job_id"]] = 0
                for v in return_val: 
                    nodeId = v["id"]
                    if j["job_id"] not in v["bids"]:
                        continue
                    
                    self.nodes[nodeId].bids[j["job_id"]] = v["bids"][j["job_id"]]                        
                    self.job_count[j["job_id"]] += v["counter"][j["job_id"]]

            for v in return_val: 
                nodeId = v["id"]
                self.nodes[nodeId].updated_cpu = v["updated_cpu"]
                self.nodes[nodeId].updated_gpu = v["updated_gpu"]
                self.nodes[nodeId].updated_bw = v["updated_bw"]
                self.nodes[nodeId].gpu_type = v["gpu_type"]
        
        return utils.calculate_utility(self.nodes, self.n_nodes, self.counter, exec_time, self.n_jobs, jobs, self.alpha, time_instant, self.use_net_topology, self.filename, self.network_t, self.gpu_types, save_on_file, failure_nodes)
    
    def terminate_node_processing(self, events):
        global nodes_thread
        
        for e in events:
            e.set()
            
        # Block until all tasks are done.
        for nt in nodes_thread:
            nt.join()
            
    def clear_screen(self):
        # Function to clear the terminal screen
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_simulation_values(self, time_instant, processed_jobs, queued_jobs: pd.DataFrame, running_jobs, batch_size):
        print()
        print("Infrastructure info")
        print(f"Number of nodes: {self.n_nodes}")
        
        for t in set(self.gpu_types):
            count = 0
            for i in self.gpu_types:
                if i == t:
                    count += 1
            print(f"Number of {t.name} GPU nodes: {count}")
        
        print()
        print("Performing simulation at time " + str(time_instant) + ".")
        print(f"# Jobs assigned: \t\t{processed_jobs}/{len(self.dataset)}")
        print(f"# Jobs currently in queue: \t{len(queued_jobs)}")
        print(f"# Jobs currently running: \t{running_jobs}")
        print(f"# Current batch size: \t\t{batch_size}")
        print()
        print("Jobs in queue stats for gpu type:")
        if len(queued_jobs) == 0:
            print("<no jobs in queue>")
        else:
            print(queued_jobs["gpu_type"].value_counts().to_dict())

            
    def print_simulation_progress(self, time_instant, job_processed, queued_jobs, running_jobs, batch_size):
        self.clear_screen()
        self.print_simulation_values(time_instant, job_processed, queued_jobs, running_jobs, batch_size) 
        
    def deallocate_jobs(self, progress_bid_events, queues, jobs_to_unallocate):
        if len(jobs_to_unallocate) > 0:
            for _, j in jobs_to_unallocate.iterrows():
                data = message_data(
                            j['job_id'],
                            j['user'],
                            j['num_gpu'],
                            j['num_cpu'],
                            j['duration'],
                            j['bw'],
                            j['gpu_type'],
                            deallocate=True
                        )
                for q in queues:
                    q.put(data)

            for e in progress_bid_events:
                e.wait()
                e.clear()       
 
    def run(self):
        # Set up nodes and related variables
        global nodes_thread
        terminate_processing_events = []
        start_events = []
        progress_bid_events = []
        use_queue = []
        manager = Manager()
        return_val = []
        queues = []
        self.setup_nodes(terminate_processing_events, start_events, use_queue, manager, return_val, queues, progress_bid_events)

        # Initialize job-related variables
        self.job_ids=[]

        # Collect node results
        start_time = time.time()
        #self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, 0, save_on_file=True)
        
        time_instant = 1
        
        job.dispatch_job(self.dataset, queues, self.t, self.failure_nodes, self.use_net_topology, self.split, self.failure_nodes, progress_bid_events)

        exec_time = time.time() - start_time
        
        # Collect node results
        a_jobs, u_jobs = self.collect_node_results(return_val, self.dataset, exec_time, time_instant, self.failure_nodes, save_on_file=False)
            
        # Terminate node processing
        self.terminate_node_processing(terminate_processing_events)

        # Save processed jobs to CSV
        jobs = pd.concat([a_jobs, u_jobs])
        #jobs.to_csv(self.filename + "_jobs_report.csv")
        
        # for _, row in jobs.iterrows():
        #     jobId = row["job_id"]
        #     winner = row["final_node_allocation"]
            #print(f"Job {jobId} winner: node {winner[0]}")
            
        count = 0
        for key in self.job_count:
            count += self.job_count[key]
            
        self.logical_network_manager.shutdown()
        self.physical_network_manager.shutdown()
            
        #print(f"Total message count: {count}")
        return count, len(a_jobs), len(u_jobs)
            
        

    