import sys
sys.path.append("/mnt/c/Users/galax/Desktop/Github Projects/consensus-algorithm-evaluation/Plebiscito")

from src.simulator import Simulator_Plebiscito
from src.config import Utility, DebugLevel, SchedulingAlgorithm
from src.dataset_builder import generate_dataset_fake

def test(run, n_jobs, n_nodes, n_failure, edge_to_add, logical_topology_name):
    #print()
    #print(f"Running Plebiscito with {n_nodes} nodes and {n_jobs} jobs.")
    
    dataset = generate_dataset_fake(entries_num=n_jobs)
    
    simulator = Simulator_Plebiscito(filename=run,
                          n_nodes=n_nodes,
                          node_bw=1000000000,
                          n_jobs=n_jobs,
                          n_client=3,
                          enable_logging=False,
                          use_net_topology=False,
                          progress_flag=False,
                          dataset=dataset,
                          alpha=1,
                          utility=Utility.LGF,
                          debug_level=DebugLevel.INFO,
                          scheduling_algorithm=SchedulingAlgorithm.FIFO,
                          decrement_factor=0.2,
                          split=True,
                          n_failures=n_failure,
                          edge_to_add=edge_to_add,
                          logical_topology_name=logical_topology_name)
    return simulator.run()