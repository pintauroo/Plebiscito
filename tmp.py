num_edges = 50
tot_cpu = 10000
tot_gpu = 500

# questa funzione ha una variabile che rispecchia il dimensionamento dell'infrastruttura di Alibaba
# e calcola quanti nodi devono essere creati per ongi tipo di servere presente da Alibaba (4 in tutto)
def compute_occurrency_per_node_type():
    node_of_type = []
    occurrences = [0.42, 0.27, 0.26, 0.05]
    sum_node = 0
    for i in range(4):
        node_of_type.append(round(num_edges*occurrences[i]))
        sum_node += round(num_edges*occurrences[i])
        
    if sum_node > num_edges:
        i = 0
        while sum_node > num_edges:
            node_of_type[i] = node_of_type[i] - 1
            sum_node -= 1
            i = (i+1)%4
            
    elif sum_node < num_edges: 
        i = 0
        while sum_node < num_edges:
            node_of_type[i] = node_of_type[i] + 1
            sum_node += 1
            i = (i+1)%4  
            
    print(f"Number of node of each type {node_of_type}. Tot: {sum_node}")
    
    return node_of_type

def compute_available_resource_per_node():
    cpu_per_type = []
    gpu_per_type = []
    occurrences = [0.32, 0.32, 0.30, 0.06]
    cpu_gpu_ratio = [0.32, 0.12, 0.48, 0.0]
    sum_cpu = 0
    sum_gpu = 0
    
    for i in range(4):
        cpu_per_type.append(tot_cpu*occurrences[i])
        sum_cpu += tot_cpu*occurrences[i]
        gpu_per_type.append(tot_gpu*occurrences[i]*cpu_gpu_ratio[i])
        sum_gpu += tot_gpu*occurrences[i]*cpu_gpu_ratio[i]
    
    print(f"CPU for each node type {cpu_per_type}. Tot: {sum_cpu}")
    print(f"GPU for each node type {gpu_per_type}. Tot: {sum_gpu}")
    return cpu_per_type, gpu_per_type 
    
def amount_of_resource_per_node(): 
    node_per_type = compute_occurrency_per_node_type()
    cpu_per_type, gpu_per_type = compute_available_resource_per_node()
        
    ret = {}
    count = 0
    d_type = 0
    sum_cpu = 0
    sum_gpu = 0
    for i in range(num_edges):
        if count > node_per_type[d_type]:
            count = 0
            d_type += 1
            
        ret[str(i)] = {"cpu": round(cpu_per_type[d_type]/node_per_type[d_type]),
                        "gpu": round(gpu_per_type[d_type]/node_per_type[d_type])}
        
        sum_cpu += round(cpu_per_type[d_type]/node_per_type[d_type])
        sum_gpu += round(gpu_per_type[d_type]/node_per_type[d_type])
        count += 1
        
    print(f"Generating infrastructure with {sum_cpu} CPUs. The request amount of CPU for all tasks is {tot_cpu}")
    print(f"Generating infrastructure with {sum_gpu} GPUs. The request amount of GPU for all tasks is {tot_gpu}")
    return ret
    
amount_of_resource_per_node()