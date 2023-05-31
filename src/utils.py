"""
This module contains utils functions to calculate all necessary stats
"""
from csv import DictWriter
import os
import src.config as c



def calculate_utility(nodes, num_edges, msg_count, time, n_req, job_ids, alpha):

    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Duplicate job IDs found!")
    
    stats = {}
    stats['nodes'] = {}
    stats['tot_utility'] = 0
    
    field_names = ['n_nodes', 'n_req', 'n_msg', 'exec_time', 'alpha', 'tot_utility', 'jaini', 'tot_bw']
    dictionary = {'n_nodes': num_edges, 'n_req' : n_req, 'n_msg' : msg_count, 'exec_time': time, 'alpha': alpha, 'tot_bw': c.tot_bw}

    # ---------------------------------------------------------
    # calculate assigned jobs
    # ---------------------------------------------------------

    count_assigned = 0
    count_unassigned = 0
    assigned_sum_cpu = 0
    assigned_sum_gpu = 0
    unassigned_sum_cpu = 0
    unassigned_sum_gpu = 0

    count = 0 
    uncount = 0 

    for job in c.job_list_instance.job_list:
        count += 1
        flag = True
        j = job['job_id']
        # Check correctness of all bids
        equal_values = True 
        for i in range(1, c.num_edges):
            if nodes[i].bids[j]['x'] != nodes[i-1].bids[j]['x']:
                uncount += 1
                # print('ROTTO' + str(nodes[i].bids[j]['auction_id']))
                equal_values = False
                break
        
        if equal_values:

                if float('-inf') in nodes[i].bids[j]['x'] and not all(x == 1 for x in nodes[i].bids[j]['x']): #check if there is a not assigned value
                    # print(c.nodes[i].bids[j]['x'])

                    
                    first = True
                    for index, node in enumerate(nodes[i].bids[j]['auction_id']):
                    
                        
                        if node != float('-inf'):
                            nodes[node].updated_cpu += float(job['num_cpu']) / float(c.layer_number)
                            nodes[node].updated_gpu += float(job['num_gpu']) / float(c.layer_number)
                            if first:
                                nodes[node].updated_bw += float(job['read_count']) / float(c.layer_number)
                                first=False
                    flag = False
                    # break
        else:
            print(c.nodes[i].bids[j])
            first = True

            for index, node in enumerate(nodes[i].bids[j]['auction_id']):
                    
                        
                if node != float('-inf'):
                    nodes[node].updated_cpu += float(job['num_cpu']) / float(c.layer_number)
                    nodes[node].updated_gpu += float(job['num_gpu']) / float(c.layer_number)
                    if first:
                        nodes[node].updated_bw += float(job['read_count']) / float(c.layer_number)
                        first=False

            flag = False
            continue
                    

        if flag:
            count_assigned += 1
            assigned_sum_cpu += float(job['num_cpu'])
            assigned_sum_gpu += float(job['num_gpu'])
        else:
            count_unassigned += 1
            unassigned_sum_cpu += float(job['num_cpu'])
            unassigned_sum_gpu += float(job['num_gpu'])
            
    print('ASSIGNED unassigned jobs')
    print(count_assigned)
    print(count_unassigned)
    field_names.append('count_assigned')
    field_names.append('count_unassigned')
    dictionary['count_assigned'] = round(count_assigned,2)
    dictionary['count_unassigned'] = round(count_unassigned,2)


    tot_used_bw = 0
    tot_used_cpu = 0
    tot_used_gpu = 0

    # ---------------------------------------------------------
    # calculate node utility, assigned jobs and leftover res
    # ---------------------------------------------------------
    for i in range(num_edges):
        field_names.append('node_'+str(i)+'_jobs')
        field_names.append('node_'+str(i)+'_utility')

        field_names.append('node_'+str(i)+'_initial_gpu')
        field_names.append('node_'+str(i)+'_updated_gpu')
        field_names.append('node_'+str(i)+'_leftover_gpu')

        

        field_names.append('node_'+str(i)+'_initial_cpu')
        field_names.append('node_'+str(i)+'_updated_cpu')
        field_names.append('node_'+str(i)+'_leftover_cpu')

        field_names.append('node_'+str(i)+'_initial_bw')
        field_names.append('node_'+str(i)+'_updated_bw')
        field_names.append('node_'+str(i)+'_leftover_bw')

        stats['nodes'][nodes[i].id] = {
            "utility": float(),
            "assigned_count": int()
        }

        stats['nodes'][nodes[i].id]['utility'] = 0
        stats['nodes'][nodes[i].id]['assigned_count'] = 0
        dictionary['node_'+str(i)+'_initial_gpu'] = round(nodes[i].initial_gpu,2)
        dictionary['node_'+str(i)+'_updated_gpu'] = round(nodes[i].updated_gpu,2)
        dictionary['node_'+str(i)+'_leftover_gpu'] = round(nodes[i].initial_gpu - nodes[i].updated_gpu,2)

        tot_used_gpu += dictionary['node_'+str(i)+'_leftover_gpu']

        dictionary['node_'+str(i)+'_initial_cpu'] = round(nodes[i].initial_cpu,2)
        dictionary['node_'+str(i)+'_updated_cpu'] = round(nodes[i].updated_cpu,2)
        dictionary['node_'+str(i)+'_leftover_cpu'] = round(nodes[i].initial_cpu - nodes[i].updated_cpu,2)

        tot_used_cpu += dictionary['node_'+str(i)+'_leftover_cpu']

        dictionary['node_'+str(i)+'_initial_bw'] = round(nodes[i].initial_bw,2)
        dictionary['node_'+str(i)+'_updated_bw'] = round(nodes[i].updated_bw,2)
        dictionary['node_'+str(i)+'_leftover_bw'] = round(nodes[i].initial_bw - nodes[i].updated_bw,2)

        tot_used_bw += dictionary['node_'+str(i)+'_leftover_bw']

        for j, job_id in enumerate(nodes[i].bids):

            if float('-inf') not in c.nodes[i].bids[job_id]['x'] and nodes[i].id in nodes[i].bids[job_id]['auction_id']:
                stats['nodes'][nodes[i].id]['assigned_count'] += 1

            # print(nodes[i].bids[job_id])
            for k, auctioner in enumerate(nodes[i].bids[job_id]['auction_id']):
                # print(nodes[i].id)
                if float('-inf') not in c.nodes[i].bids[job_id]['x'] and auctioner== nodes[i].id:
                    # print(nodes[i].bids[job_id]['bid'])
                    stats['nodes'][nodes[i].id]['utility'] += nodes[i].bids[job_id]['bid'][k]

        print(str(nodes[i].id) + ' utility: ' + str(stats['nodes'][nodes[i].id]['utility']))
        dictionary['node_'+str(i)+'_utility'] = round(stats['nodes'][nodes[i].id]['utility'],2)
        stats["tot_utility"] += stats['nodes'][nodes[i].id]['utility']

    for i in stats['nodes']:

        print('node: '+ str(i) + ' assigned jobs count: ' + str(stats['nodes'][i]['assigned_count']))
        dictionary['node_'+str(i)+'_jobs'] = round(stats['nodes'][i]['assigned_count'],2)



    #GPU metrics
    field_names.append('tot_gpu')
    field_names.append('assigned_sum_gpu')
    field_names.append('unassigned_sum_gpu')
    field_names.append('tot_used_gpu')


    dictionary['tot_gpu'] = round(c.tot_gpu,2)
    dictionary['assigned_sum_gpu'] = round(assigned_sum_gpu,2)
    dictionary['unassigned_sum_gpu'] = round(unassigned_sum_gpu,2)
    dictionary['tot_used_gpu']=round(tot_used_gpu,2)


    #CPU metrics
    field_names.append('tot_cpu')
    field_names.append('assigned_sum_cpu')
    field_names.append('unassigned_sum_cpu')
    field_names.append('tot_used_cpu')

    dictionary['tot_cpu'] = round(c.tot_cpu,2)
    dictionary['assigned_sum_cpu'] = round(assigned_sum_cpu,2)
    dictionary['unassigned_sum_cpu'] = round(unassigned_sum_cpu,2)
    dictionary['tot_used_cpu']=round(tot_used_cpu,2)


    #BW metrics
    field_names.append('tot_used_bw')
    dictionary['tot_used_bw']=round(tot_used_bw,2)
    
    # print('FPUCPU')
    # print( c.tot_cpu-tot_used_cpu)
    # print( c.tot_gpu-tot_used_gpu)
    dictionary['tot_utility'] = round(stats["tot_utility"],2)
    print(stats["tot_utility"])


    # ---------------------------------------------------------
    # calculate fairness
    # ---------------------------------------------------------
    dictionary['jaini'] = jaini_index(dictionary, num_edges)

    print('jobids = ' + str(len(job_ids)))

    print('count= ' +str(count) + ' uncount =  ' +str(uncount))

    


    write_data(field_names, dictionary)


def write_data(field_names, dictionary):
    filename = str(c.filename)+'.csv'
    # filename = 'alpha_GPU_BW.csv'
    # filename = 'alpha_BW_CPU.csv'

    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f: 
        writer = DictWriter(f, fieldnames=field_names)
    
        # Pass the dictionary as an argument to the Writerow()
        if not file_exists:
            writer.writeheader()  # write the column headers if the file doesn't exist
    
        writer.writerow(dictionary)    



def jaini_index(dictionary, num_nodes):
    data=[]
    for i in range(num_nodes):
        data.append(dictionary['node_'+str(i)+'_jobs'])

    sum_normal = 0
    sum_square = 0

    for arg in data:
        sum_normal += arg
        sum_square += arg**2

    if len(data) == 0 or sum_square == 0:
        return 1

    return sum_normal ** 2 / (len(data) * sum_square)

