"""
This module contains utils functions to calculate all necessary stats
"""
from csv import DictWriter
import os
import src.config as c
import logging

import math



def wrong_bids_calc(nodes, job):
    
    j = job['job_id']
    print('\n[WRONG BID]' + str(j))
    wrong_bids=[] # used to not replicate same action over different nodes
    wrong_ids=[]

    for curr_node in range(0, c.num_edges):
        if nodes[curr_node].bids[j]['auction_id'] not in wrong_bids:
            print('Unmatching: ' +str(c.nodes[curr_node].bids[j]['auction_id']))
            # print('Unmatching x: ' +str(c.nodes[curr_node].bids[j]['x']))
            if all(x == float('-inf') for x in nodes[curr_node].bids[j]['x']):
                continue
            else:
                # print('(wrong_bids_calc) NON matching: ' +str(wrong_ids))

                if curr_node in nodes[curr_node].bids[j]['auction_id'] and curr_node not in wrong_ids:
                    
                    wrong_ids.append(curr_node)
                    first_time = True
                    index=0
                    while index<c.layer_number:
                        if nodes[curr_node].bids[j]['auction_id'][index] == curr_node and id != float('-inf'):
                            nodes[curr_node].updated_cpu += float(job['num_cpu']) / float(c.layer_number)
                            nodes[curr_node].updated_gpu += float(job['num_gpu']) / float(c.layer_number)
                            if first_time:
                                nodes[curr_node].updated_bw += float(job['read_count']) / float(c.min_layer_number)
                                first_time = False
                        index += 1
        else:
            continue
            # print('\nAlready in wrong bids req: ' + str(j))
            # for i in range(0, c.num_edges):
            #     print(nodes[i].bids[j]['auction_id'])
            
    if c.use_net_topology:
        # release network resources between client and node        
        for curr_node in range(0, c.num_edges):
            for i, n_id in enumerate(nodes[curr_node].bids[j]['auction_id']):
                if i == 0 and n_id == curr_node:
                    c.network_t.release_bandwidth_node_and_client(curr_node, float(job['read_count']) / float(c.min_layer_number), j)
                    
        # release network resources between nodes        
        for curr_node in range(0, c.num_edges):
            prev_val = nodes[curr_node].bids[j]['auction_id'][0]
            for i, n_id in enumerate(nodes[curr_node].bids[j]['auction_id']):
                if i != 0:
                    if prev_val != n_id and n_id == curr_node:
                        c.network_t.release_bandwidth_between_nodes(curr_node, prev_val, float(job['read_count']) / float(c.min_layer_number), j)
                    prev_val = nodes[curr_node].bids[j]['auction_id'][i]


def calculate_utility(nodes, num_edges, msg_count, time, n_req, job_ids, alpha):

    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Duplicate job IDs found!")
    
    stats = {}
    stats['nodes'] = {}
    stats['tot_utility'] = 0
    
    field_names = ['n_nodes', 'n_req', 'n_msg', 'exec_time', 'alpha', 'tot_utility', 'jaini']
    dictionary = {'n_nodes': num_edges, 'n_req' : n_req, 'n_msg' : msg_count, 'exec_time': time, 'alpha': alpha}

    # ---------------------------------------------------------
    # calculate assigned jobs, update resources if job not assigned
    # ---------------------------------------------------------

    count_assigned = 0
    count_unassigned = 0
    assigned_sum_cpu = 0
    assigned_sum_gpu = 0
    assigned_sum_bw = 0
    unassigned_sum_cpu = 0
    unassigned_sum_gpu = 0
    unassigned_sum_bw = 0

    count = 0 
    count_broken = 0 
    assigned_jobs = []
    valid_bids = {}
    
    for job in c.job_list_instance.job_list:
        count += 1
        flag = True
        j = job['job_id']
        # Check correctness of all bids
        equal_values = True 
        print('job_id: ' +str(j))
        
        # for i in range(0, c.num_edges):
        #     print('nodeid: ' + str(i) + ' consensus_count: ' +str(c.nodes[i].bids[j]['consensus_count']))
        #     print('nodeid: ' + str(i) + ' deconflictions: ' +str(c.nodes[i].bids[j]['deconflictions']))
        #     print('nodeid: ' + str(i) + ' forwards: ' +str(c.nodes[i].bids[j]['forward_count']))
        #     print('')
        for i in range(1, c.num_edges):
            if nodes[i].bids[j]['auction_id'] != nodes[i-1].bids[j]['auction_id']:
                count_broken += 1
                print('BROKEN BID id: ' + str(j))
                equal_values = False
                break
        
        if equal_values: # matching auction

                if all(x == float('-inf') for x in nodes[i].bids[j]['auction_id']):
                    print('Unassigned')
                    flag = False # all unassigned
                elif float('-inf') in nodes[i].bids[j]['auction_id']: #check if there is a value not assigned 
                    # print('matching with -inf: ' +str(c.nodes[i].bids[j]['auction_id']))
                    flag = False
                    wrong_bids_calc(nodes, job)
                else:
                    print('MATCH')
                    valid_bids[j] = nodes[i].bids[j]['auction_id']
                    pass
                    # for i in range(0, c.num_edges):
                    #     print('matching: ' +str(c.nodes[i].bids[j]['auction_id']))

        else: # unmatching auctions
            # print('NON matching: ' +str(c.nodes[i].bids[j]['auction_id']))
            flag = False
            wrong_bids_calc(nodes, job)


        if flag:
            assigned_jobs.append(j)
            count_assigned += 1
            assigned_sum_cpu += float(job['num_cpu'])
            assigned_sum_gpu += float(job['num_gpu'])
            assigned_sum_bw += float(job['read_count']) 
        else:
            count_unassigned += 1
            unassigned_sum_cpu += float(job['num_cpu'])
            unassigned_sum_gpu += float(job['num_gpu'])
            unassigned_sum_bw += float(job['read_count']) 
            
    if c.use_net_topology:
        print()
        c.network_t.check_network_consistency(valid_bids)
            
    print()
    print('ASSIGNED jobs: ' +str(count_assigned))
    print('UNASSIGNED jobs: ' +str(count_unassigned))
    field_names.append('count_assigned')
    field_names.append('count_unassigned')
    dictionary['count_assigned'] = round(count_assigned,2)
    dictionary['count_unassigned'] = round(count_unassigned,2)


    # metrics lables
    field_names.append('tot_gpu')
    field_names.append('assigned_sum_gpu')
    field_names.append('tot_used_gpu')
    field_names.append('unassigned_sum_gpu')
    field_names.append('tot_cpu')
    field_names.append('assigned_sum_cpu')
    field_names.append('tot_used_cpu')
    field_names.append('unassigned_sum_cpu')
    field_names.append('tot_bw')
    field_names.append('assigned_sum_bw')
    field_names.append('tot_used_bw')
    field_names.append('unassigned_sum_bw')


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
        dictionary['node_'+str(i)+'_leftover_gpu'] = 0 if math.isclose(nodes[i].initial_gpu - nodes[i].updated_gpu, 0.0, abs_tol=1e-1) else round(nodes[i].initial_gpu - nodes[i].updated_gpu, 2)

        tot_used_gpu += dictionary['node_'+str(i)+'_leftover_gpu']

        dictionary['node_'+str(i)+'_initial_cpu'] = round(nodes[i].initial_cpu,2)
        dictionary['node_'+str(i)+'_updated_cpu'] = round(nodes[i].updated_cpu,2)
        dictionary['node_'+str(i)+'_leftover_cpu'] = 0 if math.isclose(nodes[i].initial_cpu - nodes[i].updated_cpu, 0.0, abs_tol=1e-1) else round(nodes[i].initial_cpu - nodes[i].updated_cpu,2)

        tot_used_cpu += dictionary['node_'+str(i)+'_leftover_cpu']

        dictionary['node_'+str(i)+'_initial_bw'] = round(nodes[i].initial_bw,2)
        dictionary['node_'+str(i)+'_updated_bw'] = round(nodes[i].updated_bw,2)
        dictionary['node_'+str(i)+'_leftover_bw'] = 0 if math.isclose(nodes[i].initial_bw - nodes[i].updated_bw, 0.0, abs_tol=1e-1) else round(nodes[i].initial_bw - nodes[i].updated_bw,2)

        tot_used_bw += dictionary['node_'+str(i)+'_leftover_bw']


        #calculate node assigned count and utility
        for j, job_id in enumerate(nodes[i].bids):
            
            if job_id in assigned_jobs and nodes[i].id in nodes[i].bids[job_id]['auction_id']:
                stats['nodes'][nodes[i].id]['assigned_count'] += 1

            # print(nodes[i].bids[job_id])
            for k, auctioner in enumerate(nodes[i].bids[job_id]['auction_id']):
                # print(nodes[i].id)
                if job_id in assigned_jobs and auctioner== nodes[i].id:
                    # print(nodes[i].bids[job_id]['bid'])
                    stats['nodes'][nodes[i].id]['utility'] += nodes[i].bids[job_id]['bid'][k]

        print('node: ' + str(nodes[i].id) + ' utility: ' + str(stats['nodes'][nodes[i].id]['utility']))
        dictionary['node_'+str(i)+'_utility'] = round(stats['nodes'][nodes[i].id]['utility'],2)
        stats["tot_utility"] += stats['nodes'][nodes[i].id]['utility']

    for i in stats['nodes']:

        print('node: '+ str(i) + ' assigned jobs count: ' + str(stats['nodes'][i]['assigned_count']))
        dictionary['node_'+str(i)+'_jobs'] = round(stats['nodes'][i]['assigned_count'],2)

    tot_gpu = 0
    tot_cpu = 0
    tot_bw = 0
    for n in c.nodes:
        tot_cpu += n.initial_cpu
        tot_gpu += n.initial_gpu
        tot_bw += n.initial_bw

    #GPU metrics
    dictionary['tot_gpu'] = round(tot_gpu,2)
    dictionary['assigned_sum_gpu'] = round(assigned_sum_gpu,2)
    dictionary['tot_used_gpu']=round(tot_used_gpu,2)
    dictionary['unassigned_sum_gpu'] = round(unassigned_sum_gpu,2)

    #CPU metrics
    dictionary['tot_cpu'] = round(tot_cpu,2)
    dictionary['assigned_sum_cpu'] = round(assigned_sum_cpu,2)
    dictionary['tot_used_cpu']=round(tot_used_cpu,2)
    dictionary['unassigned_sum_cpu'] = round(unassigned_sum_cpu,2)

    #BW metrics
    dictionary['tot_bw'] = round(tot_bw,2)
    dictionary['assigned_sum_bw'] = round(assigned_sum_bw,2)
    dictionary['tot_used_bw']=round(tot_used_bw,2)    
    dictionary['unassigned_sum_bw'] = round(unassigned_sum_bw,2)

    dictionary['tot_utility'] = round(stats["tot_utility"],2)
    print('total utility: ' + str(stats["tot_utility"]))


    # ---------------------------------------------------------
    # calculate fairness
    # ---------------------------------------------------------
    dictionary['jaini'] = jaini_index(dictionary, num_edges)

    print('jobs number: ' + str(len(job_ids)))

    print('count: ' +str(count) + ' count_broken: ' +str(count_broken))

    


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

