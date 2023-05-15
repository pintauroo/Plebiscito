"""
This module contains utils functions to calculate all necessary stats
"""
from csv import DictWriter
import os
import src.config as c



def calculate_utility(nodes, num_edges, msg_count, time, n_req, job_ids, alpha):
    stats = {}
    stats['nodes'] = {}
    stats['tot_utility'] = 0
    
    field_names = ['n_nodes', 'n_req', 'n_msg', 'exec_time', 'tot_utility', 'jaini', 'tot_gpu']
    dictionary = {'n_nodes': num_edges, 'n_req' : n_req, 'n_msg' : msg_count, 'exec_time': time, 'tot_gpu': c.tot_gpu}


    # ---------------------------------------------------------
    # alpha utility factor
    # ---------------------------------------------------------
    field_names.append('alpha')
    dictionary['alpha'] = alpha

    # ---------------------------------------------------------
    # calculate assigned jobs
    # ---------------------------------------------------------
    
    count_assigned = 0
    count_unassigned = 0

    for j in job_ids:

        flag = True
        for i in range(c.num_edges):
            if not all(x == 1 for x in c.nodes[i].bids[j]['x']):
                flag = False
                
        if flag:
            count_assigned += 1
        else:
            count_unassigned += 1

    field_names.append('count_assigned')
    field_names.append('count_unassigned')
    dictionary['count_assigned'] = count_assigned
    dictionary['count_unassigned'] = count_unassigned


    # ---------------------------------------------------------
    # calculate node utility and assigned jobs
    # ---------------------------------------------------------
    for i in range(num_edges):
        field_names.append('node_'+str(i)+'_jobs')
        field_names.append('node_'+str(i)+'_utility')

        stats['nodes'][nodes[i].id] = {
            "utility": float(),
            "assigned_count": int()
        }

        stats['nodes'][nodes[i].id]['utility'] = 0
        stats['nodes'][nodes[i].id]['assigned_count'] = 0

        for j, job_id in enumerate(nodes[i].bids):

            if nodes[i].id in nodes[i].bids[job_id]['auction_id']:
                stats['nodes'][nodes[i].id]['assigned_count'] += 1

            # print(nodes[i].bids[job_id])
            for k, auctioner in enumerate(nodes[i].bids[job_id]['auction_id']):
                # print(nodes[i].id)
                if auctioner== nodes[i].id:
                    # print(nodes[i].bids[job_id]['bid'])
                    stats['nodes'][nodes[i].id]['utility']+=nodes[i].bids[job_id]['bid'][k]
        print(str(nodes[i].id) + ' utility: ' + str(stats['nodes'][nodes[i].id]['utility']))
        dictionary['node_'+str(i)+'_utility'] = stats['nodes'][nodes[i].id]['utility']
        stats["tot_utility"] += stats['nodes'][nodes[i].id]['utility']

    for i in stats['nodes']:

        print('node: '+ str(i) + ' assigned jobs count: ' + str(stats['nodes'][i]['assigned_count']))
        dictionary['node_'+str(i)+'_jobs'] = stats['nodes'][i]['assigned_count']

    dictionary['tot_utility'] = stats["tot_utility"]
    print(stats["tot_utility"])


    # ---------------------------------------------------------
    # calculate fairness
    # ---------------------------------------------------------
    dictionary['jaini'] = jaini_index(dictionary, num_edges)





    write_data(field_names, dictionary)


def write_data(field_names, dictionary):
    filename = 'bidding_results.csv'
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

