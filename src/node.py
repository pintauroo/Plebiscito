'''
This module impelments the behavior of a node
'''

from queue import Empty
import time
from types import NoneType
import src.config as config
from src.network_topology import NetworkTopology
from datetime import datetime, timedelta
import copy
import logging
import math 
import threading
from threading import Event
import random
import numpy

TRACE = 5    

class InternalError(Exception):
    "Raised when the input value is less than 18"
    pass

class node:

    def __init__(self, id, seed, network_topology: NetworkTopology, use_net_topology=False):
        random.seed(seed)
        self.id = id    # unique edge node id
        self.initial_gpu = float(config.node_gpu) * random.uniform(0.1, 1)
        self.updated_gpu = self.initial_gpu# * random.uniform(0.7, 1)
        self.initial_cpu = float(config.node_cpu) * random.uniform(0.3, 0.9)
        self.updated_cpu = self.initial_cpu
        print(str(self.id) + ' gpu:' +str(self.initial_gpu) + ' cpu:' + str(self.initial_cpu))

        self.available_cpu_per_task = {}
        self.available_gpu_per_task = {}
        self.available_bw_per_task = {}

        self.last_sent_msg = {}
        self.resource_remind = {}

        # it is not possible to have NN with more than 50 layers
        self.cum_cpu_reserved = 0
        self.cum_gpu_reserved = 0
        self.cum_bw_reserved = 0
        
        if use_net_topology:
            self.network_topology = network_topology
            self.initial_bw = network_topology.get_node_direct_link_bw(self.id)
            self.bw_with_nodes = {}
            self.bw_with_client = {}
        else:
            self.initial_bw = 100000000000
            self.updated_bw = self.initial_bw
           
            
        self.use_net_topology = use_net_topology
        
        self.last_bid_timestamp = {}
        self.last_bid_timestamp_lock = threading.Lock()
        
        self.__layer_bid_lock = threading.Lock()
        self.__layer_bid = {}
        self.__layer_bid_events = {}
        
        if self.initial_gpu != 0:
            print(f"Node {self.id} CPU/GPU ratio: {self.initial_cpu/self.initial_gpu}")
        else:
            print(f"Node {self.id} CPU/GPU ratio: <inf>")
        self.counter = 0
        
        self.user_requests = []
        self.item={}
        self.bids= {}
        self.layer_bid_already = {}
        
    def set_queues(self, q, use_queue):
        self.q = q
        self.use_queue = use_queue
        self.use_queue[self.id].clear()
    def init_null(self):
        # print(self.item['duration'])
        self.bids[self.item['job_id']]={
            "count":0,
            "consensus_count":0,
            "forward_count":0,
            "deconflictions":0,
            "job_id": self.item['job_id'], 
            "user": int(), 
            "auction_id": list(), 
            "NN_gpu": self.item['NN_gpu'], 
            "NN_cpu": self.item['NN_cpu'], 
            "NN_data_size": self.item['NN_data_size'],
            "bid": list(), 
            "bid_gpu": list(), 
            "bid_cpu": list(), 
            "bid_bw": list(), 
            "timestamp": list(),
            "arrival_time":datetime.now(),
            "start_time": 0, #datetime.now(),
            "progress_time": 0, #datetime.now(),
            "duration": random.randint(1000, 10000), #self.item['duration'],
            "complete":False,
            "complete_timestamp":None,
            "N_layer_min": self.item["N_layer_min"],
            "N_layer_max": self.item["N_layer_max"],
            "edge_id": self.id, 
            "N_layer": self.item["N_layer"],
            'consensus':False,
            'clock':False,
            'rebid':False,
            "N_layer_bundle": self.item["N_layer_bundle"]


            }
        
        self.layer_bid_already[self.item['job_id']] = [False] * self.item["N_layer"]

        self.available_gpu_per_task[self.item['job_id']] = self.updated_gpu
        self.available_cpu_per_task[self.item['job_id']] = self.updated_cpu
        if not self.use_net_topology:
            self.available_bw_per_task[self.item['job_id']] = self.updated_bw
        else:
            self.bw_with_nodes[self.item['job_id']] = {}
            self.bw_with_client[self.item['job_id']] = self.network_topology.get_available_bandwidth_with_client(self.id)
        
        NN_len = len(self.item['NN_gpu'])
        
        for _ in range(0, NN_len):
            self.bids[self.item['job_id']]['bid'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid_gpu'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid_cpu'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid_bw'].append(float('-inf'))
            self.bids[self.item['job_id']]['auction_id'].append(float('-inf'))
            self.bids[self.item['job_id']]['timestamp'].append(datetime.now() - timedelta(days=1))

    def utility_function(self, avail_bw, avail_cpu, avail_gpu):
        def f(x, alpha, beta):
            if beta == 0 and x == 0:
                return 1
            
            # shouldn't happen
            if beta == 0 and x != 0:
                return 0
            
            # if beta != 0 and x == 0 is not necessary
            return math.exp(-((alpha/100) * (x - beta))**2)
            #return math.exp(-(alpha/100)*(x-beta)**2)

        if (isinstance(avail_bw, float) and avail_bw == float('inf')):
            avail_bw = self.initial_bw
        
        # we assume that every job/node has always at least one CPU
        if config.filename == 'stefano':
            x = 0
            if self.item['NN_gpu'][0] == 0:
                x = 0
            else:
                x = self.item['NN_cpu'][0]/self.item['NN_gpu'][0]
                
            beta = 0
            if avail_gpu == 0:
                beta = 0
            else:
                beta = avail_cpu/avail_gpu
            if config.a == 0:
                return f(x, 0.01, beta)
            else:
                return f(x, config.a, beta)
        elif config.filename == 'alpha_BW_CPU':
            return (config.a*(avail_bw/self.initial_bw))+((1-config.a)*(avail_cpu/self.initial_cpu)) #BW vs CPU
        elif config.filename == 'alpha_GPU_CPU':
            return (config.a*(avail_gpu/self.initial_gpu))+((1-config.a)*(avail_cpu/self.initial_cpu)) #GPU vs CPU
        elif config.filename == 'alpha_GPU_BW':
            return (config.a*(avail_gpu/self.initial_gpu))+((1-config.a)*(avail_bw/self.initial_bw)) # GPU vs BW


        # elif config.filename == 'alpha_BW_CPU':
        #     return (config.a*(self.updated_bw/config.tot_bw))+((1-config.a)*(self.updated_cpu/config.tot_cpu)) #BW vs CPU
        # elif config.filename == 'alpha_GPU_CPU':
        #     return (config.a*(self.updated_gpu/config.tot_gpu))+((1-config.a)*(self.updated_cpu/config.tot_cpu)) #GPU vs CPU
        # elif config.filename == 'alpha_GPU_BW':
        #     return (config.a*(self.updated_gpu/config.tot_gpu))+((1-config.a)*(self.updated_bw/config.tot_bw)) # GPU vs BW


    def forward_to_neighbohors(self, custom_dict=None):
        if config.enable_logging:
            self.print_node_state('FORWARD', True)
        if custom_dict == None:
            msg = {
                        "job_id": self.item['job_id'], 
                        "user": self.item['user'],
                        "edge_id": self.id, 
                        "auction_id": copy.deepcopy(self.bids[self.item['job_id']]['auction_id']), 
                        "NN_gpu": self.item['NN_gpu'],
                        "NN_cpu": self.item['NN_cpu'],
                        "NN_data_size": self.item['NN_data_size'], 
                        "bid": copy.deepcopy(self.bids[self.item['job_id']]['bid']), 
                        "timestamp": copy.deepcopy(self.bids[self.item['job_id']]['timestamp']),
                        "N_layer": self.item["N_layer"],
                        "N_layer_min": self.item["N_layer_min"],
                        "N_layer_max": self.item["N_layer_max"],
                        "N_layer_bundle": self.item["N_layer_bundle"]
                        }
        else:
            msg = {
                        "job_id": self.item['job_id'], 
                        "user": self.item['user'],
                        "edge_id": self.id, 
                        "auction_id": copy.deepcopy(custom_dict['auction_id']), 
                        "NN_gpu": self.item['NN_gpu'],
                        "NN_cpu": self.item['NN_cpu'],
                        "NN_data_size": self.item['NN_data_size'], 
                        "bid": copy.deepcopy(custom_dict['bid']), 
                        "timestamp": copy.deepcopy(custom_dict['timestamp']),
                        "N_layer": self.item["N_layer"],
                        "N_layer_min": self.item["N_layer_min"],
                        "N_layer_max": self.item["N_layer_max"],
                        "N_layer_bundle": self.item["N_layer_bundle"]
                        }
        
        if self.item['job_id'] not in self.last_sent_msg:
            self.last_sent_msg[self.item['job_id']] = msg
        elif (self.last_sent_msg[self.item['job_id']]["auction_id"] == msg["auction_id"] and \
            self.last_sent_msg[self.item['job_id']]["timestamp"] == msg["timestamp"] and \
            self.last_sent_msg[self.item['job_id']]["bid"] == msg["bid"]):
            # msg already sent before
            return
        
        for i in range(config.num_edges):
            if config.t.to()[i][self.id] and self.id != i:
                self.q[i].put(msg)
        
        self.last_sent_msg[self.item['job_id']] = msg



    def print_node_state(self, msg, bid=False, type='debug'):
        logger_method = getattr(logging, type)
        #print(str(self.item.get('auction_id')) if bid and self.item.get('auction_id') is not None else "\n")
        logger_method(str(msg) +
                    " job_id:" + str(self.item['job_id']) +
                    " NODEID:" + str(self.id) +
                    " from_edge:" + str(self.item['edge_id']) +
                    " initial GPU:" + str(self.initial_gpu) +
                    " available GPU:" + str(self.updated_gpu) +
                    " initial CPU:" + str(self.initial_cpu) +
                    " available CPU:" + str(self.updated_cpu) +
                    #" initial BW:" + str(self.initial_bw) if hasattr(self, 'initial_bw') else str(0) +
                    #" available BW:" + str(self.updated_bw) if hasattr(self, 'updated_bw') else str(0)  +
                    # "\n" + str(self.layer_bid_already[self.item['job_id']]) +
                    (("\n"+str(self.bids[self.item['job_id']]['auction_id']) if bid else "") +
                    ("\n" + str(self.item.get('auction_id')) if bid and self.item.get('auction_id') is not None else "\n"))
                    )
    
    def update_local_val(self, tmp, index, id, bid, timestamp, lst):
        tmp['job_id'] = self.item['job_id']
        tmp['auction_id'][index] = id
        tmp['bid'][index] = bid
        tmp['timestamp'][index] = timestamp
        return index + 1

    def update_local_val_new(self, tmp, index, id, bid, timestamp, lst):

        indices = []

        flag=True
        for i, v in enumerate(lst['auction_id']):
            if v == id:
                indices.append(i)
        if len(indices)>0:
            for i in indices:
                if timestamp >= lst['timestamp'][i]:
                    flag=True
                else:
                    flag=False
        if flag:
            tmp['auction_id'][index] = id
            tmp['bid'][index] = bid
            tmp['timestamp'][index] = timestamp
        else:
            pass
            # print('beccatoktm!')

        return index + 1
    
    def reset(self, index, dict, bid_time):
        dict['auction_id'][index] = float('-inf')
        dict['bid'][index]= float('-inf')
        dict['timestamp'][index] = bid_time # - timedelta(days=1)
        return index + 1




    def bid(self, enable_forward=True):
        proceed = True

        bid_on_layer = False
        NN_len = len(self.item['NN_gpu'])
        if self.use_net_topology:
            avail_bw = None
        else:
            avail_bw = self.available_bw_per_task[self.item['job_id']]
        tmp_bid = copy.deepcopy(self.bids[self.item['job_id']])
        gpu_=0
        cpu_=0
        first = False
        first_index = None
        layers = 0
        bw_with_client = False
        previous_winner_id = 0
        bid_ids = []
        bid_ids_fail = []
        bid_round = 0
        i = 0
        bid_time = datetime.now()

        if self.item['job_id'] in self.bids:                  
            while i < NN_len:
                if self.use_net_topology:
                    with self.__layer_bid_lock:
                        if bid_round >= self.__layer_bid_events[self.item["job_id"]]:
                            break

                res_cpu, res_gpu, res_bw = self.get_reserved_resources(self.item['job_id'], i)
                NN_data_size = self.item['NN_data_size'][i]
                
                if i == 0:
                    if self.use_net_topology:
                        avail_bw = self.bw_with_client[self.item['job_id']]
                        res_bw = 0
                    first_index = i
                else:
                    if self.use_net_topology and not bw_with_client and not first:
                        if tmp_bid['auction_id'][i-1] not in self.bw_with_nodes[self.item['job_id']]:                            
                            self.bw_with_nodes[self.item['job_id']][tmp_bid['auction_id'][i-1]] = self.network_topology.get_available_bandwidth_between_nodes(self.id, tmp_bid['auction_id'][i-1])
                        previous_winner_id = tmp_bid['auction_id'][i-1]
                        avail_bw = self.bw_with_nodes[self.item['job_id']][tmp_bid['auction_id'][i-1]]
                        res_bw = 0
                    first = True
                
                if  self.item['NN_gpu'][i] <= self.updated_gpu - gpu_ + res_gpu and \
                    self.item['NN_cpu'][i] <= self.updated_cpu - cpu_ + res_cpu and \
                    NN_data_size <= avail_bw + res_bw and \
                    tmp_bid['auction_id'].count(self.id)<self.item["N_layer_max"] and \
                    (self.item['N_layer_bundle'] is None or (self.item['N_layer_bundle'] is not None and layers < self.item['N_layer_bundle'])) :

                    if tmp_bid['auction_id'].count(self.id) == 0 or \
                        (tmp_bid['auction_id'].count(self.id) != 0 and i != 0 and tmp_bid['auction_id'][i-1] == self.id):
                        bid = self.utility_function(avail_bw, self.available_cpu_per_task[self.item['job_id']], self.available_gpu_per_task[self.item['job_id']])

                        if bid == 1:
                            bid -= self.id * 0.000000001
                        
                        if bid > tmp_bid['bid'][i]:# or (bid == tmp_bid['bid'][i] and self.id < tmp_bid['auction_id'][i]):
                            bid_on_layer = True

                            tmp_bid['bid'][i] = bid
                            tmp_bid['bid_gpu'][i] = self.updated_gpu
                            tmp_bid['bid_cpu'][i] = self.updated_cpu
                            tmp_bid['bid_bw'][i] = avail_bw
                                
                            gpu_ += self.item['NN_gpu'][i]
                            cpu_ += self.item['NN_cpu'][i]

                            tmp_bid['auction_id'][i]=(self.id)
                            tmp_bid['timestamp'][i] = bid_time
                            layers += 1

                            bid_ids.append(i)
                            if i == 0:
                                bw_with_client = True

                            if layers >= self.item["N_layer_bundle"]:
                                break
                            
                            i += 1
                        else:
                            bid_ids_fail.append(i)
                            if self.use_net_topology or self.item["N_layer_bundle"] is not None:
                                i += self.item["N_layer_bundle"]
                                bid_round += 1
                                first = False                           
                    else:
                        if self.use_net_topology or self.item["N_layer_bundle"] is not None:
                            i += self.item["N_layer_bundle"]
                            bid_round += 1
                            first = False            
                else:
                    if bid_on_layer:
                        break
                    if self.use_net_topology or self.item["N_layer_bundle"] is not None:
                        i += self.item["N_layer_bundle"]
                        bid_round += 1
                        first = False

            if self.id in tmp_bid['auction_id'] and \
                (first_index is None or avail_bw - self.item['NN_data_size'][first_index] >= 0) and \
                tmp_bid['auction_id'].count(self.id)>=self.item["N_layer_min"] and \
                tmp_bid['auction_id'].count(self.id)<=self.item["N_layer_max"] and \
                self.integrity_check(tmp_bid['auction_id'], 'bid') and \
                (self.item['N_layer_bundle'] is None or (self.item['N_layer_bundle'] is not None and layers == self.item['N_layer_bundle'])):

                # logging.log(TRACE, "BID NODEID:" + str(self.id) + ", auction: " + str(tmp_bid['auction_id']))
                success = False
                if self.use_net_topology:
                    if bw_with_client and self.network_topology.consume_bandwidth_node_and_client(self.id, self.item['NN_data_size'][0], self.item['job_id']):
                        success = True
                    elif not bw_with_client and self.network_topology.consume_bandwidth_between_nodes(self.id, previous_winner_id, self.item['NN_data_size'][0], self.item['job_id']):
                        success = True
                else:
                    success = True
                
                if success:
                    if config.enable_logging:
                        self.print_node_state(f"Bid succesful {tmp_bid['auction_id']}")
                    first_index = tmp_bid['auction_id'].index(self.id)
                    if not self.use_net_topology:
                        self.updated_bw -= self.item['NN_data_size'][first_index] 
                        # self.available_bw_per_task[self.item['job_id']] -= self.item['NN_data_size'][first_index] 

                    self.bids[self.item['job_id']] = copy.deepcopy(tmp_bid)

                    for i in bid_ids_fail:
                        self.release_reserved_resources(self.item["job_id"], i)
                    
                    for i in bid_ids:
                        self.release_reserved_resources(self.item["job_id"], i)
                        self.updated_gpu -= self.item['NN_gpu'][i]
                        self.updated_cpu -= self.item['NN_cpu'][i]

                    # self.available_cpu_per_task[self.item['job_id']] -= cpu_
                    # self.available_gpu_per_task[self.item['job_id']] -= gpu_

                    # if self.available_cpu_per_task[self.item['job_id']] < 0 or self.available_gpu_per_task[self.item['job_id']] < 0:
                    #     print("mannaggia la miseria")

                    
                    if enable_forward:
                        self.forward_to_neighbohors()
                    
                    if self.use_net_topology:
                        with self.__layer_bid_lock:
                            self.__layer_bid[self.item["job_id"]] = sum(1 for i in self.bids[self.item['job_id']]["auction_id"] if i != float('-inf'))
                            
                    return True
                else:
                    return False
            else:
                pass
                # self.print_node_state("bid failed " + str(tmp_bid['auction_id']), True)
        else:
            if config.enable_logging:
                self.print_node_state('Value not in dict (first_msg)', type='error')
            return False


    def lost_bid(self, index, z_kj, tmp_local, tmp_gpu, tmp_cpu, tmp_bw):        
        tmp_gpu +=  self.item['NN_gpu'][index]
        tmp_cpu +=  self.item['NN_cpu'][index]
        tmp_bw += self.item['NN_data_size'][index]
        index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
        return index, tmp_gpu, tmp_cpu, tmp_bw
    
    def deconfliction(self):
        rebroadcast = False
        k = self.item['edge_id'] # sender
        i = self.id # receiver
        self.bids[self.item['job_id']]['deconflictions']+=1
        release_to_client = False
        previous_winner_id = float('-inf')
        job_id = self.item["job_id"]
        
        tmp_local = copy.deepcopy(self.bids[self.item['job_id']])
        prev_bet = copy.deepcopy(self.bids[self.item['job_id']])
        index = 0
        reset_flag = False
        reset_ids = []
        bid_time = datetime.now()
        
        if self.use_net_topology:
            initial_count = 0
            for j in tmp_local["auction_id"]:
                if j != float('-inf'):
                    initial_count += 1

        while index < self.item["N_layer"]:
            if self.item['job_id'] in self.bids:
                z_kj = self.item['auction_id'][index]
                z_ij = tmp_local['auction_id'][index]
                y_kj = self.item['bid'][index]
                y_ij = tmp_local['bid'][index]
                t_kj = self.item['timestamp'][index]
                t_ij = tmp_local['timestamp'][index]

                if config.enable_logging:
                    logging.log(TRACE,'DECONFLICTION - NODEID(i):' + str(i) +
                                ' sender(k):' + str(k) +
                                ' z_kj:' + str(z_kj) +
                                ' z_ij:' + str(z_ij) +
                                ' y_kj:' + str(y_kj) +
                                ' y_ij:' + str(y_ij) +
                                ' t_kj:' + str(t_kj) +
                                ' t_ij:' + str(t_ij)
                                )
                if z_kj==k : 
                    if z_ij==i:
                        if (y_kj>y_ij): 
                            rebroadcast = True
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #1')
                            if index == 0:
                                release_to_client = True
                            elif previous_winner_id == float('-inf'):
                                previous_winner_id = prev_bet['auction_id'][index-1]
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])

                        elif (y_kj==y_ij and z_kj<z_ij):
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #3')
                            rebroadcast = True
                            if index == 0:
                                release_to_client = True
                            elif previous_winner_id == float('-inf'):
                                previous_winner_id = prev_bet['auction_id'][index-1]
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])

                        else:# (y_kj<y_ij):
                            rebroadcast = True
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #2')
                            index = self.update_local_val(tmp_local, index, z_ij, tmp_local['bid'][index], bid_time, self.item)
                        
                        # else:
                        #     if config.enable_logging:
                        #         logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #3else')
                        #     index+=1
                        #     rebroadcast = True

                    elif z_ij==k:
                        if  t_kj>t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#4')
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                            rebroadcast = True 
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #5 - 6')
                            index+=1
                    
                    elif z_ij == float('-inf'):
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #12')
                        index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                        rebroadcast = True

                    elif z_ij!=i and z_ij!=k:
                        if y_kj>y_ij and t_kj>=t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #7')
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                            rebroadcast = True
                        elif y_kj<y_ij and t_kj<=t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #8')
                            index += 1
                            rebroadcast = True
                        elif y_kj==y_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #9')
                            rebroadcast = True
                            index+=1
                        # elif y_kj==y_ij and z_kj<z_ij:
                            # if config.enable_logging:
                                # logging.log(TRACE, 'NODEID:'+str(self.id) +  '#9-new')
                            # index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])                  
                            # rebroadcast = True
                        elif y_kj<y_ij and t_kj>t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #10reset')
                            # index, reset_flag = self.reset(index, tmp_local)
                            index += 1
                            rebroadcast = True
                        elif y_kj>y_ij and t_kj<t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #11rest')
                            # index, reset_flag = self.reset(index, tmp_local)
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                            rebroadcast = True  
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #11else')
                            index += 1  
                            rebroadcast = True  
                    
                    else:
                        index += 1   
                        if config.enable_logging:
                            logging.log(TRACE, "eccoci")    
                
                elif z_kj==i:                                
                    if z_ij==i:
                        if t_kj>t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #13Flavio')
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                            rebroadcast = True 
                            
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #13elseFlavio')
                            index+=1
                            rebroadcast = True

                    elif z_ij==k:
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #14reset')
                        reset_ids.append(index)
                        # index = self.reset(index, self.bids[self.item['job_id']])
                        index += 1
                        reset_flag = True
                        rebroadcast = True                        

                    elif z_ij == float('-inf'):
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #16')
                        rebroadcast = True
                        index+=1
                    
                    elif z_ij!=i and z_ij!=k:
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #15')
                        rebroadcast = True
                        index+=1
                    
                    else:
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #15else')
                        rebroadcast = True
                        index+=1                
                
                elif z_kj == float('-inf'):
                    if z_ij==i:
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #31')
                        rebroadcast = True
                        index+=1
                        
                    elif z_ij==k:
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #32')
                        index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                        rebroadcast = True
                        
                    elif z_ij == float('-inf'):
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #34')
                        index+=1
                        
                    elif z_ij!=i and z_ij!=k:
                        if t_kj>t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #33')
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                            rebroadcast = True
                        else: 
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #33else')
                            index+=1
                        
                    else:
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #33elseelse')
                        index+=1
                        rebroadcast = True

                elif z_kj!=i or z_kj!=k:   
                                     
                    if z_ij==i:
                        if (y_kj>y_ij):
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#17')
                            rebroadcast = True
                            if index == 0:
                                release_to_client = True
                            elif previous_winner_id == float('-inf'):
                                previous_winner_id = prev_bet['auction_id'][index-1]
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                        elif (y_kj==y_ij and z_kj<z_ij):
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#17')
                            rebroadcast = True
                            if index == 0:
                                release_to_client = True
                            elif previous_winner_id == float('-inf'):
                                previous_winner_id = prev_bet['auction_id'][index-1]
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                        else:# (y_kj<y_ij):
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#19')
                            rebroadcast = True
                            index = self.update_local_val(tmp_local, index, z_ij, tmp_local['bid'][index], bid_time, self.item)
                        # else:
                        #     if config.enable_logging:
                        #         logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #19else')
                        #     index+=1
                        #     rebroadcast = True

                    elif z_ij==k:
                        if y_kj<y_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #20Flavio')
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                            rebroadcast = True 
                        # elif (y_kj==y_ij and z_kj<z_ij):
                        #     if config.enable_logging:
                        #         logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #3stefano')
                        #     rebroadcast = True
                        #     index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                        elif t_kj>=t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#20')
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                            rebroadcast = True
                        elif t_kj<t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#21reset')
                            # index, reset_flag = self.reset(index, tmp_local)
                            index += 1
                            rebroadcast = True

                    elif z_ij == z_kj:
                    
                        if t_kj>t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#22')
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #23 - 24')
                            index+=1
                    
                    elif z_ij == float('-inf'):
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#30')
                        index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])
                        rebroadcast = True

                    elif z_ij!=i and z_ij!=k and z_ij!=z_kj:
                        if y_kj>=y_ij and t_kj>=t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#25')
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])                   
                            rebroadcast = True
                        elif y_kj<y_ij and t_kj<=t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#26')
                            rebroadcast = True
                            index+=1
                        # elif y_kj==y_ij:# and z_kj<z_ij:
                        #     if config.enable_logging:
                        #         logging.log(TRACE, 'NODEID:'+str(self.id) +  '#27')
                        #     index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])                   
                        #     rebroadcast = True
                        # elif y_kj==y_ij:
                        #     if config.enable_logging:
                        #         logging.log(TRACE, 'NODEID:'+str(self.id) +  '#27bis')
                        #     index+=1
                        #     rebroadcast = True
                        elif y_kj<y_ij and t_kj>t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#28')
                            index = self.update_local_val(tmp_local, index, z_kj, y_kj, t_kj, self.bids[self.item['job_id']])                   
                            rebroadcast = True
                        elif y_kj>y_ij and t_kj<t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#29')
                            # index, reset_flag = self.reset(index, tmp_local)
                            index += 1
                            rebroadcast = True
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#29else')
                            index+=1
                            rebroadcast = True
                    
                    else:
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #29else2')
                        index+=1
                
                else:
                    if config.enable_logging:
                        self.print_node_state('smth wrong?', type='error')

            else:
                if config.enable_logging:
                    self.print_node_state('Value not in dict (deconfliction)', type='error')

        if self.integrity_check(tmp_local['auction_id'], 'deconfliction'):
            # tmp_local['auction_id']!=self.bids[self.item['job_id']]['auction_id']:
            # tmp_local['bid'] != self.bids[self.item['job_id']]['bid'] and \
            # tmp_local['timestamp'] != self.bids[self.item['job_id']]['timestamp']:
            # self.print_node_state(f'Deconfliction checked pass {self.id}', True)

            if reset_flag:
                self.forward_to_neighbohors(tmp_local)
                for i in reset_ids:
                    _ = self.reset(i, tmp_local, bid_time)
                self.bids[self.item['job_id']] = copy.deepcopy(tmp_local)
                return False, False             

            cpu = 0
            gpu = 0
            bw = 0

            first_1 = False
            first_2 = False
            for i in range(len(tmp_local["auction_id"])):
                if tmp_local["auction_id"][i] == self.id and prev_bet["auction_id"][i] == self.id:
                    if i != 0 and tmp_local["auction_id"][i-1] != prev_bet["auction_id"][i-1]: 
                        if self.use_net_topology:
                            print(f"Failure in node {self.id} job_bid {job_id}. Deconfliction failed. Exiting ...")
                            raise InternalError
                elif tmp_local["auction_id"][i] == self.id and prev_bet["auction_id"][i] != self.id:
                    # self.release_reserved_resources(self.item['job_id'], i)
                    cpu -= self.item['NN_cpu'][i]
                    gpu -= self.item['NN_gpu'][i]
                    if not first_1:
                        bw -= self.item['NN_data_size'][i]
                        first_1 = True
                elif tmp_local["auction_id"][i] != self.id and prev_bet["auction_id"][i] == self.id:
                    cpu += self.item['NN_cpu'][i]
                    gpu += self.item['NN_gpu'][i]
                    if not first_2:
                        bw += self.item['NN_data_size'][i]
                        first_2 = True
                    
            self.updated_cpu += cpu
            self.updated_gpu += gpu

            if self.use_net_topology:
                if release_to_client:
                    self.network_topology.release_bandwidth_node_and_client(self.id, bw, self.item['job_id'])
                elif previous_winner_id != float('-inf'):
                    self.network_topology.release_bandwidth_between_nodes(previous_winner_id, self.id, bw, self.item['job_id'])      
            else:
                self.updated_bw += bw

            self.bids[self.item['job_id']] = copy.deepcopy(tmp_local)
            
            if self.use_net_topology:
                with self.__layer_bid_lock:
                    self.__layer_bid[self.item["job_id"]] = sum(1 for i in self.bids[self.item['job_id']]["auction_id"] if i != float('-inf'))

            return rebroadcast, False
        else:
            if self.use_net_topology:
                print(f"Failure in node {self.id}. Deconfliction failed. Exiting ...")
                raise InternalError

            cpu = 0
            gpu = 0
            bw = 0
            first_1 = False

            for i in range(len(self.item["auction_id"])):
                if self.item["auction_id"][i] == self.id and prev_bet["auction_id"][i] == self.id:
                    pass
                elif self.item["auction_id"][i] == self.id and prev_bet["auction_id"][i] != self.id:
                    cpu -= self.item['NN_cpu'][i]
                    gpu -= self.item['NN_gpu'][i]
                    if not first_1:
                        bw -= self.item['NN_data_size'][i]
                        first_1 = True
                elif self.item["auction_id"][i] != self.id and prev_bet["auction_id"][i] == self.id:
                    self.remind_to_release_resources(self.item["job_id"], self.item['NN_cpu'][i], self.item['NN_gpu'][i], bw, [i])
       
            self.updated_cpu += cpu
            self.updated_gpu += gpu
            self.updated_bw += bw            
            
            for key in self.item:
                self.bids[self.item['job_id']][key] = copy.deepcopy(self.item[key])
                
            self.forward_to_neighbohors()
                    
            return False, True       

       
    def remind_to_release_resources(self, job_id, cpu, gpu, bw, idx):
        if job_id not in self.resource_remind:
            self.resource_remind[job_id] = {}
            self.resource_remind[job_id]["idx"] = []

        self.resource_remind[job_id]["cpu"] = cpu
        self.resource_remind[job_id]["gpu"] = gpu
        self.resource_remind[job_id]["bw"] = bw
        self.resource_remind[job_id]["idx"] += idx

        for id in idx:
            self.cum_cpu_reserved += cpu
            self.cum_gpu_reserved += gpu
            self.cum_bw_reserved += bw

    def get_reserved_resources(self, job_id, index):
        if job_id not in self.resource_remind:
            return 0, 0, 0
        
        found = 0
        for id in self.resource_remind[job_id]["idx"]:
            if id == index:
                found += 1
        
        if found == 0:
            return 0, 0, 0

        return math.ceil(self.cum_cpu_reserved), math.ceil(self.cum_gpu_reserved), math.ceil(self.cum_bw_reserved)

    def release_reserved_resources(self, job_id, index):
        if job_id not in self.resource_remind:
            return False
        found = 0
        for id in self.resource_remind[job_id]["idx"]:
            if id == index:
                # print(self.resource_remind[job_id]["cpu"])
                self.updated_cpu += self.resource_remind[job_id]["cpu"]
                self.updated_gpu += self.resource_remind[job_id]["gpu"]
                self.updated_bw += self.resource_remind[job_id]["bw"]

                self.cum_cpu_reserved -= self.resource_remind[job_id]["cpu"]
                self.cum_gpu_reserved -= self.resource_remind[job_id]["gpu"]
                self.cum_bw_reserved -= self.resource_remind[job_id]["bw"]

                found += 1
                
        if found > 0:
            for _ in range(found):
                self.resource_remind[job_id]["idx"].remove(index)
            return True
        else:
            return False

    def update_bid(self):
        if self.item['job_id'] in self.bids:
        
            # Consensus check
            if  self.bids[self.item['job_id']]['auction_id']==self.item['auction_id'] and \
                self.bids[self.item['job_id']]['bid'] == self.item['bid'] and \
                self.bids[self.item['job_id']]['timestamp'] == self.item['timestamp']:
                    
                    if float('-inf') in self.bids[self.item['job_id']]['auction_id']:
                        if self.id not in self.bids[self.item['job_id']]['auction_id']: 
                            self.bid()
                        else:
                            self.forward_to_neighbohors()
                    else:
                        if config.enable_logging:
                            self.print_node_state('Consensus -', True)
                            self.bids[self.item['job_id']]['consensus_count']+=1
                            # pass
                    
            else:
                if config.enable_logging:
                    self.print_node_state('BEFORE', True)
                rebroadcast, integrity_fail = self.deconfliction()

                success = False
                proceed = True

                if not integrity_fail and self.id not in self.bids[self.item['job_id']]['auction_id']:
                    success = self.bid()
                    
                if not success and rebroadcast:
                    self.forward_to_neighbohors()
                elif float('-inf') in self.bids[self.item['job_id']]['auction_id']:
                    self.forward_to_neighbohors()


        else:
            if config.enable_logging:
                self.print_node_state('Value not in dict (update_bid)', type='error')

    def new_msg(self):
        if str(self.id) == str(self.item['edge_id']):
            if config.enable_logging:
                self.print_node_state('This was not supposed to be received', type='error')

        elif self.item['job_id'] in self.bids:
            if self.integrity_check(self.item['auction_id'], 'new msg'):
                self.update_bid()
            else:
                print('new_msg' + str(self.item) + '\n' + str(self.bids[self.item['job_id']]))
        else:
            if config.enable_logging:
                self.print_node_state('Value not in dict (new_msg)', type='error')

    def integrity_check_old(self, bid, msg):
        curr_val = bid[0]
        curr_count = 1
        for i in range(1, len(bid)):
            if curr_val != float('-inf'):
                if bid[i] == curr_val:
                    curr_count += 1
                else:
                    if curr_count < self.item["N_layer_min"] or curr_count > self.item["N_layer_max"]:
                        self.print_node_state(str(msg) + ' DISCARD BROKEN MSG ' + str(bid))
                        return False
                    
                    curr_val = bid[i]
                    curr_count = 1
        
        if curr_count < self.item["N_layer_min"] or curr_count > self.item["N_layer_max"]:
            if config.enable_logging:
                self.print_node_state(str(msg) + ' DISCARD BROKEN MSG ' + str(bid))
            return False
        
        return True
    
    # def integrity_check(self, bid, msg):
    #     curr_val = bid[0]
    #     curr_count = 1
    #     for i in range(1, len(bid)):
            
    #         if bid[i] == curr_val:
    #             curr_count += 1
    #         else:
    #             if (curr_count < self.item["N_layer_min"] or curr_count > self.item["N_layer_max"]) and curr_val != float('-inf'):
    #                 self.print_node_state(str(msg) + ' DISCARD BROKEN MSG ' + str(bid))
    #                 return False
                    
    #             curr_val = bid[i]
    #             curr_count = 1
        
    #     if curr_count < self.item["N_layer_min"] or curr_count > self.item["N_layer_max"] and curr_val != float('-inf'):
    #         if config.enable_logging:
    #             self.print_node_state(str(msg) + ' DISCARD BROKEN MSG ' + str(bid))
    #         return False
        
    #     return True

    def integrity_check(self, bid, msg):
        min_ = self.item["N_layer_min"]
        max_ = self.item["N_layer_max"]
        curr_val = bid[0]
        curr_count = 1
        prev_values = [curr_val]  # List to store previously encountered values

        for i in range(1, len(bid)):
            if bid[i] == curr_val:
                curr_count += 1
            else:
                if (curr_count < min_ or curr_count > max_) and curr_val != float('-inf'):
                    if config.enable_logging:
                        self.print_node_state(str(msg) + ' 1 DISCARD BROKEN MSG ' + str(bid))
                    return False

                if bid[i] in prev_values:  # Check if current value is repeated
                    if config.enable_logging:
                        self.print_node_state(str(msg) + ' 2 DISCARD BROKEN MSG ' + str(bid))
                    return False

                curr_val = bid[i]
                curr_count = 1
                prev_values.append(curr_val)  # Add current value to the list

        if curr_count < min_ or curr_count > max_ and curr_val != float('-inf'):
            if config.enable_logging:    
                self.print_node_state(str(msg) + ' 3 DISCARD BROKEN MSG ' + str(bid))
            return False

        return True
    
    def garbage_collection(self, end_event):
        while not end_event.is_set():
            time.sleep(0.3)
            
            with self.last_bid_timestamp_lock:
                for key in self.last_bid_timestamp:
                    if (datetime.now() - self.last_bid_timestamp[key]["timestamp"]).total_seconds() > 1:
                        found = False
                        inf_found = False
                        for index in range(len(self.bids[key]['auction_id'])):
                            if self.bids[key]['auction_id'][index] == self.id:
                                found = True
                            elif self.bids[key]['auction_id'][index] == float('-inf'):
                                inf_found = True
                        if found and inf_found:
                            print(f"Node {self.id} -- Garbage collection on job bid {key} {self.bids[key]['auction_id']}", flush=True)
                            for index in range(len(self.bids[key]['auction_id'])):
                                if self.bids[key]['auction_id'][index] == self.id:
                                    self.updated_cpu += self.last_bid_timestamp[key]["item"]['NN_cpu'][index]
                                    self.updated_gpu += self.last_bid_timestamp[key]["item"]['NN_gpu'][index]
                                self.bids[key]['auction_id'][index] = float('-inf')
                                
                            self.updated_bw += self.last_bid_timestamp[key]["item"]['NN_data_size'][0]
                        elif inf_found:
                            for id in range(len(self.bids[key]['auction_id'])):
                                self.bids[key]['auction_id'][id] = float('-inf')
            
    def progress_bid_rounds(self, item): 
        prev_n_bet = 0    
        item['edge_id'] = float('-inf')
        item['auction_id'] = [float('-inf') for _ in range(len(item["NN_gpu"]))]
        item['bid'] = [float('-inf') for _ in range(len(item["NN_gpu"]))]
        item['timestamp'] = [datetime.now() - timedelta(days=1) for _ in range(len(item["NN_gpu"]))]
        
        while True:
            time.sleep(12)
            with self.__layer_bid_lock:
                if self.__layer_bid[item["job_id"]] == prev_n_bet:
                    break
                
                prev_n_bet = self.__layer_bid[item["job_id"]]
                
                if self.__layer_bid[item["job_id"]] < len(item["NN_gpu"]):
                    self.__layer_bid_events[item["job_id"]] += 1
                    self.q[self.id].put(item)
                    # print(f"Push {job_id} node {self.id}")
                else:
                    break

    def progress_time(self):

        # print('\nKTM' + str(self.id))



        for job in self.bids:
                    #   self.q[self.id].qsize()<=1) and \
            # print((datetime.now() - self.bids[job]['arrival_time']).total_seconds() )

            equal_values=True

            for i in range(1, config.num_edges):
                # Use get() to retrieve the 'auction_id' value for the current node and the previous node
                auction_id_i = config.nodes[i].bids.get(job, {}).get('auction_id')
                auction_id_prev = config.nodes[i - 1].bids.get(job, {}).get('auction_id')

                # Check if either 'auction_id_i' or 'auction_id_prev' is None (key not found)
                if auction_id_i is None or auction_id_prev is None:
                    continue  # Skip this iteration and move to the next iteration of the loop

                if auction_id_i != auction_id_prev:
                    equal_values = False
                    break

            if equal_values and float('-inf') not in self.bids[job]['auction_id'] and\
                not self.bids[job]['complete'] and\
                self.id in self.bids[job]['auction_id']:
                # (self.bids[job]['count'] > config.num_edges*len(self.bids[job]['auction_id']) or self.bids[job]['consensus'])  and \
                # (datetime.now() - self.bids[job]['arrival_time']).total_seconds() > 1 and\
                
                
                # print('node:' + str(self.id) + ' TIME, job: ' + str(job) + ' auction: '+str(self.bids[job]['auction_id'])+ ' time:' + str(self.bids[job]['progress_time']) +' count:' + str(self.bids[job]['count']) + ' q:' + str(self.q[self.id].qsize()) )



                
                self.bids[job]['progress_time'] += 1 # timedelta(seconds=1)
                self.bids[job]['clock'] = True 

                if int((self.bids[job]['progress_time'])) >= int(self.bids[job]['duration']) and \
                    not self.bids[job]['complete']:

                    # print('node:' + str(self.id) + ' DONE ' + str(job) + ' count:' + str(self.bids[job]['count']) + ' q:' + str(self.q[self.id].qsize()) )
                    self.bids[job]['complete']=True
                    self.bids[job]['complete_timestamp']=datetime.now()



                    if self.id in self.bids[job]['auction_id']:
                        for ind, id in enumerate(self.bids[job]['auction_id']):
                            if self.id==id:
                                # print('node:' + str(self.id) + ' available GPU:' + str(self.updated_gpu)+ ' initial CPU:' + str(self.initial_gpu) +' available CPU:' + str(self.updated_cpu)+' initial CPU:' + str(self.initial_cpu) )

                                self.updated_gpu+=self.bids[job]['NN_gpu'][ind]
                                self.updated_cpu+=self.bids[job]['NN_cpu'][ind]

                        print('node:' + str(self.id) + ' COMPLETE, job: ' + str(job) + ' available GPU:' + str(self.updated_gpu)+ ' initial CPU:' + str(self.initial_gpu) +' available CPU:' + str(self.updated_cpu)+' initial CPU:' + str(self.initial_cpu) )



                    # REBIDDING

                    for job_rebid in self.bids:

                        if float('-inf') in self.bids[job_rebid]['auction_id'] :
                            # not self.bids[job_rebid]['complete'] and \
                            # not self.bids[job_rebid]['clock'] :
                            # (datetime.now() - self.bids[job_rebid]['arrival_time']).total_seconds() > 2:
                            
                            # (datetime.now() - self.bids[job_rebid]['arrival_time']).total_seconds() > 1 :
                            print('node:' + str(self.id) + ' REBIDDING, job: ' + str(job_rebid) + ' auction: '+str(self.bids[job_rebid]['auction_id'])+' count:' + str(self.bids[job_rebid]['count']) + ' q:' + str(self.q[self.id].qsize()) )
                            # self.bids[job_rebid]['rebid']=True
                            # for i, _ in enumerate(self.bids[job_rebid]['auction_id']):

                            #     if self.bids[job_rebid]['auction_id'][i] == self.id:

                            #         self.updated_gpu+=self.bids[job_rebid]['NN_gpu'][i]
                            #         self.updated_cpu+=self.bids[job_rebid]['NN_cpu'][i]


                            #     self.bids[job_rebid]['auction_id'][i] = float('-inf')
                            #     self.bids[job_rebid]['bid'][i]= float('-inf')
                            #     self.bids[job_rebid]['timestamp'][i]=datetime.now()
                            msg = {
                                    "job_id": self.bids[job_rebid]['job_id'], 
                                    "user": self.bids[job_rebid]['user'],
                                    "edge_id": self.id, 
                                    "auction_id": copy.deepcopy(self.bids[job_rebid]['auction_id']), 
                                    "NN_gpu": self.bids[job_rebid]['NN_gpu'],
                                    "NN_cpu": self.bids[job_rebid]['NN_cpu'],
                                    "NN_data_size": self.bids[job_rebid]['NN_data_size'], 
                                    "bid": copy.deepcopy(self.bids[job_rebid]['bid']), 
                                    "timestamp": copy.deepcopy(self.bids[job_rebid]['timestamp']),
                                    "N_layer": self.bids[job_rebid]["N_layer"],
                                    "N_layer_min": self.bids[job_rebid]["N_layer_min"],
                                    "N_layer_max": self.bids[job_rebid]["N_layer_max"],
                                    "N_layer_bundle": self.bids[job_rebid]["N_layer_bundle"],
                                    "rebid":True
                                    }


                            for i in range(config.num_edges):
                                    self.q[i].put(msg)
                                    # break

                    # self.completed_jobs[job] = self.bids[job]
                    # self.completed_jobs[job]['completed_timestamp'] = datetime.now()
                    # del self.bids[job]

                else:


                    if self.id in self.bids[job]['auction_id']:
                        if self.q[self.id].qsize()<=1:
                            # print('push')
                            self.q[self.id].put({'progress_time':True})






    def work(self, event, notify_start, ret_val):
        notify_start.set()
        if self.use_net_topology:
            timeout = 15
        else:
            timeout = 5
        terminate_garbage_collect = Event()
        # t = threading.Thread(target=self.garbage_collection, args=(terminate_garbage_collect,))
        # t.start()
        
        while True:

            
            try: 
                
                self.item = self.q[self.id].get(timeout=timeout)
                if 'progress_time' in self.item:
                    # print(str(self.id)+str('\n'))
                    self.progress_time()
                else:
                    self.counter += 1
                    
                    with self.last_bid_timestamp_lock:
                        # self.last_bid_timestamp[self.item['job_id']] = {
                        #     "timestamp": datetime.now(),
                        #     "item": copy.deepcopy(self.item)
                        # }
                        self.use_queue[self.id].set()     

                            
                        
                        #print(self.item)
                        #print(self.item['user'] not in self.user_requests)
                        #print(self.item['edge_id'] is None)
                        # check msg type
                        
                        flag = False
                        # new request from client
                        if self.item['edge_id'] is None:
                            flag = True

                        if self.item['job_id'] not in self.bids:
                            if config.enable_logging:
                                self.print_node_state('IF1 q:' + str(self.q[self.id].qsize()))

                            self.init_null()
                            
                            if self.use_net_topology:    
                                with self.__layer_bid_lock:
                                    self.__layer_bid[self.item["job_id"]] = 0

                                    self.__layer_bid_events[self.item["job_id"]] = 1
                            
                                threading.Thread(target=self.progress_bid_rounds, args=(copy.deepcopy(self.item),)).start()
                                
                            self.bid(flag)

                        
                        if not flag:
                            if config.enable_logging:
                                self.print_node_state('IF2 q:' + str(self.q[self.id].qsize()))
                            # if not self.bids[self.item['job_id']]['complete'] and \
                            #    not self.bids[self.item['job_id']]['clock'] :
                            if self.id not in self.item['auction_id']:
                                self.bid(False)

                            self.update_bid()
                            # else:
                            #     print('kitemmuorten!')
                        
                        if config.progress_flag:
                            if 'rebid' in self.item:
                                self.bids[self.item['job_id']]['arrival_time'] = datetime.now()

                            self.progress_time()

                        self.bids[self.item['job_id']]['start_time'] = 0                            
                        self.bids[self.item['job_id']]['count'] += 1
                        

                        self.q[self.id].task_done()
                    
            except Empty:
                # the exception is raised if the timeout in the queue.get() expires.
                # the break statement must be executed only if the event has been set 
                # by the main thread (i.e., no more task will be submitted)
                self.use_queue[self.id].clear()
                
                all_finished = True
                for id, e in enumerate(self.use_queue):
                    if e.is_set():
                        all_finished = False
                        # print(f"Waiting for node {id} to finish")
                        
                if all_finished:
                    if event.is_set():
                        for j_key in self.resource_remind:
                            times = len(self.resource_remind[j_key]["idx"])
                            self.updated_cpu += self.resource_remind[j_key]["cpu"] * times
                            self.updated_gpu += self.resource_remind[j_key]["gpu"] * times
                            self.updated_bw += self.resource_remind[j_key]["bw"]
                            
                        with self.last_bid_timestamp_lock:
                            if self.use_net_topology:
                                self.updated_bw = self.network_topology.get_node_direct_link_bw(self.id)
                                
                            ret_val["id"] = self.id
                            ret_val["bids"] = copy.deepcopy(self.bids)
                            ret_val["counter"] = self.counter
                            ret_val["updated_cpu"] = self.updated_cpu
                            ret_val["updated_gpu"] = self.updated_gpu
                            ret_val["updated_bw"] = self.updated_bw
                            print('\nnode:' + str(self.id) + ' available GPU:' + str(self.updated_gpu)+ ' initial CPU:' + str(self.initial_gpu) +' available CPU:' + str(self.updated_cpu)+' initial CPU:' + str(self.initial_cpu) )

                            
                        print(f"Node {self.id}: received end processing signal", flush=True)
                        
                        terminate_garbage_collect.set()
                        # t.join()

                        if int(self.updated_cpu) > int(self.initial_cpu):
                            print(f"Node {self.id} -- Mannaggia updated={self.updated_cpu} initial={self.initial_cpu}", flush=True)
                        break               

                # print(str(self.q.qsize()) +" polpetta - user:"+ str(self.id) + " job_id: "  + str(self.item['job_id'])  + " from " + str(self.item['user']))

      
      
