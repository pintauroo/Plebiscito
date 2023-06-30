'''
This module impelments the behavior of a node
'''

import queue
from queue import Empty
import sys
import time
import src.config as config
from datetime import datetime, timedelta
import copy
import logging
import math 
import threading
from threading import Event
import random
import numpy

TRACE = 5    

class node:

    def __init__(self, id, seed, network_topology, use_net_topology=False):
        random.seed(seed)
        self.id = id    # unique edge node id
        self.initial_gpu = float(config.node_gpu) * random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        self.updated_gpu = self.initial_gpu# * random.uniform(0.7, 1)
        self.initial_cpu = float(config.node_cpu) * random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.updated_cpu = self.initial_cpu
        
        if use_net_topology:
            self.network_topology = network_topology
            self.initial_bw = network_topology.get_node_direct_link_bw(self.id)
        else:
            self.initial_bw = network_topology.get_node_direct_link_bw(self.id)
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

    def utility_function(self, avail_bw):
        def f(x, alpha, beta):
            if beta == 0 and x == 0:
                return 1
            
            # shouldn't happen
            if beta == 0 and x != 0:
                return 0
            
            # if beta != 0 and x == 0 is not necessary
            return math.exp(-((alpha/100) * (x - beta))**2)
            #return math.exp(-(alpha/100)*(x-beta)**2)
        
        # we assume that every job/node has always at least one CPU
        if config.filename == 'stefano':
            x = 0
            if self.item['NN_gpu'][0] == 0:
                x = 0
            else:
                x = self.item['NN_cpu'][0]/self.item['NN_gpu'][0]
                
            beta = 0
            if self.updated_gpu == 0:
                beta = 0
            else:
                beta = self.updated_cpu/self.updated_gpu
                
            return f(x, config.a, beta)
        elif config.filename == 'alpha_BW_CPU':
            return (config.a*(avail_bw/self.initial_bw))+((1-config.a)*(self.updated_cpu/self.initial_cpu)) #BW vs CPU
        elif config.filename == 'alpha_GPU_CPU':
            return (config.a*(self.updated_gpu/self.initial_gpu))+((1-config.a)*(self.updated_cpu/self.initial_cpu)) #GPU vs CPU
        elif config.filename == 'alpha_GPU_BW':
            return (config.a*(self.updated_gpu/self.initial_gpu))+((1-config.a)*(avail_bw/self.initial_bw)) # GPU vs BW


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
                        "x": copy.deepcopy(self.bids[self.item['job_id']]['x']), 
                        "timestamp": copy.deepcopy(self.bids[self.item['job_id']]['timestamp'])
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
                        "x": copy.deepcopy(custom_dict['x']), 
                        "timestamp": copy.deepcopy(custom_dict['timestamp'])
                        }
        for i in range(config.num_edges):
            if config.t.to()[i][self.id] and self.id != i:
                self.q[i].put(msg)
                # logging.debug("FORWARD NODEID:" + str(self.id) + " to " + str(i) + " " + str(self.bids[self.item['job_id']]['auction_id']))



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
                    (("\n"+str(self.bids[self.item['job_id']]['auction_id']) if bid else "") +
                    ("\n" + str(self.item.get('auction_id')) if bid and self.item.get('auction_id') is not None else "\n"))
                    )
    
    def update_local_val(self, tmp, index, id, bid, timestamp):
        tmp['job_id'] = self.item['job_id']
        tmp['auction_id'][index] = id
        tmp['x'][index] = 1
        tmp['bid'][index] = bid
        tmp['timestamp'][index] = timestamp
        return index + 1
    
    def reset(self, index):
        self.bids[self.item['job_id']]['auction_id'][index] = float('-inf')
        self.bids[self.item['job_id']]['x'][index] = 0
        self.bids[self.item['job_id']]['bid'][index]= float('-inf')
        self.bids[self.item['job_id']]['timestamp'][index]=datetime.now() - timedelta(days=1)
        return index + 1


    def init_null(self):
        #print("hello")
        self.bids[self.item['job_id']]={
            "count":int(),
            "consensus_count":int(),
            "forward_count":int(),
            "deconflictions":int(),
            "job_id": self.item['job_id'], 
            "user": int(), 
            "auction_id": list(), 
            "NN_gpu": self.item['NN_gpu'], 
            "NN_cpu": self.item['NN_cpu'], 
            "bid": list(), 
            "bid_gpu": list(), 
            "bid_cpu": list(), 
            "bid_bw": list(), 
            "x": list(), 
            "timestamp": list()
            }
        
        self.layer_bid_already[self.item['job_id']] = [False] * config.layer_number
        
        NN_len = len(self.item['NN_gpu'])
        
        for _ in range(0, NN_len):
            self.bids[self.item['job_id']]['x'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid_gpu'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid_cpu'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid_bw'].append(float('-inf'))
            self.bids[self.item['job_id']]['auction_id'].append(float('-inf'))
            self.bids[self.item['job_id']]['timestamp'].append(datetime.now() - timedelta(days=1))

    def bid(self, enable_forward=True):
        proceed = True
        if self.use_net_topology:
            proceed = self.__layer_bid_events[self.item["job_id"]].is_set()
        if not proceed:
            return False

        sequence = True
        NN_len = len(self.item['NN_gpu'])
        if self.use_net_topology:
            avail_bw = None
        else:
            avail_bw = self.updated_bw
        tmp_bid = copy.deepcopy(self.bids[self.item['job_id']])
        tmp_layer_bid_already = self.layer_bid_already[self.item['job_id']]
        first = True
        gpu_=0
        cpu_=0
        job_id_counter = 0
        first_index = None
        layers = 0
        bw_with_client = False
        previous_winner_id = 0

        if self.item['job_id'] in self.bids:                  
            for i in range(0, NN_len):
                if tmp_bid['auction_id'][i] == float('-inf') and not tmp_layer_bid_already[i]:

                    if i == 0:
                        if avail_bw == None:
                            avail_bw = self.network_topology.get_available_bandwidth_with_client(self.id)
                        NN_data_size = self.item['NN_data_size'][i]
                        first_index = i
                        bw_with_client = True
                    else:
                        if first and not bw_with_client:
                            previous_winner_id = tmp_bid['auction_id'][i-1]
                            NN_data_size = self.item['NN_data_size'][i]
                            if avail_bw == None:
                                avail_bw = self.network_topology.get_available_bandwidth_between_nodes(self.id, tmp_bid['auction_id'][i-1])
                            first = False
                    
                    if  sequence==True and \
                        self.item['NN_gpu'][i] <= self.updated_gpu - gpu_ and \
                        self.item['NN_cpu'][i] <= self.updated_cpu - cpu_ and \
                        NN_data_size <= avail_bw and \
                        tmp_bid['auction_id'].count(self.id)<config.max_layer_number:
                            
                            tmp_bid['bid'][i] = self.utility_function(avail_bw=avail_bw)
                            tmp_bid['bid_gpu'][i] = self.updated_gpu
                            tmp_bid['bid_cpu'][i] = self.updated_cpu
                            tmp_bid['bid_bw'][i] = avail_bw
                            tmp_layer_bid_already[i] = True
                            
                            gpu_ += self.item['NN_gpu'][i]
                            cpu_ += self.item['NN_cpu'][i]

                            tmp_bid['x'][i]=(1)
                            tmp_bid['auction_id'][i]=(self.id)
                            tmp_bid['timestamp'][i] = datetime.now()
                            layers += 1
                    else:
                        sequence = False
                        tmp_bid['x'][i]=(float('-inf'))
                        tmp_bid['bid'][i]=(float('-inf'))
                        tmp_bid['auction_id'][i]=(float('-inf'))
                        tmp_bid['timestamp'][i] = (datetime.now() - timedelta(days=1))


            if self.id in tmp_bid['auction_id'] and \
                self.updated_cpu - cpu_ >= 0 and \
                self.updated_gpu - gpu_ >= 0 and \
                (first_index is None or avail_bw - self.item['NN_data_size'][first_index] >= 0) and \
                tmp_bid['auction_id'].count(self.id)>=config.min_layer_number and \
                tmp_bid['auction_id'].count(self.id)<=config.max_layer_number and \
                self.integrity_check(tmp_bid['auction_id'], 'bid'):
                # print(tmp_bid['auction_id'])
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
                    self.layer_bid_already[self.item['job_id']] = tmp_layer_bid_already
                    first_index = tmp_bid['auction_id'].index(self.id)
                    if not self.use_net_topology:
                        self.updated_bw -= self.item['NN_data_size'][first_index] 

                    self.bids[self.item['job_id']] = copy.deepcopy(tmp_bid)

                    self.updated_gpu -= gpu_
                    self.updated_cpu -= cpu_

                    job_id_counter += 1
                    self.bids[self.item['job_id']]['count'] = job_id_counter
                    
                    if enable_forward:
                        self.forward_to_neighbohors()
                    
                    if self.use_net_topology:
                        self.__layer_bid_events[self.item["job_id"]].clear()
                        with self.__layer_bid_lock:
                            self.__layer_bid[self.item["job_id"]] += layers
                    
                    return True
                else:
                    return False
        else:
            if config.enable_logging:
                self.print_node_state('Value not in dict (first_msg)', type='error')
            return False


    def lost_bid(self, index, z_kj, tmp_local, tmp_gpu, tmp_cpu, tmp_bw):
        first_time = True
        
        while index<config.layer_number and self.item['auction_id'][index] == z_kj:
            tmp_gpu +=  self.item['NN_gpu'][index]
            tmp_cpu +=  self.item['NN_cpu'][index]
            if first_time:
                tmp_bw += self.item['NN_data_size'][index]
                first_time = False
            index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
        return index, tmp_gpu, tmp_cpu, tmp_bw
    
    def deconfliction(self):
        rebroadcast = False
        k = self.item['edge_id'] # sender
        i = self.id # receiver
        self.bids[self.item['job_id']]['deconflictions']+=1
        release_to_client = False
        previous_winner_id = float('-inf')
        
        tmp_local = copy.deepcopy(self.bids[self.item['job_id']])
        prev_bet = copy.deepcopy(self.bids[self.item['job_id']])
        tmp_gpu = 0 
        tmp_cpu = 0 
        tmp_bw = 0
        index = 0
        
        if self.use_net_topology:
            initial_count = 0
            for j in tmp_local["auction_id"]:
                if j != float('-inf'):
                    initial_count += 1

        while index < config.layer_number:
            if self.item['job_id'] in self.bids:
                z_kj = self.item['auction_id'][index]
                z_ij = tmp_local['auction_id'][index]
                y_kj = self.item['bid'][index]
                y_ij = tmp_local['bid'][index]
                t_kj = self.item['timestamp'][index]
                t_ij = tmp_local['timestamp'][index]

                # logging.log(TRACE,'DECONFLICTION - NODEID(i):' + str(i) +
                #               ' sender(k):' + str(k) +
                #               ' z_kj:' + str(z_kj) +
                #               ' z_ij:' + str(z_ij) +
                #               ' y_kj:' + str(y_kj) +
                #               ' y_ij:' + str(y_ij) +
                #               ' t_kj:' + str(t_kj) +
                #               ' t_ij:' + str(t_ij)
                #                )
                if z_kj==k : 
                    if z_ij==i:
                        if (y_kj>y_ij): 
                            rebroadcast = True
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #1-#2')
                            if index == 0:
                                release_to_client = True
                            else:
                                previous_winner_id = prev_bet['auction_id'][index-1]
                            index, tmp_gpu, tmp_cpu, tmp_bw = self.lost_bid(index, z_kj, tmp_local, tmp_gpu, tmp_cpu, tmp_bw)
                        elif (y_kj==y_ij and z_kj<z_ij):
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #3')
                            rebroadcast = True
                            if index == 0:
                                release_to_client = True
                            else:
                                previous_winner_id = prev_bet['auction_id'][index-1]
                            index, tmp_gpu, tmp_cpu, tmp_bw = self.lost_bid(index, z_kj, tmp_local, tmp_gpu, tmp_cpu, tmp_bw)           
                        elif (y_kj<y_ij):
                            rebroadcast = True
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #1-#2')
                            while index<config.layer_number and tmp_local['auction_id'][index]  == z_ij:
                                index = self.update_local_val(tmp_local, index, z_ij, tmp_local['bid'][index], datetime.now())
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #3else')
                            index+=1

                    elif  z_ij==k:
                        if  t_kj>t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#4')
                            index = self.update_local_val(tmp_local, index, k, self.item['bid'][index], t_kj)
                            rebroadcast = True
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #4else')
                            index+=1
                    
                    elif  z_ij == float('-inf'):
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #12')
                        index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], t_kj)
                        rebroadcast = True

                    elif z_ij!=i and z_ij!=k:
                        if y_kj>y_ij and t_kj>=t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #7')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        elif y_kj<y_ij and t_kj>=t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #8')
                            rebroadcast = True
                            index+=1
                        elif y_kj==y_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #9else')
                            rebroadcast = True
                            index+=1
                        elif y_kj<y_ij and t_kj<t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #10')
                            index += 1
                            rebroadcast = True
                        elif y_kj>y_ij and t_kj<t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #11')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True  
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #11else')
                            index += 1                   
                    
                    else:
                        if config.enable_logging:
                            logging.log(TRACE, "eccoci")    
                
                elif z_kj==i:                                
                    if z_ij==i:
                        if t_kj>t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #13Flavio')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True 
                            
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #13elseFlavio')
                            index+=1
                    
                    elif z_ij==k:
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #14')
                        index = self.reset(index)
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
                        index = self.reset(index)
                        rebroadcast = True
                        
                    elif z_ij == float('-inf'):
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #34')
                        index+=1
                        
                    elif z_ij!=i and z_ij!=k:
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #33')
                        index = self.reset(index)
                        rebroadcast = True
                        
                    else:
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #33else')
                        index+=1

                elif z_kj!=i or z_kj!=k:   
                                     
                    if z_ij==i:
                        if (y_kj>y_ij):
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#17')
                            rebroadcast = True
                            if index == 0:
                                release_to_client = True
                            else:
                                previous_winner_id = prev_bet['auction_id'][index-1]
                            index, tmp_gpu, tmp_cpu, tmp_bw = self.lost_bid(index, z_kj, tmp_local, tmp_gpu, tmp_cpu, tmp_bw)
                        elif (y_kj==y_ij and z_kj<z_ij):
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#17')
                            rebroadcast = True
                            if index == 0:
                                release_to_client = True
                            else:
                                previous_winner_id = prev_bet['auction_id'][index-1]
                            index, tmp_gpu, tmp_cpu, tmp_bw = self.lost_bid(index, z_kj, tmp_local, tmp_gpu, tmp_cpu, tmp_bw)
                        elif (y_kj<y_ij):
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#19')
                            rebroadcast = True
                            while index<config.layer_number and tmp_local['auction_id'][index]  == z_ij:
                                index = self.update_local_val(tmp_local, index, z_ij, tmp_local['bid'][index], datetime.now())
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #19else')
                            index+=1
                            rebroadcast = True

                    elif z_ij==k:
                        
                        if y_kj>y_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #20Flavio')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True 
                        elif (y_kj==y_ij and z_kj<z_ij):
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #3stefano')
                            rebroadcast = True
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                        elif t_kj>t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#20')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        elif t_kj<t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#21')
                            index = self.reset(index)
                            rebroadcast = True
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #21else')
                            index+=1

                    elif z_ij == z_kj:
                    
                        if t_kj>t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#22')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #22else')
                            index+=1
                    
                    elif z_ij == float('-inf'):
                        if config.enable_logging:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#30')
                        index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                        rebroadcast = True

                    elif z_ij!=i and z_ij!=k and z_ij!=z_kj:
                        if y_kj>y_ij and t_kj>=t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#25')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])                   
                            rebroadcast = True
                        elif y_kj<y_ij and t_kj<t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#26')
                            rebroadcast = True
                            index+=1
                        elif y_kj==y_ij and z_kj<z_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#27')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])                   
                            rebroadcast = True
                        elif y_kj==y_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#27')
                            index+=1
                        elif y_kj<y_ij and t_kj>t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#28')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        elif y_kj>y_ij and t_kj<t_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#29')
                            index+=1
                            rebroadcast = True
                        else:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#29else')
                            index+=1
                    
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
            
            if self.use_net_topology:
                if release_to_client:
                    self.network_topology.release_bandwidth_node_and_client(self.id, tmp_bw, self.item['job_id'])
                elif previous_winner_id != float('-inf'):
                    self.network_topology.release_bandwidth_between_nodes(previous_winner_id, self.id, tmp_bw, self.item['job_id'])  
            else:
                self.updated_bw += tmp_bw                  

            self.bids[self.item['job_id']] = copy.deepcopy(tmp_local)
            self.updated_gpu += tmp_gpu
            self.updated_cpu += tmp_cpu
            
            if self.use_net_topology:
                count = 0
                for i in tmp_local["auction_id"]:
                    if i != float('-inf'):
                        count += 1
                if initial_count != count:
                    self.__layer_bid_events[self.item["job_id"]].clear()
                    with self.__layer_bid_lock:
                        self.__layer_bid[self.item["job_id"]] = count

            return rebroadcast, False
        else:
            if self.use_net_topology:
                print(f"Failure in node {self.id}. Deconfliction failed. Exiting ...")
                sys.exit(1)
            
            self.forward_to_neighbohors(custom_dict=copy.deepcopy(self.item))

            cpu = 0
            gpu = 0
            bw = 0
            first = False
            
            for id, b in enumerate(self.item["auction_id"]):
                if b == self.id:
                    cpu -= self.item['NN_cpu'][id]
                    gpu -= self.item['NN_gpu'][id]
                    if not first:
                        bw -= self.item['NN_data_size'][id]
                    first = True
            
            first = False
            for id, b in enumerate(prev_bet["auction_id"]):
                if b == self.id:
                    cpu += self.item['NN_cpu'][id]
                    gpu += self.item['NN_gpu'][id]
                    if not first:
                        bw += self.item['NN_data_size'][id]
                    first = True    
                    
            self.updated_cpu += cpu
            self.updated_gpu += gpu
            self.updated_bw += bw
            
            for key in self.item:
                self.bids[self.item['job_id']][key] = copy.deepcopy(self.item[key])
                    
            return False, True       

       
    def update_bid(self):
        if self.item['job_id'] in self.bids:
        
            # Consensus check
            if  self.bids[self.item['job_id']]['auction_id']==self.item['auction_id'] and \
                self.bids[self.item['job_id']]['bid'] == self.item['bid'] and \
                self.bids[self.item['job_id']]['timestamp'] == self.item['timestamp']:
                    
                    if  self.id not in self.bids[self.item['job_id']]['auction_id'] and \
                        float('-inf') in self.bids[self.item['job_id']]['auction_id']:
                            self.bid()
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
                if self.use_net_topology:
                    proceed == self.__layer_bid_events[self.item["job_id"]].is_set()
                if not integrity_fail and self.id not in self.bids[self.item['job_id']]['auction_id'] and float('-inf') in self.bids[self.item['job_id']]['auction_id'] and proceed:
                    success = self.bid()
                    
                if not success and rebroadcast:
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
                    if curr_count < config.min_layer_number or curr_count > config.max_layer_number:
                        self.print_node_state(str(msg) + ' DISCARD BROKEN MSG ' + str(bid))
                        return False
                    
                    curr_val = bid[i]
                    curr_count = 1
        
        if curr_count < config.min_layer_number or curr_count > config.max_layer_number:
            if config.enable_logging:
                self.print_node_state(str(msg) + ' DISCARD BROKEN MSG ' + str(bid))
            return False
        
        return True
    
    def integrity_check(self, bid, msg):
        curr_val = bid[0]
        curr_count = 1
        for i in range(1, len(bid)):
            
            if bid[i] == curr_val:
                curr_count += 1
            else:
                if (curr_count < config.min_layer_number or curr_count > config.max_layer_number) and curr_val != float('-inf'):
                    self.print_node_state(str(msg) + ' DISCARD BROKEN MSG ' + str(bid))
                    return False
                    
                curr_val = bid[i]
                curr_count = 1
        
        if curr_count < config.min_layer_number or curr_count > config.max_layer_number:
            if config.enable_logging:
                self.print_node_state(str(msg) + ' DISCARD BROKEN MSG ' + str(bid))
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
                                self.bids[key]['x'][index] = float('-inf')
                                
                            self.updated_bw += self.last_bid_timestamp[key]["item"]['NN_data_size'][0]
                        elif inf_found:
                            for id in range(len(self.bids[key]['auction_id'])):
                                self.bids[key]['auction_id'][id] = float('-inf')
                                self.bids[key]['x'][index] = float('-inf')
            
    def progress_bid_rounds(self, item): 
        prev_n_bet = 0    
        item['edge_id'] = float('-inf')
        item['auction_id'] = [float('-inf') for _ in range(len(item["NN_gpu"]))]
        item['bid'] = [float('-inf') for _ in range(len(item["NN_gpu"]))]
        item['timestamp'] = [datetime.now() - timedelta(days=1) for _ in range(len(item["NN_gpu"]))]
        
        while True:
            time.sleep(13)
            with self.__layer_bid_lock:
                if self.__layer_bid[item["job_id"]] == prev_n_bet:
                    break
                
                prev_n_bet = self.__layer_bid[item["job_id"]]
                
                if self.__layer_bid[item["job_id"]] < len(item["NN_gpu"]):
                    self.__layer_bid_events[self.item["job_id"]].set()
                    self.q[self.id].put(item)
                else:
                    break
        
    
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
                        self.init_null()
                        
                        if self.use_net_topology:    
                            with self.__layer_bid_lock:
                                self.__layer_bid[self.item["job_id"]] = 0

                            e = Event()
                            e.set()
                            self.__layer_bid_events[self.item["job_id"]] = e 
                        
                            threading.Thread(target=self.progress_bid_rounds, args=(copy.deepcopy(self.item),)).start()
                            
                        if config.enable_logging:
                            self.print_node_state('IF1 q:' + str(self.q[self.id].qsize()))
                        
                        self.bid(flag)
                    
                    if not flag:
                        if config.enable_logging:
                            self.print_node_state('IF2 q:' + str(self.q[self.id].qsize()))
                        
                        self.update_bid()

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
                        with self.last_bid_timestamp_lock:
                            if self.use_net_topology:
                                self.updated_bw = self.network_topology.get_node_direct_link_bw(self.id)
                                
                            ret_val["id"] = self.id
                            ret_val["bids"] = copy.deepcopy(self.bids)
                            ret_val["counter"] = self.counter
                            ret_val["updated_cpu"] = self.updated_cpu
                            ret_val["updated_gpu"] = self.updated_gpu
                            ret_val["updated_bw"] = self.updated_bw
                            
                        print(f"Node {self.id}: received end processing signal", flush=True)
                        
                        terminate_garbage_collect.set()
                        # t.join()
                        if self.updated_cpu > self.initial_cpu:
                            print(f"Node {self.id} -- Mannaggia", flush=True)
                        break               

                # print(str(self.q.qsize()) +" polpetta - user:"+ str(self.id) + " job_id: "  + str(self.item['job_id'])  + " from " + str(self.item['user']))

      
      
