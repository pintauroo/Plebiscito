'''
This module impelments the behavior of a node
'''

import queue
import sys
import time
import src.config as config
from datetime import datetime, timedelta
import copy
import logging
import math 
import random

TRACE = 5


class node:

    def __init__(self, id):
        self.id = id    # unique edge node id
        self.initial_gpu = float(config.node_gpu) # * random.uniform(0.7, 1)
        self.updated_gpu = self.initial_gpu 
        self.initial_cpu = float(config.node_cpu) # * random.uniform(0.7, 1)
        self.updated_cpu = self.initial_cpu 
        self.initial_bw = config.t.b
        self.updated_bw = self.initial_bw
        
        self.counter = 0
        
        self.user_requests = []
        self.item={}
        self.bids= {}
        self.layer_bid_already = {}


    def append_data(self, d):
        self.q.put(d)

    def join_queue(self):
        self.q.join()
        
    def set_queues(self, q, use_queue):
        self.q = q
        self.use_queue = use_queue

    def utility_function(self):
        def f(x, alpha, beta):
            if beta == 0 and x == 0:
                return 1
            
            # shouldn't happen
            if beta == 0 and x != 0:
                return 0
            
            # if beta != 0 and x == 0 is not necessary
            
            return math.exp(-(alpha/20)*(x-beta)**2)
        
        # we assume that every job/node has always at least one CPU
        if config.filename == 'stefano':
            x = 0
            if self.item['NN_gpu'][0] == 0:
                x = 0
            else:
                x = self.item['NN_cpu'][0]/self.item['NN_gpu'][0]
                
            beta = 0
            if self.initial_gpu == 0:
                beta = 0
            else:
                beta = self.initial_cpu/self.initial_gpu
                
            return f(x, config.a, beta)
        # elif config.filename == 'alpha_BW_CPU':
        #     return (config.a*(self.updated_bw/self.initial_bw))+((1-config.a)*(self.updated_cpu/self.initial_cpu)) #BW vs CPU
        # elif config.filename == 'alpha_GPU_CPU':
        #     return (config.a*(self.updated_gpu/self.initial_gpu))+((1-config.a)*(self.updated_cpu/self.initial_cpu)) #GPU vs CPU
        # elif config.filename == 'alpha_GPU_BW':
        #     return (config.a*(self.updated_gpu/self.initial_gpu))+((1-config.a)*(self.updated_bw/self.initial_bw)) # GPU vs BW


        elif config.filename == 'alpha_BW_CPU':
            return (config.a*(self.updated_bw/config.tot_bw))+((1-config.a)*(self.updated_cpu/config.tot_cpu)) #BW vs CPU
        elif config.filename == 'alpha_GPU_CPU':
            return (config.a*(self.updated_gpu/config.tot_gpu))+((1-config.a)*(self.updated_cpu/config.tot_cpu)) #GPU vs CPU
        elif config.filename == 'alpha_GPU_BW':
            return (config.a*(self.updated_gpu/config.tot_gpu))+((1-config.a)*(self.updated_bw/config.tot_bw)) # GPU vs BW


    def forward_to_neighbohors(self):
        self.print_node_state('FORWARD', True)
        self.bids[self.item['job_id']]['forward_count']+=1
        msg={
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
        for i in range(config.num_edges):
            if config.t.to()[i][self.id] and self.id != i:
                config.nodes[i].append_data(msg)



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
                    " initial BW:" + str(self.initial_bw) +
                    " available BW:" + str(self.updated_bw) +
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

    def bid(self):

        sequence = True
        NN_len = len(self.item['NN_gpu'])
        avail_bw = self.updated_bw
        tmp_bid = copy.deepcopy(self.bids[self.item['job_id']])
        tmp_layer_bid_already = self.layer_bid_already[self.item['job_id']]
        first = True
        gpu_=0
        cpu_=0
        first_index = None
        layers = 0

        if self.item['job_id'] in self.bids:
            
            for i in range(0, NN_len):
                if tmp_bid['auction_id'][i] == self.id:
                    layers = layers + 1
                    
            for i in range(0, NN_len):
                if tmp_bid['auction_id'][i] == float('-inf') and not tmp_layer_bid_already[i]:

                    if first:
                        NN_data_size = self.item['NN_data_size'][i]
                        first_index = i
                        first = False
                    
                    if  sequence==True and \
                        self.item['NN_gpu'][i] <= self.updated_gpu - gpu_ and \
                        self.item['NN_cpu'][i] <= self.updated_cpu - cpu_ and \
                        NN_data_size <= avail_bw and \
                        tmp_bid['auction_id'].count(self.id)<config.max_layer_number:
                            
                            tmp_bid['bid'][i] = self.utility_function()
                            tmp_bid['bid_gpu'][i] = self.updated_gpu
                            tmp_bid['bid_cpu'][i] = self.updated_cpu
                            tmp_bid['bid_bw'][i] = self.updated_bw
                            tmp_layer_bid_already[i] = True
                            
                            gpu_ += self.item['NN_gpu'][i]
                            cpu_ += self.item['NN_cpu'][i]

                            tmp_bid['x'][i]=(1)
                            tmp_bid['auction_id'][i]=(self.id)
                            tmp_bid['timestamp'][i] = datetime.now()
                    else:
                        sequence = False
                        tmp_bid['x'][i]=(float('-inf'))
                        tmp_bid['bid'][i]=(float('-inf'))
                        tmp_bid['auction_id'][i]=(float('-inf'))
                        tmp_bid['timestamp'][i] = (datetime.now() - timedelta(days=1))


            if self.id in tmp_bid['auction_id'] and \
                self.updated_cpu - cpu_ >= 0 and \
                self.updated_gpu - gpu_ >= 0 and \
                (first_index is None or self.updated_bw - self.item['NN_data_size'][first_index] >= 0) and \
                tmp_bid['auction_id'].count(self.id)>=config.min_layer_number and \
                tmp_bid['auction_id'].count(self.id)<=config.max_layer_number and \
                self.integrity_check(tmp_bid['auction_id'], 'bid'):
                # print(tmp_bid['auction_id'])
                # logging.log(TRACE, "BID NODEID:" + str(self.id) + ", auction: " + str(tmp_bid['auction_id']))
                
                self.layer_bid_already[self.item['job_id']] = tmp_layer_bid_already
                first_index = tmp_bid['auction_id'].index(self.id)

                self.bids[self.item['job_id']] = copy.deepcopy(tmp_bid)

                self.updated_bw -= self.item['NN_data_size'][first_index]  
                self.updated_gpu -= gpu_
                self.updated_cpu -= cpu_

                job_id_counter += 1
                self.bids[self.item['job_id']]['count'] = job_id_counter
                
                self.forward_to_neighbohors()
        else:
            if config.enable_logging:
                self.print_node_state('Value not in dict (first_msg)', type='error')


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
        
        tmp_local = copy.deepcopy(self.bids[self.item['job_id']])
        tmp_gpu = 0 
        tmp_cpu = 0 
        tmp_bw = 0
        index = 0

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
                            index, tmp_gpu, tmp_cpu, tmp_bw = self.lost_bid(index, z_kj, tmp_local, tmp_gpu, tmp_cpu, tmp_bw)
                        elif (y_kj==y_ij and z_kj<z_ij):
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #3')
                            rebroadcast = True
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
                        if (y_kj>y_ij) or (y_kj==y_ij and z_kj<z_ij):
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  '#17')
                            rebroadcast = True
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

                    elif z_ij==k:
                        
                        if y_kj<y_ij:
                            if config.enable_logging:
                                logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #20Flavio')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True 
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
                        elif y_kj==y_ij :
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

            self.bids[self.item['job_id']] = copy.deepcopy(tmp_local)
            # print(tmp_local['auction_id'])
            self.updated_gpu += tmp_gpu
            self.updated_cpu += tmp_cpu
            self.updated_bw += tmp_bw

            return rebroadcast
        else:
            return False            



       
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
                rebroadcast = self.deconfliction()

                if self.id not in self.bids[self.item['job_id']]['auction_id'] and float('-inf') in self.bids[self.item['job_id']]['auction_id']:
                    self.bid()
                    
                elif rebroadcast:
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

    def work(self, event, ret_val):
        retry = 0
        while True:
            try: 
                self.item = self.q[self.id].get(timeout=2)
                self.counter += 1

                if self.item['job_id'] not in self.bids:
                    self.init_null()
                
                # check msg type
                if self.item['edge_id'] is not None and self.item['user'] in self.user_requests:
                    if config.enable_logging:
                        self.print_node_state('IF1 q:' + str(self.q.qsize())) # edge to edge request
                    self.new_msg()

                elif self.item['edge_id'] is None and self.item['user'] not in self.user_requests:
                    if config.enable_logging:
                        self.print_node_state('IF2 q:' + str(self.q.qsize())) # brand new request from client
                    self.user_requests.append(self.item['user'])
                    self.bid()

                elif self.item['edge_id'] is not None and self.item['user'] not in self.user_requests:
                    if config.enable_logging:
                        self.print_node_state('IF3 q:' + str(self.q.qsize())) # edge anticipated client request
                    self.user_requests.append(self.item['user'])
                    self.new_msg()

                elif self.item['edge_id'] is None and self.item['user'] in self.user_requests:
                    if config.enable_logging:
                        self.print_node_state('IF4 q:' + str(self.q.qsize())) # client after edge request
                    self.bid()

                self.q[self.id].task_done()
            except:
                # the exception is raised if the timeout in the queue.get() expires.
                # the break statement must be executed only if the event has been set 
                # by the main thread (i.e., no more task will be submitted)
                if retry < 2:
                    self.use_queue[self.id].clear()
                    if self.q[self.id].qsize() != 0:
                        retry = 0
                        self.use_queue[self.id].set()
                    else:
                        retry +=1
                
                if event.is_set():
                    ret_val["id"] = self.id
                    ret_val["bids"] = copy.deepcopy(self.bids)
                    ret_val["counter"] = self.counter
                    ret_val["updated_cpu"] = self.updated_cpu
                    ret_val["updated_gpu"] = self.updated_gpu
                    ret_val["updated_bw"] = self.updated_bw
                    # Event has occurred, handle it in your desired way
                    print(f"Node {self.id}: received end processing signal", flush=True)
                    break

                

                # print(str(self.q.qsize()) +" polpetta - user:"+ str(self.id) + " job_id: "  + str(self.item['job_id'])  + " from " + str(self.item['user']))

      
      
