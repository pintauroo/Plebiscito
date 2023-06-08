'''
This module impelments the behavior of a node
'''

import queue
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
        self.initial_gpu = float(config.node_gpu) * random.uniform(0.7, 1)
        self.updated_gpu = self.initial_gpu * random.uniform(0.7, 1)
        self.initial_cpu = float(config.node_cpu) * random.uniform(0.7, 1)
        self.updated_cpu = self.initial_cpu * random.uniform(0.7, 1)
        self.initial_bw = config.t.b
        self.updated_bw = self.initial_bw
        
        self.q = queue.Queue()
        self.user_requests = []
        self.item={}
        self.bids= {}
        self.layer_bid_already = {}


    def append_data(self, d):
        self.q.put(d)

    def join_queue(self):
        self.q.join()

    def utility_function(self):
        def f(x, alpha, beta):
            return math.exp(-(alpha/2)*(x-beta)**2)
        
        if config.filename == 'stefano':
            return f(self.item['NN_cpu'][0]/self.item['NN_gpu'][0], config.a, self.initial_cpu/self.initial_gpu)
        elif config.filename == 'alpha_BW_CPU':
            return (config.a*(self.updated_bw/config.tot_bw))+((1-config.a)*(self.updated_cpu/config.tot_cpu)) #BW vs CPU
        elif config.filename == 'alpha_GPU_CPU':
            return (config.a*(self.updated_gpu/config.tot_gpu))+((1-config.a)*(self.updated_cpu/config.tot_cpu)) #GPU vs CPU
        elif config.filename == 'alpha_GPU_BW':
            return (config.a*(self.updated_gpu/config.tot_gpu))+((1-config.a)*(self.updated_bw/config.tot_bw)) # GPU vs BW


    def forward_to_neighbohors(self):
        self.print_node_state('FORWARD', True)
        for i in range(config.num_edges):
            if config.t.to()[i][self.id] and self.id != i:
                config.nodes[i].append_data({
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
                    })
                # logging.debug("FORWARD NODEID:" + str(self.id) + " to " + str(i) + " " + str(self.bids[self.item['job_id']]['auction_id']))



    def print_node_state(self, msg, bid=False, type='debug'):
        logger_method = getattr(logging, type)
        #print(str(self.item.get('auction_id')) if bid and self.item.get('auction_id') is not None else "\n")
        logger_method(str(msg) +
                    " job_id:" + str(self.item['job_id']) +
                    " NODEID:" + str(self.id) +
                    " from_edge:" + str(self.item['edge_id']) +
                    " available GPU:" + str(self.updated_gpu) +
                    " available CPU:" + str(self.updated_cpu) +
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
        job_id_counter = self.bids[self.item['job_id']]['count']

        if self.item['job_id'] in self.bids and job_id_counter<config.max_layer_number:

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

                self.updated_bw -= self.item['NN_data_size'][first_index]  
                self.updated_gpu -= gpu_
                self.updated_cpu -= cpu_

                self.bids[self.item['job_id']] = tmp_bid
                job_id_counter += 1
                self.bids[self.item['job_id']]['count'] = job_id_counter
                
                self.forward_to_neighbohors()
        else:
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
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #1-#2')
                            index, tmp_gpu, tmp_cpu, tmp_bw = self.lost_bid(index, z_kj, tmp_local, tmp_gpu, tmp_cpu, tmp_bw)
                        elif (y_kj==y_ij and z_kj<z_ij):
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #3')
                            rebroadcast = True
                            index, tmp_gpu, tmp_cpu, tmp_bw = self.lost_bid(index, z_kj, tmp_local, tmp_gpu, tmp_cpu, tmp_bw)           
                        elif (y_kj<y_ij):
                            rebroadcast = True
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #1-#2')
                            while index<config.layer_number and tmp_local['auction_id'][index]  == z_ij:
                                index = self.update_local_val(tmp_local, index, z_ij, tmp_local['bid'][index], datetime.now())
                        else:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #3else')
                            index+=1
                        # if (y_kj>y_ij) or (y_kj==y_ij and z_kj<z_ij):
                        #     rebroadcast = True
                        #     logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #1-#2')
                        #     index, tmp_gpu, tmp_cpu, tmp_bw = self.lost_bid(index, z_kj, tmp_local, tmp_gpu, tmp_cpu, tmp_bw)
                        # elif (y_kj<y_ij):
                        #     logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #3')
                        #     rebroadcast = True
                        #     while index<config.layer_number and tmp_local['auction_id'][index]  == z_ij:
                        #         index = self.update_local_val(tmp_local, index, z_ij, tmp_local['bid'][index], datetime.now())

                        # else:
                        #     logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #3else')
                        #     index+=1

                    elif  z_ij==k:
                        if  t_kj>t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#4')
                            index = self.update_local_val(tmp_local, index, k, self.item['bid'][index], t_kj)

                        else:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #4else')
                            index+=1
                    
                    elif  z_ij == float('-inf'):
                        logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #12')
                        index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], t_kj)
                        rebroadcast = True

                    elif z_ij!=i and z_ij!=k:
                        if y_kj>y_ij and t_kj>=t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #7')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        elif y_kj<y_ij and t_kj>=t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #8')
                            rebroadcast = True
                            index+=1
                        elif y_kj==y_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #9else')
                            rebroadcast = True
                            index+=1
                        elif y_kj<y_ij and t_kj<t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #10')
                            index += 1
                            rebroadcast = True
                        elif y_kj>y_ij and t_kj<t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #11')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True  
                        else:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #11else')
                            index += 1                   
                    
                    else:
                        logging.log(TRACE, "eccoci")    
                
                elif z_kj==i:                                
                    if z_ij==i:
                        if t_kj>t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #13Flavio')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True 
                            
                        else:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #13elseFlavio')
                            index+=1
                    
                    elif z_ij==k:
                        logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #14')
                        index = self.reset(index)                        

                    elif z_ij == float('-inf'):
                        logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #16')
                        rebroadcast = True
                        index+=1
                    
                    elif z_ij!=i and z_ij!=k:
                        logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #15')
                        rebroadcast = True
                        index+=1
                    
                    else:
                        logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #15else')
                        rebroadcast = True
                        index+=1                
                
                elif z_kj == float('-inf'):
                    if z_ij==i:
                        logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #31')
                        rebroadcast = True
                        index+=1
                        
                    elif z_ij==k:
                        logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #32')
                        index = self.reset(index)
                        rebroadcast = True
                        
                    elif z_ij == float('-inf'):
                        logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #34')
                        index+=1
                        
                    elif z_ij!=i and z_ij!=k:
                        logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #33')
                        index = self.reset(index)
                        rebroadcast = True
                        
                    else:
                        logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #33else')
                        index+=1

                elif z_kj!=i or z_kj!=k:   
                                     
                    if z_ij==i:
                        if (y_kj>y_ij) or (y_kj==y_ij and z_kj<z_ij):
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#17')
                            rebroadcast = True
                            index, tmp_gpu, tmp_cpu, tmp_bw = self.lost_bid(index, z_kj, tmp_local, tmp_gpu, tmp_cpu, tmp_bw)
                        elif (y_kj<y_ij):
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#19')
                            rebroadcast = True
                            while index<config.layer_number and tmp_local['auction_id'][index]  == z_ij:
                                index = self.update_local_val(tmp_local, index, z_ij, tmp_local['bid'][index], datetime.now())
                        else:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #19else')
                            index+=1

                    elif z_ij==k:
                        
                        if y_kj<y_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #20Flavio')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True 
                        elif t_kj>t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#20')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        elif t_kj<t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#21')
                            index = self.reset(index)
                            rebroadcast = True
                        else:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #21else')
                            index+=1

                    elif z_ij == z_kj:
                    
                        if t_kj>t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#22')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                        else:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #22else')
                            index+=1
                    
                    elif z_ij == float('-inf'):
                        logging.log(TRACE, 'NODEID:'+str(self.id) +  '#30')
                        index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                        rebroadcast = True

                    elif z_ij!=i and z_ij!=k and z_ij!=z_kj:
                        if y_kj>y_ij and t_kj>=t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#25')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])                   
                            rebroadcast = True
                        elif y_kj<y_ij and t_kj<t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#26')
                            rebroadcast = True
                            index+=1
                        elif y_kj==y_ij and z_kj<z_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#27')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])                   
                            rebroadcast = True
                        elif y_kj==y_ij :
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#27')
                            index+=1
                        elif y_kj<y_ij and t_kj>t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#28')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(tmp_local, index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        elif y_kj>y_ij and t_kj<t_ij:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#29')
                            index+=1
                            rebroadcast = True
                        else:
                            logging.log(TRACE, 'NODEID:'+str(self.id) +  '#29else')
                            index+=1
                    
                    else:
                        logging.log(TRACE, 'NODEID:'+str(self.id) +  ' #29else2')
                        index+=1
                
                else:
                    self.print_node_state('smth wrong?', type='error')
                    pass

            else:
                self.print_node_state('Value not in dict (deconfliction)', type='error')

        if self.integrity_check(tmp_local['auction_id'], 'deconfliction') and \
            tmp_local['auction_id']!=self.bids[self.item['job_id']]['auction_id']:
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
                            self.print_node_state('Consensus -', True)
                            # pass
                    
            else:
                self.print_node_state('BEFORE', True)
                rebroadcast = self.deconfliction()

                if self.id not in self.bids[self.item['job_id']]['auction_id'] and float('-inf') in self.bids[self.item['job_id']]['auction_id']:
                    self.bid()
                    
                elif rebroadcast:# and self.integrity_check(self.item['auction_id'], 'update_bid'): 
                    self.forward_to_neighbohors()
                # self.print_node_state('AFTER', True)

        else:
            self.print_node_state('Value not in dict (update_bid)', type='error')

    def new_msg(self):
        if str(self.id) == str(self.item['edge_id']):
            self.print_node_state('This was not supposed to be received', type='error')

        elif self.item['job_id'] in self.bids:
            if self.integrity_check(self.item['auction_id'], 'new msg'):
                self.update_bid()
            else:
                print('new_msg' + str(self.item) + '\n' + str(self.bids[self.item['job_id']]))
        else:
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
                        # print(bid)
                        return False
                    
                    curr_val = bid[i]
                    curr_count = 1
        
        if curr_count < config.min_layer_number or curr_count > config.max_layer_number:
            self.print_node_state(str(msg) + ' DISCARD BROKEN MSG ' + str(bid))
            return False
        
        return True
    
    def integrity_check(self, bid, msg):
        curr_count = {}
        for i in range(0, len(bid)):
            if bid[i] == float('-inf'):
                continue
            
            if bid[i] not in curr_count:
                curr_count[bid[i]] = 1
            else:
                curr_count[bid[i]] += 1
            
        for key in curr_count:
            if curr_count[key] < config.min_layer_number or curr_count[key] > config.max_layer_number:
                self.print_node_state(str(msg) + ' DISCARD BROKEN MSG ' + str(bid))
                return False
                
        return True

    def work(self, event):
        while True:
            try: 
                self.item = self.q.get(timeout=5)
                config.counter += 1

                if self.item['job_id'] not in self.bids:
                    self.init_null()
                
                # check msg type
                if self.item['edge_id'] is not None and self.item['user'] in self.user_requests:
                    self.print_node_state('IF1 q:' + str(self.q.qsize())) # edge to edge request
                    self.new_msg()

                elif self.item['edge_id'] is None and self.item['user'] not in self.user_requests:
                    self.print_node_state('IF2 q:' + str(self.q.qsize())) # brand new request from client
                    self.user_requests.append(self.item['user'])
                    self.bid()

                elif self.item['edge_id'] is not None and self.item['user'] not in self.user_requests:
                    self.print_node_state('IF3 q:' + str(self.q.qsize())) # edge anticipated client request
                    self.user_requests.append(self.item['user'])
                    self.new_msg()

                elif self.item['edge_id'] is None and self.item['user'] in self.user_requests:
                    self.print_node_state('IF4 q:' + str(self.q.qsize())) # client after edge request
                    self.bid()

                self.q.task_done()
            except:
                # the exception is raised if the timeout in the queue.get() expires.
                # the break statement must be executed only if the event has been set 
                # by the main thread (i.e., no more task will be submitted)
                if event.is_set():
                    break

                

                # print(str(self.q.qsize()) +" polpetta - user:"+ str(self.id) + " job_id: "  + str(self.item['job_id'])  + " from " + str(self.item['user']))

      
      