'''
This module impelments the behavior of a node
'''

import queue
import src.config as config
from datetime import datetime, timedelta
import copy
import logging
import math 

TRACE = 5


class node:

    def __init__(self, id):
        self.id = id    # unique edge node id
        self.initial_gpu = float(config.node_gpu)
        self.updated_gpu = self.initial_gpu
        self.initial_cpu = float(config.node_cpu)
        self.updated_cpu = self.initial_cpu
        self.initial_bw = config.t.b
        self.updated_bw = self.initial_bw
        
        self.q = queue.Queue()
        self.user_requests = []
        self.item={}
        self.bids= {}
        self.tmp_item = {}
        self.layers = 0
        self.sequence=True

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
            # return (config.a*(self.updated_bw[self.item['user']][self.id]/config.tot_bw))+((1-config.a)*(self.updated_cpu/config.tot_cpu)) #BW vs CPU
            return (config.a*(self.updated_bw/config.tot_bw))+((1-config.a)*(self.updated_cpu/config.tot_cpu)) #BW vs CPU
        elif config.filename == 'alpha_GPU_CPU':
            return (config.a*(self.updated_gpu/config.tot_gpu))+((1-config.a)*(self.updated_cpu/config.tot_cpu)) #GPU vs CPU
        elif config.filename == 'alpha_GPU_BW':
            return (config.a*(self.updated_gpu/config.tot_gpu))+((1-config.a)*(self.updated_bw/config.tot_bw)) # GPU vs BW
            # return (config.a*(self.updated_gpu/config.tot_gpu))+((1-config.a)*(self.updated_bw[self.item['user']][self.id]/config.tot_bw)) # GPU vs BW

    def forward_to_neighbohors(self):
        for i in range(config.num_edges):
            if config.t.to()[i][self.id] and self.id != i:
                config.nodes[i].append_data({
                    "job_id": self.item['job_id'], 
                    "user": self.item['user'],
                    "edge_id": self.id, 
                    "auction_id": self.bids[self.item['job_id']]['auction_id'], 
                    "NN_gpu": self.item['NN_gpu'],
                    "NN_cpu": self.item['NN_cpu'],
                    "NN_data_size": self.item['NN_data_size'], 
                    "bid": self.bids[self.item['job_id']]['bid'], 
                    "x": self.bids[self.item['job_id']]['x'], 
                    "timestamp": self.bids[self.item['job_id']]['timestamp']
                    })
                logging.debug("FORWARD " + str(self.id) + " to " + str(i) + " " + str(self.bids[self.item['job_id']]['auction_id']))



    def print_node_state(self, msg, bid=False, type='debug'):
        logger_method = getattr(logging, type)
        logger_method(str(msg) +
                    " - edge_id:" + str(self.id) +
                    " job_id:" + str(self.item['job_id']) +
                    " from_edge:" + str(self.item['edge_id']) +
                    " available GPU:" + str(self.updated_gpu) +
                    " available CPU:" + str(self.updated_cpu) +
                    " available BW:" + str(self.updated_bw) +
                    (("\n"+str(self.bids[self.item['job_id']]['auction_id']) if bid else "") +
                    ("\n"+str(self.item['auction_id']) if bid else "\n"))
                    )
    
    def update_local_val(self, index, id, bid, timestamp):
        self.bids[self.item['job_id']]['job_id'] = self.item['job_id']
        self.bids[self.item['job_id']]['auction_id'][index] = id
        self.bids[self.item['job_id']]['x'][index] = 1
        self.bids[self.item['job_id']]['bid'][index]= bid
        self.bids[self.item['job_id']]['timestamp'][index]=timestamp
        return index + 1
    
    def reset(self, index):
        self.bids[self.item['job_id']]['auction_id'][index] = float('-inf')
        self.bids[self.item['job_id']]['x'][index] = 0
        self.bids[self.item['job_id']]['bid'][index]= float('-inf')
        self.bids[self.item['job_id']]['timestamp'][index]=datetime.now() - timedelta(days=1)
        return index + 1

    def unbid(self):
        for i, id in enumerate(self.bids[self.item['job_id']]['auction_id']):
            if id == self.id:
                self.updated_gpu+=self.bids[self.item['job_id']]['NN_gpu'][i]
                self.updated_cpu+=self.bids[self.item['job_id']]['NN_cpu'][i]
                self.updated_bw += self.item['NN_data_size'][i]
                # self.updated_bw[self.item['user']][self.id] += self.item['NN_data_size'][i]
                self.bids[self.item['job_id']]['auction_id'][i]=float('-inf')
                self.bids[self.item['job_id']]['x'][i]=0
                self.bids[self.item['job_id']]['bid'][i]=float('-inf')

    def init_null(self):
        self.print_node_state('INITNULL')

        self.bids[self.item['job_id']]={
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

        NN_len = len(self.item['NN_gpu'])
        
        for _ in range(0, NN_len):
            self.bids[self.item['job_id']]['x'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid_gpu'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid_cpu'].append(float('-inf'))
            self.bids[self.item['job_id']]['bid_bw'].append(float('-inf'))
            self.bids[self.item['job_id']]['auction_id'].append(float('-inf'))
            self.bids[self.item['job_id']]['timestamp'].append(datetime.now() - timedelta(days=1))


    def first_msg(self):
        
        if self.item['job_id'] in self.bids:
            self.sequence = True
            self.layers = 0
            NN_len = len(self.item['NN_gpu'])
            required_bw =self.item['NN_data_size'][0]
            avail_bw = self.updated_bw

            if required_bw<=avail_bw: # TODO node id needed
                for i in range(0, NN_len):
                    if  self.sequence==True and \
                        self.item['NN_gpu'][i]<=self.updated_gpu and \
                        self.item['NN_cpu'][i]<=self.updated_cpu and \
                        self.layers<config.max_layer_number:
                            self.bids[self.item['job_id']]['bid'][i] = self.utility_function()
                            self.bids[self.item['job_id']]['bid_gpu'][i] = self.updated_gpu
                            self.bids[self.item['job_id']]['bid_cpu'][i] = self.updated_cpu
                            self.bids[self.item['job_id']]['bid_bw'][i] = self.updated_bw # TODO update with BW
                            self.updated_gpu = self.updated_gpu-self.item['NN_gpu'][i]
                            self.updated_cpu = self.updated_cpu-self.item['NN_cpu'][i]
                            self.bids[self.item['job_id']]['x'][i] = 1
                            self.bids[self.item['job_id']]['auction_id'][i] = self.id
                            self.bids[self.item['job_id']]['timestamp'][i] = datetime.now()
                            self.layers+=1
                    else:
                        self.sequence = False
                        self.bids[self.item['job_id']]['x'][i]  = float('-inf')
                        self.bids[self.item['job_id']]['bid'][i] = float('-inf')
                        self.bids[self.item['job_id']]['auction_id'][i] = float('-inf')
                        self.bids[self.item['job_id']]['timestamp'][i] = datetime.now() - timedelta(days=1)

            if self.bids[self.item['job_id']]['auction_id'].count(self.id)<config.min_layer_number:
                self.unbid()
            else:
                first_index = self.bids[self.item['job_id']]['auction_id'].index(self.id)
                # last_index = len(self.bids[self.item['job_id']]['auction_id']) - self.bids[self.item['job_id']]['auction_id'][::-1].index(self.id) - 1
                self.updated_bw -= self.item['NN_data_size'][first_index]  
                
                self.forward_to_neighbohors()
        else:
            self.print_node_state('Value not in dict (first_msg)', type='error')
     
    

                        
    def rebid(self):
        
        self.print_node_state('REBID')

        if self.item['job_id'] in self.bids:

            self.sequence=True
            self.layers = 0
            NN_len = len(self.item['NN_gpu'])
            # avail_bw = self.updated_bw[self.item['user']][self.id]
            avail_bw = self.updated_bw




            for i in range(0, NN_len):
                if self.bids[self.item['job_id']]['auction_id'][i] == float('-inf'):
                    logging.log(TRACE, "RIF1: " + str(self.id) + ", avail_gpu:" + str(self.updated_gpu) + ", avail_cpu:" + str(self.updated_cpu) + ", BIDDING on: " + str(i) +", NN_len:" +  str(NN_len) +  ", Layer_resources" + str(self.item['NN_gpu'][i]))
                    required_bw = self.item['NN_data_size'][i]
                    
                    if  self.sequence==True and\
                        self.item['NN_gpu'][i] <= self.updated_gpu and \
                        self.item['NN_cpu'][i] <= self.updated_cpu and \
                        self.item['NN_data_size'][i] <= avail_bw and \
                        self.layers<config.max_layer_number:
                            
                            logging.log(TRACE, "RIF2 NODEID: " + str(self.id) + ", avail_gpu:" + str(self.updated_gpu) + ", avail_cpu:" + str(self.updated_cpu) + ", BIDDING on: " + str(i) +", NN_len:" +  str(NN_len) +  ", Layer_resources" + str(self.item['NN_gpu'][i]))
                            self.bids[self.item['job_id']]['bid'][i] = self.utility_function()
                            self.bids[self.item['job_id']]['bid_gpu'][i] = self.updated_gpu
                            self.bids[self.item['job_id']]['bid_cpu'][i] = self.updated_cpu
                            self.bids[self.item['job_id']]['bid_bw'][i] = self.updated_bw # TODO update with BW
                            self.updated_gpu = self.updated_gpu-self.item['NN_gpu'][i]
                            self.updated_cpu = self.updated_cpu-self.item['NN_cpu'][i]
                            self.bids[self.item['job_id']]['x'][i]=(1)
                            self.bids[self.item['job_id']]['auction_id'][i]=(self.id)
                            self.bids[self.item['job_id']]['timestamp'][i] = datetime.now()
                            self.layers+=1
                    else:
                        self.sequence = False
                        self.bids[self.item['job_id']]['x'][i]=(float('-inf'))
                        self.bids[self.item['job_id']]['bid'][i]=(float('-inf'))
                        self.bids[self.item['job_id']]['auction_id'][i]=(float('-inf'))
                        self.bids[self.item['job_id']]['timestamp'][i] = (datetime.now() - timedelta(days=1))


            if self.bids[self.item['job_id']]['auction_id'].count(self.id)<config.min_layer_number:
                self.unbid()
            else:
                first_index = self.bids[self.item['job_id']]['auction_id'].index(self.id)
                # last_index = len(self.bids[self.item['job_id']]['auction_id']) - self.bids[self.item['job_id']]['auction_id'][::-1].index(self.id) - 1
                self.updated_bw -= self.item['NN_data_size'][first_index]  
                
                self.forward_to_neighbohors()
        else:
            self.print_node_state('Value not in dict (rebid)', type='error')

     





    def deconfliction(self):
        rebroadcast = False
        k = self.item['edge_id'] # sender
        i = self.id # receiver
        index=0
        while index < config.layer_number:
            if self.item['job_id'] in self.bids:
                z_kj = self.item['auction_id'][index]
                z_ij = self.bids[self.item['job_id']]['auction_id'][index]
                y_kj = self.item['bid'][index]
                y_ij = self.bids[self.item['job_id']]['bid'][index]
                t_kj = self.item['timestamp'][index]
                t_ij = self.bids[self.item['job_id']]['timestamp'][index]

                logging.log(TRACE,' edge_id(i):' + str(i) +
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
                        if (y_kj>y_ij) or (y_kj==y_ij and z_kj<z_ij):
                            rebroadcast = True
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #1-#2')

                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                self.updated_gpu = self.updated_gpu +  self.item['NN_gpu'][index]
                                self.updated_cpu = self.updated_cpu +  self.item['NN_cpu'][index]
                                # self.updated_bw[self.item['user']][self.id] = self.updated_bw[self.item['user']][self.id] + self.item['NN_data_size'][index]
                                self.updated_bw = self.updated_bw + self.item['NN_data_size'][index]
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                        elif (y_kj<y_ij):
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #3')
                            rebroadcast = True
                            while index<config.layer_number and self.bids[self.item['job_id']]['auction_id'][index]  == z_ij:
                                index = self.update_local_val(index, z_ij, self.bids[self.item['job_id']]['bid'][index], datetime.now())

                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #3else')
                            index+=1

                    elif  z_ij==k:
                        if  t_kj>t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#4')
                            index = self.update_local_val(index, k, self.item['bid'][index], t_kj)

                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #4else')
                            index+=1
                    
                    elif  z_ij == float('-inf'):
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #12')
                        index = self.update_local_val(index, z_kj, self.item['bid'][index], t_kj)
                        rebroadcast = True

                    elif z_ij!=i and z_ij!=k:
                        if y_kj>y_ij and t_kj>=t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #7')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        elif y_kj<y_ij and t_kj>t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #8')
                            rebroadcast = True
                            index+=1
                        elif y_kj==y_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #9else')
                            rebroadcast = True
                            index+=1
                        elif y_kj<y_ij and t_kj<t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #10')
                            index += 1
                            rebroadcast = True
                        elif y_kj>y_ij and t_kj<t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #11')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True  
                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #11else')
                            index += 1                   
                        
                elif z_kj==i:                                
                    if z_ij==i:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #13')
                        index+=1
                    elif z_ij==k:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #14')
                        index = self.reset(index)                        

                    elif z_ij == float('-inf'):
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #16')
                        rebroadcast = True
                        index+=1
                    elif z_ij!=i and z_ij!=k:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #15')
                        rebroadcast = True
                        index+=1
                    else:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #15else')
                        rebroadcast = True
                        index+=1                
                
                elif z_kj == float('-inf'):
                    if z_ij==i:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #31')
                        rebroadcast = True
                        index+=1
                    elif z_ij==k:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #32')
                        index = self.reset(index)
                        rebroadcast = True
                    elif z_ij == float('-inf'):
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #34')
                        index+=1
                    elif z_ij!=i and z_ij!=k:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #33')
                        if t_kj>t_ij:
                            index = self.reset(index)
                            rebroadcast = True
                        else:
                            index+=1
                    else:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #33else')
                        index+=1

                elif z_kj!=i or z_kj!=k:                    
                    if z_ij==i:
                        if (y_kj>y_ij) or (y_kj==y_ij and z_kj<z_ij):
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#17')
                            rebroadcast = True
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                self.updated_gpu = self.updated_gpu +  self.item['NN_gpu'][index]
                                self.updated_cpu = self.updated_cpu +  self.item['NN_cpu'][index]
                                # self.updated_bw[self.item['user']][self.id] = self.updated_bw[self.item['user']][self.id] + self.item['NN_data_size'][index]
                                self.updated_bw = self.updated_bw + self.item['NN_data_size'][index]
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                        elif (y_kj<y_ij):
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#19')
                            rebroadcast = True
                            while index<config.layer_number and self.bids[self.item['job_id']]['auction_id'][index]  == z_ij:
                                index = self.update_local_val(index, z_ij, self.bids[self.item['job_id']]['bid'][index], datetime.now())
                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #19else')

                            index+=1

                    elif z_ij==k:

                        if t_kj>t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#20')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        elif t_kj<t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#21')
                            index = self.reset(index)
                            rebroadcast = True
                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #21else')
                            index+=1

                    elif z_ij == z_kj:
                    
                        if t_kj>t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#22')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #22else')
                            index+=1
                    elif  z_ij == float('-inf'):
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  '#30')
                        index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                        rebroadcast = True


                    elif   z_ij!=i and z_ij!=k and z_ij!=z_kj:
                        if y_kj>y_ij and t_kj>=t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#25')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])                   
                            rebroadcast = True
                        elif y_kj<y_ij and t_kj<=t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#26')
                            rebroadcast = True
                            index+=1
                        elif y_kj==y_ij and z_kj<z_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#27')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])                   
                            rebroadcast = True
                        elif y_kj==y_ij :
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#27')
                            index+=1
                        elif y_kj<y_ij and t_kj>t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#28')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        elif y_kj>y_ij and t_kj<t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#29')
                            index+=1
                            rebroadcast = True
                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#29else')
                            index+=1
                    else:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #29else2')
                        index+=1
                else:
                    self.print_node_state('smth wrong?', type='error')
                    pass

            else:
                self.print_node_state('Value not in dict (deconfliction)', type='error')

                # logging.error("Value not in dict (deconfliction) - user:"+ str(self.id) + " " + str(self.item['job_id'])+"job_id: "  + str(self.item['job_id']) + " node: " + str(self.id) + " from " + str(self.item['user']))
        
        # print("End deconfliction - user:"+ str(self.id) + " job_id: "  + str(self.item['job_id'])  + " from " + str(self.item['user']))

        return rebroadcast
                



       
    def update_bid(self):


        if self.item['job_id'] in self.bids:
        
            #check consensus
            if self.bids[self.item['job_id']]['auction_id']==self.item['auction_id'] and self.bids[self.item['job_id']]['bid'] == self.item['bid'] and self.bids[self.item['job_id']]['timestamp'] == self.item['timestamp']:
                if self.id not in self.bids[self.item['job_id']]['auction_id'] and float('-inf') in self.bids[self.item['job_id']]['auction_id']:
                    self.rebid()
                else:
                    self.print_node_state('Consensus -')
                    pass
                
            else:
                self.print_node_state('BEFORE', True)
                rebroadcast = self.deconfliction()

                if self.id not in self.bids[self.item['job_id']]['auction_id'] and float('-inf') in self.bids[self.item['job_id']]['auction_id']:
                    self.rebid()
                    
                elif rebroadcast: 
                    self.forward_to_neighbohors()
                self.print_node_state('AFTER', True)

        else:
            self.print_node_state('Value not in dict (update_bid)', type='error')



    def new_msg(self):
        if str(self.id) == str(self.item['edge_id']):
            self.print_node_state('This was not supposed to be received', type='error')

        elif self.item['job_id'] in self.bids:
            if self.integrity_check(self.item['auction_id']):
                self.update_bid()
        else:
            self.print_node_state('Value not in dict (new_msg)', type='error')
     
                     


    def integrity_check(self, bid):
        curr_val = bid[0]
        curr_count = 1
        for i in range(1, len(bid)):
            
            if bid[i] == curr_val:
                curr_count += 1
            else:
                if curr_count < config.min_layer_number or curr_count > config.max_layer_number:
                    self.print_node_state('DISCARD BROKEN MSG' + str(bid))
                    return False
                
                curr_val = bid[i]
                curr_count = 1
        
        return True



    def work(self):
        while True:
            if(self.q.empty() == False):

                config.counter += 1
                self.item=None
                self.item = copy.deepcopy(self.q.get())

                if self.item['job_id'] not in self.bids:
                    self.init_null()
                
                # check msg type
                if self.item['edge_id'] is not None and self.item['user'] in self.user_requests:
                    self.print_node_state('IF1 q:' + str(self.q.qsize()), True) # edge to edge request
                    self.new_msg()

                elif self.item['edge_id'] is None and self.item['user'] not in self.user_requests:
                    self.print_node_state('IF2 q:' + str(self.q.qsize())) # brand new request from client
                    self.user_requests.append(self.item['user'])
                    self.first_msg()


                elif self.item['edge_id'] is not None and self.item['user'] not in self.user_requests:
                    self.print_node_state('IF3 q:' + str(self.q.qsize())) # edge anticipated client request
                    self.user_requests.append(self.item['user'])
                    self.new_msg()

                elif self.item['edge_id'] is None and self.item['user'] in self.user_requests:
                    self.print_node_state('IF4 q:' + str(self.q.qsize())) # client after edge request
                    self.rebid()

                



      
                self.q.task_done()

                # print(str(self.q.qsize()) +" polpetta - user:"+ str(self.id) + " job_id: "  + str(self.item['job_id'])  + " from " + str(self.item['user']))

      
      