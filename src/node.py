'''
This module impelments the behavior of an edge node
'''

import queue
import time
import random
import src.config as config
from datetime import datetime, timedelta
import copy
import logging

TRACE = 5


class node:

    def __init__(self, id):
        self.id = id    # unique edge node id
        self.initial_resources = config.node_max_res
        self.updated_resources = self.initial_resources
        self.q = queue.Queue()
        self.client_id_requests = []
        self.item={}
        self.bids= {}
        self.tmp_item = {}
        self.layers = 0
        self.sequence=True


    def getID(self):
        return self.id
    
    def get_available_res(self):
        return self.updated_resources

    def append_data(self, d):
        self.q.put(d)

    def join_queue(self):
        self.q.join()

    def forward_to_neighbohors(self):
        # print("FORWARD - client_id:"+ str(self.getID()) + " req_id: "  + str(self.item['req_id'])  + " from " + str(self.item['client_id']))

        for i in range(config.num_edges):
            if config.m[i][self.getID()] and self.getID() != i:
                config.nodes[i].append_data({
                    "req_id": self.item['req_id'], 
                    "client_id": self.item['client_id'],
                    "edge_id": self.getID(), 
                    "auction_id": self.bids[self.item['req_id']]['auction_id'], 
                    "NN_resources": self.item['NN_resources'],
                    "NN_data_size": self.item['NN_data_size'], 
                    "bid": self.bids[self.item['req_id']]['bid'], 
                    "x": self.bids[self.item['req_id']]['x'], 
                    "timestamp": self.bids[self.item['req_id']]['timestamp']
                    })
                logging.debug(str(self.getID()) + " to " + str(i) + " " + str(self.bids[self.item['req_id']]['auction_id']))



    def print_node_state(self, msg, bid=False):
        logging.debug(str(msg) +
                    " - edge_id:" + str(self.getID()) +
                    " req_id:" + str(self.item['req_id']) +
                    " from_edge:" + str(self.item['edge_id']) +
                    " available resources:" + str(self.get_available_res()) +
                    (("\n"+str(self.bids[self.item['req_id']]['auction_id']) if bid else "") +
                    ("\n"+str(self.item['auction_id']) if bid else "\n"))
                    )


    # def update_msg(self):
        
    #     self.tmp_item['client_id']=self.getID()
    #     self.tmp_item['req_id']=self.item['req_id']
    #     self.tmp_item['NN_resources']=self.item['NN_resources']
    #     self.tmp_item['bid']=self.bids[self.item['req_id']]['bid']
    #     self.tmp_item['auction_id']=self.bids[self.item['req_id']]['auction_id']
    #     self.tmp_item['x']=self.bids[self.item['req_id']]['x']
    #     self.tmp_item['timestamp']=self.bids[self.item['req_id']]['timestamp']

    # def hard_reset(self):
    #     self.bids[self.item['req_id']]['req_id'] = float('-inf')
    #     self.bids[self.item['req_id']]['auction_id'] = []
    #     self.bids[self.item['req_id']]['x']= []
    #     self.bids[self.item['req_id']]['bid']= []
    #     self.bids[self.item['req_id']]['timestamp']=[]
    #     self.updated_resources=self.initial_resources


    def update_local_val(self, index, id, bid, timestamp):
        # print("indexxxxxx: "+str(index))
        self.bids[self.item['req_id']]['req_id'] = self.item['req_id']
        self.bids[self.item['req_id']]['auction_id'][index] = id
        self.bids[self.item['req_id']]['x'][index] = 1
        self.bids[self.item['req_id']]['bid'][index]= bid
        self.bids[self.item['req_id']]['timestamp'][index]=timestamp
        return index + 1
    
    def reset(self, index):
        self.bids[self.item['req_id']]['auction_id'][index] = float('-inf')
        self.bids[self.item['req_id']]['x'][index] = 0
        self.bids[self.item['req_id']]['bid'][index]= float('-inf')
        self.bids[self.item['req_id']]['timestamp'][index]=datetime.now() - timedelta(days=1)
        return index + 1

    # def reset_local(self, item, index):
    #     self.bids[item['req_id']]['auction_id'][index] = float('-inf')
    #     self.bids[item['req_id']]['x'][index] = float('-inf')
    #     self.bids[item['req_id']]['bid'][index] = float('-inf')
    #     self.bids[item['req_id']]['timestamp'][index] = float('-inf')


    def unbid(self):
        for i, id in enumerate(self.bids[self.item['req_id']]['auction_id']):
            if id == self.getID():
                self.updated_resources+=self.bids[self.item['req_id']]['NN_resources'][i]
                self.bids[self.item['req_id']]['auction_id'][i]=float('-inf')
                self.bids[self.item['req_id']]['x'][i]=0
                self.bids[self.item['req_id']]['bid'][i]=float('-inf')


    def init_null(self):
        self.bids[self.item['req_id']]={
            "req_id": self.item['req_id'], 
            "client_id": int(), 
            "auction_id": list(), 
            "NN_resources": self.item['NN_resources'], 
            "bid": list(), 
            "bid_res": list(), 
            "bid_bw": list(), 
            "x": list(), 
            "timestamp": list()
            }

        NN_len = len(self.item['NN_resources'])
        
        for _ in range(0, NN_len):
            self.bids[self.item['req_id']]['x'].append(float('-inf'))
            self.bids[self.item['req_id']]['bid'].append(float('-inf'))
            self.bids[self.item['req_id']]['bid_res'].append(float('-inf'))
            self.bids[self.item['req_id']]['bid_bw'].append(float('-inf'))
            self.bids[self.item['req_id']]['auction_id'].append(float('-inf'))
            self.bids[self.item['req_id']]['timestamp'].append(datetime.now() - timedelta(days=1))


    def first_msg(self):
        

        if self.item['req_id'] in self.bids:
            self.sequence = True
            self.layers = 0
            NN_len = len(self.item['NN_resources'])
            avail_bw =self.item['NN_data_size'][0]
            output_size = config.b[self.item['client_id']][self.id]
            # print(avail_bw)
            # print(output_size)
            if avail_bw<=output_size: # TODO node id needed
                for i in range(0, NN_len):
                    #bid if there are enough resources
                    if self.sequence==True and self.item['NN_resources'][i]<=self.updated_resources and self.layers<config.max_layer_number:
                        self.bids[self.item['req_id']]['bid'][i] = (config.a*self.updated_resources)+((1-config.a)*config.b[self.item['client_id']][self.id])
                        self.bids[self.item['req_id']]['bid_res'][i] = self.updated_resources
                        self.bids[self.item['req_id']]['bid_bw'][i] = self.updated_resources
                        self.updated_resources = self.updated_resources-self.item['NN_resources'][i]
                        self.bids[self.item['req_id']]['x'][i] = 1
                        self.bids[self.item['req_id']]['auction_id'][i] = self.getID()
                        self.bids[self.item['req_id']]['timestamp'][i] = datetime.now()
                        self.layers+=1
                    else:
                        self.sequence = False
                        self.bids[self.item['req_id']]['x'][i]  = float('-inf')
                        self.bids[self.item['req_id']]['bid'][i] = float('-inf')
                        self.bids[self.item['req_id']]['auction_id'][i] = float('-inf')
                        self.bids[self.item['req_id']]['timestamp'][i] = datetime.now() - timedelta(days=1)

            if self.bids[self.item['req_id']]['auction_id'].count(self.getID())<config.min_layer_number:
                self.unbid()
            else:
                config.b[self.item['client_id']][self.id] -= self.item['NN_data_size'][0] # TODO else update bandwidth 

            self.forward_to_neighbohors()
        else:
            raise logging.error("Value not in dict (first_msg) - client_id:"+ str(self.getID()) + " " + str(self.item['req_id'])+" req_id: "  + str(self.item['req_id']) + " node: " + str(self.getID()) + " from " + str(self.item['client_id']))
     
    

                        
    def rebid(self):

        # print("REBID - client_id:"+ str(self.getID()) + " req_id: "  + str(self.item['req_id'])  + " from " + str(self.item['client_id']))
        
        self.print_node_state('REBID')
        if self.item['req_id'] in self.bids:
            logging.log(TRACE, 'rebida')
            self.sequence=True
            self.layers = 0
            NN_len = len(self.item['NN_resources'])
            if self.item['NN_data_size'][0]<=config.b[self.item['client_id']][self.id]: # TODO node id needed
                logging.log(TRACE, 'rebidB')
                for i in range(0, NN_len):

                    if self.bids[self.item['req_id']]['auction_id'][i] == float('-inf'):
                        logging.log(TRACE, "RIF1: " + str(self.getID()) + ", avail_res:" + str(self.updated_resources) + ", BIDDING on: " + str(i) +", NN_len:" +  str(NN_len) +  ", Layer_resources" + str(self.item['NN_resources'][i]))
                        if self.sequence==True and self.item['NN_resources'][i] <= self.updated_resources and  self.layers<config.max_layer_number:
                            logging.log(TRACE, "RIF2 NODEID: " + str(self.getID()) + ", avail_res:" + str(self.updated_resources) + ", BIDDING on: " + str(i) +", NN_len:" +  str(NN_len) +  ", Layer_resources" + str(self.item['NN_resources'][i]))
                            self.bids[self.item['req_id']]['bid'][i] = (config.a*self.updated_resources)+((1-config.a)*config.b[self.item['client_id']][self.id])
                            self.bids[self.item['req_id']]['bid_res'][i] = self.updated_resources
                            self.bids[self.item['req_id']]['bid_bw'][i] = self.updated_resources=self.updated_resources-self.item['NN_resources'][i]
                            self.bids[self.item['req_id']]['x'][i]=(1)
                            self.bids[self.item['req_id']]['auction_id'][i]=(self.getID())
                            self.bids[self.item['req_id']]['timestamp'][i] = datetime.now()
                            self.layers+=1
                        else:
                            self.sequence = False
                            self.bids[self.item['req_id']]['x'][i]=(float('-inf'))
                            self.bids[self.item['req_id']]['bid'][i]=(float('-inf'))
                            self.bids[self.item['req_id']]['auction_id'][i]=(float('-inf'))
                            self.bids[self.item['req_id']]['timestamp'][i] = (datetime.now() - timedelta(days=1))


            if self.bids[self.item['req_id']]['auction_id'].count(self.getID())<config.min_layer_number:
                self.unbid()
            else:
                config.b[self.item['client_id']][self.id] -= self.item['NN_data_size'][self.bids[self.item['req_id']]['auction_id'].index(self.getID())] # TODO else update bandwidth 

                self.forward_to_neighbohors()
        else:
            raise logging.error("Value not in dict (rebid) - client_id:"+ str(self.getID()) + " " +str(self.item['req_id'])+" req_id: "  + str(self.item['req_id']) + " node: " + str(self.getID()) + " from " + str(self.item['client_id']))
     





    def deconfliction(self):
        
        # print("Start deconfliction - client_id:"+ str(self.getID()) + " req_id: "  + str(self.item['req_id']) + " from " + str(self.item['client_id']))

        rebroadcast = False
        k = self.item['edge_id'] # sender
        i = self.getID() # receiver
        index=0
        while index < config.layer_number:
            # print(str(self.getID())+str(index))
            if self.item['req_id'] in self.bids:
                # print(str(self.getID())+str(self.item['req_id']))
                z_kj = self.item['auction_id'][index]
                z_ij = self.bids[self.item['req_id']]['auction_id'][index]
                y_kj = self.item['bid'][index]
                y_ij = self.bids[self.item['req_id']]['bid'][index]
                t_kj = self.item['timestamp'][index]
                t_ij = self.bids[self.item['req_id']]['timestamp'][index]

                logging.log(TRACE,' edge_id(i):' + str(i) +
                              ' sender(k):' + str(k) +
                              ' z_kj:' + str(z_kj) +
                              ' z_ij:' + str(z_ij) +
                              ' y_kj:' + str(y_kj) +
                              ' y_ij:' + str(y_ij) +
                              ' t_kj:' + str(t_kj) +
                              ' t_ij:' + str(t_ij)
                               )
                # print("index:" + str(index)+" client_id:" + str(self.getID()) + " count" +  str(self.i))

                #z_kj==k
                if z_kj==k : 
                    if z_ij==i:
                        #1 - 2
                        if (y_kj>y_ij) or (y_kj==y_ij and z_kj<z_ij):
                            rebroadcast = True
                            # print("#1-#2")
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #1-#2')

                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                self.updated_resources = self.updated_resources +  self.item['NN_resources'][index]
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                                # print("1-2update bid: " +str(self.updated_resources))
                        #3
                        elif (y_kj<y_ij):
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #3')
                            rebroadcast = True

                            # print(str(self.item['auction_id'][index]) +" "+ str(z_ij))
                            while index<config.layer_number and self.bids[self.item['req_id']]['auction_id'][index]  == z_ij:
                                index = self.update_local_val(index, z_ij, self.bids[self.item['req_id']]['bid'][index], datetime.now())
                                # print("3update bid: " +str(self.updated_resources))

                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #3else')
                            index+=1

                    elif  z_ij==k:
                        #4
                        if  t_kj>t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#4')
                            # rebroadcast = False
                            index = self.update_local_val(index, k, self.item['bid'][index], t_kj)
                        # #5
                        # elif t_kj==t_ij:
                        #     # # rebroadcast = False
                        #     index+=1
                        # #6
                        # elif t_kj<t_ij:
                        #     # # rebroadcast = False
                        #     index+=1
                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #4else')
                            # # rebroadcast = False
                            index+=1
                    
                    #12
                    elif  z_ij == float('-inf'):
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #12')
                        index = self.update_local_val(index, z_kj, self.item['bid'][index], t_kj)
                        rebroadcast = True

                    elif z_ij!=i and z_ij!=k:
                        #7 
                        if y_kj>y_ij and t_kj>=t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #7')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        #8 xx
                        elif y_kj<y_ij and t_kj>t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #8')
                            rebroadcast = True
                            index+=1
                        #9
                        elif y_kj==y_ij and z_kj<z_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #9')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        elif y_kj==y_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #9else')
                            rebroadcast = True
                            index+=1
                        #10
                        elif y_kj<y_ij and t_kj<t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #10')
                            index += 1
                            # index = self.reset(index)
                            rebroadcast = True
                        #11
                        elif y_kj>y_ij and t_kj<t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #11')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True  
                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #11else')
                            index += 1                   
                        
                #z_kj==i
                elif z_kj==i:
                    # print("z_kj==i")
                                
                    #13
                    if z_ij==i:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #13')
                        # if t_kj==t_ij:
                        #     # # rebroadcast = False
                        #     index+=1
                        # else:
                        #     # # rebroadcast = False
                        #     index+=1
                        # # rebroadcast = False
                        index+=1
                    #14
                    elif z_ij==k:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #14')
                        index = self.reset(index)                        

                    #16
                    elif z_ij == float('-inf'):
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #16')
                        rebroadcast = True
                        index+=1
                    #15
                    elif z_ij!=i and z_ij!=k:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #15')
                        rebroadcast = True
                        index+=1
                    else:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #15else')
                        rebroadcast = True
                        index+=1                
                
                #z_kj == float('-inf')
                elif z_kj == float('-inf'):
                    # print("z_kj == float('-inf')")
                    #31
                    if z_ij==i:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #31')
                        rebroadcast = True
                        index+=1
                    #32
                    elif z_ij==k:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #32')
                        index = self.reset(index)
                        rebroadcast = True
                    #34
                    elif z_ij == float('-inf'):
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #34')
                        # self.rebid()
                        # # # rebroadcast = False
                        index+=1
                    #33
                    elif z_ij!=i and z_ij!=k:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #33')
                        if t_kj>t_ij:
                            index = self.reset(index)
                            rebroadcast = True
                        else:
                            index+=1
                    else:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #33else')
                        # # rebroadcast = False
                        index+=1

                #z_kj!=i or z_kj!=k
                elif z_kj!=i or z_kj!=k:
                    # print("z_kj!=i or z_kj!=k")
                    
                    if z_ij==i:
                        #17 - 18
                        if (y_kj>y_ij) or (y_kj==y_ij and z_kj<z_ij):
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#17')
                            rebroadcast = True
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                self.updated_resources = self.updated_resources +  self.item['NN_resources'][index]
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            # print("update bid: " +str(self.updated_resources))
                        # elif :
                        #     # print("#18")
                        #     rebroadcast = True
                        #     while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                        #         index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                        #19
                        elif (y_kj<y_ij):
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#19')
                            rebroadcast = True
                            # print(str(self.item['auction_id'][index]) +" "+ str(z_ij))
                            while index<config.layer_number and self.bids[self.item['req_id']]['auction_id'][index]  == z_ij:
                                index = self.update_local_val(index, z_ij, self.bids[self.item['req_id']]['bid'][index], datetime.now())
                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #19else')
                            # print("--------------")
                            rebroadcast = True
                            index+=1

                    elif z_ij==k:
                        #20
                        # print("BUG?: " + str(index)+" client_id:" + str(self.getID()) )
                        if t_kj>t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#20')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        #21
                        elif t_kj<t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#21')
                            index = self.reset(index)
                            rebroadcast = True
                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #21else')
                            # print("#else")
                            # # rebroadcast = False
                            index+=1

                    elif z_ij == z_kj:
                    
                        #22
                        if t_kj>t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#22')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            # # rebroadcast = False
                        # #23
                        # elif t_kj==t_ij:
                        #     # # rebroadcast = False
                        #     index+=1
                        # #24
                        # elif t_kj<t_ij:
                        #     # # rebroadcast = False
                        #     index+=1
                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #22else')
                            # print("#23" + str(t_kj) +str(t_ij))
                            # # rebroadcast = False
                            index+=1
                    #30 
                    elif  z_ij == float('-inf'):
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  '#30')
                        index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                        rebroadcast = True


                    elif   z_ij!=i and z_ij!=k and z_ij!=z_kj:
                        #25
                        if y_kj>y_ij and t_kj>=t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#25')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])                   
                            rebroadcast = True
                        #26
                        elif y_kj<y_ij and t_kj<=t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#26')
                            rebroadcast = True
                            index+=1
                        #27
                        elif y_kj==y_ij and z_kj<z_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#27')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])                   
                            rebroadcast = True

                        elif y_kj==y_ij :
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#27')
                            # rebroadcast = True
                            index+=1
                        #28 
                        elif y_kj<y_ij and t_kj>t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#28')
                            while index<config.layer_number and self.item['auction_id'][index] == z_kj:
                                # print(self.getID())
                                index = self.update_local_val(index, z_kj, self.item['bid'][index], self.item['timestamp'][index])
                            rebroadcast = True
                        #29
                        elif y_kj>y_ij and t_kj<t_ij:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#29')
                            index+=1
                            # index = self.reset(index)
                            rebroadcast = True
                        else:
                            logging.log(TRACE, 'edge_id:'+str(self.id) +  '#29else')
                            # # rebroadcast = False
                            index+=1
                    else:
                        logging.log(TRACE, 'edge_id:'+str(self.id) +  ' #29else2')
                        index+=1
                else:
                    logging.error("smth wrong?")
                    pass
                


                
            else:
                raise logging.error("Value not in dict (deconfliction) - client_id:"+ str(self.getID()) + " " + str(self.item['req_id'])+"req_id: "  + str(self.item['req_id']) + " node: " + str(self.getID()) + " from " + str(self.item['client_id']))
        
        # print("End deconfliction - client_id:"+ str(self.getID()) + " req_id: "  + str(self.item['req_id'])  + " from " + str(self.item['client_id']))

        return rebroadcast
                



       
    def update_bid(self):


        if self.item['req_id'] in self.bids:
        
            #check consensus
            if self.bids[self.item['req_id']]['auction_id']==self.item['auction_id'] and self.bids[self.item['req_id']]['bid'] == self.item['bid'] and self.bids[self.item['req_id']]['timestamp'] == self.item['timestamp']:
                if self.getID() not in self.bids[self.item['req_id']]['auction_id'] and float('-inf') in self.bids[self.item['req_id']]['auction_id']:
                    self.rebid()
                else:
                    
                    logging.debug("Consensus - client_id:"+ str(self.getID()) + " req_id: "  + str(self.item['req_id']) + " from " + str(self.item['client_id']))
                    pass
            else:
                self.print_node_state('BEFORE', True)
                # logging.debug("BEFORE req_id: "  + str(self.item['req_id']) +" client_id:" + str(self.getID()) +" from node:" + str(self.item['client_id'])   + " avail res:" + str(self.updated_resources) + "\n" + str(self.bids[self.item['req_id']]['auction_id'])+ "\n" +str(self.item['auction_id'])+ "\n")

                # print(self.bids[self.item['req_id']])
                rebroadcast = self.deconfliction()


                if self.getID() not in self.bids[self.item['req_id']]['auction_id'] and float('-inf') in self.bids[self.item['req_id']]['auction_id']:
                    self.rebid()
                    
                elif rebroadcast: # or float('-inf') in self.bids[self.item['req_id']]['auction_id']:
                    # print("REBROADCAST req_id: "  + str(self.item['req_id']) +" client_id:" + str(self.getID()) +" from node:" + str(self.item['client_id'])   + " avail res:" + str(self.updated_resources) + "\n" + str(self.bids[self.item['req_id']]['auction_id'])+ "\n" +str(self.item['auction_id'])+ "\n")
                    self.forward_to_neighbohors()
                self.print_node_state('AFTER', True)

                # logging.debug("AFTER req_id: "  + str(self.item['req_id']) +" client_id:" + str(self.getID()) +" from node:" + str(self.item['client_id'])   + " avail res:" + str(self.updated_resources) + "\n" + str(self.bids[self.item['req_id']]['auction_id'])+ "\n" +str(self.item['auction_id'])+ "\n")
        else:
            raise logging.error("Value not in dict (new_msg) - client_id:"+ str(self.getID()) + " " +str(self.item['req_id'])+"req_id: "  + str(self.item['req_id']) + " from " + str(self.item['client_id']))



    def new_msg(self):
        if str(self.getID()) == str(self.item['edge_id']):
            raise logging.error("This was not supposed to be received")
        elif self.item['req_id'] in self.bids:
            if self.integrity_check(self.item['auction_id']):
                self.update_bid()
        else:
            raise logging.error("Value not in dict (new_msg) - client_id:"+ str(self.getID()) + " req_id: "  + str(self.item['req_id']) + " from " + str(self.item['client_id']) +str(self.item))
     
                     


    def integrity_check(self, bid):
        curr_val = bid[0]
        curr_count = 1
        for i in range(1, len(bid)):
            
            if bid[i] == curr_val:
                curr_count += 1
            else:
                if curr_count < config.min_layer_number or curr_count > config.max_layer_number:
                    print('DISCARD BROKEN MSG' + str(bid))
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
                # print(str(self.id)+str(self.item))
                # print(self.item['req_id'])

                
                
                if self.item['edge_id'] is not None and self.item['client_id'] in self.client_id_requests:
                    self.print_node_state('IF1') # edge to edge request
                    self.new_msg()

                elif self.item['edge_id'] is None and self.item['client_id'] not in self.client_id_requests:
                    self.print_node_state('IF2') # brand new request from client
                    self.client_id_requests.append(self.item['client_id'])
                    if self.item['req_id'] not in self.bids:
                        self.init_null()
                    
                    self.first_msg()


                elif self.item['edge_id'] is not None and self.item['client_id'] not in self.client_id_requests:
                    self.print_node_state('IF3') # edge anticipated client request
                    self.client_id_requests.append(self.item['client_id'])
                    if self.item['req_id'] not in self.bids:
                        self.init_null()

                    self.new_msg()

                elif self.item['edge_id'] is None and self.item['client_id'] in self.client_id_requests:
                    self.print_node_state('IF4') # client after edge request
                    if self.item['req_id'] not in self.bids:
                        self.init_null()

                    self.rebid()

                

                # if self.item['client_id'] in self.client_id_requests:
                #     if self.item['edge_id'] is None: #msg rcvd from client
                        
                #     elif self.item['edge_id'] is not None: #msg rcvd from edge srv
                        
                # else:
                    
                #     elif self.item['edge_id'] is not None: #msg rcvd from edge srv
                        
                    


                
                # if self.item['req_id'] in self.bids:
                    
                #     if self.item['client_id'] != 999:
                #         # print("IF1  - client_id:"+ str(self.getID()) + " req_id: "  + str(self.item['req_id']) + " from " + str(self.item['client_id']))
                #         self.new_msg()

                #     elif self.item['client_id'] == 999:
                #         # print("IF4  - client_id:"+ str(self.getID()) + " req_id: "  + str(self.item['req_id'])  + " from " + str(self.item['client_id']))
                #         self.rebid()
                # else:
                #     # # print(self.bids.keys())

                #     self.init_null()

                #     #edge receives request from another node for the first time
                #     if self.item['client_id'] != 999:
                #         # print("IF2  - client_id:"+ str(self.getID()) + " req_id: "  + str(self.item['req_id']) + " from " + str(self.item['client_id']))
                #         self.new_msg()

                #     #edge receives new request from client for the first time
                #     elif self.item['client_id'] == 999:
                #         # print("IF3  - client_id:"+ str(self.getID()) + " req_id: "  + str(self.item['req_id']) + " from " + str(self.item['client_id']))
                #         self.first_msg()

                # print(self.q.qsize())
                # if self.q.qsize()>1:
                #     print("big")
                #     self.q.task_done()
                # else:
                #     print("small")
                #     time.sleep(1)
                #     self.q.task_done()
                self.q.task_done()

                # print(str(self.q.qsize()) +" polpetta - client_id:"+ str(self.getID()) + " req_id: "  + str(self.item['req_id'])  + " from " + str(self.item['client_id']))

      
      