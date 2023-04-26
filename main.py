import threading
import datetime
import src.config as c
import time
import random
import sys

import time

import logging
TRACE = 5
logging.addLevelName(TRACE, "TRACE")
logging.basicConfig(filename='debug.log', level=TRACE, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

# # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# console_handler = logging.StreamHandler()
# file_handler = logging.FileHandler('debug.log')
# logging.getLogger('').addHandler(console_handler)
# logging.getLogger('').addHandler(file_handler)

# Log a message
# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
# logging.error('This is an error message')
# logging.critical('This is a critical message')

logging.debug('Clients number: ' + str(c.num_clients))
logging.debug('Edges number: ' + str(c.num_edges))
logging.debug('Requests number: ' + str(c.bid_requests))

for i in range(c.num_edges):
    c.nodes.append(threading.Thread(target=c.nodes[i].work, daemon=True).start())

start_time = time.time()


# if sys.argv[1]:
#     c.bid_requests = int(sys.argv[1])
# else:
# c.bid_requests = int(sys.argv[1])
data={}
req_id=0
for j in range(c.num_clients):
    for i in range(c.bid_requests):
        
        # for k in range(c.num_edges):

            c.nodes[0].append_data(c.message_data(req_id, j))
    req_id+=1

        # data[j] = c.message_data(i, j) #j is client request, i is request index
        # print(data[j]['req_id'])
    # time.sleep(0.5)
    # for i in range(100):
    # for i in range(c.num_edges):
    #     # if random.randint(0, 1000) > 500 and j%c.num_edges:
    #         # if random.randint(0, 1000) > 500 :
    #     c.nodes[i].append_data(data)
        # time.sleep(0.5)

# print('data')
# print(data)

# for j in range(c.num_edges):
#         # if random.randint(0, 1000) > 500 and j%c.num_edges:
#             # if random.randint(0, 1000) > 500 :
#     # for j in range(c.bid_requests):
#     for i in range(c.bid_requests):
#         print(data[i]['req_id'])

#         c.nodes[j].append_data(data[i])
        
# c.nodes[1].append_data(c.message_data(0, 999, datetime.datetime.now()))
# time.sleep(1)

# c.nodes[1].append_data(c.message_data(1, 999, datetime.datetime.now()))



# Block until all tasks are done.
for i in range(c.num_edges):
    # print("process-end" + str(c.nodes[i].getID()))
    c.nodes[i].join_queue()

logging.info("Run time: %s" % (time.time() - start_time))
print("Run time: %s" % (time.time() - start_time))



    

for j in range(req_id):
    print('\n')
    logging.info("RESULTS req:" +str(j))
    for i in range(c.num_edges):
        logging.info(c.nodes[i].bids[j]['auction_id'])
        print(c.nodes[i].bids[j]['auction_id'])

#         # print(c.nodes[i].bids)
#         # print('id: ' + str(i)+ "avl res: "+ str(c.nodes[i].updated_resources))
        
#         # print(vars(c.nodes[i].bids))
#         # print(c.nodes[i].__dict__)

logging.info('Tot messages: '+str(c.counter))
print('Tot messages: '+str(c.counter))