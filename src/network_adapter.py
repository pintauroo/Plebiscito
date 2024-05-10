import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class NetworkAdapterInterface:
    def broadcast(self, nodeID, msg):
        """Broadcast a message to all nodes except the nodeID specified."""
        pass

class DummyNetworkAdapter(NetworkAdapterInterface):
    def __init__(self, queues):
        self.queues = queues
        
    def broadcast(self, nodeID, msg):
        for i in range(len(self.queues)):
            if i != nodeID:
                self.queues[i].put(msg)
                
                
class TCPNetworkAdapter(NetworkAdapterInterface):

    class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.message_queue = kwargs.pop('message_queue')
            super().__init__(*args, **kwargs)
        
        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            self.message_queue.put(post_data)
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Message received and saved.')
    
    def __init__(self, ips, queue):
        self.ips = ips
        self.queues = queue
        
        self.server_thread = threading.Thread(target=TCPNetworkAdapter.runHTTPServer, kwargs={'message_queue': queue})
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # start listening to TCP connections
        
        
    def broadcast(self, nodeID, msg):
        for i in range(len(self.queues)):
            if i != nodeID:
                self.queues[i].put(msg)
                
    def runHTTPServer(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=9999, message_queue=None):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 80))
        server_address = (s.getsockname()[0], port)
        httpd = server_class(server_address, handler_class)
        httpd.RequestHandlerClass.message_queue = message_queue
        print(f"Starting server on port {port}...")
        httpd.serve_forever()