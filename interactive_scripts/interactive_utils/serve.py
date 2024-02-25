import os
import json
import asyncio
import websockets
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Define a class that extends SimpleHTTPRequestHandler to add CORS headers
class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def http_server():
    directory_to_serve = '/scr/priyasun/robot-lightning/interactive_scripts/interactive_utils'
    os.chdir(directory_to_serve)
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    print(f"Starting HTTP server on port 8080")
    httpd.serve_forever()


if __name__ == "__main__":
    http_server()
