from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
server_address = ('', 8080)  # Choose your desired port number
httpd = HTTPServer(server_address, CORSRequestHandler)
print('Serving HTTP with CORS enabled on port 8080...')
httpd.serve_forever()
