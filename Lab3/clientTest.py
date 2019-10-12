import socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('192.168.1.249', 8080))
client.send(b"I am CLIENT\n")
from_server = client.recv(4096)
client.close()
print(from_server)
