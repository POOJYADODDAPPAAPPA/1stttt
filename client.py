import socket

hostName = socket.gethostname()
host = socket.gethostbyname(hostName)
port = 1234

client_socket = socket.socket()
client_socket.connect((host, port))

msg = input("Enter msg:")

while msg.lower().strip() != 'bye':
	client_socket.send(msg.encode())
	data = client_socket.recv(1024).decode()
	print("Received from server:",data)
	msg = input("Enter msg to send to server:")
