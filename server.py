import socket

hostName = socket.gethostname()
host = socket.gethostbyname(hostName)
port = 1234

server_socket = socket.socket()
server_socket.bind((host, port))
print("Listening at ", host)
server_socket.listen(2)
con, address = server_socket.accept()

print("Connection from", str(address))

while True:
	data = con.recv(1024).decode()
	if not data:
		break
	print("From connected users:", data)
	data = input("Enter msg:")
	con.send(data.encode())
con.close()
