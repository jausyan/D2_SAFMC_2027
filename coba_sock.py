import socket
from time import sleep

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)  # UDP
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    #  sock.settimeout(2)
    while True:
        sock.sendto(bytes("1", "utf-8"), ("255.255.255.255", 5005))
        sleep(1)

main()
