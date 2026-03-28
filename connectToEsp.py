import socket
import sys

class EspCommunication:
    def __init__(self, ip="0.0.0.0", port=4210):
        """Initializes the UDP socket for hardware communication."""
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Prevent "Address already in use" errors during rapid restarts
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Make the socket non-blocking For running alongside a video feed)
         # 5ms timeout
        self.sock.settimeout(0.005)
        self.is_bound = False

    def start(self):
        """Binds the socket to begin listening for the ESP32."""
        try:
            self.sock.bind((self.ip, self.port))
            self.is_bound = True
            print(f"Successfully bound to port {self.port}.")
            print("Listening for incoming UDP packets...\n")
        except socket.error as err:
            print(f"Socket Binding Failed. Error: {err}")
            sys.exit() 

    def read_packet(self):
        """Checks for new data. Returns the decoded string, or None if empty/error."""
        if not self.is_bound:
            return None

        try:
            # Buffer size is 1024 bytes
            data, addr = self.sock.recvfrom(1024)
            data = data.decode('utf-8')
            print('Data ' + data)
            return data
        
        # If no packet arrived turing this iteration
        except socket.timeout:
            return None
        except UnicodeDecodeError as decode_err:
            print(f"Decode Error: Corrupted data received. Details: {decode_err}")
            return None
        except socket.error as net_err:
            print(f"Network Error during reception: {net_err}")
            return None

    def stop(self):
        """Cleans up and releases the port."""
        print("Closing socket connection.")
        self.sock.close()