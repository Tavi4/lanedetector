import socket
import select
import pickle
import datetime

from typing import * #  * = Any, Optional

class ObjectSocketParams:
    """Class containing parameters that are used for socket operations

    Attributes:
        OBJECT_HEADER_SIZE_BYTES (int): Header size for objects sent in bytes
        DEFAULT_TIMEOUT_S (int): Default timeout for received data in seconds
        CHUNK_SIZE_BYTES (int): Chunk size for received data in bytes
    """
    OBJECT_HEADER_SIZE_BYTES = 4
    DEFAULT_TIMEOUT_S = 1
    CHUNK_SIZE_BYTES = 1024

class ObjectSenderSocket:
    """Class that sends objects over a TCP connection

    Attributes:
        ip (str): IP address for sender socket
        port (int): Port number for sender socket
        sock (socket.socket): The primary socket for sender
        conn (socket.socket): The connection socket to receiver
        print_when_awaiting_receiver ((optional)bool): Whether to print logs for receiver or not
        print_when_sending_object (bool): Whether to print logs for sending object or not
    """
    ip: str
    port: int
    sock: socket.socket
    conn: socket.socket
    print_when_awaiting_receiver: bool
    print_when_sending_object: bool

    def __init__(self, ip: str, port: int,
                 print_when_awaiting_receiver: bool = False,
                 print_when_sending_object: bool = False):
        """Initializes the ObjectSenderSocket.

        Args:
            ip (str): IP address for sender socket
            port (int): Port number for sender socket
            print_when_awaiting_receiver (bool): Whether to print logs for receiver or not. Defaults is False
            print_when_sending_object (bool): Whether to print logs for sending object or not. Defaults is False
        """
        self.ip = ip
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.ip, self.port))
        self.conn = None

        self.print_when_awaiting_receiver = print_when_awaiting_receiver
        self.print_when_sending_object = print_when_sending_object

        self.await_receiver_conection()

    def await_receiver_conection(self):
        """Function that waits for the receiver to connect"""
        if self.print_when_awaiting_receiver:
            print(f'[{datetime.datetime.now()}][ObjectSenderSocket/{self.ip}:{self.port}] awaiting receiver connection...')

        self.sock.listen(1)
        self.conn, _ = self.sock.accept()

        if self.print_when_awaiting_receiver:
            print(f'[{datetime.datetime.now()}][ObjectSenderSocket/{self.ip}:{self.port}] receiver connected')

    def close(self):
        """Function that closes the connection for the receiver"""
        self.conn.close()
        self.conn = None

    def is_connected(self) -> bool:
        """Function that checks if the connection is established

        Returns:
            bool: True if the connection is established, False otherwise
        """

        return self.conn is not None


    def send_object(self, obj: Any):
        """Function that sends object to the receiver
        Args:

            obj (Any): Object to be sent
        """

        data = pickle.dumps(obj)
        data_size = len(data)
        data_size_encoded = data_size.to_bytes(ObjectSocketParams.OBJECT_HEADER_SIZE_BYTES, 'little')
        self.conn.sendall(data_size_encoded)
        self.conn.sendall(data)
        if self.print_when_sending_object:
            print(f'[{datetime.datetime.now()}][ObjectSenderSocket/{self.ip}:{self.port}] Sent object of size {data_size} bytes.')



class ObjectReceiverSocket:
    """Class that sends objects over a TCP connection

    Attributes:

        ip (str): IP address for receiver socket
        port (int): Port number for receiver socket
        conn (socket.socket): The connection socket to receiver
        print_when_connecting_to_sender (bool): Whether to print logs for receiver or not. Defaults is False
        print_when_receiving_object (bool): Whether to print logs for sending object or not. Defaults is False
    """

    ip: str
    port: int
    conn: socket.socket
    print_when_connecting_to_sender: bool
    print_when_receiving_object: bool

    def __init__(self, ip: str, port: int,
                 print_when_connecting_to_sender: bool = False,
                 print_when_receiving_object: bool = False):
        """Initializes the ObjectReceiverSocket

        Args:
            ip (str): IP address for receiver socket
            port (int): Port number for receiver socket
            print_when_connecting_to_sender (bool): Whether to print logs for receiver or not. Defaults is False
            print_when_receiving_object (bool): Whether to print logs for sending object or not. Defaults is False
        """
        self.ip = ip
        self.port = port
        self.print_when_connecting_to_sender = print_when_connecting_to_sender
        self.print_when_receiving_object = print_when_receiving_object

        self.connect_to_sender()

    def connect_to_sender(self):
        """Function that connects to the sender"""
        if self.print_when_connecting_to_sender:
            print(f'[{datetime.datetime.now()}][ObjectReceiverSocket/{self.ip}:{self.port}] connecting to sender...')

        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.ip, self.port))

        if self.print_when_connecting_to_sender:
            print(f'[{datetime.datetime.now()}][ObjectReceiverSocket/{self.ip}:{self.port}] connected to sender')

    def close(self):
        """Function that closes the connection for the receiver"""
        self.conn.close()
        self.conn = None

    def is_connected(self) -> bool:
        """Function that checks if the connection is established

        Returns:
            bool: True if the connection is established, False otherwise
        """
        return self.conn is not None

    def recv_object(self) -> Any:
        """Function that receives object from the sender

        Returns:
            Any: Object to be received
        """
        obj_size_bytes = self._recv_object_size()
        data = self._recv_all(obj_size_bytes)
        obj = pickle.loads(data)
        if self.print_when_receiving_object:
            print(f'[{datetime.datetime.now()}][ObjectReceiverSocket/{self.ip}:{self.port}] Received object of size {obj_size_bytes} bytes.')
        return obj

    def _recv_with_timeout(self, n_bytes: int, timeout_s: float = ObjectSocketParams.DEFAULT_TIMEOUT_S) -> Optional[bytes]:
        """Function that receives objects alongside a timeout

        Args:

            n_bytes (int): Number of bytes to receive
            timeout_s (float): Timeout in seconds

        Returns:
            Optional[bytes]: Object to be received if timeout doesn't occur
        """
        rlist, _1, _2 = select.select([self.conn], [], [], timeout_s)
        if rlist:
            data = self.conn.recv(n_bytes)
            return data
        else:
            return None  # Only returned on timeout

    def _recv_all(self, n_bytes: int, timeout_s: float = ObjectSocketParams.DEFAULT_TIMEOUT_S) -> bytes:
        """Function that receives bytes as objects

        Args:
            n_bytes (int): Number of bytes to be received
            timeout_s (float): Timeout in seconds

        Returns:
            bytes: Object to be received
        """
        data = []
        left_to_recv = n_bytes
        while left_to_recv > 0:
            desired_chunk_size = min(ObjectSocketParams.CHUNK_SIZE_BYTES, left_to_recv)
            chunk = self._recv_with_timeout(desired_chunk_size, timeout_s)
            if chunk is not None:
                data += [chunk]
                left_to_recv -= len(chunk)
            else:  # no more data incoming, timeout
                bytes_received = sum(map(len, data))
                raise socket.error(f'Timeout elapsed without any new data being received. '
                                   f'{bytes_received} / {n_bytes} bytes received.')
        data = b''.join(data)
        return data

    def _recv_object_size(self) -> int:
        """Function that returns the size of the object

        Returns:

            int: Size of the object
        """
        data = self._recv_all(ObjectSocketParams.OBJECT_HEADER_SIZE_BYTES)
        obj_size_bytes = int.from_bytes(data, 'little')
        return obj_size_bytes


