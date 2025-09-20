import socket
import threading

class TCPMessageClient:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.client_socket = None
        self.connected = False
        self.on_message_received = None  # 回调函数，用于处理接收到的消息

    def connect_to_server(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 设置连接超时
            self.client_socket.settimeout(10.0)
            self.client_socket.connect((self.host, self.port))
            self.connected = True
            print(f"成功连接到服务器 {self.host}:{self.port}")
            
            # 启动接收消息的线程
            receive_thread = threading.Thread(target=self.receive_messages)
            receive_thread.daemon = True
            receive_thread.start()
            
            return True
        except Exception as e:
            print(f"连接服务器失败: {e}")
            return False

    def receive_messages(self):
        while self.connected:
            try:
                # 设置接收超时
                self.client_socket.settimeout(30.0)
                message = self.client_socket.recv(1024).decode('utf-8')
                if message:
                    # 调用回调函数处理消息
                    if self.on_message_received:
                        self.on_message_received(message)
                else:
                    # 服务器关闭连接
                    break
            except socket.timeout:
                # 超时继续循环
                continue
            except Exception as e:
                if self.connected:
                    print(f"接收消息时出错: {e}")
                break
        
        # 连接断开后清理
        self.disconnect()

    def send_message(self, message):
        """发送单条消息的方法"""
        if not self.connected or not self.client_socket:
            print("未连接到服务器")
            return False
            
        try:
            self.client_socket.send(message.encode('utf-8'))
            return True
        except Exception as e:
            print(f"发送消息时出错: {e}")
            self.disconnect()
            return False

    def disconnect(self):
        self.connected = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        print("已断开与服务器的连接")