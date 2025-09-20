import socket
import os

class TCPFileClient:
    def __init__(self, host='localhost', port=12346):
        self.host = host
        self.port = port
        self.client_socket = None
        self.connected = False

    def connect_to_server(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 设置连接超时
            self.client_socket.settimeout(10.0)
            self.client_socket.connect((self.host, self.port))
            self.connected = True
            print(f"成功连接到文件服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接文件服务器失败: {e}")
            return False

    def send_file(self, file_path):
        if not self.connected:
            print("未连接到服务器")
            return False
            
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在")
            return False
            
        try:
            # 获取文件名和大小
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            # 发送文件信息
            file_info = f"{filename}|{file_size}"
            self.client_socket.send(file_info.encode('utf-8'))
            
            # 等待服务器确认
            self.client_socket.settimeout(30.0)
            response = self.client_socket.recv(1024).decode('utf-8')
            if response != "READY":
                print(f"服务器未准备好接收文件: {response}")
                return False
                
            # 发送文件数据
            with open(file_path, 'rb') as f:
                bytes_sent = 0
                while bytes_sent < file_size:
                    data = f.read(1024)
                    if not data:
                        break
                    self.client_socket.send(data)
                    bytes_sent += len(data)
            
            # 等待服务器确认文件接收完成
            try:
                confirmation = self.client_socket.recv(1024).decode('utf-8')
                print(confirmation)
            except:
                print(f"文件 {filename} 发送完成 ({bytes_sent} 字节)")
            
            return True
        except Exception as e:
            print(f"发送文件时出错: {e}")
            return False

    def disconnect(self):
        self.connected = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        print("已断开与文件服务器的连接")