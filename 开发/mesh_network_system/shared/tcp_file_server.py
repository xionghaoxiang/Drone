import socket
import os
import threading

class TCPFileServer:
    def __init__(self, host='0.0.0.0', port=12346):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False

    def start_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"TCP文件服务器启动在 {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"客户端 {address} 已连接（文件传输）")
                    
                    # 为每个客户端创建一个线程
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        print(f"接受客户端连接时出错: {e}")
        except Exception as e:
            print(f"启动文件服务器时出错: {e}")
        finally:
            self.stop_server()

    def handle_client(self, client_socket, address):
        try:
            # 增加超时设置
            client_socket.settimeout(30.0)
            
            # 接收文件信息
            file_info = client_socket.recv(1024).decode('utf-8')
            if not file_info or '|' not in file_info:
                print(f"客户端 {address} 发送了无效的文件信息")
                client_socket.close()
                return
                
            filename, file_size = file_info.split('|')
            file_size = int(file_size)
            
            print(f"从 {address} 接收文件: {filename} ({file_size} 字节)")
            
            # 确认准备接收
            client_socket.send("READY".encode('utf-8'))
            
            # 接收文件数据
            filename = "received_" + filename  # 避免文件名冲突
            with open(filename, 'wb') as f:
                bytes_received = 0
                while bytes_received < file_size:
                    data = client_socket.recv(min(1024, file_size - bytes_received))
                    if not data:
                        break
                    f.write(data)
                    bytes_received += len(data)
                    
            print(f"文件 {filename} 接收完成 ({bytes_received} 字节)")
            
            # 发送确认消息
            client_socket.send(f"文件 {filename} 接收完成".encode('utf-8'))
            
        except socket.timeout:
            print(f"客户端 {address} 文件传输超时")
        except Exception as e:
            print(f"处理客户端 {address} 文件传输时出错: {e}")
        finally:
            client_socket.close()
            print(f"客户端 {address} 文件传输连接已断开")

    def stop_server(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("TCP文件服务器已停止")