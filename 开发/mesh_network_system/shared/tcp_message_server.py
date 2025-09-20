import socket
import threading

class TCPMessageServer:
    def __init__(self, host='0.0.0.0', port=12345):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []
        self.running = False
        self.on_client_connected = None
        self.on_client_disconnected = None
        self.on_message_received = None

    def start_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"TCP消息服务器启动在 {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"客户端 {address} 已连接")
                    
                    # 调用客户端连接回调
                    if self.on_client_connected:
                        self.on_client_connected(address)
                    
                    # 为每个客户端创建一个线程
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                    self.clients.append((client_socket, address))
                except Exception as e:
                    if self.running:
                        print(f"接受客户端连接时出错: {e}")
        except Exception as e:
            print(f"启动服务器时出错: {e}")
        finally:
            self.stop_server()

    def handle_client(self, client_socket, address):
        try:
            while self.running:
                # 设置接收超时，避免永久阻塞
                client_socket.settimeout(60.0)
                message = client_socket.recv(1024).decode('utf-8')
                if not message:
                    break
                    
                print(f"来自 {address} 的消息: {message}")
                
                # 调用消息接收回调
                if self.on_message_received:
                    self.on_message_received(address, message)
                
                # 广播消息给所有其他客户端
                self.broadcast_message(f"{address}: {message}", client_socket)
        except socket.timeout:
            print(f"客户端 {address} 连接超时")
        except Exception as e:
            print(f"处理客户端 {address} 时出错: {e}")
        finally:
            # 从客户端列表中移除
            self.clients = [(sock, addr) for sock, addr in self.clients if sock != client_socket]
            client_socket.close()
            print(f"客户端 {address} 已断开连接")
            
            # 调用客户端断开连接回调
            if self.on_client_disconnected:
                self.on_client_disconnected(address)

    def broadcast_message(self, message, sender_socket):
        # 发送消息给所有连接的客户端（除了发送者）
        disconnected_clients = []
        for client_socket, address in self.clients:
            if client_socket != sender_socket:
                try:
                    client_socket.send(message.encode('utf-8'))
                except:
                    # 如果发送失败，记录断开连接的客户端
                    disconnected_clients.append((client_socket, address))
        
        # 移除所有断开连接的客户端
        for client_socket, address in disconnected_clients:
            self.clients = [(sock, addr) for sock, addr in self.clients if sock != client_socket]
            client_socket.close()
            
            # 调用客户端断开连接回调
            if self.on_client_disconnected:
                self.on_client_disconnected(address)

    def stop_server(self):
        self.running = False
        # 关闭所有客户端连接
        for client_socket, address in self.clients:
            try:
                client_socket.close()
            except:
                pass
        self.clients = []
        
        if self.server_socket:
            self.server_socket.close()
        print("TCP消息服务器已停止")