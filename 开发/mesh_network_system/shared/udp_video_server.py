import socket
import threading
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("注意: OpenCV未安装，将只显示基本信息")

class UDPVideoServer:
    def __init__(self, host='0.0.0.0', port=12347):
        self.host = host
        self.port = port
        self.sock = None
        self.running = False
        self.clients = set()
        self.use_real_video = OPENCV_AVAILABLE
        self.on_status_update = None  # 状态更新回调

    def start_server(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 设置套接字选项，允许地址重用
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.running = True
            
            if self.on_status_update:
                self.on_status_update(f"UDP视频服务器启动在 {self.host}:{self.port}")
            
            if self.use_real_video:
                if self.on_status_update:
                    self.on_status_update("准备显示实时视频...")
            else:
                if self.on_status_update:
                    self.on_status_update("OpenCV不可用，将只显示基本信息...")
            
            while self.running:
                try:
                    data, client_address = self.sock.recvfrom(65536)  # 增大缓冲区以适应视频帧
                    
                    # 添加客户端到集合
                    self.clients.add(client_address)
                    
                    if self.use_real_video:
                        # 处理真实视频数据
                        self.process_video_frame(data, client_address)
                    else:
                        # 处理模拟视频数据
                        self.process_simulated_data(data, client_address)
                    
                except socket.error as e:
                    if self.running:
                        if self.on_status_update:
                            self.on_status_update(f"套接字错误: {e}")
                        break
                except Exception as e:
                    if self.running:
                        if self.on_status_update:
                            self.on_status_update(f"处理视频数据时出错: {e}")
        except Exception as e:
            if self.on_status_update:
                self.on_status_update(f"启动视频服务器时出错: {e}")
        finally:
            self.stop_server()

    def process_video_frame(self, data, client_address):
        """处理真实视频帧"""
        try:
            # 尝试解码JPEG图像
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # 显示视频帧
                window_name = f'来自 {client_address} 的视频'
                cv2.imshow(window_name, img)
                
                # 必须调用waitKey才能更新显示，但不阻塞太久
                # 使用1ms等待时间，确保窗口能及时刷新
                cv2.waitKey(1) & 0xFF
                
                # 发送确认回复
                if self.sock:
                    response = f"ACK: 接收到视频帧，大小 {img.shape}".encode('utf-8')
                    self.sock.sendto(response, client_address)
            else:
                # 可能是模拟数据
                decoded_data = data.decode('utf-8', errors='ignore')
                if self.on_status_update:
                    self.on_status_update(f"从 {client_address} 接收到数据: {decoded_data[:50]}...")
                
                # 回复客户端
                if self.sock:
                    response = f"ACK: {decoded_data[:20]}".encode('utf-8')
                    self.sock.sendto(response, client_address)
                
        except Exception as e:
            # 如果解码失败，当作模拟数据处理
            try:
                decoded_data = data.decode('utf-8')
                if self.on_status_update:
                    self.on_status_update(f"从 {client_address} 接收到模拟数据: {decoded_data[:50]}...")
                
                # 回复客户端
                if self.sock:
                    response = f"ACK: {decoded_data[:20]}".encode('utf-8')
                    self.sock.sendto(response, client_address)
            except:
                if self.on_status_update:
                    self.on_status_update(f"从 {client_address} 接收到无法识别的数据")

    def process_simulated_data(self, data, client_address):
        """处理模拟视频数据"""
        try:
            decoded_data = data.decode('utf-8')
            if self.on_status_update:
                self.on_status_update(f"从 {client_address} 接收到数据: {decoded_data[:50]}...")
            
            # 回复客户端
            if self.sock:
                response = f"ACK: {decoded_data[:20]}".encode('utf-8')
                self.sock.sendto(response, client_address)
        except Exception as e:
            if self.on_status_update:
                self.on_status_update(f"处理来自 {client_address} 的数据时出错: {e}")

    def stop_server(self):
        self.running = False
        if self.sock:
            self.sock.close()
        if self.use_real_video:
            cv2.destroyAllWindows()
        if self.on_status_update:
            self.on_status_update("UDP视频服务器已停止")