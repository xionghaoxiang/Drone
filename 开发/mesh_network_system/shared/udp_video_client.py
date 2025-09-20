import socket
import threading
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("注意: OpenCV未安装，将使用模拟视频数据")

class UDPVideoClient:
    def __init__(self, host='localhost', port=12347):
        self.host = host
        self.port = port
        self.sock = None
        self.running = False
        self.use_real_video = OPENCV_AVAILABLE
        self.on_status_update = None  # 状态更新回调

    def start_client(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.running = True
            
            if self.on_status_update:
                self.on_status_update(f"UDP视频客户端已启动，连接到 {self.host}:{self.port}")
            
            if self.use_real_video:
                if self.on_status_update:
                    self.on_status_update("开始捕获和发送实时视频...")
                # 发送真实视频数据
                self.send_real_video_data()
            else:
                if self.on_status_update:
                    self.on_status_update("OpenCV不可用，开始发送模拟视频数据...")
                # 发送模拟视频数据
                self.send_video_data()
            
        except Exception as e:
            if self.on_status_update:
                self.on_status_update(f"启动视频客户端时出错: {e}")
        finally:
            self.stop_client()

    def send_real_video_data(self):
        """发送真实视频数据"""
        cap = None
        try:
            # 打开默认摄像头
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                if self.on_status_update:
                    self.on_status_update("无法打开摄像头，切换到模拟视频数据")
                self.send_video_data()
                return
            
            if self.on_status_update:
                self.on_status_update("摄像头已打开，开始传输视频")
            
            while self.running:
                # 读取视频帧
                ret, frame = cap.read()
                if not ret:
                    if self.on_status_update:
                        self.on_status_update("无法读取视频帧")
                    break
                
                # 调整帧大小以减少数据量
                frame = cv2.resize(frame, (320, 240))
                
                # 编码图像为JPEG格式
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                
                # 发送数据到服务器
                if self.sock:
                    self.sock.sendto(buffer.tobytes(), (self.host, self.port))
                
                # 控制帧率
                import time
                time.sleep(0.03)  # 约30 FPS
                
        except Exception as e:
            if self.running and self.on_status_update:
                self.on_status_update(f"发送视频数据时出错: {e}")
        finally:
            if cap:
                cap.release()

    def send_video_data(self):
        """发送模拟视频数据（原实现）"""
        try:
            frame_count = 0
            while self.running:
                # 模拟视频帧数据
                frame_data = f"VIDEO_FRAME_{frame_count}".encode('utf-8')
                
                # 发送数据到服务器
                if self.sock:
                    self.sock.sendto(frame_data, (self.host, self.port))
                
                frame_count += 1
                
                # 控制发送速度
                import time
                time.sleep(0.1)  # 10 FPS
                
        except Exception as e:
            if self.running and self.on_status_update:
                self.on_status_update(f"发送视频数据时出错: {e}")

    def stop_client(self):
        self.running = False
        if self.sock:
            self.sock.close()
        if self.on_status_update:
            self.on_status_update("UDP视频客户端已停止")