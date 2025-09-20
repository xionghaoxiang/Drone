import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# 添加共享模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

# 导入网络模块
from tcp_message_server import TCPMessageServer
from tcp_file_server import TCPFileServer
from udp_video_server import UDPVideoServer

class MeshNetworkServerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("自组网终端传输系统 - 服务器")
        self.root.geometry("500x400")
        
        # 服务器实例
        self.message_server = None
        self.file_server = None
        self.video_server = None
        
        # 服务器线程
        self.message_thread = None
        self.file_thread = None
        self.video_thread = None
        
        # 服务器运行状态
        self.message_server_running = False
        self.file_server_running = False
        self.video_server_running = False
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 主标题
        title_label = tk.Label(self.root, text="自组网终端服务器", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # IP和端口设置框架
        settings_frame = ttk.LabelFrame(self.root, text="服务器设置")
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        # IP地址输入
        ip_frame = ttk.Frame(settings_frame)
        ip_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(ip_frame, text="监听IP:").pack(side="left", padx=5)
        self.ip_entry = ttk.Entry(ip_frame)
        self.ip_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.ip_entry.insert(0, "0.0.0.0")
        
        # 端口设置框架
        port_frame = ttk.Frame(settings_frame)
        port_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(port_frame, text="消息端口:").pack(side="left", padx=5)
        self.message_port_entry = ttk.Entry(port_frame, width=10)
        self.message_port_entry.pack(side="left", padx=5)
        self.message_port_entry.insert(0, "12345")
        
        tk.Label(port_frame, text="文件端口:").pack(side="left", padx=5)
        self.file_port_entry = ttk.Entry(port_frame, width=10)
        self.file_port_entry.pack(side="left", padx=5)
        self.file_port_entry.insert(0, "12346")
        
        tk.Label(port_frame, text="视频端口:").pack(side="left", padx=5)
        self.video_port_entry = ttk.Entry(port_frame, width=10)
        self.video_port_entry.pack(side="left", padx=5)
        self.video_port_entry.insert(0, "12347")
        
        # 服务器控制按钮框架
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        # TCP消息服务器按钮
        self.message_start_btn = ttk.Button(button_frame, text="启动消息服务器", command=self.start_message_server)
        self.message_start_btn.pack(side="left", padx=5)
        
        self.message_stop_btn = ttk.Button(button_frame, text="停止消息服务器", command=self.stop_message_server, state="disabled")
        self.message_stop_btn.pack(side="left", padx=5)
        
        # TCP文件服务器按钮
        self.file_start_btn = ttk.Button(button_frame, text="启动文件服务器", command=self.start_file_server)
        self.file_start_btn.pack(side="left", padx=5)
        
        self.file_stop_btn = ttk.Button(button_frame, text="停止文件服务器", command=self.stop_file_server, state="disabled")
        self.file_stop_btn.pack(side="left", padx=5)
        
        # UDP视频服务器按钮
        self.video_start_btn = ttk.Button(button_frame, text="启动视频服务器", command=self.start_video_server)
        self.video_start_btn.pack(side="left", padx=5)
        
        self.video_stop_btn = ttk.Button(button_frame, text="停止视频服务器", command=self.stop_video_server, state="disabled")
        self.video_stop_btn.pack(side="left", padx=5)
        
        # 日志显示区域
        log_frame = ttk.LabelFrame(self.root, text="服务器日志")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 日志滚动条
        log_scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        log_scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
        
    def log_message(self, message):
        """在日志区域添加消息"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        
    def start_message_server(self):
        """启动TCP消息服务器"""
        try:
            host = self.ip_entry.get()
            port = int(self.message_port_entry.get())
            
            self.message_server = TCPMessageServer(host=host, port=port)
            # 设置回调函数
            self.message_server.on_client_connected = lambda addr: self.log_message(f"消息客户端 {addr} 已连接")
            self.message_server.on_client_disconnected = lambda addr: self.log_message(f"消息客户端 {addr} 已断开")
            self.message_server.on_message_received = lambda addr, msg: self.log_message(f"来自 {addr} 的消息: {msg}")
            
            self.message_thread = threading.Thread(target=self.message_server.start_server)
            self.message_thread.daemon = True
            self.message_thread.start()
            
            self.message_server_running = True
            self.message_start_btn.config(state="disabled")
            self.message_stop_btn.config(state="normal")
            self.status_var.set(f"消息服务器运行在 {host}:{port}")
            self.log_message(f"消息服务器启动在 {host}:{port}")
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的端口号")
        except Exception as e:
            messagebox.showerror("错误", f"启动消息服务器失败: {str(e)}")
            
    def stop_message_server(self):
        """停止TCP消息服务器"""
        if self.message_server:
            self.message_server.stop_server()
            
        self.message_server_running = False
        self.message_start_btn.config(state="normal")
        self.message_stop_btn.config(state="disabled")
        self.status_var.set("消息服务器已停止")
        self.log_message("消息服务器已停止")
        
    def start_file_server(self):
        """启动TCP文件服务器"""
        try:
            host = self.ip_entry.get()
            port = int(self.file_port_entry.get())
            
            self.file_server = TCPFileServer(host=host, port=port)
            self.file_thread = threading.Thread(target=self.file_server.start_server)
            self.file_thread.daemon = True
            self.file_thread.start()
            
            self.file_server_running = True
            self.file_start_btn.config(state="disabled")
            self.file_stop_btn.config(state="normal")
            self.status_var.set(f"文件服务器运行在 {host}:{port}")
            self.log_message(f"文件服务器启动在 {host}:{port}")
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的端口号")
        except Exception as e:
            messagebox.showerror("错误", f"启动文件服务器失败: {str(e)}")
            
    def stop_file_server(self):
        """停止TCP文件服务器"""
        if self.file_server:
            self.file_server.stop_server()
            
        self.file_server_running = False
        self.file_start_btn.config(state="normal")
        self.file_stop_btn.config(state="disabled")
        self.status_var.set("文件服务器已停止")
        self.log_message("文件服务器已停止")
        
    def start_video_server(self):
        """启动UDP视频服务器"""
        try:
            host = self.ip_entry.get()
            port = int(self.video_port_entry.get())
            
            self.video_server = UDPVideoServer(host=host, port=port)
            # 设置状态更新回调
            self.video_server.on_status_update = self.log_message
            
            self.video_thread = threading.Thread(target=self.video_server.start_server)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            self.video_server_running = True
            self.video_start_btn.config(state="disabled")
            self.video_stop_btn.config(state="normal")
            self.status_var.set(f"视频服务器运行在 {host}:{port}")
            self.log_message(f"视频服务器启动在 {host}:{port}")
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的端口号")
        except Exception as e:
            messagebox.showerror("错误", f"启动视频服务器失败: {str(e)}")
            
    def stop_video_server(self):
        """停止UDP视频服务器"""
        if self.video_server:
            self.video_server.stop_server()
            
        self.video_server_running = False
        self.video_start_btn.config(state="normal")
        self.video_stop_btn.config(state="disabled")
        self.status_var.set("视频服务器已停止")
        self.log_message("视频服务器已停止")


def main():
    root = tk.Tk()
    app = MeshNetworkServerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()