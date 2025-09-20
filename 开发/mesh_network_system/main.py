import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import socket
import threading

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入OpenCV
try:
    import cv2
    import numpy as np

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("注意: OpenCV未安装，视频功能将使用模拟数据")


class IntegratedMeshNetworkSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("自组网终端传输系统")
        self.root.geometry("700x600")

        # TCP消息传输相关
        self.message_client = None
        self.message_server = None
        self.message_connected = False
        self.message_server_running = False
        self.message_clients = []  # 服务器端的客户端列表

        # TCP文件传输相关
        self.file_client = None
        self.file_server = None
        self.file_connected = False
        self.file_server_running = False

        # UDP视频传输相关
        self.video_client = None
        self.video_server = None
        self.video_running = False
        self.video_server_running = False
        self.use_real_video = OPENCV_AVAILABLE

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 主标题
        title_label = tk.Label(self.root, text="自组网终端传输系统", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Notebook用于创建标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # 创建各个功能标签页
        self.create_network_settings_tab()
        self.create_message_tab()
        self.create_file_tab()
        self.create_video_tab()
        self.create_log_tab()

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")

    def create_network_settings_tab(self):
        """创建网络设置标签页"""
        network_frame = ttk.Frame(self.notebook)
        self.notebook.add(network_frame, text="网络设置")

        # 服务器设置
        server_frame = ttk.LabelFrame(network_frame, text="服务器设置")
        server_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(server_frame, text="监听IP:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.server_ip_entry = ttk.Entry(server_frame, width=15)
        self.server_ip_entry.grid(row=0, column=1, padx=5, pady=5)
        self.server_ip_entry.insert(0, "0.0.0.0")

        tk.Label(server_frame, text="消息端口:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.server_message_port_entry = ttk.Entry(server_frame, width=8)
        self.server_message_port_entry.grid(row=0, column=3, padx=5, pady=5)
        self.server_message_port_entry.insert(0, "12345")

        tk.Label(server_frame, text="文件端口:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.server_file_port_entry = ttk.Entry(server_frame, width=8)
        self.server_file_port_entry.grid(row=1, column=1, padx=5, pady=5)
        self.server_file_port_entry.insert(0, "12346")

        tk.Label(server_frame, text="视频端口:").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.server_video_port_entry = ttk.Entry(server_frame, width=8)
        self.server_video_port_entry.grid(row=1, column=3, padx=5, pady=5)
        self.server_video_port_entry.insert(0, "12347")

        # 客户端设置
        client_frame = ttk.LabelFrame(network_frame, text="客户端设置")
        client_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(client_frame, text="服务器IP:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.client_ip_entry = ttk.Entry(client_frame, width=15)
        self.client_ip_entry.grid(row=0, column=1, padx=5, pady=5)
        self.client_ip_entry.insert(0, "localhost")

        tk.Label(client_frame, text="消息端口:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.client_message_port_entry = ttk.Entry(client_frame, width=8)
        self.client_message_port_entry.grid(row=0, column=3, padx=5, pady=5)
        self.client_message_port_entry.insert(0, "12345")

        tk.Label(client_frame, text="文件端口:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.client_file_port_entry = ttk.Entry(client_frame, width=8)
        self.client_file_port_entry.grid(row=1, column=1, padx=5, pady=5)
        self.client_file_port_entry.insert(0, "12346")

        tk.Label(client_frame, text="视频端口:").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.client_video_port_entry = ttk.Entry(client_frame, width=8)
        self.client_video_port_entry.grid(row=1, column=3, padx=5, pady=5)
        self.client_video_port_entry.insert(0, "12347")

        # 控制按钮
        control_frame = ttk.Frame(network_frame)
        control_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(control_frame, text="应用设置", command=self.apply_settings).pack(side="left", padx=5)
        ttk.Button(control_frame, text="重置为默认值", command=self.reset_settings).pack(side="left", padx=5)

    def create_message_tab(self):
        """创建消息传输标签页"""
        message_frame = ttk.Frame(self.notebook)
        self.notebook.add(message_frame, text="消息传输")

        # 服务器控制
        server_frame = ttk.LabelFrame(message_frame, text="消息服务器")
        server_frame.pack(fill="x", padx=10, pady=5)

        self.start_message_server_btn = ttk.Button(server_frame, text="启动消息服务器",
                                                   command=self.start_message_server)
        self.start_message_server_btn.pack(side="left", padx=5, pady=5)

        self.stop_message_server_btn = ttk.Button(server_frame, text="停止消息服务器", command=self.stop_message_server,
                                                  state="disabled")
        self.stop_message_server_btn.pack(side="left", padx=5, pady=5)

        # 客户端控制
        client_frame = ttk.LabelFrame(message_frame, text="消息客户端")
        client_frame.pack(fill="x", padx=10, pady=5)

        self.connect_message_btn = ttk.Button(client_frame, text="连接消息服务器", command=self.connect_message_server)
        self.connect_message_btn.pack(side="left", padx=5, pady=5)

        self.disconnect_message_btn = ttk.Button(client_frame, text="断开消息连接",
                                                 command=self.disconnect_message_server, state="disabled")
        self.disconnect_message_btn.pack(side="left", padx=5, pady=5)

        # 消息显示区域
        message_display_frame = ttk.LabelFrame(message_frame, text="消息显示")
        message_display_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.message_text = tk.Text(message_display_frame, height=10)
        self.message_text.pack(fill="both", expand=True, padx=5, pady=5)

        # 消息滚动条
        message_scrollbar = ttk.Scrollbar(self.message_text, command=self.message_text.yview)
        message_scrollbar.pack(side="right", fill="y")
        self.message_text.config(yscrollcommand=message_scrollbar.set)

        # 消息输入区域
        input_frame = ttk.Frame(message_display_frame)
        input_frame.pack(fill="x", padx=5, pady=5)

        self.message_entry = ttk.Entry(input_frame)
        self.message_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.message_entry.bind("<Return>", self.send_message)

        self.send_btn = ttk.Button(input_frame, text="发送", command=self.send_message, state="disabled")
        self.send_btn.pack(side="right", padx=5)

    def create_file_tab(self):
        """创建文件传输标签页"""
        file_frame = ttk.Frame(self.notebook)
        self.notebook.add(file_frame, text="文件传输")

        # 服务器控制
        server_frame = ttk.LabelFrame(file_frame, text="文件服务器")
        server_frame.pack(fill="x", padx=10, pady=5)

        self.start_file_server_btn = ttk.Button(server_frame, text="启动文件服务器", command=self.start_file_server)
        self.start_file_server_btn.pack(side="left", padx=5, pady=5)

        self.stop_file_server_btn = ttk.Button(server_frame, text="停止文件服务器", command=self.stop_file_server,
                                               state="disabled")
        self.stop_file_server_btn.pack(side="left", padx=5, pady=5)

        # 客户端控制
        client_frame = ttk.LabelFrame(file_frame, text="文件客户端")
        client_frame.pack(fill="x", padx=10, pady=5)

        self.connect_file_btn = ttk.Button(client_frame, text="连接文件服务器", command=self.connect_file_server)
        self.connect_file_btn.pack(side="left", padx=5, pady=5)

        self.disconnect_file_btn = ttk.Button(client_frame, text="断开文件连接", command=self.disconnect_file_server,
                                              state="disabled")
        self.disconnect_file_btn.pack(side="left", padx=5, pady=5)

        # 文件传输区域
        transfer_frame = ttk.LabelFrame(file_frame, text="文件传输")
        transfer_frame.pack(fill="x", padx=10, pady=5)

        self.file_path_entry = ttk.Entry(transfer_frame)
        self.file_path_entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        self.browse_btn = ttk.Button(transfer_frame, text="浏览", command=self.browse_file, state="disabled")
        self.browse_btn.pack(side="left", padx=5, pady=5)

        self.send_file_btn = ttk.Button(transfer_frame, text="发送文件", command=self.send_file, state="disabled")
        self.send_file_btn.pack(side="left", padx=5, pady=5)

    def create_video_tab(self):
        """创建视频传输标签页"""
        video_frame = ttk.Frame(self.notebook)
        self.notebook.add(video_frame, text="视频传输")

        # 服务器控制
        server_frame = ttk.LabelFrame(video_frame, text="视频服务器")
        server_frame.pack(fill="x", padx=10, pady=5)

        self.start_video_server_btn = ttk.Button(server_frame, text="启动视频服务器", command=self.start_video_server)
        self.start_video_server_btn.pack(side="left", padx=5, pady=5)

        self.stop_video_server_btn = ttk.Button(server_frame, text="停止视频服务器", command=self.stop_video_server,
                                                state="disabled")
        self.stop_video_server_btn.pack(side="left", padx=5, pady=5)

        # 客户端控制
        client_frame = ttk.LabelFrame(video_frame, text="视频客户端")
        client_frame.pack(fill="x", padx=10, pady=5)

        self.start_video_client_btn = ttk.Button(client_frame, text="开始视频传输", command=self.start_video_client)
        self.start_video_client_btn.pack(side="left", padx=5, pady=5)

        self.stop_video_client_btn = ttk.Button(client_frame, text="停止视频传输", command=self.stop_video_client,
                                                state="disabled")
        self.stop_video_client_btn.pack(side="left", padx=5, pady=5)

        # 视频状态显示
        status_frame = ttk.LabelFrame(video_frame, text="视频状态")
        status_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.video_status_text = tk.Text(status_frame, height=8)
        self.video_status_text.pack(fill="both", expand=True, padx=5, pady=5)

    def create_log_tab(self):
        """创建日志标签页"""
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="系统日志")

        # 日志显示区域
        log_display_frame = ttk.LabelFrame(log_frame, text="日志信息")
        log_display_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_text = tk.Text(log_display_frame, height=15)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # 日志滚动条
        log_scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        log_scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=log_scrollbar.set)

        # 清除日志按钮
        clear_frame = ttk.Frame(log_display_frame)
        clear_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(clear_frame, text="清除日志", command=self.clear_log).pack(side="right")

    def apply_settings(self):
        """应用网络设置"""
        # 这里可以添加验证逻辑
        messagebox.showinfo("设置", "网络设置已应用")
        self.log_message("网络设置已应用")

    def reset_settings(self):
        """重置为默认设置"""
        self.server_ip_entry.delete(0, tk.END)
        self.server_ip_entry.insert(0, "0.0.0.0")
        self.server_message_port_entry.delete(0, tk.END)
        self.server_message_port_entry.insert(0, "12345")
        self.server_file_port_entry.delete(0, tk.END)
        self.server_file_port_entry.insert(0, "12346")
        self.server_video_port_entry.delete(0, tk.END)
        self.server_video_port_entry.insert(0, "12347")

        self.client_ip_entry.delete(0, tk.END)
        self.client_ip_entry.insert(0, "localhost")
        self.client_message_port_entry.delete(0, tk.END)
        self.client_message_port_entry.insert(0, "12345")
        self.client_file_port_entry.delete(0, tk.END)
        self.client_file_port_entry.insert(0, "12346")
        self.client_video_port_entry.delete(0, tk.END)
        self.client_video_port_entry.insert(0, "12347")

        messagebox.showinfo("设置", "已重置为默认设置")
        self.log_message("已重置为默认网络设置")

    # TCP消息传输相关方法
    def start_message_server(self):
        """启动TCP消息服务器"""
        try:
            host = self.server_ip_entry.get()
            port = int(self.server_message_port_entry.get())

            self.message_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.message_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.message_server.bind((host, port))
            self.message_server.listen(5)
            self.message_server_running = True

            # 启动服务器线程
            server_thread = threading.Thread(target=self.run_message_server)
            server_thread.daemon = True
            server_thread.start()

            self.start_message_server_btn.config(state="disabled")
            self.stop_message_server_btn.config(state="normal")
            self.status_var.set(f"消息服务器运行在 {host}:{port}")
            self.log_message(f"消息服务器启动在 {host}:{port}")
            self.message_text.insert(tk.END, f"消息服务器启动在 {host}:{port}\n")
            self.message_text.see(tk.END)

        except Exception as e:
            messagebox.showerror("错误", f"启动消息服务器失败: {str(e)}")

    def run_message_server(self):
        """运行消息服务器"""
        try:
            while self.message_server_running:
                try:
                    client_socket, address = self.message_server.accept()
                    self.log_message(f"消息客户端 {address} 已连接")
                    self.message_text.insert(tk.END, f"消息客户端 {address} 已连接\n")
                    self.message_text.see(tk.END)

                    # 为每个客户端创建线程
                    client_thread = threading.Thread(target=self.handle_message_client, args=(client_socket, address))
                    client_thread.daemon = True
                    client_thread.start()

                    self.message_clients.append(client_socket)
                except Exception as e:
                    if self.message_server_running:
                        self.log_message(f"接受消息客户端连接时出错: {e}")
        except Exception as e:
            self.log_message(f"消息服务器运行出错: {e}")

    def handle_message_client(self, client_socket, address):
        """处理消息客户端"""
        try:
            while self.message_server_running:
                try:
                    message = client_socket.recv(1024).decode('utf-8')
                    if not message:
                        break

                    self.log_message(f"来自 {address} 的消息: {message}")
                    self.message_text.insert(tk.END, f"{address}: {message}\n")
                    self.message_text.see(tk.END)

                    # 广播消息给其他客户端
                    self.broadcast_message(f"{address}: {message}", client_socket)
                except Exception as e:
                    break
        finally:
            if client_socket in self.message_clients:
                self.message_clients.remove(client_socket)
            client_socket.close()
            self.log_message(f"消息客户端 {address} 已断开")
            self.message_text.insert(tk.END, f"消息客户端 {address} 已断开\n")
            self.message_text.see(tk.END)

    def broadcast_message(self, message, sender_socket):
        """广播消息"""
        disconnected_clients = []
        for client in self.message_clients:
            if client != sender_socket:
                try:
                    client.send(message.encode('utf-8'))
                except:
                    disconnected_clients.append(client)

        # 移除断开的客户端
        for client in disconnected_clients:
            if client in self.message_clients:
                self.message_clients.remove(client)

    def stop_message_server(self):
        """停止TCP消息服务器"""
        self.message_server_running = False
        if self.message_server:
            self.message_server.close()

        # 关闭所有客户端连接
        for client in self.message_clients:
            try:
                client.close()
            except:
                pass
        self.message_clients = []

        self.start_message_server_btn.config(state="normal")
        self.stop_message_server_btn.config(state="disabled")
        self.status_var.set("消息服务器已停止")
        self.log_message("消息服务器已停止")
        self.message_text.insert(tk.END, "消息服务器已停止\n")
        self.message_text.see(tk.END)

    def connect_message_server(self):
        """连接TCP消息服务器"""
        try:
            host = self.client_ip_entry.get()
            port = int(self.client_message_port_entry.get())

            self.message_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.message_client.connect((host, port))
            self.message_connected = True

            # 启动接收消息线程
            receive_thread = threading.Thread(target=self.receive_messages)
            receive_thread.daemon = True
            receive_thread.start()

            self.connect_message_btn.config(state="disabled")
            self.disconnect_message_btn.config(state="normal")
            self.send_btn.config(state="normal")
            self.status_var.set(f"已连接到消息服务器 {host}:{port}")
            self.log_message(f"已连接到消息服务器 {host}:{port}")
            self.message_text.insert(tk.END, f"已连接到消息服务器 {host}:{port}\n")
            self.message_text.see(tk.END)

        except Exception as e:
            messagebox.showerror("错误", f"连接消息服务器失败: {str(e)}")

    def receive_messages(self):
        """接收消息"""
        while self.message_connected:
            try:
                message = self.message_client.recv(1024).decode('utf-8')
                if message:
                    self.message_text.insert(tk.END, f"{message}\n")
                    self.message_text.see(tk.END)
                else:
                    break
            except Exception as e:
                if self.message_connected:
                    self.log_message(f"接收消息时出错: {e}")
                break
        self.disconnect_message_server()

    def send_message(self, event=None):
        """发送消息"""
        if not self.message_connected:
            messagebox.showwarning("警告", "请先连接到消息服务器")
            return

        message = self.message_entry.get()
        if message:
            try:
                self.message_client.send(message.encode('utf-8'))
                self.message_text.insert(tk.END, f"我: {message}\n")
                self.message_text.see(tk.END)
                self.message_entry.delete(0, tk.END)
            except Exception as e:
                messagebox.showerror("错误", f"发送消息失败: {str(e)}")

    def disconnect_message_server(self):
        """断开消息服务器连接"""
        self.message_connected = False
        if self.message_client:
            self.message_client.close()

        self.connect_message_btn.config(state="normal")
        self.disconnect_message_btn.config(state="disabled")
        self.send_btn.config(state="disabled")
        self.status_var.set("已断开消息服务器连接")
        self.log_message("已断开消息服务器连接")
        self.message_text.insert(tk.END, "已断开消息服务器连接\n")
        self.message_text.see(tk.END)

    # TCP文件传输相关方法
    def start_file_server(self):
        """启动TCP文件服务器"""
        try:
            host = self.server_ip_entry.get()
            port = int(self.server_file_port_entry.get())

            self.file_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.file_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.file_server.bind((host, port))
            self.file_server.listen(5)
            self.file_server_running = True

            # 启动服务器线程
            server_thread = threading.Thread(target=self.run_file_server)
            server_thread.daemon = True
            server_thread.start()

            self.start_file_server_btn.config(state="disabled")
            self.stop_file_server_btn.config(state="normal")
            self.status_var.set(f"文件服务器运行在 {host}:{port}")
            self.log_message(f"文件服务器启动在 {host}:{port}")

        except Exception as e:
            messagebox.showerror("错误", f"启动文件服务器失败: {str(e)}")

    def run_file_server(self):
        """运行文件服务器"""
        try:
            while self.file_server_running:
                try:
                    client_socket, address = self.file_server.accept()
                    self.log_message(f"文件客户端 {address} 已连接")

                    # 为每个客户端创建线程
                    client_thread = threading.Thread(target=self.handle_file_client, args=(client_socket, address))
                    client_thread.daemon = True
                    client_thread.start()
                except Exception as e:
                    if self.file_server_running:
                        self.log_message(f"接受文件客户端连接时出错: {e}")
        except Exception as e:
            self.log_message(f"文件服务器运行出错: {e}")

    def handle_file_client(self, client_socket, address):
        """处理文件客户端"""
        try:
            # 接收文件信息
            file_info = client_socket.recv(1024).decode('utf-8')
            if '|' not in file_info:
                client_socket.close()
                return

            filename, file_size = file_info.split('|')
            file_size = int(file_size)

            self.log_message(f"从 {address} 接收文件: {filename} ({file_size} 字节)")

            # 确认准备接收
            client_socket.send("READY".encode('utf-8'))

            # 接收文件数据
            filename = "received_" + filename  # 避免文件名冲突
            with open(filename, 'wb') as f:
                bytes_received = 0
                while bytes_received < file_size:
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    f.write(data)
                    bytes_received += len(data)

            self.log_message(f"文件 {filename} 接收完成 ({bytes_received} 字节)")
            client_socket.send(f"文件 {filename} 接收完成".encode('utf-8'))

        except Exception as e:
            self.log_message(f"处理文件客户端 {address} 时出错: {e}")
        finally:
            client_socket.close()
            self.log_message(f"文件客户端 {address} 已断开")

    def stop_file_server(self):
        """停止TCP文件服务器"""
        self.file_server_running = False
        if self.file_server:
            self.file_server.close()

        self.start_file_server_btn.config(state="normal")
        self.stop_file_server_btn.config(state="disabled")
        self.status_var.set("文件服务器已停止")
        self.log_message("文件服务器已停止")

    def connect_file_server(self):
        """连接TCP文件服务器"""
        try:
            host = self.client_ip_entry.get()
            port = int(self.client_file_port_entry.get())

            self.file_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.file_client.connect((host, port))
            self.file_connected = True

            self.connect_file_btn.config(state="disabled")
            self.disconnect_file_btn.config(state="normal")
            self.browse_btn.config(state="normal")
            self.send_file_btn.config(state="normal")
            self.status_var.set(f"已连接到文件服务器 {host}:{port}")
            self.log_message(f"已连接到文件服务器 {host}:{port}")

        except Exception as e:
            messagebox.showerror("错误", f"连接文件服务器失败: {str(e)}")

    def disconnect_file_server(self):
        """断开文件服务器连接"""
        self.file_connected = False
        if self.file_client:
            self.file_client.close()

        self.connect_file_btn.config(state="normal")
        self.disconnect_file_btn.config(state="disabled")
        self.browse_btn.config(state="disabled")
        self.send_file_btn.config(state="disabled")
        self.status_var.set("已断开文件服务器连接")
        self.log_message("已断开文件服务器连接")

    def browse_file(self):
        """浏览文件"""
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)

    def send_file(self):
        """发送文件"""
        if not self.file_connected:
            messagebox.showwarning("警告", "请先连接到文件服务器")
            return

        file_path = self.file_path_entry.get()
        if not file_path:
            messagebox.showwarning("警告", "请选择要发送的文件")
            return

        try:
            # 在单独线程中发送文件，避免阻塞UI
            send_thread = threading.Thread(target=self._send_file_thread, args=(file_path,))
            send_thread.daemon = True
            send_thread.start()
        except Exception as e:
            messagebox.showerror("错误", f"文件发送失败: {str(e)}")

    def _send_file_thread(self, file_path):
        """发送文件线程"""
        try:
            import os
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            # 发送文件信息
            file_info = f"{filename}|{file_size}"
            self.file_client.send(file_info.encode('utf-8'))

            # 等待服务器确认
            response = self.file_client.recv(1024).decode('utf-8')
            if response != "READY":
                self.root.after(0, lambda: messagebox.showerror("错误", f"服务器未准备好接收文件: {response}"))
                return

            # 发送文件数据
            with open(file_path, 'rb') as f:
                bytes_sent = 0
                while bytes_sent < file_size:
                    data = f.read(1024)
                    if not data:
                        break
                    self.file_client.send(data)
                    bytes_sent += len(data)

            # 等待服务器确认文件接收完成
            try:
                confirmation = self.file_client.recv(1024).decode('utf-8')
                self.root.after(0, lambda: messagebox.showinfo("成功", confirmation))
            except:
                self.root.after(0, lambda: messagebox.showinfo("成功", f"文件 {filename} 发送完成 ({bytes_sent} 字节)"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"发送文件时出错: {str(e)}"))

    # UDP视频传输相关方法
    def start_video_server(self):
        """启动UDP视频服务器"""
        try:
            host = self.server_ip_entry.get()
            port = int(self.server_video_port_entry.get())

            self.video_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.video_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.video_server.bind((host, port))
            self.video_server_running = True

            # 启动服务器线程
            server_thread = threading.Thread(target=self.run_video_server)
            server_thread.daemon = True
            server_thread.start()

            self.start_video_server_btn.config(state="disabled")
            self.stop_video_server_btn.config(state="normal")
            self.status_var.set(f"视频服务器运行在 {host}:{port}")
            self.log_message(f"视频服务器启动在 {host}:{port}")
            self.video_status_text.insert(tk.END, f"视频服务器启动在 {host}:{port}\n")
            if not self.use_real_video:
                self.video_status_text.insert(tk.END, "OpenCV不可用，将只显示基本信息\n")
            self.video_status_text.see(tk.END)

        except Exception as e:
            messagebox.showerror("错误", f"启动视频服务器失败: {str(e)}")

    def run_video_server(self):
        """运行视频服务器"""
        try:
            while self.video_server_running:
                try:
                    data, client_address = self.video_server.recvfrom(65536)

                    if self.use_real_video:
                        # 处理真实视频数据
                        self.process_video_frame(data, client_address)
                    else:
                        # 处理模拟视频数据
                        self.process_simulated_video_data(data, client_address)

                except socket.error as e:
                    if self.video_server_running:
                        self.log_message(f"视频服务器套接字错误: {e}")
                        break
                except Exception as e:
                    if self.video_server_running:
                        self.log_message(f"处理视频数据时出错: {e}")
        except Exception as e:
            self.log_message(f"视频服务器运行出错: {e}")
        finally:
            if self.use_real_video:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass

    def process_video_frame(self, data, client_address):
        """处理视频帧"""
        try:
            if self.use_real_video:
                # 尝试解码JPEG图像
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    # 显示视频帧
                    window_name = f'来自 {client_address} 的视频'
                    cv2.imshow(window_name, img)

                    # 必须调用waitKey才能更新显示
                    cv2.waitKey(1) & 0xFF

                    # 发送确认回复
                    response = f"ACK: 接收到视频帧".encode('utf-8')
                    self.video_server.sendto(response, client_address)
                else:
                    # 可能是模拟数据
                    decoded_data = data.decode('utf-8', errors='ignore')
                    self.video_status_text.insert(tk.END, f"从 {client_address} 接收到数据: {decoded_data[:50]}...\n")
                    self.video_status_text.see(tk.END)

                    # 回复客户端
                    response = f"ACK: {decoded_data[:20]}".encode('utf-8')
                    self.video_server.sendto(response, client_address)
        except Exception as e:
            # 如果解码失败，当作模拟数据处理
            try:
                decoded_data = data.decode('utf-8')
                self.video_status_text.insert(tk.END, f"从 {client_address} 接收到模拟数据: {decoded_data[:50]}...\n")
                self.video_status_text.see(tk.END)

                # 回复客户端
                response = f"ACK: {decoded_data[:20]}".encode('utf-8')
                self.video_server.sendto(response, client_address)
            except:
                self.video_status_text.insert(tk.END, f"从 {client_address} 接收到无法识别的数据\n")
                self.video_status_text.see(tk.END)

    def process_simulated_video_data(self, data, client_address):
        """处理模拟视频数据"""
        try:
            decoded_data = data.decode('utf-8')
            self.video_status_text.insert(tk.END, f"从 {client_address} 接收到数据: {decoded_data[:50]}...\n")
            self.video_status_text.see(tk.END)

            # 回复客户端
            response = f"ACK: {decoded_data[:20]}".encode('utf-8')
            self.video_server.sendto(response, client_address)
        except Exception as e:
            self.log_message(f"处理来自 {client_address} 的模拟数据时出错: {e}")

    def stop_video_server(self):
        """停止UDP视频服务器"""
        self.video_server_running = False
        if self.video_server:
            self.video_server.close()

        if self.use_real_video:
            try:
                cv2.destroyAllWindows()
            except:
                pass

        self.start_video_server_btn.config(state="normal")
        self.stop_video_server_btn.config(state="disabled")
        self.status_var.set("视频服务器已停止")
        self.log_message("视频服务器已停止")
        self.video_status_text.insert(tk.END, "视频服务器已停止\n")
        self.video_status_text.see(tk.END)

    def start_video_client(self):
        """启动UDP视频客户端"""
        try:
            host = self.client_ip_entry.get()
            port = int(self.client_video_port_entry.get())

            self.video_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.video_running = True

            # 启动客户端线程
            client_thread = threading.Thread(target=self.run_video_client, args=(host, port))
            client_thread.daemon = True
            client_thread.start()

            self.start_video_client_btn.config(state="disabled")
            self.stop_video_client_btn.config(state="normal")
            self.status_var.set("视频传输进行中...")
            self.log_message(f"UDP视频客户端已启动，连接到 {host}:{port}")
            self.video_status_text.insert(tk.END, f"UDP视频客户端已启动，连接到 {host}:{port}\n")

            if self.use_real_video:
                self.video_status_text.insert(tk.END, "开始捕获和发送实时视频...\n")
            else:
                self.video_status_text.insert(tk.END, "OpenCV不可用，开始发送模拟视频数据...\n")
            self.video_status_text.see(tk.END)

        except Exception as e:
            messagebox.showerror("错误", f"启动视频客户端失败: {str(e)}")

    def run_video_client(self, host, port):
        """运行视频客户端"""
        try:
            if self.use_real_video:
                self.send_real_video_data(host, port)
            else:
                self.send_simulated_video_data(host, port)
        except Exception as e:
            if self.video_running:
                self.log_message(f"视频客户端运行出错: {e}")
        finally:
            if self.video_client:
                self.video_client.close()

    def send_real_video_data(self, host, port):
        """发送真实视频数据"""
        cap = None
        try:
            # 打开默认摄像头
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                self.video_status_text.insert(tk.END, "无法打开摄像头，切换到模拟视频数据\n")
                self.video_status_text.see(tk.END)
                self.send_simulated_video_data(host, port)
                return

            self.video_status_text.insert(tk.END, "摄像头已打开，开始传输视频\n")
            self.video_status_text.see(tk.END)

            while self.video_running:
                # 读取视频帧
                ret, frame = cap.read()
                if not ret:
                    self.video_status_text.insert(tk.END, "无法读取视频帧\n")
                    self.video_status_text.see(tk.END)
                    break

                # 调整帧大小以减少数据量
                frame = cv2.resize(frame, (320, 240))

                # 编码图像为JPEG格式
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

                # 发送数据到服务器
                if self.video_client:
                    self.video_client.sendto(buffer.tobytes(), (host, port))

                # 控制帧率
                import time
                time.sleep(0.03)  # 约30 FPS

        except Exception as e:
            if self.video_running:
                self.log_message(f"发送视频数据时出错: {e}")
        finally:
            if cap:
                cap.release()

    def send_simulated_video_data(self, host, port):
        """发送模拟视频数据"""
        try:
            frame_count = 0
            while self.video_running:
                # 模拟视频帧数据
                frame_data = f"VIDEO_FRAME_{frame_count}".encode('utf-8')

                # 发送数据到服务器
                if self.video_client:
                    self.video_client.sendto(frame_data, (host, port))

                frame_count += 1

                # 控制发送速度
                import time
                time.sleep(0.1)  # 10 FPS

        except Exception as e:
            if self.video_running:
                self.log_message(f"发送模拟视频数据时出错: {e}")

    def stop_video_client(self):
        """停止UDP视频客户端"""
        self.video_running = False

        self.start_video_client_btn.config(state="normal")
        self.stop_video_client_btn.config(state="disabled")
        self.status_var.set("视频传输已停止")
        self.log_message("UDP视频客户端已停止")
        self.video_status_text.insert(tk.END, "视频传输已停止\n")
        self.video_status_text.see(tk.END)

    # 日志和通用方法
    def log_message(self, message):
        """记录日志消息"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def clear_log(self):
        """清除日志"""
        self.log_text.delete(1.0, tk.END)


def main():
    root = tk.Tk()
    app = IntegratedMeshNetworkSystem(root)
    root.mainloop()


if __name__ == "__main__":
    main()