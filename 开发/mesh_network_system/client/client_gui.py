import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading

# 添加共享模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

# 导入网络模块
from tcp_message_client import TCPMessageClient
from tcp_file_client import TCPFileClient
from udp_video_client import UDPVideoClient

class MeshNetworkClientGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("自组网终端传输系统 - 客户端")
        self.root.geometry("600x500")
        
        # 客户端实例
        self.message_client = None
        self.file_client = None
        self.video_client = None
        
        # 客户端连接状态
        self.message_connected = False
        self.file_connected = False
        self.video_running = False
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 主标题
        title_label = tk.Label(self.root, text="自组网终端传输系统", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # IP和端口设置框架
        settings_frame = ttk.LabelFrame(self.root, text="连接设置")
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        # IP地址输入
        ip_frame = ttk.Frame(settings_frame)
        ip_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(ip_frame, text="服务器IP:").pack(side="left", padx=5)
        self.ip_entry = ttk.Entry(ip_frame)
        self.ip_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.ip_entry.insert(0, "localhost")
        
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
        
        # 功能按钮框架
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        # TCP消息传输按钮
        self.message_connect_btn = ttk.Button(button_frame, text="连接消息服务器", command=self.connect_message_server)
        self.message_connect_btn.pack(side="left", padx=5)
        
        self.message_disconnect_btn = ttk.Button(button_frame, text="断开消息连接", command=self.disconnect_message_server, state="disabled")
        self.message_disconnect_btn.pack(side="left", padx=5)
        
        # TCP文件传输按钮
        self.file_connect_btn = ttk.Button(button_frame, text="连接文件服务器", command=self.connect_file_server)
        self.file_connect_btn.pack(side="left", padx=5)
        
        self.file_disconnect_btn = ttk.Button(button_frame, text="断开文件连接", command=self.disconnect_file_server, state="disabled")
        self.file_disconnect_btn.pack(side="left", padx=5)
        
        # UDP视频传输按钮
        self.video_start_btn = ttk.Button(button_frame, text="开始视频传输", command=self.start_video_transmission)
        self.video_start_btn.pack(side="left", padx=5)
        
        self.video_stop_btn = ttk.Button(button_frame, text="停止视频传输", command=self.stop_video_transmission, state="disabled")
        self.video_stop_btn.pack(side="left", padx=5)
        
        # 消息传输框架
        message_frame = ttk.LabelFrame(self.root, text="消息传输")
        message_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 消息显示区域
        self.message_text = tk.Text(message_frame, height=8)
        self.message_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 消息滚动条
        message_scrollbar = ttk.Scrollbar(self.message_text, command=self.message_text.yview)
        message_scrollbar.pack(side="right", fill="y")
        self.message_text.config(yscrollcommand=message_scrollbar.set)
        
        # 消息输入区域
        input_frame = ttk.Frame(message_frame)
        input_frame.pack(fill="x", padx=5, pady=5)
        
        self.message_entry = ttk.Entry(input_frame)
        self.message_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.message_entry.bind("<Return>", self.send_message)
        
        self.send_btn = ttk.Button(input_frame, text="发送", command=self.send_message, state="disabled")
        self.send_btn.pack(side="right", padx=5)
        
        # 文件传输框架
        file_frame = ttk.LabelFrame(self.root, text="文件传输")
        file_frame.pack(fill="x", padx=10, pady=5)
        
        self.file_path_entry = ttk.Entry(file_frame)
        self.file_path_entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        self.browse_btn = ttk.Button(file_frame, text="浏览", command=self.browse_file, state="disabled")
        self.browse_btn.pack(side="left", padx=5, pady=5)
        
        self.send_file_btn = ttk.Button(file_frame, text="发送文件", command=self.send_file, state="disabled")
        self.send_file_btn.pack(side="left", padx=5, pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
        
    def connect_message_server(self):
        try:
            host = self.ip_entry.get()
            port = int(self.message_port_entry.get())
            
            # 创建消息客户端
            self.message_client = TCPMessageClient(host, port)
            # 设置消息接收回调
            self.message_client.on_message_received = self.display_message
            
            # 连接服务器
            if self.message_client.connect_to_server():
                self.message_connected = True
                self.message_connect_btn.config(state="disabled")
                self.message_disconnect_btn.config(state="normal")
                self.send_btn.config(state="normal")
                self.status_var.set(f"消息服务器 {host}:{port} 已连接")
                self.message_text.insert(tk.END, f"已连接到消息服务器 {host}:{port}\n")
                self.message_text.see(tk.END)
            else:
                messagebox.showerror("连接失败", "无法连接到消息服务器")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的端口号")
        except Exception as e:
            messagebox.showerror("错误", f"连接失败: {str(e)}")
            
    def disconnect_message_server(self):
        self.message_connected = False
        self.message_connect_btn.config(state="normal")
        self.message_disconnect_btn.config(state="disabled")
        self.send_btn.config(state="disabled")
        self.status_var.set("消息服务器已断开")
        self.message_text.insert(tk.END, "已断开消息服务器连接\n")
        self.message_text.see(tk.END)
        
        if self.message_client:
            self.message_client.disconnect()
        
    def display_message(self, message):
        """在消息区域显示消息"""
        self.message_text.insert(tk.END, f"{message}\n")
        self.message_text.see(tk.END)
        
    def send_message(self, event=None):
        if not self.message_connected:
            messagebox.showwarning("警告", "请先连接到消息服务器")
            return
            
        message = self.message_entry.get()
        if message:
            if self.message_client and self.message_client.send_message(message):
                self.message_text.insert(tk.END, f"我: {message}\n")
                self.message_text.see(tk.END)
                self.message_entry.delete(0, tk.END)
            else:
                messagebox.showerror("错误", "发送消息失败")
            
    def connect_file_server(self):
        try:
            host = self.ip_entry.get()
            port = int(self.file_port_entry.get())
            
            # 创建文件客户端
            self.file_client = TCPFileClient(host, port)
            
            # 连接服务器
            if self.file_client.connect_to_server():
                self.file_connected = True
                self.file_connect_btn.config(state="disabled")
                self.file_disconnect_btn.config(state="normal")
                self.browse_btn.config(state="normal")
                self.send_file_btn.config(state="normal")
                self.status_var.set(f"文件服务器 {host}:{port} 已连接")
                self.message_text.insert(tk.END, f"已连接到文件服务器 {host}:{port}\n")
                self.message_text.see(tk.END)
            else:
                messagebox.showerror("连接失败", "无法连接到文件服务器")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的端口号")
        except Exception as e:
            messagebox.showerror("错误", f"连接失败: {str(e)}")
            
    def disconnect_file_server(self):
        self.file_connected = False
        self.file_connect_btn.config(state="normal")
        self.file_disconnect_btn.config(state="disabled")
        self.browse_btn.config(state="disabled")
        self.send_file_btn.config(state="disabled")
        self.status_var.set("文件服务器已断开")
        self.message_text.insert(tk.END, "已断开文件服务器连接\n")
        self.message_text.see(tk.END)
        
        if self.file_client:
            self.file_client.disconnect()
        
    def browse_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)
            
    def send_file(self):
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
        try:
            self.root.after(0, lambda: self.message_text.insert(tk.END, f"正在发送文件: {file_path}\n"))
            self.root.after(0, lambda: self.message_text.see(tk.END))
            
            # 发送文件
            if self.file_client and self.file_client.send_file(file_path):
                self.root.after(0, lambda: messagebox.showinfo("成功", "文件发送完成"))
                self.root.after(0, lambda: self.message_text.insert(tk.END, f"文件发送完成: {file_path}\n"))
            else:
                self.root.after(0, lambda: messagebox.showerror("错误", "文件发送失败"))
            self.root.after(0, lambda: self.message_text.see(tk.END))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"文件发送失败: {str(e)}"))
            
    def start_video_transmission(self):
        try:
            host = self.ip_entry.get()
            port = int(self.video_port_entry.get())
            
            # 创建视频客户端
            self.video_client = UDPVideoClient(host, port)
            # 设置状态更新回调
            self.video_client.on_status_update = self.display_message
            
            # 在新线程中启动视频客户端
            self.video_client_thread = threading.Thread(target=self.video_client.start_client)
            self.video_client_thread.daemon = True
            self.video_client_thread.start()
            
            self.video_running = True
            self.video_start_btn.config(state="disabled")
            self.video_stop_btn.config(state="normal")
            self.status_var.set("视频传输进行中...")
            self.message_text.insert(tk.END, "开始视频传输\n")
            self.message_text.see(tk.END)
        except ValueError:
            messagebox.showerror("错误", "请输入有效的端口号")
        except Exception as e:
            messagebox.showerror("错误", f"启动视频传输失败: {str(e)}")
            
    def stop_video_transmission(self):
        """停止视频传输"""
        self.video_running = False
        if self.video_client:
            self.video_client.stop_client()
            
        self.video_start_btn.config(state="normal")
        self.video_stop_btn.config(state="disabled")
        self.status_var.set("视频传输已停止")
        self.message_text.insert(tk.END, "视频传输已停止\n")
        self.message_text.see(tk.END)


def main():
    root = tk.Tk()
    app = MeshNetworkClientGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()