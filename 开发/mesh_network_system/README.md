# 自组网终端传输系统

一个基于Python的网络通信系统，支持TCP消息传输、TCP文件传输和UDP视频传输。

## 功能特性

1. **TCP消息传输** - 可靠的消息通信
2. **TCP文件传输** - 稳定的文件传输
3. **UDP视频传输** - 实时视频通信
4. **图形用户界面** - 直观易用的操作界面

## 系统要求

- Python 3.6 或更高版本
- Windows/Linux/macOS 操作系统
- 摄像头设备（用于视频传输功能）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 目录结构

```
mesh_network_system/
├── client/              # 客户端代码
│   ├── client_gui.py    # 客户端图形界面
│   └── run_client.py    # 客户端启动脚本
├── server/              # 服务器端代码
│   ├── server_gui.py    # 服务器端图形界面
│   └── run_server.py    # 服务器端启动脚本
├── shared/              # 共享网络模块
│   ├── tcp_message_client.py  # TCP消息客户端
│   ├── tcp_message_server.py  # TCP消息服务器
│   ├── tcp_file_client.py     # TCP文件客户端
│   ├── tcp_file_server.py     # TCP文件服务器
│   ├── udp_video_client.py    # UDP视频客户端
│   └── udp_video_server.py    # UDP视频服务器
├── requirements.txt     # 依赖包列表
└── README.md            # 项目说明文档
```

## 使用方法

### 启动服务器

```bash
cd server
python run_server.py
```

在服务器图形界面中：

1. 设置监听IP地址和端口号（默认为0.0.0.0和对应端口）
2. 分别启动消息、文件、视频服务器或一键启动所有功能
3. 查看服务器日志了解运行状态

### 启动客户端

```bash
cd client
python run_client.py
```

在客户端图形界面中：

1. 设置服务器IP地址和端口号
2. 分别连接消息、文件、视频服务
3. 使用对应功能进行通信

### 图形界面操作

#### 服务器端GUI：

- 设置监听IP地址和端口号
- 独立控制消息、文件、视频服务的启停
- 实时日志显示与运行状态反馈

#### 客户端GUI：

- 设置服务器IP地址和端口号
- 连接和断开各种服务
- 实时消息发送和接收
- 文件选择和传输
- 视频传输控制

## 端口说明

- TCP消息传输默认端口：12345
- TCP文件传输默认端口：12346
- UDP视频传输默认端口：12347

## 注意事项

1. 视频功能需要摄像头设备和相关驱动支持
2. 防火墙可能会阻止网络通信，请确保相应端口已开放
3. 如果使用默认端口，请确保它们未被其他程序占用
