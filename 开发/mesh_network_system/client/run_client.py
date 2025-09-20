import sys
import os
import tkinter as tk

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client_gui import MeshNetworkClientGUI

def main():
    root = tk.Tk()
    app = MeshNetworkClientGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()