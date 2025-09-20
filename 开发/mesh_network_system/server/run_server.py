import sys
import os
import tkinter as tk

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server_gui import MeshNetworkServerGUI

def main():
    root = tk.Tk()
    app = MeshNetworkServerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()