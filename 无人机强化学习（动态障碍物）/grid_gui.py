import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os
import random

# 导入环境类
from environment import GridEnvironment

class GridWorldGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("无人机路径规划环境设置")
        self.root.geometry("1000x700")
        
        # 默认网格大小
        self.grid_size = 20
        self.cell_size = 20  # 每个格子的像素大小
        
        # 环境对象
        self.env = GridEnvironment(self.grid_size)
        
        # 当前模式：0-设置起点, 1-设置终点, 2-设置静态障碍物, 3-设置动态障碍物
        self.mode = tk.StringVar(value="start")
        
        # 动态障碍物参数
        self.dynamic_obstacle_size = tk.IntVar(value=4)
        self.dynamic_obstacle_count = tk.IntVar(value=2)
        
        # 用于存储动态障碍物轨迹点（每个障碍物独立）
        self.dynamic_obstacle_trajectories = []  # 存储每个障碍物的轨迹点列表
        self.current_trajectory_points = []  # 当前正在设置的轨迹点
        self.selected_dynamic_obstacle = None  # 当前选中的动态障碍物索引
        
        # 创建界面
        self.create_widgets()
        
        # 初始化网格显示
        self.draw_grid()
        
    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 模式选择
        mode_frame = ttk.LabelFrame(control_frame, text="操作模式")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(mode_frame, text="设置起点", variable=self.mode, value="start").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="设置终点", variable=self.mode, value="goal").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="设置静态障碍物", variable=self.mode, value="static").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="设置动态障碍物轨迹点", variable=self.mode, value="dynamic").pack(anchor=tk.W)
        
        # 动态障碍物设置
        dynamic_frame = ttk.LabelFrame(control_frame, text="动态障碍物参数")
        dynamic_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(dynamic_frame, text="障碍物大小:").pack(anchor=tk.W)
        ttk.Spinbox(dynamic_frame, from_=3, to=6, textvariable=self.dynamic_obstacle_size, width=10).pack(anchor=tk.W)
        
        ttk.Label(dynamic_frame, text="障碍物数量:").pack(anchor=tk.W)
        ttk.Spinbox(dynamic_frame, from_=1, to=4, textvariable=self.dynamic_obstacle_count, width=10).pack(anchor=tk.W)
        
        ttk.Button(dynamic_frame, text="添加动态障碍物", command=self.add_dynamic_obstacle).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(dynamic_frame, text="清空当前轨迹点", command=self.clear_current_trajectory).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(dynamic_frame, text="完成当前轨迹", command=self.finish_current_trajectory).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(dynamic_frame, text="清除所有动态障碍物", command=self.clear_all_dynamic_obstacles).pack(fill=tk.X, pady=(5, 0))
        
        # 动态障碍物列表
        self.dynamic_listbox = tk.Listbox(dynamic_frame, height=4)
        self.dynamic_listbox.pack(fill=tk.X, pady=(5, 0))
        self.dynamic_listbox.bind('<<ListboxSelect>>', self.on_dynamic_select)
        
        # 网格大小设置
        size_frame = ttk.LabelFrame(control_frame, text="网格大小")
        size_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.size_var = tk.IntVar(value=self.grid_size)
        ttk.Spinbox(size_frame, from_=10, to=30, textvariable=self.size_var, width=10).pack(anchor=tk.W)
        ttk.Button(size_frame, text="应用网格大小", command=self.apply_grid_size).pack(anchor=tk.W, pady=(5, 0))
        
        # 控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="清空环境", command=self.clear_environment).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="保存环境配置", command=self.save_environment_config).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="加载环境配置", command=self.load_environment_config).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="运行强化学习", command=self.run_reinforcement_learning).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="可视化环境", command=self.visualize_environment).pack(fill=tk.X)
        
        # 状态显示
        status_frame = ttk.LabelFrame(control_frame, text="状态信息")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="就绪")
        self.status_label.pack(anchor=tk.W)
        
        # 右侧网格显示区域
        grid_frame = ttk.Frame(main_frame)
        grid_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, grid_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标点击事件
        self.canvas.mpl_connect('button_press_event', self.on_grid_click)
        
    def add_dynamic_obstacle(self):
        """添加一个新的动态障碍物"""
        # 创建一个新的动态障碍物条目
        index = len(self.dynamic_obstacle_trajectories)
        self.dynamic_obstacle_trajectories.append([])
        self.dynamic_listbox.insert(tk.END, f"动态障碍物 {index+1}")
        self.status_label.config(text=f"已添加动态障碍物 {index+1}，请设置其轨迹点")
        
    def clear_current_trajectory(self):
        """清空当前轨迹点"""
        self.current_trajectory_points = []
        self.draw_grid()
        self.status_label.config(text="已清空当前轨迹点")
        
    def finish_current_trajectory(self):
        """完成当前轨迹设置"""
        if not self.dynamic_obstacle_trajectories:
            messagebox.showwarning("警告", "请先添加动态障碍物")
            return
            
        if len(self.current_trajectory_points) < 2:
            messagebox.showwarning("警告", "请至少设置2个轨迹点")
            return
            
        # 将当前轨迹点分配给选中的动态障碍物，如果没有选中则分配给最后一个
        if self.selected_dynamic_obstacle is not None:
            index = self.selected_dynamic_obstacle
        else:
            index = len(self.dynamic_obstacle_trajectories) - 1
            
        self.dynamic_obstacle_trajectories[index] = self.current_trajectory_points.copy()
        self.current_trajectory_points = []
        
        # 更新环境中的动态障碍物
        self.update_environment_dynamic_obstacles()
        
        self.draw_grid()
        self.status_label.config(text=f"已完成动态障碍物 {index+1} 的轨迹设置")
        
    def clear_all_dynamic_obstacles(self):
        """清除所有动态障碍物"""
        self.dynamic_obstacle_trajectories = []
        self.current_trajectory_points = []
        self.selected_dynamic_obstacle = None
        self.dynamic_listbox.delete(0, tk.END)
        self.env.dynamic_obstacles = []
        self.draw_grid()
        self.status_label.config(text="已清除所有动态障碍物")
        
    def update_environment_dynamic_obstacles(self):
        """更新环境中的动态障碍物"""
        # 清除现有的动态障碍物
        self.env.dynamic_obstacles = []
        
        # 为每个有轨迹的动态障碍物创建对象
        for i, trajectory in enumerate(self.dynamic_obstacle_trajectories):
            if len(trajectory) >= 2:  # 至少需要2个点
                size = self.dynamic_obstacle_size.get()
                
                dynamic_obstacle = {
                    'trajectory': trajectory.copy(),
                    'current_index': 0,  # 初始位置为轨迹的第一个点
                    'size': size,
                    'direction': 1  # 初始方向为正向
                }
                self.env.dynamic_obstacles.append(dynamic_obstacle)
        
    def on_dynamic_select(self, event):
        """当选择动态障碍物时"""
        selection = self.dynamic_listbox.curselection()
        if selection:
            self.selected_dynamic_obstacle = selection[0]
            # 显示选中的动态障碍物的轨迹点
            if self.selected_dynamic_obstacle < len(self.dynamic_obstacle_trajectories):
                self.current_trajectory_points = self.dynamic_obstacle_trajectories[self.selected_dynamic_obstacle].copy()
                self.draw_grid()
                self.status_label.config(text=f"正在编辑动态障碍物 {self.selected_dynamic_obstacle+1} 的轨迹")
        
    def apply_grid_size(self):
        """应用新的网格大小"""
        new_size = self.size_var.get()
        if new_size != self.grid_size:
            self.grid_size = new_size
            self.env = GridEnvironment(self.grid_size)
            self.dynamic_obstacle_trajectories = []
            self.current_trajectory_points = []
            self.selected_dynamic_obstacle = None
            self.dynamic_listbox.delete(0, tk.END)
            self.draw_grid()
            self.status_label.config(text=f"网格大小已更新为 {self.grid_size}x{self.grid_size}")
        
    def clear_environment(self):
        """清空环境"""
        self.env = GridEnvironment(self.grid_size)
        self.dynamic_obstacle_trajectories = []
        self.current_trajectory_points = []
        self.selected_dynamic_obstacle = None
        self.dynamic_listbox.delete(0, tk.END)
        self.draw_grid()
        self.status_label.config(text="环境已清空")
        
    def save_environment_config(self):
        """保存环境配置到文件"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            config = {
                "grid_size": self.env.size,
                "start": self.env.start,
                "goal": self.env.goal,
                "static_obstacles": self.env.obstacles,
                "grid_data": self.env.grid.tolist(),
                "dynamic_obstacles": self.env.dynamic_obstacles,
                "dynamic_obstacle_trajectories": self.dynamic_obstacle_trajectories
            }
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.status_label.config(text=f"环境配置已保存到 {file_path}")
            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存配置文件: {str(e)}")
        
    def load_environment_config(self):
        """从文件加载环境配置"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                # 创建新环境
                self.grid_size = config["grid_size"]
                self.env = GridEnvironment(self.grid_size)
                
                # 设置起点和终点
                self.env.start = tuple(config["start"])
                self.env.goal = tuple(config["goal"])
                
                # 设置静态障碍物
                self.env.obstacles = [tuple(obs) for obs in config["static_obstacles"]]
                self.env.grid = np.array(config["grid_data"])
                
                # 设置动态障碍物
                self.env.dynamic_obstacles = config["dynamic_obstacles"]
                
                # 设置动态障碍物轨迹
                self.dynamic_obstacle_trajectories = []
                trajectories_data = config.get("dynamic_obstacle_trajectories", [])
                for trajectory in trajectories_data:
                    self.dynamic_obstacle_trajectories.append([tuple(point) for point in trajectory])
                
                # 更新列表框
                self.dynamic_listbox.delete(0, tk.END)
                for i in range(len(self.dynamic_obstacle_trajectories)):
                    self.dynamic_listbox.insert(tk.END, f"动态障碍物 {i+1}")
                
                # 重置其他状态
                self.current_trajectory_points = []
                self.selected_dynamic_obstacle = None
                
                # 更新界面
                self.size_var.set(self.grid_size)
                self.draw_grid()
                self.status_label.config(text=f"环境配置已从 {file_path} 加载")
                
            except Exception as e:
                messagebox.showerror("加载失败", f"无法加载配置文件: {str(e)}")
        
    def run_reinforcement_learning(self):
        """运行强化学习算法"""
        try:
            # 导入强化学习算法
            from rl_runner import run_reinforcement_learning, visualize_results
            
            # 运行强化学习
            self.status_label.config(text="正在运行强化学习...")
            self.root.update()
            
            results = run_reinforcement_learning(self.env)
            
            if results:
                self.status_label.config(text="强化学习完成，正在显示结果...")
                visualize_results(self.env, results)
                self.show_results(results)
            else:
                messagebox.showerror("运行失败", "强化学习运行失败")
                
        except Exception as e:
            messagebox.showerror("运行失败", f"无法运行强化学习算法: {str(e)}")
        
    def show_results(self, results):
        """显示强化学习结果"""
        # 这里可以显示结果图表或更新GUI
        q_reward = results['q_learning']['rewards'][-1] if results['q_learning']['rewards'] else 'N/A'
        sarsa_reward = results['sarsa']['rewards'][-1] if results['sarsa']['rewards'] else 'N/A'
        pg_reward = results['policy_gradient']['rewards'][-1] if results['policy_gradient']['rewards'] else 'N/A'
        dqn_reward = results['dqn']['rewards'][-1] if results['dqn']['rewards'] else 'N/A'
        
        messagebox.showinfo("训练完成", 
                           f"强化学习训练已完成!\n"
                           f"Q-Learning最终奖励: {q_reward}\n"
                           f"SARSA最终奖励: {sarsa_reward}\n"
                           f"Policy Gradient最终奖励: {pg_reward}\n"
                           f"DQN最终奖励: {dqn_reward}")
        self.status_label.config(text="强化学习训练完成")
        
    def visualize_environment(self):
        """可视化当前环境"""
        self.env.visualize(title="当前环境配置")
        
    def draw_grid(self):
        """绘制网格"""
        self.ax.clear()
        
        # 创建显示网格
        display_grid = np.zeros((self.grid_size, self.grid_size))
        
        # 标记静态障碍物
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.env.grid[i, j] == 1:
                    display_grid[i, j] = 1  # 静态障碍物用1表示
                    
        # 标记动态障碍物
        dynamic_positions = self.env.get_dynamic_obstacle_positions()
        for x, y in dynamic_positions:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                display_grid[x, y] = 0.5  # 动态障碍物用0.5表示
                
        # 显示网格
        self.ax.imshow(display_grid, cmap='binary', origin='lower')
        
        # 标记起点和终点
        self.ax.plot(self.env.start[1], self.env.start[0], 'go', markersize=10, label='起点')
        self.ax.plot(self.env.goal[1], self.env.goal[0], 'r*', markersize=15, label='终点')
        
        # 绘制已设置的动态障碍物轨迹
        for i, trajectory in enumerate(self.dynamic_obstacle_trajectories):
            if trajectory:
                traj_x = [p[1] for p in trajectory]
                traj_y = [p[0] for p in trajectory]
                line_style = '--' if self.selected_dynamic_obstacle == i else '--'
                alpha = 1.0 if self.selected_dynamic_obstacle == i else 0.5
                self.ax.plot(traj_x, traj_y, 'b' + line_style, linewidth=2, alpha=alpha)
                for j, (x, y) in enumerate(trajectory):
                    marker_color = 'ro' if self.selected_dynamic_obstacle == i else 'bo'
                    self.ax.plot(y, x, marker_color, markersize=6)
                    if self.selected_dynamic_obstacle == i:
                        self.ax.text(y, x, str(j+1), color='white', fontsize=8, ha='center', va='center')
        
        # 绘制当前正在设置的轨迹点
        if self.current_trajectory_points:
            traj_x = [p[1] for p in self.current_trajectory_points]
            traj_y = [p[0] for p in self.current_trajectory_points]
            self.ax.plot(traj_x, traj_y, 'g--', linewidth=2, alpha=0.7)
            for i, (x, y) in enumerate(self.current_trajectory_points):
                self.ax.plot(y, x, 'go', markersize=6)
                self.ax.text(y, x, str(i+1), color='white', fontsize=8, ha='center', va='center')
        
        # 绘制动态障碍物轨迹
        for obstacle in self.env.dynamic_obstacles:
            if 'trajectory' in obstacle:
                trajectory_x = [pos[1] for pos in obstacle['trajectory']]
                trajectory_y = [pos[0] for pos in obstacle['trajectory']]
                self.ax.plot(trajectory_x, trajectory_y, 'c--', alpha=0.5, linewidth=1)
                
                # 标记动态障碍物当前位置
                if 'current_index' in obstacle and obstacle['current_index'] < len(obstacle['trajectory']):
                    center_x, center_y = obstacle['trajectory'][obstacle['current_index']]
                    self.ax.plot(center_y, center_x, 'rx', markersize=8)
        
        self.ax.set_xlim(-0.5, self.grid_size-0.5)
        self.ax.set_ylim(-0.5, self.grid_size-0.5)
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.grid(True)
        self.ax.legend()
        self.ax.set_title('无人机路径规划环境')
        
        self.canvas.draw()
        
    def on_grid_click(self, event):
        """处理网格点击事件"""
        if event.inaxes != self.ax:
            return
            
        # 获取点击的网格坐标
        x = int(round(event.ydata))
        y = int(round(event.xdata))
        
        # 检查坐标是否有效
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return
            
        mode = self.mode.get()
        
        if mode == "start":
            # 设置起点
            self.env.start = (x, y)
            self.status_label.config(text=f"起点设置为 ({x}, {y})")
        elif mode == "goal":
            # 设置终点
            self.env.goal = (x, y)
            self.status_label.config(text=f"终点设置为 ({x}, {y})")
        elif mode == "static":
            # 设置静态障碍物（切换状态）
            if self.env.grid[x, y] == 1:
                self.env.grid[x, y] = 0  # 移除障碍物
                # 从障碍物列表中移除
                self.env.obstacles = [obs for obs in self.env.obstacles if not (obs[0] <= x < obs[0] + obs[2] and obs[1] <= y < obs[1] + obs[3])]
                self.status_label.config(text=f"移除静态障碍物 ({x}, {y})")
            else:
                self.env.grid[x, y] = 1  # 添加障碍物
                # 添加到障碍物列表
                self.env.obstacles.append((x, y, 1, 1))
                self.status_label.config(text=f"添加静态障碍物 ({x}, {y})")
        elif mode == "dynamic":
            # 添加动态障碍物轨迹点
            if (x, y) not in self.current_trajectory_points:
                self.current_trajectory_points.append((x, y))
                self.status_label.config(text=f"添加轨迹点 ({x}, {y})")
            else:
                self.current_trajectory_points.remove((x, y))
                self.status_label.config(text=f"移除轨迹点 ({x}, {y})")
            
        self.draw_grid()

def main():
    root = tk.Tk()
    app = GridWorldGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()