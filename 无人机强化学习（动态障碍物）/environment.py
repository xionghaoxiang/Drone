import numpy as np
import matplotlib.pyplot as plt
import random

class GridEnvironment:
    def __init__(self, size=20):
        self.size = size
        self.grid = np.zeros((size, size))
        self.start = (0, 0)  # 左下角
        self.goal = (size-1, size-1)  # 右上角
        self.obstacles = []
        self.dynamic_obstacles = []  # 动态障碍物列表
        self.timestep = 0  # 时间步
        
    def add_obstacle(self, x, y, width, height):
        """添加静态障碍物"""
        for i in range(x, x + width):
            for j in range(y, y + height):
                if 0 <= i < self.size and 0 <= j < self.size:
                    self.grid[i, j] = 1
        self.obstacles.append((x, y, width, height))
    
    def add_dynamic_obstacles(self, num_obstacles=2):
        """添加固定轨迹往复运动的动态障碍物，初始位置固定"""
        for i in range(num_obstacles):
            # 固定大小（4-6个网格）
            size = random.randint(2, 2)
            
            # 固定移动方向和轨迹
            if i == 0:
                # 第一个障碍物水平移动
                row = 4  # 固定在第4行
                # 限制移动范围，确保不会阻挡起点到终点的主要路径
                trajectory = [(row, col) for col in range(4, self.size - 4)]
                start_index = 0  # 固定初始位置在最左端
                direction = 1    # 初始移动方向向右
            else:
                # 第二个障碍物垂直移动
                col = 16  # 固定在第16列
                # 限制移动范围，确保不会阻挡起点到终点的主要路径
                trajectory = [(row, col) for row in range(4, self.size - 4)]
                start_index = 0  # 固定初始位置在最上端
                direction = 1    # 初始移动方向向下
            
            dynamic_obstacle = {
                'trajectory': trajectory,
                'current_index': start_index,
                'size': size,
                'direction': direction  # 1表示正向移动，-1表示反向移动
            }
            self.dynamic_obstacles.append(dynamic_obstacle)
    
    def update_dynamic_obstacles(self):
        """更新动态障碍物位置"""
        self.timestep += 1
        for obstacle in self.dynamic_obstacles:
            # 移动到下一个位置
            obstacle['current_index'] += obstacle['direction']
            
            # 如果到达轨迹的末端，改变方向
            if obstacle['current_index'] >= len(obstacle['trajectory']) - 1:
                obstacle['current_index'] = len(obstacle['trajectory']) - 1
                obstacle['direction'] = -1
            elif obstacle['current_index'] <= 0:
                obstacle['current_index'] = 0
                obstacle['direction'] = 1
    
    def get_dynamic_obstacle_positions(self):
        """获取当前动态障碍物占据的位置"""
        positions = []
        for obstacle in self.dynamic_obstacles:
            # 获取当前轨迹点
            center_x, center_y = obstacle['trajectory'][obstacle['current_index']]
            size = obstacle['size']
            
            # 计算障碍物占据的所有网格位置（以中心点为中心）
            half_size = size // 2
            for i in range(center_x - half_size, center_x + half_size + 1):
                for j in range(center_y - half_size, center_y + half_size + 1):
                    if 0 <= i < self.size and 0 <= j < self.size:
                        positions.append((i, j))
        return positions
    
    def is_valid_position(self, x, y):
        """检查位置是否有效（不在障碍物中且在边界内）"""
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        # 检查是否在静态障碍物中
        if self.grid[x, y] == 1:
            return False
        # 检查是否在动态障碍物中
        dynamic_positions = self.get_dynamic_obstacle_positions()
        if (x, y) in dynamic_positions:
            return False
        return True
    
    def visualize(self, path=None, title='Grid Environment'):
        """可视化环境"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 创建一个临时网格用于显示，包含动态障碍物
        display_grid = self.grid.copy()
        dynamic_positions = self.get_dynamic_obstacle_positions()
        for x, y in dynamic_positions:
            if 0 <= x < self.size and 0 <= y < self.size:
                display_grid[x, y] = 0.5  # 动态障碍物用不同的值表示
        
        ax.imshow(display_grid, cmap='binary', origin='lower')
        
        # 标记起点和终点
        ax.plot(self.start[1], self.start[0], 'go', markersize=10, label='Start')
        ax.plot(self.goal[1], self.goal[0], 'ro', markersize=10, label='Goal')
        
        # 绘制动态障碍物轨迹
        for obstacle in self.dynamic_obstacles:
            trajectory_x = [pos[1] for pos in obstacle['trajectory']]
            trajectory_y = [pos[0] for pos in obstacle['trajectory']]
            ax.plot(trajectory_x, trajectory_y, 'c--', alpha=0.5, linewidth=1)
        
        # 标记路径
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
        
        # 标记动态障碍物中心
        for obstacle in self.dynamic_obstacles:
            center_x, center_y = obstacle['trajectory'][obstacle['current_index']]
            ax.plot(center_y, center_x, 'rx', markersize=8)
        
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.grid(True)
        ax.legend()
        plt.title(title)
        plt.show()