import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

class DQN(nn.Module):
    """深度Q网络模型"""
    def __init__(self, input_size=4, hidden_size=128, output_size=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.9, epsilon=0.1, 
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 神经网络
        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # 更新目标网络的频率
        self.target_update_freq = 100
        self.step_count = 0
        
    def get_state(self, position):
        """将位置转换为状态向量"""
        # 状态包括当前位置和目标位置的归一化坐标
        if isinstance(position, tuple):
            x, y = position
        else:
            x, y = position[0], position[1]
            
        state = torch.FloatTensor([
            x / self.env.size,
            y / self.env.size,
            self.env.goal[0] / self.env.size,
            self.env.goal[1] / self.env.size
        ]).to(self.device)
        return state
    
    def get_action(self, state):
        """选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # 随机动作
        else:
            with torch.no_grad():
                # 确保state是torch张量
                if not isinstance(state, torch.Tensor):
                    state = self.get_state(state)
                # 确保是二维张量(batch_size, feature_dim)
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                q_values = self.q_network(state)
                return q_values.argmax(dim=1).item()
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.stack([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * (~dones))
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 降低探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_next_state(self, state, action):
        """获取下一个状态"""
        # 将状态转换为位置
        x, y = None, None
        
        if isinstance(state, tuple) and len(state) >= 2:
            x, y = state[0], state[1]
        elif isinstance(state, torch.Tensor):
            if state.dim() == 0:
                # 标量张量
                x, y = int(state.item()), int(state.item())
            elif state.dim() == 1 and len(state) >= 2:
                x = int(state[0].item() * self.env.size)
                y = int(state[1].item() * self.env.size)
            else:
                # 其他情况默认为0
                x, y = 0, 0
        elif isinstance(state, (list, np.ndarray)) and len(state) >= 2:
            x = int(state[0] * self.env.size)
            y = int(state[1] * self.env.size)
        elif hasattr(state, '__getitem__') and len(state) >= 2:
            x = int(state[0] * self.env.size) if isinstance(state[0], (int, float)) else state[0]
            y = int(state[1] * self.env.size) if isinstance(state[1], (int, float)) else state[1]
        else:
            # 如果以上都不匹配，默认使用起始位置
            x, y = self.env.start[0], self.env.start[1]
            
        # 确保x和y是整数
        if not isinstance(x, int):
            x = int(x) if isinstance(x, (float, np.floating)) else 0
        if not isinstance(y, int):
            y = int(y) if isinstance(y, (float, np.floating)) else 0
        
        if action == 0:  # 上
            next_x, next_y = x - 1, y
        elif action == 1:  # 右
            next_x, next_y = x, y + 1
        elif action == 2:  # 下
            next_x, next_y = x + 1, y
        else:  # 左
            next_x, next_y = x, y - 1
            
        # 检查边界和障碍物
        if self.env.is_valid_position(next_x, next_y):
            return (next_x, next_y)
        else:
            return (x, y)  # 保持原位
    
    def train(self, episodes=1000):
        """训练智能体"""
        rewards_history = []
        
        # 初始化目标网络
        self.update_target_network()
        
        for episode in range(episodes):
            # 每个episode开始时更新动态障碍物
            self.env.update_dynamic_obstacles()
            
            # 初始化状态
            state = self.env.start
            state_tensor = self.get_state(state)
            total_reward = 0
            steps = 0
            
            while state != self.env.goal:
                action = self.get_action(state_tensor)
                next_state = self.get_next_state(state, action)
                
                # 计算奖励
                if next_state == self.env.goal:
                    reward = 100
                    done = True
                elif state == next_state:  # 碰撞或边界
                    reward = -10
                    done = False
                else:
                    reward = -1
                    done = False
                
                next_state_tensor = self.get_state(next_state)
                
                # 存储经验
                self.remember(state_tensor, action, reward, next_state_tensor, done)
                
                # 经验回放
                self.replay()
                
                # 更新状态
                state = next_state
                state_tensor = next_state_tensor
                total_reward += reward
                steps += 1
                
                # 更新计数器
                self.step_count += 1
                
                # 更新目标网络
                if self.step_count % self.target_update_freq == 0:
                    self.update_target_network()
                
                if steps > 1000:  # 防止无限循环
                    break
            
            rewards_history.append(total_reward)
            
            # 每200个episode输出一次当前最优路径（使用贪婪策略找到的路径）
            # 修改为从第200个episode开始显示图像
            if episode % 200 == 0 and episode >= 200:
                print(f"DQN Episode {episode}, Reward: {total_reward}")
                # 可视化当前最优路径（使用贪婪策略找到的路径）
                path = self.find_path()
                # 验证路径的合法性
                valid_path = self.validate_path(path)
                if valid_path:
                    self.env.visualize(path, f'DQN Path - Episode {episode}')
        
        return rewards_history
    
    def validate_path(self, path):
        """验证路径是否合法（不穿越障碍物）"""
        if not path:
            return False
        
        # 检查起点和终点
        if not self.env.is_valid_position(path[0][0], path[0][1]) and path[0] != self.env.start:
            return False
        if not self.env.is_valid_position(path[-1][0], path[-1][1]) and path[-1] != self.env.goal:
            return False
        
        # 使用visited_states集合记录已访问状态，防止智能体陷入A→B→C→A类复杂循环
        visited_states = set()
        
        # 检查路径中相邻状态之间的移动是否合法
        for i in range(len(path) - 1):
            state = path[i]
            next_state = path[i + 1]
            
            # 检查当前位置是否是障碍物
            if not self.env.is_valid_position(state[0], state[1]) and state != self.env.start:
                return False
                
            # 检查下一个位置是否是障碍物
            if not self.env.is_valid_position(next_state[0], next_state[1]) and next_state != self.env.goal:
                return False
                
            # 检查从当前状态到下一个状态的移动是否合法
            action = None
            x, y = state
            nx, ny = next_state
            
            if nx == x - 1 and ny == y:  # 上
                action = 0
            elif nx == x and ny == y + 1:  # 右
                action = 1
            elif nx == x + 1 and ny == y:  # 下
                action = 2
            elif nx == x and ny == y - 1:  # 左
                action = 3
            else:
                # 如果移动不是四个基本方向之一，则非法
                return False
            
            # 检查该动作是否会导致碰撞或边界
            if not self.env.is_valid_position(nx, ny):
                return False
            
            # 检查是否形成循环
            if state in visited_states:
                return False
            visited_states.add(state)
        
        # 最后检查终点是否已访问
        if path[-1] in visited_states:
            return False
        
        return True
    
    def find_path(self):
        """找到从起点到终点的路径"""
        path = []
        state = self.env.start
        visited_states = set()
        
        # 在查找路径时需要考虑当前动态障碍物的位置
        max_steps = 1000
        steps = 0
        
        while state != self.env.goal and steps < max_steps:
            state_tensor = self.get_state(state)
            with torch.no_grad():
                # 确保state_tensor是正确的维度
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
            
            next_state = self.get_next_state(state, action)
            
            # 检查是否形成循环
            if next_state in visited_states:
                # 尝试其他动作
                best_action = action
                best_q_value = q_values[0, best_action].item() if q_values.dim() > 1 else q_values[best_action].item()
                found_better = False
                for a in range(4):
                    test_next = self.get_next_state(state, a)
                    if test_next not in visited_states:
                        test_q_value = q_values[0, a].item() if q_values.dim() > 1 else q_values[a].item()
                        if test_q_value > best_q_value:
                            best_action = a
                            best_q_value = test_q_value
                            next_state = test_next
                            found_better = True
                
                # 如果所有动作都会导致循环，就跳出
                if not found_better:
                    break
            
            path.append(state)
            visited_states.add(state)
            state = next_state
            steps += 1
            
        # 只有当最后一步是终点时才添加终点
        if state == self.env.goal:
            path.append(self.env.goal)
        return path

# 测试DQN算法
if __name__ == "__main__":
    from environment import GridEnvironment
    import matplotlib.pyplot as plt
    
    # 创建环境
    env = GridEnvironment(size=20)

    # 添加静态障碍物（至少10个，分散布置，减小部分障碍物面积）
    obstacles = [
        (3, 3, 3, 3),  # 左上区域 (减小面积 4x4 -> 3x3)
        (15, 3, 3, 3),  # 右上区域 (减小面积 4x4 -> 3x3)
        (3, 15, 3, 3),  # 左下区域 (减小面积 4x4 -> 3x3)
        (15, 15, 3, 3),  # 右下区域 (减小面积 4x4 -> 3x3)
        (8, 8, 2, 2),  # 中心偏左 (减小面积 3x3 -> 2x2)
        (12, 8, 2, 2),  # 中心偏右 (减小面积 3x3 -> 2x2)
        (8, 12, 2, 2),  # 中心偏下 (减小面积 3x3 -> 2x2)
        (12, 12, 2, 2),  # 中心偏上 (减小面积 3x3 -> 2x2)
        (5, 10, 2, 1),  # 左侧中部 (减小面积 2x2 -> 2x1)
        (15, 10, 2, 1)  # 右侧中部 (减小面积 2x2 -> 2x1)
    ]

    # 在边界添加额外的静态障碍物，但避开起点(0,0)和终点(19,19)附近区域
    # 上边界障碍物 (避开起点附近区域)
    #obstacles.append((0, 4, 1, 2))  # 上边界左侧
    # obstacles.append((0, 14, 1, 2))  # 上边界右侧，避开终点正上方

    # 下边界障碍物 (避开终点附近区域)
    #obstacles.append((19, 2, 1, 2))  # 下边界左侧，避开终点正下方
    # obstacles.append((19, 14, 1, 2)) # 下边界右侧

    # 左边界障碍物 (避开起点附近区域)
    #obstacles.append((2, 0, 2, 1))  # 左边界上侧，避开起点正左方
    # obstacles.append((14, 0, 2, 1))  # 左边界下侧

    # 右边界障碍物 (避开终点附近区域)
    #obstacles.append((2, 19, 2, 1))  # 右边界上侧，避开终点正右方
    # obstacles.append((14, 19, 2, 1)) # 右边界下侧

    # 添加障碍物到环境中
    for obs in obstacles:
        env.add_obstacle(*obs)
    # 添加固定轨迹往复运动的动态障碍物
    env.add_dynamic_obstacles(2)
    
    # 创建DQN智能体
    dqn_agent = DQNAgent(env)
    
    # 训练智能体
    print("Training DQN agent...")
    dqn_rewards = dqn_agent.train(episodes=1000)
    
    # 找到路径（在查找路径前更新动态障碍物位置）
    env.update_dynamic_obstacles()  # 确保我们看到的是当前的动态障碍物位置
    dqn_path = dqn_agent.find_path()
    
    # 验证路径合法性
    dqn_valid = dqn_agent.validate_path(dqn_path)
    
    print(f"DQN path valid: {dqn_valid}")
    
    # 可视化结果
    if dqn_path and dqn_valid:
        env.visualize(dqn_path, 'DQN Path Planning')
    elif dqn_path:
        print("Warning: DQN generated an invalid path")
        env.visualize(dqn_path, 'DQN Path Planning (Invalid)')
    else:
        env.visualize([], 'DQN Path Planning (No Path Found)')
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 5))
    
    # 计算滑动平均以更好地显示趋势
    window_size = 50
    dqn_rewards_smooth = np.convolve(dqn_rewards, np.ones(window_size)/window_size, mode='valid')
    
    episodes_range = np.arange(len(dqn_rewards_smooth))
    plt.plot(episodes_range, dqn_rewards_smooth, label='DQN', color='green', linewidth=2)
    plt.title('DQN Cumulative Reward over Episodes', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward (Smoothed)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # 输出统计信息
    print("\n=== DQN Performance ===")
    print(f"DQN - Path length: {len(dqn_path)}")
    print(f"DQN - Final reward: {dqn_rewards[-1]}")
    print(f"DQN - Average reward (last 100 episodes): {np.mean(dqn_rewards[-100:]):.2f}")