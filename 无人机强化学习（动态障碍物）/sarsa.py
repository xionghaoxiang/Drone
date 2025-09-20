import numpy as np
import random

class SarsaAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((env.size, env.size, 4))  # 4个动作：上、右、下、左
        
    def get_action(self, state):
        """选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # 随机动作
        else:
            return np.argmax(self.q_table[state[0], state[1]])
    
    def update_q_value(self, state, action, reward, next_state, next_action):
        """更新Q值 (SARSA更新规则)"""
        current_q = self.q_table[state[0], state[1], action]
        next_q = self.q_table[next_state[0], next_state[1], next_action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.q_table[state[0], state[1], action] = new_q
    
    def get_next_state(self, state, action):
        """获取下一个状态"""
        x, y = state
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
            return state  # 保持原位
    
    def train(self, episodes=1000):
        """训练智能体"""
        rewards_history = []
        
        for episode in range(episodes):
            # 每个episode开始时更新动态障碍物
            self.env.update_dynamic_obstacles()
            
            state = self.env.start
            action = self.get_action(state)  # 初始动作
            total_reward = 0
            steps = 0
            
            while state != self.env.goal:
                next_state = self.get_next_state(state, action)
                
                # 计算奖励
                if next_state == self.env.goal:
                    reward = 100
                elif state == next_state:  # 碰撞或边界
                    reward = -10
                else:
                    reward = -1
                
                # 选择下一个动作 (SARSA的关键区别)
                next_action = self.get_action(next_state)
                
                # 更新Q值
                self.update_q_value(state, action, reward, next_state, next_action)
                
                # 转移到下一个状态和动作
                state = next_state
                action = next_action
                
                total_reward += reward
                steps += 1
                
                if steps > 1000:  # 防止无限循环
                    break
            
            rewards_history.append(total_reward)
            
            # 每200个episode输出一次当前最优路径（使用贪婪策略找到的路径）
            # 修改为从第200个episode开始显示图像
            if episode % 200 == 0 and episode >= 200:
                print(f"Episode {episode}, Reward: {total_reward}")
                # 可视化当前最优路径（使用贪婪策略找到的路径）
                path = self.find_path()
                # 验证路径的合法性
                valid_path = self.validate_path(path)
                if valid_path:
                    self.env.visualize(path, f'SARSA Path - Episode {episode}')
        
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
        
        # 添加路径合法性检查，考虑当前动态障碍物的位置
        while state != self.env.goal:
            # 获取最优动作
            action = np.argmax(self.q_table[state[0], state[1]])
            
            # 获取下一个状态
            next_state = self.get_next_state(state, action)
            
            # 检查是否形成循环
            if next_state in visited_states:
                # 尝试其他动作
                best_action = action
                for a in range(4):
                    test_next = self.get_next_state(state, a)
                    if (self.q_table[state[0], state[1], a] > self.q_table[state[0], state[1], best_action] and 
                        test_next not in visited_states):
                        best_action = a
                        next_state = test_next
                
                # 如果所有动作都会导致循环，就跳出
                if best_action == action and next_state in visited_states:
                    break
            
            # 检查下一步是否合法（不在障碍物中且在边界内）
            if not self.env.is_valid_position(next_state[0], next_state[1]):
                # 如果下一步不合法，尝试其他动作
                valid_actions = []
                for a in range(4):
                    test_next = self.get_next_state(state, a)
                    if self.env.is_valid_position(test_next[0], test_next[1]) and test_next not in visited_states:
                        valid_actions.append(a)
                
                # 如果有合法动作，选择Q值最大的
                if valid_actions:
                    best_action = max(valid_actions, key=lambda a: self.q_table[state[0], state[1], a])
                    next_state = self.get_next_state(state, best_action)
                else:
                    # 没有合法动作，终止搜索
                    break
            
            # 添加当前状态到路径
            path.append(state)
            visited_states.add(state)
            state = next_state
            
            # 防止无限循环
            if len(path) > 1000:
                break
        
        # 确保终点被包含
        if state == self.env.goal and self.env.is_valid_position(state[0], state[1]):
            path.append(state)
        
        return path