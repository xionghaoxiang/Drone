import numpy as np
import random

class PolicyGradientAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.8, epsilon_decay=0.995, epsilon_min=0.5):
        # 修改：添加探索机制
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 策略网络（使用简单的线性函数近似）
        self.weights = np.random.randn(env.size * env.size, 4) * 0.01
        
        # 存储episode的经验
        self.states = []
        self.actions = []
        self.rewards = []
        
    def get_state_index(self, position):
        """将位置转换为状态索引"""
        return position[0] * self.env.size + position[1]
    
    def get_action(self, state):
        """根据策略选择动作"""
        # 添加探索机制 - ε-greedy策略
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # 随机动作
        else:
            state_idx = self.get_state_index(state)
            logits = self.weights[state_idx]
            # 计算softmax概率
            exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
            probs = exp_logits / np.sum(exp_logits)
            
            # 根据概率选择动作
            return np.random.choice(4, p=probs)
    
    def get_action_probs(self, state):
        """获取动作概率分布"""
        state_idx = self.get_state_index(state)
        logits = self.weights[state_idx]
        # 计算softmax概率
        exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
        probs = exp_logits / np.sum(exp_logits)
        return probs
    
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
    
    def store_experience(self, state, action, reward):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def discount_rewards(self, rewards):
        """计算折扣奖励"""
        discounted = np.zeros_like(rewards, dtype=float)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted[t] = running_add
        return discounted
    
    def update_policy(self):
        """更新策略"""
        if len(self.rewards) == 0:
            return
            
        # 计算折扣奖励
        discounted_rewards = self.discount_rewards(self.rewards)
        
        # 标准化奖励以提高稳定性
        if np.std(discounted_rewards) > 0:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)
        
        # 更新策略
        for i in range(len(self.states)):
            state_idx = self.get_state_index(self.states[i])
            action = self.actions[i]
            reward = discounted_rewards[i]
            
            # 计算当前策略下的动作概率
            probs = self.get_action_probs(self.states[i])
            
            # 计算梯度
            dsoftmax = probs.copy()
            dsoftmax[action] -= 1
            
            # 更新权重
            self.weights[state_idx] -= self.learning_rate * reward * dsoftmax
        
        # 清空经验池
        self.states = []
        self.actions = []
        self.rewards = []
        
        # 降低探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, episodes=1000):
        """训练智能体"""
        rewards_history = []
        
        for episode in range(episodes):
            # 每个episode开始时更新动态障碍物
            self.env.update_dynamic_obstacles()
            
            # 初始化状态
            state = self.env.start
            total_reward = 0
            steps = 0
            
            while state != self.env.goal:
                action = self.get_action(state)
                next_state = self.get_next_state(state, action)
                
                # 修改奖励函数 - 增加正向激励
                if next_state == self.env.goal:
                    reward = 100
                elif state == next_state:  # 碰撞或边界
                    reward = -10
                else:
                    # 增加小步移动的正向奖励，鼓励探索
                    reward = -1
                
                # 存储经验
                self.store_experience(state, action, reward)
                
                # 更新状态
                state = next_state
                total_reward += reward
                steps += 1
                
                if steps > 1000:  # 防止无限循环
                    break
            
            # 更新策略
            self.update_policy()
            rewards_history.append(total_reward)
            
            # 每200个episode输出一次当前最优路径（使用贪婪策略找到的路径）
            # 修改为从第200个episode开始显示图像
            if episode % 200 == 0 and episode >= 200:
                print(f"Policy Gradient Episode {episode}, Reward: {total_reward}")
                # 可视化当前最优路径（使用贪婪策略找到的路径）
                path = self.find_path()
                # 验证路径的合法性
                valid_path = self.validate_path(path)
                if valid_path:
                    self.env.visualize(path, f'Policy Gradient Path - Episode {episode}')
        
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
        """找到从起点到终点的路径（使用确定性策略）"""
        path = []
        state = self.env.start
        visited_states = set()
        
        # 在查找路径时需要考虑当前动态障碍物的位置
        max_steps = 1000
        steps = 0
        
        while state != self.env.goal and steps < max_steps:
            # 使用确定性策略（选择概率最高的动作）
            state_idx = self.get_state_index(state)
            logits = self.weights[state_idx]
            action = np.argmax(logits)
            
            next_state = self.get_next_state(state, action)
            
            # 检查是否形成循环
            if next_state in visited_states:
                # 尝试其他动作
                best_action = action
                best_logit = logits[best_action]
                found_better = False
                for a in range(4):
                    test_logit = logits[a]
                    test_next = self.get_next_state(state, a)
                    if test_next not in visited_states and test_logit > best_logit:
                        best_action = a
                        best_logit = test_logit
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