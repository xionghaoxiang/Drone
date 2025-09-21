"""
奖励函数模块
包含多种不同的奖励函数设计，用于强化学习算法的性能对比
"""

import numpy as np

def reward_function_v1(env, state, action, next_state, goal_reached=False, collision=False):
    """
    奖励函数版本1：基础奖励函数（当前实现）
    
    Args:
        env: 环境对象
        state: 当前状态
        action: 执行的动作
        next_state: 下一状态
        goal_reached: 是否到达目标
        collision: 是否发生碰撞
    
    Returns:
        reward: 计算得到的奖励值
    """
    if goal_reached:
        return 100
    elif collision:
        return -10
    else:
        return -1

def reward_function_v2(env, state, action, next_state, goal_reached=False, collision=False):
    """
    奖励函数版本2：距离奖励函数
    基于到目标点的欧几里得距离提供奖励，鼓励智能体向目标移动
    
    Args:
        env: 环境对象
        state: 当前状态
        action: 执行的动作
        next_state: 下一状态
        goal_reached: 是否到达目标
        collision: 是否发生碰撞
    
    Returns:
        reward: 计算得到的奖励值
    """
    if goal_reached:
        return 100
    elif collision:
        return -10
    else:
        # 计算当前状态到目标的距离
        current_distance = np.sqrt((state[0] - env.goal[0])**2 + (state[1] - env.goal[1])**2)
        # 计算下一状态到目标的距离
        next_distance = np.sqrt((next_state[0] - env.goal[0])**2 + (next_state[1] - env.goal[1])**2)
        
        # 如果更接近目标，给予正奖励；否则给予负奖励
        distance_improvement = current_distance - next_distance
        distance_reward = 5 * distance_improvement  # 缩放因子
        
        # 基础步数惩罚
        step_penalty = -0.1
        
        return distance_reward + step_penalty

def reward_function_v3(env, state, action, next_state, goal_reached=False, collision=False):
    """
    奖励函数版本3：步数敏感奖励函数
    强烈惩罚步数，鼓励寻找最短路径
    
    Args:
        env: 环境对象
        state: 当前状态
        action: 执行的动作
        next_state: 下一状态
        goal_reached: 是否到达目标
        collision: 是否发生碰撞
    
    Returns:
        reward: 计算得到的奖励值
    """
    if goal_reached:
        return 100
    elif collision:
        return -5
    else:
        # 更强的步数惩罚
        step_penalty = -1.5
        return step_penalty

def reward_function_v4(env, state, action, next_state, goal_reached=False, collision=False):
    """
    奖励函数版本4：综合奖励函数
    结合距离奖励、步数惩罚和安全奖励
    
    Args:
        env: 环境对象
        state: 当前状态
        action: 执行的动作
        next_state: 下一状态
        goal_reached: 是否到达目标
        collision: 是否发生碰撞
    
    Returns:
        reward: 计算得到的奖励值
    """
    if goal_reached:
        # 根据路径长度调整奖励
        return 100
    elif collision:
        return -20  # 更强的碰撞惩罚
    else:
        # 距离奖励
        current_distance = np.sqrt((state[0] - env.goal[0])**2 + (state[1] - env.goal[1])**2)
        next_distance = np.sqrt((next_state[0] - env.goal[0])**2 + (next_state[1] - env.goal[1])**2)
        distance_improvement = current_distance - next_distance
        distance_reward = 2 * distance_improvement
        
        # 步数惩罚
        step_penalty = -0.5
        
        # 安全奖励（远离障碍物）
        safety_reward = 0
        min_distance_to_obstacle = float('inf')
        
        # 计算到最近障碍物的距离
        for i in range(env.size):
            for j in range(env.size):
                if not env.is_valid_position(i, j) and (i, j) != env.goal:
                    obstacle_distance = np.sqrt((next_state[0] - i)**2 + (next_state[1] - j)**2)
                    min_distance_to_obstacle = min(min_distance_to_obstacle, obstacle_distance)
        
        # 如果离障碍物较远，给予安全奖励
        if min_distance_to_obstacle > 3:
            safety_reward = 1
        
        return distance_reward + step_penalty + safety_reward

def reward_function_v5(env, state, action, next_state, goal_reached=False, collision=False):
    """
    奖励函数版本5：时间惩罚奖励函数
    随着时间步增加，增加步数惩罚，鼓励快速到达目标
    
    Args:
        env: 环境对象
        state: 当前状态
        action: 执行的动作
        next_state: 下一状态
        goal_reached: 是否到达目标
        collision: 是否发生碰撞
    
    Returns:
        reward: 计算得到的奖励值
    """
    # 这个函数需要在环境中跟踪episode_steps
    if not hasattr(env, 'episode_steps'):
        env.episode_steps = 0
    
    if goal_reached:
        reward = 100
        env.episode_steps = 0  # 重置计数器
        return reward
    elif collision:
        reward = -10
        env.episode_steps = 0  # 重置计数器
        return reward
    else:
        # 基础奖励
        base_reward = -1
        
        # 时间惩罚（随着步数增加而增加）
        time_penalty = -0.01 * env.episode_steps
        
        # 更新步数计数器
        env.episode_steps += 1
        
        return base_reward + time_penalty

# 奖励函数字典，方便在算法中调用
REWARD_FUNCTIONS = {
    'v1': reward_function_v1,
    'v2': reward_function_v2,
    'v3': reward_function_v3,
    'v4': reward_function_v4,
    'v5': reward_function_v5
}

def get_reward_function(version):
    """
    获取指定版本的奖励函数
    
    Args:
        version: 奖励函数版本 ('v1', 'v2', 'v3', 'v4', 'v5')
    
    Returns:
        对应的奖励函数
    """
    return REWARD_FUNCTIONS.get(version, reward_function_v1)