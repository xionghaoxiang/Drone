import numpy as np
import matplotlib.pyplot as plt
import time

def run_reinforcement_learning(env, reward_version='v2'):
    """
    在给定环境中运行所有强化学习算法
    
    参数:
    env: GridEnvironment对象
    reward_version: 奖励函数版本
    
    返回:
    dict: 包含所有算法结果的字典
    """
    try:
        # 导入强化学习算法
        from q_learning import QLearningAgent
        from sarsa import SarsaAgent
        from policy_gradient import PolicyGradientAgent
        from dqn import DQNAgent
        
        # 创建智能体
        q_agent = QLearningAgent(env, reward_version=reward_version)
        sarsa_agent = SarsaAgent(env, reward_version=reward_version)
        pg_agent = PolicyGradientAgent(env, reward_version=reward_version)
        dqn_agent = DQNAgent(env, reward_version=reward_version)
        
        # 训练智能体 (将训练次数从200提高到1000)
        print("正在训练Q-Learning智能体...")
        start_time = time.time()
        q_rewards = q_agent.train(episodes=1000)
        q_time = time.time() - start_time
        
        print("正在训练SARSA智能体...")
        start_time = time.time()
        sarsa_rewards = sarsa_agent.train(episodes=1000)
        sarsa_time = time.time() - start_time
        
        print("正在训练Policy Gradient智能体...")
        start_time = time.time()
        pg_rewards = pg_agent.train(episodes=1000)
        pg_time = time.time() - start_time
        
        print("正在训练DQN智能体...")
        start_time = time.time()
        dqn_rewards = dqn_agent.train(episodes=1000)
        dqn_time = time.time() - start_time
        
        # 更新动态障碍物位置并查找路径
        env.update_dynamic_obstacles()
        q_path = q_agent.find_path()
        sarsa_path = sarsa_agent.find_path()
        pg_path = pg_agent.find_path()
        dqn_path = dqn_agent.find_path()
        
        # 验证路径
        q_valid = q_agent.validate_path(q_path)
        sarsa_valid = sarsa_agent.validate_path(sarsa_path)
        pg_valid = pg_agent.validate_path(pg_path)
        dqn_valid = dqn_agent.validate_path(dqn_path)
        
        # 返回结果
        results = {
            'q_learning': {
                'rewards': q_rewards,
                'path': q_path,
                'valid': q_valid,
                'training_time': q_time
            },
            'sarsa': {
                'rewards': sarsa_rewards,
                'path': sarsa_path,
                'valid': sarsa_valid,
                'training_time': sarsa_time
            },
            'policy_gradient': {
                'rewards': pg_rewards,
                'path': pg_path,
                'valid': pg_valid,
                'training_time': pg_time
            },
            'dqn': {
                'rewards': dqn_rewards,
                'path': dqn_path,
                'valid': dqn_valid,
                'training_time': dqn_time
            }
        }
        
        return results
        
    except Exception as e:
        print(f"运行强化学习算法时出错: {str(e)}")
        return None

def visualize_results(env, results):
    """
    可视化强化学习结果
    
    参数:
    env: GridEnvironment对象
    results: run_reinforcement_learning返回的结果字典
    """
    if not results:
        print("没有结果可显示")
        return
    
    # 创建图表显示路径 (现在有4个算法，改为2行2列)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Q-Learning路径
    ax1 = axes[0]
    display_grid_for_visualization(ax1, env)
    if results['q_learning']['path'] and results['q_learning']['valid']:
        path_x = [p[1] for p in results['q_learning']['path']]
        path_y = [p[0] for p in results['q_learning']['path']]
        ax1.plot(path_x, path_y, 'b-', linewidth=2, marker='o', markersize=4, 
                label=f'Q-Learning Path (Length: {len(results["q_learning"]["path"])})')
    ax1.set_title('Q-Learning Path')
    ax1.legend()
    
    # SARSA路径
    ax2 = axes[1]
    display_grid_for_visualization(ax2, env)
    if results['sarsa']['path'] and results['sarsa']['valid']:
        path_x = [p[1] for p in results['sarsa']['path']]
        path_y = [p[0] for p in results['sarsa']['path']]
        ax2.plot(path_x, path_y, 'g-', linewidth=2, marker='s', markersize=4,
                label=f'SARSA Path (Length: {len(results["sarsa"]["path"])})')
    ax2.set_title('SARSA Path')
    ax2.legend()
    
    # Policy Gradient路径
    ax3 = axes[2]
    display_grid_for_visualization(ax3, env)
    if results['policy_gradient']['path'] and results['policy_gradient']['valid']:
        path_x = [p[1] for p in results['policy_gradient']['path']]
        path_y = [p[0] for p in results['policy_gradient']['path']]
        ax3.plot(path_x, path_y, 'r-', linewidth=2, marker='^', markersize=4,
                label=f'Policy Gradient Path (Length: {len(results["policy_gradient"]["path"])})')
    ax3.set_title('Policy Gradient Path')
    ax3.legend()
    
    # DQN路径
    ax4 = axes[3]
    display_grid_for_visualization(ax4, env)
    if results['dqn']['path'] and results['dqn']['valid']:
        path_x = [p[1] for p in results['dqn']['path']]
        path_y = [p[0] for p in results['dqn']['path']]
        ax4.plot(path_x, path_y, 'm-', linewidth=2, marker='d', markersize=4,
                label=f'DQN Path (Length: {len(results["dqn"]["path"])})')
    ax4.set_title('DQN Path')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 6))
    window_size = 10
    if len(results['q_learning']['rewards']) >= window_size:
        q_rewards_smooth = np.convolve(results['q_learning']['rewards'], 
                                      np.ones(window_size)/window_size, mode='valid')
        plt.plot(q_rewards_smooth, label='Q-Learning')
        
    if len(results['sarsa']['rewards']) >= window_size:
        sarsa_rewards_smooth = np.convolve(results['sarsa']['rewards'], 
                                          np.ones(window_size)/window_size, mode='valid')
        plt.plot(sarsa_rewards_smooth, label='SARSA')
        
    if len(results['policy_gradient']['rewards']) >= window_size:
        pg_rewards_smooth = np.convolve(results['policy_gradient']['rewards'], 
                                       np.ones(window_size)/window_size, mode='valid')
        plt.plot(pg_rewards_smooth, label='Policy Gradient')
        
    if len(results['dqn']['rewards']) >= window_size:
        dqn_rewards_smooth = np.convolve(results['dqn']['rewards'], 
                                       np.ones(window_size)/window_size, mode='valid')
        plt.plot(dqn_rewards_smooth, label='DQN')
    
    plt.title('Cumulative Reward over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 绘制训练时间对比图
    plt.figure(figsize=(10, 6))
    algorithms = ['Q-Learning', 'SARSA', 'Policy Gradient', 'DQN']
    training_times = [
        results['q_learning']['training_time'],
        results['sarsa']['training_time'],
        results['policy_gradient']['training_time'],
        results['dqn']['training_time']
    ]
    
    bars = plt.bar(algorithms, training_times, color=['blue', 'green', 'red', 'magenta'])
    plt.title('Training Time Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Training Time (seconds)')
    
    # 在柱状图上显示数值
    for bar, time_val in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.grid(True, axis='y')
    plt.show()

def display_grid_for_visualization(ax, env):
    """
    为可视化准备网格显示
    
    参数:
    ax: matplotlib轴对象
    env: GridEnvironment对象
    """
    # 创建显示网格
    display_grid = env.grid.copy()
    dynamic_positions = env.get_dynamic_obstacle_positions()
    for x, y in dynamic_positions:
        if 0 <= x < env.size and 0 <= y < env.size:
            display_grid[x, y] = 0.5  # 动态障碍物用0.5表示
    
    ax.imshow(display_grid, cmap='binary', origin='lower')
    
    # 标记起点和终点
    ax.plot(env.start[1], env.start[0], 'go', markersize=10, label='Start')
    ax.plot(env.goal[1], env.goal[0], 'r*', markersize=15, label='Goal')
    
    # 绘制动态障碍物轨迹
    for obstacle in env.dynamic_obstacles:
        if 'trajectory' in obstacle:
            trajectory_x = [pos[1] for pos in obstacle['trajectory']]
            trajectory_y = [pos[0] for pos in obstacle['trajectory']]
            ax.plot(trajectory_x, trajectory_y, 'c--', alpha=0.5, linewidth=1)
            
            # 标记动态障碍物当前位置
            if 'current_index' in obstacle and obstacle['current_index'] < len(obstacle['trajectory']):
                center_x, center_y = obstacle['trajectory'][obstacle['current_index']]
                ax.plot(center_y, center_x, 'rx', markersize=8)
    
    ax.set_xlim(-0.5, env.size-0.5)
    ax.set_ylim(-0.5, env.size-0.5)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.grid(True)