from environment import GridEnvironment
from q_learning import QLearningAgent
from sarsa import SarsaAgent
from policy_gradient import PolicyGradientAgent
from dqn import DQNAgent
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 创建环境
    env = GridEnvironment(size=20)
    
    # 添加静态障碍物（至少10个，分散布置，减小部分障碍物面积）
    obstacles = [
        (3, 3, 3, 3),   # 左上区域 (减小面积 4x4 -> 3x3)
        (15, 3, 3, 3),  # 右上区域 (减小面积 4x4 -> 3x3)
        (3, 15, 3, 3),  # 左下区域 (减小面积 4x4 -> 3x3)
        (15, 15, 3, 3), # 右下区域 (减小面积 4x4 -> 3x3)
        (8, 8, 2, 2),   # 中心偏左 (减小面积 3x3 -> 2x2)
        (12, 8, 2, 2),  # 中心偏右 (减小面积 3x3 -> 2x2)
        (8, 12, 2, 2),  # 中心偏下 (减小面积 3x3 -> 2x2)
        (12, 12, 2, 2), # 中心偏上 (减小面积 3x3 -> 2x2)
        (5, 10, 2, 1),  # 左侧中部 (减小面积 2x2 -> 2x1)
        (15, 10, 2, 1)  # 右侧中部 (减小面积 2x2 -> 2x1)
    ]
    
    # 在边界添加额外的静态障碍物，但避开起点(0,0)和终点(19,19)附近区域
    # 上边界障碍物 (避开起点附近区域)
    #obstacles.append((0, 4, 1, 2))   # 下左
    obstacles.append((0, 14, 1, 2))  # 下右
    
    # 下边界障碍物 (避开终点附近区域)
    #obstacles.append((19, 2, 1, 2))  # 上左
    #obstacles.append((19, 14, 1, 2)) # 上右
    
    # 左边界障碍物 (避开起点附近区域)
    #obstacles.append((2, 0, 2, 1))   # 左边界下
    obstacles.append((14, 0, 2, 1))  # 左边界上
    
    # 右边界障碍物 (避开终点附近区域)
    #obstacles.append((2, 19, 2, 1))  # 右边界下侧
    #obstacles.append((14, 19, 2, 1)) # 右边界上侧
    
    # 添加障碍物到环境中
    for obs in obstacles:
        env.add_obstacle(*obs)
    
    # 添加固定轨迹往复运动的动态障碍物
    env.add_dynamic_obstacles(2)
    
    # 创建四个智能体（Q-learning、SARSA、策略梯度和DQN）
    q_agent = QLearningAgent(env)
    sarsa_agent = SarsaAgent(env)
    pg_agent = PolicyGradientAgent(env)
    dqn_agent = DQNAgent(env)
    
    # 训练智能体，将训练次数改为1000次
    print("Training Q-Learning agent...")
    q_rewards = q_agent.train(episodes=1000)
    
    print("\nTraining SARSA agent...")
    sarsa_rewards = sarsa_agent.train(episodes=1000)
    
    print("\nTraining Policy Gradient agent...")
    pg_rewards = pg_agent.train(episodes=1000)
    
    print("\nTraining DQN agent...")
    dqn_rewards = dqn_agent.train(episodes=1000)      #1000次太慢了
    
    # 找到路径（在查找路径前更新动态障碍物位置）
    env.update_dynamic_obstacles()  # 确保我们看到的是当前的动态障碍物位置
    q_path = q_agent.find_path()
    sarsa_path = sarsa_agent.find_path()
    pg_path = pg_agent.find_path()
    dqn_path = dqn_agent.find_path()
    
    # 验证路径合法性
    q_valid = q_agent.validate_path(q_path)
    sarsa_valid = sarsa_agent.validate_path(sarsa_path)
    pg_valid = pg_agent.validate_path(pg_path)
    dqn_valid = dqn_agent.validate_path(dqn_path)
    
    print(f"Q-learning path valid: {q_valid}")
    print(f"SARSA path valid: {sarsa_valid}")
    print(f"Policy Gradient path valid: {pg_valid}")
    print(f"DQN path valid: {dqn_valid}")
    
    # 可视化结果 - 对比四种算法的路径
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    # Q-learning路径
    ax1 = axes[0]
    ax1.imshow(env.grid, cmap='binary', origin='lower')
    ax1.plot(env.start[1], env.start[0], 'go', markersize=12, label='Start')
    ax1.plot(env.goal[1], env.goal[0], 'r*', markersize=15, label='Goal')
    
    # 绘制静态障碍物（包括边界）
    for (row, col, h, w) in env.obstacles:
        rect = plt.Rectangle((col - 0.5, row - 0.5), w, h, linewidth=1, 
                           edgecolor='gray', facecolor='gray', alpha=0.7)
        ax1.add_patch(rect)
    
    # 绘制动态障碍物轨迹
    for obstacle in env.dynamic_obstacles:
        trajectory_x = [pos[1] for pos in obstacle['trajectory']]
        trajectory_y = [pos[0] for pos in obstacle['trajectory']]
        ax1.plot(trajectory_x, trajectory_y, 'c--', alpha=0.5, linewidth=1)
    
    if q_path and q_valid:
        path_x = [p[1] for p in q_path]
        path_y = [p[0] for p in q_path]
        ax1.plot(path_x, path_y, 'b-', linewidth=2.5, marker='o', markersize=4, 
                label=f'Q-Learning Path (Length: {len(q_path)})')
    elif q_path:
        print("Warning: Q-learning generated an invalid path")
        # 即使路径无效，也显示以供调试
        path_x = [p[1] for p in q_path]
        path_y = [p[0] for p in q_path]
        ax1.plot(path_x, path_y, 'b--', linewidth=1.5, marker='o', markersize=2, 
                label=f'Q-Learning Path (Invalid, Length: {len(q_path)})')
    
    ax1.set_xlim(-0.5, env.size-0.5)
    ax1.set_ylim(-0.5, env.size-0.5)
    ax1.set_xticks(range(env.size))
    ax1.set_yticks(range(env.size))
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    ax1.set_title('Q-Learning Path Planning', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    
    # SARSA路径
    ax2 = axes[1]
    ax2.imshow(env.grid, cmap='binary', origin='lower')
    ax2.plot(env.start[1], env.start[0], 'go', markersize=12, label='Start')
    ax2.plot(env.goal[1], env.goal[0], 'r*', markersize=15, label='Goal')
    
    # 绘制静态障碍物（包括边界）
    for (row, col, h, w) in env.obstacles:
        rect = plt.Rectangle((col - 0.5, row - 0.5), w, h, linewidth=1, 
                           edgecolor='gray', facecolor='gray', alpha=0.7)
        ax2.add_patch(rect)
    
    # 绘制动态障碍物轨迹
    for obstacle in env.dynamic_obstacles:
        trajectory_x = [pos[1] for pos in obstacle['trajectory']]
        trajectory_y = [pos[0] for pos in obstacle['trajectory']]
        ax2.plot(trajectory_x, trajectory_y, 'c--', alpha=0.5, linewidth=1)
    
    if sarsa_path and sarsa_valid:
        path_x = [p[1] for p in sarsa_path]
        path_y = [p[0] for p in sarsa_path]
        ax2.plot(path_x, path_y, 'm-', linewidth=2.5, marker='s', markersize=4,
                label=f'SARSA Path (Length: {len(sarsa_path)})')
    elif sarsa_path:
        print("Warning: SARSA generated an invalid path")
        # 即使路径无效，也显示以供调试
        path_x = [p[1] for p in sarsa_path]
        path_y = [p[0] for p in sarsa_path]
        ax2.plot(path_x, path_y, 'm--', linewidth=1.5, marker='s', markersize=2,
                label=f'SARSA Path (Invalid, Length: {len(sarsa_path)})')
        
        # 增加路径点的可见性，确保至少显示起点和终点
        if len(sarsa_path) >= 2:
            ax2.plot([path_x[0], path_x[-1]], [path_y[0], path_y[-1]], 'm--', linewidth=1.5)
            ax2.plot(path_x[0], path_y[0], 'ms', markersize=6, alpha=0.7)
            ax2.plot(path_x[-1], path_y[-1], 'ms', markersize=6, alpha=0.7)
    else:
        # 如果没有路径，至少显示起点和终点
        ax2.plot(env.start[1], env.start[0], 'go', markersize=12)
        ax2.plot(env.goal[1], env.goal[0], 'r*', markersize=15)
        
    ax2.set_xlim(-0.5, env.size-0.5)
    ax2.set_ylim(-0.5, env.size-0.5)
    ax2.set_xticks(range(env.size))
    ax2.set_yticks(range(env.size))
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left')
    ax2.set_title('SARSA Path Planning', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    
    # Policy Gradient路径
    ax3 = axes[2]
    ax3.imshow(env.grid, cmap='binary', origin='lower')
    ax3.plot(env.start[1], env.start[0], 'go', markersize=12, label='Start')
    ax3.plot(env.goal[1], env.goal[0], 'r*', markersize=15, label='Goal')
    
    # 绘制静态障碍物（包括边界）
    for (row, col, h, w) in env.obstacles:
        rect = plt.Rectangle((col - 0.5, row - 0.5), w, h, linewidth=1, 
                           edgecolor='gray', facecolor='gray', alpha=0.7)
        ax3.add_patch(rect)
    
    # 绘制动态障碍物轨迹
    for obstacle in env.dynamic_obstacles:
        trajectory_x = [pos[1] for pos in obstacle['trajectory']]
        trajectory_y = [pos[0] for pos in obstacle['trajectory']]
        ax3.plot(trajectory_x, trajectory_y, 'c--', alpha=0.5, linewidth=1)
    
    if pg_path and pg_valid:
        path_x = [p[1] for p in pg_path]
        path_y = [p[0] for p in pg_path]
        ax3.plot(path_x, path_y, 'orange', linewidth=2.5, marker='^', markersize=4,
                label=f'Policy Gradient Path (Length: {len(pg_path)})')
    elif pg_path:
        print("Warning: Policy Gradient generated an invalid path")
        # 即使路径无效，也显示以供调试
        path_x = [p[1] for p in pg_path]
        path_y = [p[0] for p in pg_path]
        ax3.plot(path_x, path_y, 'orange', linewidth=1.5, marker='^', markersize=2, linestyle='--',
                label=f'Policy Gradient Path (Invalid, Length: {len(pg_path)})')
        
        # 增加路径点的可见性，确保至少显示起点和终点
        if len(pg_path) >= 2:
            ax3.plot([path_x[0], path_x[-1]], [path_y[0], path_y[-1]], 'orange', linewidth=1.5, linestyle='--')
            ax3.plot(path_x[0], path_y[0], 'o', color='orange', markersize=6, alpha=0.7)
            ax3.plot(path_x[-1], path_y[-1], 'o', color='orange', markersize=6, alpha=0.7)
    else:
        # 如果没有路径，至少显示起点和终点
        ax3.plot(env.start[1], env.start[0], 'go', markersize=12)
        ax3.plot(env.goal[1], env.goal[0], 'r*', markersize=15)
        
    ax3.set_xlim(-0.5, env.size-0.5)
    ax3.set_ylim(-0.5, env.size-0.5)
    ax3.set_xticks(range(env.size))
    ax3.set_yticks(range(env.size))
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left')
    ax3.set_title('Policy Gradient Path Planning', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')

    # DQN路径
    ax4 = axes[3]
    ax4.imshow(env.grid, cmap='binary', origin='lower')
    ax4.plot(env.start[1], env.start[0], 'go', markersize=12, label='Start')
    ax4.plot(env.goal[1], env.goal[0], 'r*', markersize=15, label='Goal')
    
    # 绘制静态障碍物（包括边界）
    for (row, col, h, w) in env.obstacles:
        rect = plt.Rectangle((col - 0.5, row - 0.5), w, h, linewidth=1, 
                           edgecolor='gray', facecolor='gray', alpha=0.7)
        ax4.add_patch(rect)
    
    # 绘制动态障碍物轨迹
    for obstacle in env.dynamic_obstacles:
        trajectory_x = [pos[1] for pos in obstacle['trajectory']]
        trajectory_y = [pos[0] for pos in obstacle['trajectory']]
        ax4.plot(trajectory_x, trajectory_y, 'c--', alpha=0.5, linewidth=1)
    
    if dqn_path and dqn_valid:
        path_x = [p[1] for p in dqn_path]
        path_y = [p[0] for p in dqn_path]
        ax4.plot(path_x, path_y, 'g-', linewidth=2.5, marker='d', markersize=4,
                label=f'DQN Path (Length: {len(dqn_path)})')
    elif dqn_path:
        print("Warning: DQN generated an invalid path")
        # 即使路径无效，也显示以供调试
        path_x = [p[1] for p in dqn_path]
        path_y = [p[0] for p in dqn_path]
        ax4.plot(path_x, path_y, 'g--', linewidth=1.5, marker='d', markersize=2, linestyle='--',
                label=f'DQN Path (Invalid, Length: {len(dqn_path)})')
        
        # 增加路径点的可见性，确保至少显示起点和终点
        if len(dqn_path) >= 2:
            ax4.plot([path_x[0], path_x[-1]], [path_y[0], path_y[-1]], 'g--', linewidth=1.5, linestyle='--')
            ax4.plot(path_x[0], path_y[0], 'd', color='green', markersize=6, alpha=0.7)
            ax4.plot(path_x[-1], path_y[-1], 'd', color='green', markersize=6, alpha=0.7)
    else:
        # 如果没有路径，至少显示起点和终点
        ax4.plot(env.start[1], env.start[0], 'go', markersize=12)
        ax4.plot(env.goal[1], env.goal[0], 'r*', markersize=15)
        
    ax4.set_xlim(-0.5, env.size-0.5)
    ax4.set_ylim(-0.5, env.size-0.5)
    ax4.set_xticks(range(env.size))
    ax4.set_yticks(range(env.size))
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(loc='upper left')
    ax4.set_title('DQN Path Planning', fontsize=14, fontweight='bold')
    ax4.set_xlabel('X Coordinate')
    ax4.set_ylabel('Y Coordinate')
    
    plt.tight_layout()
    plt.show()
    
    # 绘制奖励曲线对比
    plt.figure(figsize=(12, 6))
    
    # 计算滑动平均以更好地显示趋势
    window_size = 50
    q_rewards_smooth = np.convolve(q_rewards, np.ones(window_size)/window_size, mode='valid')
    sarsa_rewards_smooth = np.convolve(sarsa_rewards, np.ones(window_size)/window_size, mode='valid')
    pg_rewards_smooth = np.convolve(pg_rewards, np.ones(window_size)/window_size, mode='valid')
    dqn_rewards_smooth = np.convolve(dqn_rewards, np.ones(window_size)/window_size, mode='valid')
    
    episodes_range = np.arange(len(q_rewards_smooth))
    plt.plot(episodes_range, q_rewards_smooth, label='Q-Learning', color='blue', linewidth=2)
    plt.plot(episodes_range, sarsa_rewards_smooth, label='SARSA', color='magenta', linewidth=2)
    plt.plot(episodes_range, pg_rewards_smooth, label='Policy Gradient', color='orange', linewidth=2)
    plt.plot(episodes_range, dqn_rewards_smooth, label='DQN', color='green', linewidth=2)
    plt.title('Comparison of Cumulative Reward over Episodes', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward (Smoothed)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # 输出统计信息
    print("\n=== Performance Comparison ===")
    print(f"Q-Learning - Path length: {len(q_path)}")
    print(f"Q-Learning - Final reward: {q_rewards[-1]}")
    print(f"Q-Learning - Average reward (last 100 episodes): {np.mean(q_rewards[-100:]):.2f}")
    
    print(f"\nSARSA - Path length: {len(sarsa_path)}")
    print(f"SARSA - Final reward: {sarsa_rewards[-1]}")
    print(f"SARSA - Average reward (last 100 episodes): {np.mean(sarsa_rewards[-100:]):.2f}")
    
    print(f"\nPolicy Gradient - Path length: {len(pg_path)}")
    print(f"Policy Gradient - Final reward: {pg_rewards[-1]}")
    print(f"Policy Gradient - Average reward (last 100 episodes): {np.mean(pg_rewards[-100:]):.2f}")
    
    print(f"\nDQN - Path length: {len(dqn_path)}")
    print(f"DQN - Final reward: {dqn_rewards[-1]}")
    print(f"DQN - Average reward (last 100 episodes): {np.mean(dqn_rewards[-100:]):.2f}")

if __name__ == "__main__":
    main()