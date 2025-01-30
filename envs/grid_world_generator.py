# grid_world.py

import numpy as np

class GridWorld:
    def __init__(self, grid_matrix, default_reward=-1, terminal_rewards=None):
        """
        初始化网格世界环境。

        :param grid_matrix: n x m 的矩阵，定义网格的结构和奖励。
                            使用以下规则定义矩阵元素：
                            - 'S' : 起始位置
                            - 'G' : 目标位置
                            - 'X' : 障碍物
                            - 数值 : 特定位置的奖励
                            - 其他 : 默认奖励
        :param default_reward: 默认步进奖励，默认为-1。
        :param terminal_rewards: 字典，指定特定位置的终止奖励。
        """
        self.grid_matrix = grid_matrix
        self.grid_size = (len(grid_matrix), len(grid_matrix[0]))
        self.default_reward = default_reward
        self.terminal_rewards = terminal_rewards if terminal_rewards else {}
        
        # 动作空间：0: 上, 1: 右, 2: 下, 3: 左
        self.action_space = 4
        self.state_space = self.grid_size[0] * self.grid_size[1]

        # 解析网格矩阵，找到起始位置和目标位置
        self.start = None
        self.goal = None
        self.obstacles = []
        self.reward_map = {}  # {state: reward}

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                cell = grid_matrix[i][j]
                state = self._pos_to_state((i, j))
                if cell == 'S':
                    self.start = (i, j)
                elif cell == 'G':
                    self.goal = (i, j)
                    self.reward_map[state] = self.terminal_rewards.get((i, j), 10)
                elif cell == 'X':
                    self.obstacles.append((i, j))
                elif isinstance(cell, (int, float)):
                    self.reward_map[state] = cell

        if self.start is None:
            raise ValueError("Grid must have a start position marked with 'S'.")
        if self.goal is None:
            raise ValueError("Grid must have a goal position marked with 'G'.")

        self.reset()

    def reset(self):
        """
        重置环境到起始状态。

        :return: 初始状态的整数表示。
        """
        self.agent_pos = self.start
        self.done = False
        return self._pos_to_state(self.agent_pos)

    def step(self, action):
        """
        执行动作，返回下一个状态、奖励、是否结束以及额外信息。

        :param action: 动作，整数0-3。
        :return: next_state, reward, done, info
        """
        if self.done:
            raise Exception("Episode has finished. Please reset the environment.")

        # 计算新位置
        new_pos = self._move(self.agent_pos, action)

        # 检查是否撞墙或进入障碍物
        if self._is_valid(new_pos):
            self.agent_pos = new_pos
        else:
            # 撞墙或障碍物，位置不变
            pass

        # 检查是否达到终止状态
        if self.agent_pos == self.goal or self.agent_pos in self.terminal_rewards:
            reward = self.reward_map.get(self._pos_to_state(self.agent_pos), self.default_reward)
            self.done = True
        else:
            state = self._pos_to_state(self.agent_pos)
            reward = self.reward_map.get(state, self.default_reward)
            self.done = False

        next_state = self._pos_to_state(self.agent_pos)
        info = {}

        return next_state, reward, self.done, info

    def _move(self, position, action):
        """
        根据动作计算新位置。

        :param position: 当前的位置，元组(x, y)。
        :param action: 动作，整数0-3。
        :return: 新的位置，元组(x, y)。
        """
        x, y = position
        if action == 0:  # 上
            x -= 1
        elif action == 1:  # 右
            y += 1
        elif action == 2:  # 下
            x += 1
        elif action == 3:  # 左
            y -= 1
        else:
            raise ValueError("Invalid action.")

        return (x, y)

    def _is_valid(self, position):
        """
        检查位置是否有效（在网格内且不是障碍物）。

        :param position: 位置，元组(x, y)。
        :return: 如果有效返回True，否则返回False。
        """
        x, y = position
        # 检查边界
        if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
            return False
        # 检查障碍物
        if position in self.obstacles:
            return False
        return True

    def _pos_to_state(self, position):
        """
        将位置转换为状态编号。

        :param position: 位置，元组(x, y)。
        :return: 状态编号，整数。
        """
        x, y = position
        return x * self.grid_size[1] + y

    def _state_to_pos(self, state):
        """
        将状态编号转换为位置。

        :param state: 状态编号，整数。
        :return: 位置，元组(x, y)。
        """
        x = state // self.grid_size[1]
        y = state % self.grid_size[1]
        return (x, y)

    def render(self):
        """
        打印当前网格的状态。
        """
        grid = [['O' for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]

        # 设置障碍物
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'

        # 设置奖励位置
        for state, reward in self.reward_map.items():
            pos = self._state_to_pos(state)
            if grid[pos[0]][pos[1]] == 'O':
                grid[pos[0]][pos[1]] = f"{reward}"

        # 设置目标
        grid[self.goal[0]][self.goal[1]] = 'G'

        # 设置智能体
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'

        # 打印网格
        for row in grid:
            print(' '.join(row))
        print()




# if __name__ == "__main__":
#     # 示例用法
#     # 定义一个5x5的网格矩阵
#     # 'S' - 起始位置
#     # 'G' - 目标位置
#     # 'X' - 障碍物
#     # 数值 - 特定位置的奖励
#     grid_matrix = [
#         ['S', 'O', 'O', 'O', 'O'],
#         ['O', 'X',  -5, 'X', 'O'],
#         ['O', 'O', 'X', 'O', 'O'],
#         ['O', 'O', 'O', 'X', 'O'],
#         ['O', 'O', 'O', 'O', 'G']
#     ]

#     # 定义终止奖励（例如目标位置）
#     terminal_rewards = {
#         (4, 4): 10
#     }

#     env = GridWorld(
#         grid_matrix=grid_matrix,
#         default_reward=-1,
#         terminal_rewards=terminal_rewards
#     )

#     state = env.reset()
#     env.render()

#     # 执行动作的示例：向右 (1), 向下 (2), ...
#     actions = [1, 2, 1, 2, 1, 2, 1, 0]
#     for action in actions:
#         next_state, reward, done, info = env.step(action)
#         env.render()
#         print(f"Action: {action}, Reward: {reward}, Done: {done}")
#         if done:
#             print("Reached the goal or terminal state!")
#             break
