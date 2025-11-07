"""
DRL雙機器人探索模擬環境
基於原始DRL Robot Exploration論文的雙機器人版本
修改以用於論文比較實驗

關鍵參數設置 (與你的論文保持一致):
- sensor_range: 50 (感測器範圍)
- movement_step: 5 (移動步長 / step size)
- robot_size: 2
- local_size: 40
"""

from scipy import spatial
from skimage import io
import numpy as np
import numpy.ma as ma
import sys
from scipy import ndimage
import matplotlib.pyplot as plt
from random import shuffle
import os

class DualRobotEnv:
    """雙機器人探索環境 - 兩個機器人使用DRL控制器獨立決策"""
    
    def __init__(self, index_map, train, plot):
        self.mode = train
        self.plot = plot
        
        # 地圖設置
        if self.mode:
            self.map_dir = '../DungeonMaps/train'
        else:
            self.map_dir = '../DungeonMaps/test'
        
        self.map_list = os.listdir(self.map_dir)
        self.map_number = np.size(self.map_list)
        if self.mode:
            shuffle(self.map_list)
        
        self.li_map = index_map
        
        # 初始化全域地圖
        self.global_map, initial_position = self.map_setup(
            self.map_dir + '/' + self.map_list[self.li_map])
        
        # 為兩個機器人設置不同的起始位置
        self.robot1_position = initial_position.copy()
        self.robot2_position = self.find_second_start_position(initial_position)
        
        # 共享地圖 (兩個機器人共享觀察結果)
        self.shared_op_map = np.ones(self.global_map.shape) * 127
        
        self.map_size = np.shape(self.global_map)
        self.finish_percent = 0.985
        self.resolution = 1
        
        # **關鍵參數 - 與你的論文參數完全一致**
        self.sensor_range = 50  # 感測器範圍 (你的ROBOT_CONFIG)
        self.robot_size = 2     # 機器人尺寸
        self.local_size = 40    # 局部地圖大小
        self.movement_step = 5  # 移動步長 (step size)
        
        # 機器人狀態
        self.robot1_old_position = np.zeros([2])
        self.robot2_old_position = np.zeros([2])
        
        # 動作空間 (從DRL論文 - 50個動作)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        action_file = current_dir + '/action_points.csv'
        if os.path.exists(action_file):
            self.action_space = np.genfromtxt(action_file, delimiter=",")
        else:
            # 生成50個動作點
            self.action_space = self.generate_action_points(50)
        
        # 地圖相關
        self.t = self.map_points(self.global_map)
        self.free_tree = spatial.KDTree(self.free_points(self.global_map).tolist())
        
        # 視覺化
        if self.plot:
            self.robot1_xPoint = np.array([self.robot1_position[0]])
            self.robot1_yPoint = np.array([self.robot1_position[1]])
            self.robot2_xPoint = np.array([self.robot2_position[0]])
            self.robot2_yPoint = np.array([self.robot2_position[1]])
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
    
    def generate_action_points(self, num_actions):
        """生成動作空間點 (極座標方式)"""
        actions = []
        radius = 30  # 動作半徑
        for i in range(num_actions):
            angle = 2 * np.pi * i / num_actions
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            actions.append([x, y])
        return np.array(actions)
    
    def find_second_start_position(self, first_position):
        """為第二個機器人找到合適的起始位置"""
        free_pts = self.free_points(self.global_map)
        distances = np.sqrt(np.sum((free_pts - first_position)**2, axis=1))
        sorted_indices = np.argsort(distances)
        idx = sorted_indices[len(sorted_indices)//3]  
        return free_pts[idx].copy()
    
    def map_setup(self, map_path):
        """從圖片載入地圖"""
        map_img = io.imread(map_path)
        if len(map_img.shape) > 2:
            map_img = map_img[:, :, 0]
        
        global_map = (map_img > 150).astype(int)
        
        free_indices = np.where(global_map == 1)
        start_idx = len(free_indices[0]) // 2
        start_position = np.array([free_indices[1][start_idx], 
                                   free_indices[0][start_idx]], dtype=float)
        
        return global_map, start_position
    
    def map_points(self, map_data):
        """獲取地圖所有點的座標"""
        indices = np.indices(map_data.shape)
        points = np.column_stack([indices[1].ravel(), indices[0].ravel()])
        return points
    
    def free_points(self, map_data):
        """獲取所有自由空間點"""
        free_indices = np.where(map_data.ravel() == 1)[0]
        return self.t[free_indices]
    
    def begin(self):
        """初始化環境"""
        # 兩個機器人同時進行初始感測
        self.shared_op_map = self.inverse_sensor(
            self.robot1_position, self.sensor_range, 
            self.shared_op_map, self.global_map)
        self.shared_op_map = self.inverse_sensor(
            self.robot2_position, self.sensor_range, 
            self.shared_op_map, self.global_map)
        
        # 生成帶有機器人的地圖
        step_map = self.create_combined_robot_map()
        
        # 為兩個機器人分別生成局部地圖
        map_local_1 = self.local_map(
            self.robot1_position, step_map, self.map_size, 
            self.sensor_range + self.local_size)
        map_local_2 = self.local_map(
            self.robot2_position, step_map, self.map_size, 
            self.sensor_range + self.local_size)
        
        if self.plot:
            self.plot_env()
        
        return map_local_1, map_local_2
    
    def step(self, action_index_1, action_index_2):
        """
        兩個機器人同時執行動作
        使用DRL論文原始的控制策略
        
        返回:
            tuple: (map_local_1, map_local_2, reward_1, reward_2, 
                   terminal_1, terminal_2, complete, new_location_1, new_location_2)
        """
        # 保存舊狀態
        self.robot1_old_position = self.robot1_position.copy()
        self.robot2_old_position = self.robot2_position.copy()
        old_shared_op_map = self.shared_op_map.copy()
        
        # 執行動作 (使用movement_step=5)
        self.take_action(action_index_1, self.robot1_position)
        self.take_action(action_index_2, self.robot2_position)
        
        # 碰撞檢查
        collision_1, coll_idx_1 = self.collision_check(
            self.robot1_old_position, self.robot1_position, 
            self.map_size, self.global_map)
        collision_2, coll_idx_2 = self.collision_check(
            self.robot2_old_position, self.robot2_position, 
            self.map_size, self.global_map)
        
        # 處理碰撞
        if coll_idx_1:
            self.robot1_position = self.nearest_free(self.free_tree, collision_1)
        if coll_idx_2:
            self.robot2_position = self.nearest_free(self.free_tree, collision_2)
        
        # 更新共享地圖 (兩個機器人的感測器數據都更新到同一張地圖)
        self.shared_op_map = self.inverse_sensor(
            self.robot1_position, self.sensor_range, 
            self.shared_op_map, self.global_map)
        self.shared_op_map = self.inverse_sensor(
            self.robot2_position, self.sensor_range, 
            self.shared_op_map, self.global_map)
        
        # 創建帶有兩個機器人的地圖
        step_map = self.create_combined_robot_map()
        
        # 生成局部地圖
        map_local_1 = self.local_map(
            self.robot1_position, step_map, self.map_size, 
            self.sensor_range + self.local_size)
        map_local_2 = self.local_map(
            self.robot2_position, step_map, self.map_size, 
            self.sensor_range + self.local_size)
        
        # 計算獎勵
        reward_1 = self.get_reward(old_shared_op_map, self.shared_op_map, coll_idx_1)
        reward_2 = self.get_reward(old_shared_op_map, self.shared_op_map, coll_idx_2)
        
        # 判斷終止條件
        terminal_1 = False
        terminal_2 = False
        new_location_1 = False
        new_location_2 = False
        
        # 低獎勵檢查 (與DRL論文一致)
        if reward_1 <= 0.02 and not coll_idx_1:
            reward_1 = -0.8
            new_location_1 = True
            terminal_1 = True
        
        if reward_2 <= 0.02 and not coll_idx_2:
            reward_2 = -0.8
            new_location_2 = True
            terminal_2 = True
        
        # 碰撞處理 (與DRL論文一致)
        if coll_idx_1:
            if self.mode:  
                new_location_1 = True
                terminal_1 = True
                self.robot1_position = self.robot1_old_position.copy()
        
        if coll_idx_2:
            if self.mode: 
                new_location_2 = True
                terminal_2 = True
                self.robot2_position = self.robot2_old_position.copy()
        
        # 檢查是否完成探索
        complete = self.check_complete()
        
        if self.plot:
            self.robot1_xPoint = np.append(self.robot1_xPoint, self.robot1_position[0])
            self.robot1_yPoint = np.append(self.robot1_yPoint, self.robot1_position[1])
            self.robot2_xPoint = np.append(self.robot2_xPoint, self.robot2_position[0])
            self.robot2_yPoint = np.append(self.robot2_yPoint, self.robot2_position[1])
            self.plot_env()
        
        return (map_local_1, map_local_2, reward_1, reward_2, 
                terminal_1, terminal_2, complete, new_location_1, new_location_2)
    
    def create_combined_robot_map(self):
        """創建包含兩個機器人的地圖"""
        map_copy = self.shared_op_map.copy()
        
        # 添加機器人1
        robot1_points = self.range_search(
            self.robot1_position, self.robot_size, self.t)
        for i in range(robot1_points.shape[0]):
            rob_loc = np.int32(robot1_points[i, :])
            rob_loc = np.flipud(rob_loc)
            if 0 <= rob_loc[0] < map_copy.shape[0] and 0 <= rob_loc[1] < map_copy.shape[1]:
                map_copy[tuple(rob_loc)] = 76
        
        # 添加機器人2
        robot2_points = self.range_search(
            self.robot2_position, self.robot_size, self.t)
        for i in range(robot2_points.shape[0]):
            rob_loc = np.int32(robot2_points[i, :])
            rob_loc = np.flipud(rob_loc)
            if 0 <= rob_loc[0] < map_copy.shape[0] and 0 <= rob_loc[1] < map_copy.shape[1]:
                map_copy[tuple(rob_loc)] = 50  
        
        return map_copy
    
    def take_action(self, action_index, robot_position):
        """
        執行動作 - 使用與你論文相同的movement_step=5
        這是關鍵的比較參數之一
        """
        action = self.action_space[action_index]
        # 正規化並縮放到movement_step
        action_norm = action / np.linalg.norm(action) * self.movement_step
        robot_position[0] += action_norm[0]
        robot_position[1] += action_norm[1]
    
    def collision_check(self, start_point, end_point, map_size, map_glo):
        """Bresenham直線碰撞檢測 (與DRL論文一致)"""
        x0, y0 = start_point.round().astype(int)
        x1, y1 = end_point.round().astype(int)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        coll_points = np.ones((1, 2), np.uint8) * -1

        while 0 <= x < map_size[1] and 0 <= y < map_size[0]:
            k = map_glo[y, x]
            if k == 1:  
                coll_points[0, 0] = x
                coll_points[0, 1] = y
                break

            if x == end_point[0] and y == end_point[1]:
                break

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        
        if np.sum(coll_points) == -2:
            coll_index = False
        else:
            coll_index = True

        return coll_points, coll_index
    
    def inverse_sensor(self, robot_position, sensor_range, op_map, map_glo):
        """
        反向感測器模型 - sensor_range=50 (你的論文參數)
        這是另一個關鍵的比較參數
        """
        x, y = int(robot_position[0]), int(robot_position[1])
        
        for dx in range(-sensor_range, sensor_range + 1):
            for dy in range(-sensor_range, sensor_range + 1):
                dist = np.sqrt(dx**2 + dy**2)
                if dist <= sensor_range:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < op_map.shape[1] and 0 <= ny < op_map.shape[0]:
                        if map_glo[ny, nx] == 1:
                            op_map[ny, nx] = 0  
                        else:
                            op_map[ny, nx] = 255  
        
        return op_map
    
    def local_map(self, robot_position, global_map, map_size, local_range):
        """提取局部地圖"""
        x, y = int(robot_position[0]), int(robot_position[1])
        x_min = max(0, x - local_range)
        x_max = min(map_size[1], x + local_range)
        y_min = max(0, y - local_range)
        y_max = min(map_size[0], y + local_range)
        
        local = global_map[y_min:y_max, x_min:x_max]
        
        # Resize to fixed size (與DRL論文一致)
        from skimage.transform import resize
        local_resized = resize(local, (120, 120))  
        
        return local_resized
    
    def get_reward(self, old_op_map, op_map, coll_index):
        """
        計算獎勵 - 與DRL論文一致
        """
        if not coll_index:
            new_explored = np.size(np.where(op_map == 255)) - np.size(np.where(old_op_map == 255))
            reward = float(new_explored) / 14000  
            if reward > 1:
                reward = 1
        else:
            reward = -1  
        
        return reward
    
    def check_complete(self):
        """檢查是否完成探索"""
        total_free = np.size(np.where(self.global_map == 1))
        explored_free = np.size(np.where(self.shared_op_map == 255))
        
        if total_free > 0:
            explored_ratio = float(explored_free) / float(total_free)
            return explored_ratio >= self.finish_percent
        return False
    
    def nearest_free(self, tree, point):
        """找到最近的自由空間點"""
        pts = np.atleast_2d(point)
        index = tuple(tree.query(pts)[1])
        nearest = tree.data[index]
        return nearest[0]
    
    def range_search(self, position, r, points):
        """搜尋半徑內的所有點"""
        nvar = position.shape[0]
        r2 = r ** 2
        s = 0
        for d in range(nvar):
            s += (points[:, d] - position[d]) ** 2
        idx = np.nonzero(s <= r2)
        idx = np.asarray(idx).ravel()
        inrange_points = points[idx, :]
        return inrange_points
    
    def plot_env(self):
        """視覺化環境 - 顯示兩個機器人"""
        self.ax.cla()
        self.ax.imshow(self.shared_op_map, cmap='gray')
        self.ax.axis((0, self.map_size[1], self.map_size[0], 0))
        
        # 機器人1 (藍色)
        self.ax.plot(self.robot1_xPoint, self.robot1_yPoint, 'b-', linewidth=2, label='Robot 1 (DRL)')
        self.ax.plot(self.robot1_position[0], self.robot1_position[1], 'bo', markersize=10)
        
        # 機器人2 (紅色)
        self.ax.plot(self.robot2_xPoint, self.robot2_yPoint, 'r-', linewidth=2, label='Robot 2 (DRL)')
        self.ax.plot(self.robot2_position[0], self.robot2_position[1], 'ro', markersize=10)
        
        self.ax.legend()
        self.ax.set_title(f'Dual Robot DRL Exploration (sensor_range={self.sensor_range}, step_size={self.movement_step})')
        plt.pause(0.01)
    
    def reset(self, map_index=None):
        """重置環境"""
        if map_index is not None:
            self.li_map = map_index
        else:
            self.li_map = (self.li_map + 1) % self.map_number
        
        self.global_map, initial_position = self.map_setup(
            self.map_dir + '/' + self.map_list[self.li_map])
        
        self.robot1_position = initial_position.copy()
        self.robot2_position = self.find_second_start_position(initial_position)
        
        self.shared_op_map = np.ones(self.global_map.shape) * 127
        self.t = self.map_points(self.global_map)
        self.free_tree = spatial.KDTree(self.free_points(self.global_map).tolist())
        
        if self.plot:
            self.robot1_xPoint = np.array([self.robot1_position[0]])
            self.robot1_yPoint = np.array([self.robot1_position[1]])
            self.robot2_xPoint = np.array([self.robot2_position[0]])
            self.robot2_yPoint = np.array([self.robot2_position[1]])
        
        return self.begin()