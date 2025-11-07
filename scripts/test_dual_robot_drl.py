"""
雙機器人DRL探索測試腳本
使用原始DRL論文的CNN控制器分別控制兩個機器人

關鍵設置:
- 每個機器人使用獨立的DRL控制器 (相同的網路結構,但可以獨立決策)
- 共享環境地圖
- 比較參數與你的論文一致: sensor_range=50, movement_step=5
"""

import os
import tensorflow as tf
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import dual_robot_simulation as robot_sim

# ===== 關鍵參數設置 (與DRL論文一致) =====
TRAIN = False
PLOT = True
ACTIONS = 50  # 動作空間大小
GAMMA = 0.99  # 折扣因子

# 網路參數 (與DRL論文的CNN一致)
network_dir = "../saved_networks/cnn_" + str(ACTIONS)

def create_CNN(num_actions):
    """
    創建CNN網路 (與DRL論文圖3的結構一致)
    輸入: 局部地圖 (灰階圖像)
    輸出: 每個動作的Q值
    """
    # 輸入層 - 局部地圖
    s = tf.keras.layers.Input(shape=(120, 120, 1), name='state_input')
    
    # 卷積層1
    conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(s)
    
    # 卷積層2
    conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
    
    # 卷積層3
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
    
    # 展平
    flatten = tf.keras.layers.Flatten()(conv3)
    
    # 全連接層
    fc1 = tf.keras.layers.Dense(512, activation='relu')(flatten)
    
    # Dropout層 (防止過擬合,與DRL論文一致)
    dropout = tf.keras.layers.Dropout(0.5)(fc1)
    
    # 輸出層 - Q值
    readout = tf.keras.layers.Dense(num_actions, activation='linear')(dropout)
    
    model = tf.keras.Model(inputs=s, outputs=readout)
    return model

def preprocess_state(state):
    """
    預處理狀態 (與DRL論文一致)
    """
    # 確保是灰階圖像
    if len(state.shape) == 2:
        state = np.expand_dims(state, axis=-1)
    
    # 正規化到 [0, 1]
    state = state.astype(np.float32) / 255.0
    
    return state

def select_action_epsilon_greedy(model, state, epsilon):
    """
    使用epsilon-greedy策略選擇動作 (與DRL論文一致)
    """
    if np.random.random() < epsilon:
        # 探索: 隨機選擇動作
        return np.random.randint(0, ACTIONS)
    else:
        # 利用: 選擇Q值最大的動作
        state_batch = np.expand_dims(state, axis=0)
        q_values = model.predict(state_batch, verbose=0)[0]
        return np.argmax(q_values)

def select_action_bayesian(model, state, dropout_rate):
    """
    使用Bayesian策略選擇動作 (與DRL論文的訓練策略一致)
    通過dropout層來表示動作的不確定性
    """
    if dropout_rate < 1.0:
        # 使用dropout來選擇最不確定的動作
        state_batch = np.expand_dims(state, axis=0)
        
        # 進行多次前向傳播以獲取不確定性
        num_samples = 10
        q_samples = []
        for _ in range(num_samples):
            q_values = model.predict(state_batch, verbose=0)[0]
            q_samples.append(q_values)
        
        q_samples = np.array(q_samples)
        q_std = np.std(q_samples, axis=0)
        
        # 選擇最不確定的動作
        return np.argmax(q_std)
    else:
        # 完全利用: 選擇Q值最大的動作
        state_batch = np.expand_dims(state, axis=0)
        q_values = model.predict(state_batch, verbose=0)[0]
        return np.argmax(q_values)

def test_dual_robot_drl():
    """
    測試雙機器人DRL探索
    """
    print("="*60)
    print("雙機器人DRL探索測試")
    print("="*60)
    print(f"關鍵參數設置:")
    print(f"  - 動作空間大小: {ACTIONS}")
    print(f"  - 感測器範圍: 50 (sensor_range)")
    print(f"  - 移動步長: 5 (movement_step / step size)")
    print(f"  - 機器人尺寸: 2 (robot_size)")
    print(f"  - 局部地圖大小: 40 (local_size)")
    print("="*60)
    
    # 創建環境
    env = robot_sim.DualRobotEnv(index_map=0, train=TRAIN, plot=PLOT)
    
    # 創建兩個DRL控制器 (使用相同的網路結構)
    print("\n創建DRL控制器...")
    model_robot1 = create_CNN(ACTIONS)
    model_robot2 = create_CNN(ACTIONS)
    
    # 如果有預訓練模型,載入權重
    if os.path.exists(network_dir):
        try:
            model_robot1.load_weights(network_dir + '/model_robot1.h5')
            model_robot2.load_weights(network_dir + '/model_robot2.h5')
            print("成功載入預訓練模型")
        except:
            print("警告: 無法載入預訓練模型,使用隨機初始化")
    else:
        print("警告: 未找到預訓練模型,使用隨機初始化")
    
    model_robot1.compile(optimizer='adam', loss='mse')
    model_robot2.compile(optimizer='adam', loss='mse')
    
    # 初始化環境
    print("\n開始探索...")
    state1, state2 = env.begin()
    state1 = preprocess_state(state1)
    state2 = preprocess_state(state2)
    
    # 測試參數
    epsilon = 0.05  # 測試時使用小的探索率
    max_steps = 5000
    step_count = 0
    total_reward_1 = 0
    total_reward_2 = 0
    
    # 統計資訊
    collision_count_1 = 0
    collision_count_2 = 0
    
    print(f"\n開始測試 (最大步數: {max_steps})")
    print("-"*60)
    
    while step_count < max_steps:
        # 機器人1選擇動作
        action1 = select_action_epsilon_greedy(model_robot1, state1, epsilon)
        
        # 機器人2選擇動作
        action2 = select_action_epsilon_greedy(model_robot2, state2, epsilon)
        
        # 執行動作
        result = env.step(action1, action2)
        (next_state1, next_state2, reward1, reward2, 
         terminal1, terminal2, complete, new_loc1, new_loc2) = result
        
        # 預處理下一個狀態
        next_state1 = preprocess_state(next_state1)
        next_state2 = preprocess_state(next_state2)
        
        # 累積獎勵
        total_reward_1 += reward1
        total_reward_2 += reward2
        
        # 統計碰撞
        if reward1 == -1:
            collision_count_1 += 1
        if reward2 == -1:
            collision_count_2 += 1
        
        # 更新狀態
        state1 = next_state1
        state2 = next_state2
        
        step_count += 1
        
        # 每100步顯示進度
        if step_count % 100 == 0:
            explored_ratio = np.size(np.where(env.shared_op_map == 255)) / np.size(np.where(env.global_map == 1))
            print(f"步數: {step_count:4d} | 探索率: {explored_ratio:.2%} | "
                  f"獎勵(R1): {total_reward_1:6.2f} | 獎勵(R2): {total_reward_2:6.2f} | "
                  f"碰撞(R1): {collision_count_1:3d} | 碰撞(R2): {collision_count_2:3d}")
        
        # 檢查是否完成
        if complete:
            print(f"\n探索完成! 總步數: {step_count}")
            break
        
        # 如果需要重新定位
        if new_loc1 or new_loc2:
            if TRAIN:
                # 訓練模式: 重新開始
                state1, state2 = env.reset()
                state1 = preprocess_state(state1)
                state2 = preprocess_state(state2)
            # 測試模式: 繼續
    
    # 最終統計
    print("\n" + "="*60)
    print("測試完成!")
    print("="*60)
    final_explored = np.size(np.where(env.shared_op_map == 255)) / np.size(np.where(env.global_map == 1))
    print(f"最終探索率: {final_explored:.2%}")
    print(f"總步數: {step_count}")
    print(f"機器人1總獎勵: {total_reward_1:.2f}")
    print(f"機器人2總獎勵: {total_reward_2:.2f}")
    print(f"總獎勵 (兩機器人): {total_reward_1 + total_reward_2:.2f}")
    print(f"機器人1碰撞次數: {collision_count_1}")
    print(f"機器人2碰撞次數: {collision_count_2}")
    print(f"平均步數獎勵: {(total_reward_1 + total_reward_2) / step_count:.4f}")
    print("="*60)
    
    if PLOT:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    test_dual_robot_drl()