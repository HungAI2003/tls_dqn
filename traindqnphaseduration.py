from __future__ import absolute_import, print_function

import csv
import datetime
import os
import random
import sys
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traci
from sumolib import checkBinary

from sumoClass import SumoClass


class PhaseTracker:
    def __init__(self, log_dir="phase_logs"):
        self.phase_selections = []
        self.phase_durations = defaultdict(int)
        self.phase_counts = defaultdict(int)
        
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{self.log_dir}/phase_log_{timestamp}.csv"
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Step", "CurrentPhase", "SelectedPhase", "Duration", "TransitionPhase", "TransitionTime"])
    
    def log_selection(self, episode, step, current_phase, selected_phase, duration, transition_phase=None, transition_time=None):

        self.phase_selections.append((episode, step, current_phase, selected_phase, duration, transition_phase, transition_time))

        self.phase_counts[selected_phase] += 1
        self.phase_durations[duration] += 1
        
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, step, current_phase, selected_phase, duration, transition_phase, transition_time])
        
        if transition_phase is not None:
            print(f"[Ep {episode}, Step {step}] Chuyển từ pha {current_phase} -> {selected_phase} qua pha vàng {transition_phase} ({transition_time} bước)")
        else:
            print(f"[Ep {episode}, Step {step}] Giữ nguyên pha {selected_phase} trong {duration} bước")

def calculate_reward(traci, previous_state=None):
    co2_emission1 = 0
    
    queue_length = (traci.edge.getLastStepHaltingNumber('1i') + 
                    traci.edge.getLastStepHaltingNumber('2i') + 
                    traci.edge.getLastStepHaltingNumber('3i') + 
                    traci.edge.getLastStepHaltingNumber('4i'))
    
    co2_emission1 += (traci.edge.getFuelConsumption('1i') + 
                     traci.edge.getFuelConsumption('2i') + 
                     traci.edge.getFuelConsumption('3i') + 
                     traci.edge.getFuelConsumption('4i'))
    co2_emission = co2_emission1 / 1000
    
    waiting_time = (traci.edge.getWaitingTime('1i') + 
                    traci.edge.getWaitingTime('2i') + 
                    traci.edge.getWaitingTime('3i') + 
                    traci.edge.getWaitingTime('4i'))
    
    throughput = (traci.edge.getLastStepVehicleNumber('1i') + 
                traci.edge.getLastStepVehicleNumber('2i') + 
                traci.edge.getLastStepVehicleNumber('3i') + 
                traci.edge.getLastStepVehicleNumber('4i'))
    
    avg_speed = (traci.edge.getLastStepMeanSpeed('1i') + 
                traci.edge.getLastStepMeanSpeed('2i') + 
                traci.edge.getLastStepMeanSpeed('3i') + 
                traci.edge.getLastStepMeanSpeed('4i')) / 4.0
    
    co2_penalty = -0.25 * co2_emission
    queue_penalty = -0.4 * queue_length
    waiting_penalty = -0.45 * waiting_time
    speed_reward = 0.5 * avg_speed  
    throughput_reward = 0.2 * throughput  
    
    reward = co2_penalty + queue_penalty + waiting_penalty + speed_reward + throughput_reward
    
    if queue_length > 35 and avg_speed < 1.0:
        reward -= 50  
    
    return reward, co2_emission, queue_length, waiting_time, avg_speed, throughput

def calculate_transition_time(avg_speed, queue_length):
    # """
    # Tính toán thời gian chuyển tiếp dựa trên tốc độ và chiều dài hàng đợi
    # Trả về số bước mô phỏng cho đèn vàng
    # """
    base_time = 3
    
    # Tăng thời gian nếu tốc độ cao (xe cần nhiều thời gian để dừng an toàn)
    if avg_speed > 10.0:  # m/s
        base_time += 2
    elif avg_speed > 5.0:
        base_time += 1
    
    # Giảm thời gian nếu hàng đợi dài (ưu tiên giải phóng tắc nghẽn)
    if queue_length > 15:
        base_time = max(2, base_time - 1)
    
    # Giữ thời gian chuyển tiếp trong khoảng hợp lý: 2-6 bước
    return max(2, min(6, base_time))

class DQN(nn.Module):
    def __init__(self, action_size=24):
        super(DQN, self).__init__()
        #  input (1,12,12)
        self.conv1_b1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2_b1 = nn.Conv2d(16, 32, kernel_size=2, stride=1)

        self.conv1_b2 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2_b2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        
        self.fc1 = nn.Linear(1032, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)  

    def forward(self, x1, x2, x3):

        x1 = F.relu(self.conv1_b1(x1))
        x1 = F.relu(self.conv2_b1(x1))
        x1 = torch.flatten(x1, 1)
        
        x2 = F.relu(self.conv1_b2(x2))
        x2 = F.relu(self.conv2_b2(x2))
        x2 = torch.flatten(x2, 1)
        
        x3 = torch.flatten(x3, 1)
        
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TLSClass:
    def __init__(self):
        self.gamma = 0.95
        self.epsilon = 1.0
        self.learning_rate = 0.001
        self.memory = deque(maxlen=10000)
        
        self.phases = [0, 1, 2, 3, 4, 5]  # 6 pha chính
        
        self.durations = [15, 20, 30, 40]  # 4 tùy chọn thời gian
        
        self.action_size = len(self.phases) * len(self.durations)  # 6 * 4 = 24 hành động
        
        self.action_map = {}
        idx = 0
        for phase in self.phases:
            for duration in self.durations:
                self.action_map[idx] = (phase, duration)
                idx += 1
        
        self.model = DQN(action_size=self.action_size)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        
        self.last_avg_speed = 0
        self.last_queue_length = 0

    def decay_epsilon(self):
        # """Giảm dần epsilon sau mỗi episode"""
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            phase, duration = self.get_phase_and_duration(action)
            print(f"[Khám phá] Chọn ngẫu nhiên hành động {action}: Pha {phase}, Thời gian {duration}")
            return action
        else:
            state0_array = np.array(state[0])
            if state0_array.ndim == 3:
                input1 = torch.tensor(state0_array, dtype=torch.float32) \
                            .unsqueeze(0).permute(0, 3, 1, 2)
            elif state0_array.ndim == 4:
                input1 = torch.tensor(state0_array, dtype=torch.float32) \
                            .permute(0, 3, 1, 2)
            elif state0_array.ndim == 5:
                new_shape = (state0_array.shape[0],
                            state0_array.shape[1],
                            state0_array.shape[2],
                            state0_array.shape[3] * state0_array.shape[4])
                input1 = torch.tensor(state0_array.reshape(new_shape), dtype=torch.float32) \
                            .permute(0, 3, 1, 2)
            else:
                raise ValueError("Unexpected dimensions for state[0]: " + str(state0_array.shape))
            
            state1_array = np.array(state[1])
            if state1_array.ndim == 3:
                input2 = torch.tensor(state1_array, dtype=torch.float32) \
                            .unsqueeze(0).permute(0, 3, 1, 2)
            elif state1_array.ndim == 4:
                input2 = torch.tensor(state1_array, dtype=torch.float32) \
                            .permute(0, 3, 1, 2)
            else:
                raise ValueError("Unexpected dimensions for state[1]: " + str(state1_array.shape))
            
            input3 = torch.tensor(state[2], dtype=torch.float32)
            if input3.ndim == 1:
                input3 = input3.unsqueeze(0)  # Tạo batch_size = 1
            elif input3.ndim == 2 and input3.shape[0] != 1:
                input3 = input3.reshape(1, -1)  # Tạo batch_size = 1        
            
            with torch.no_grad():
                q_values = self.model(input1, input2, input3)
            
            action = torch.argmax(q_values).item()
            phase, duration = self.get_phase_and_duration(action)
            print(f"[Khai thác] Chọn tối ưu hành động {action}: Pha {phase}, Thời gian {duration}")
            return action

    def get_phase_and_duration(self, action):
        return self.action_map[action]
        
    def update_traffic_stats(self, avg_speed, queue_length):
        # """
        # Cập nhật thống kê lưu lượng để tính toán thời gian chuyển tiếp
        # """
        self.last_avg_speed = avg_speed
        self.last_queue_length = queue_length

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = [m[0] for m in minibatch]
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch], dtype=np.float32)
        next_states = [m[3] for m in minibatch]
        dones = np.array([m[4] for m in minibatch], dtype=np.float32)
        
        def process_state(x):
            t = torch.tensor(x, dtype=torch.float32)
            if t.dim() == 3:
                return t.permute(2, 0, 1)
            elif t.dim() == 4:
                return t.squeeze(0).permute(2, 0, 1)
            else:
                raise ValueError("Unexpected tensor dimension: " + str(t.dim()))
        
        input1_batch = torch.stack([process_state(s[0]) for s in states])
        input2_batch = torch.stack([process_state(s[1]) for s in states])
        
        input3_batch = torch.stack([
            torch.tensor(s[2], dtype=torch.float32).view(-1)
            for s in states
        ])
        
        next_input1_batch = torch.stack([process_state(ns[0]) for ns in next_states])
        next_input2_batch = torch.stack([process_state(ns[1]) for ns in next_states])
        next_input3_batch = torch.stack([
            torch.tensor(ns[2], dtype=torch.float32).view(-1)
            for ns in next_states
        ])
        
        current_q = self.model(input1_batch, input2_batch, input3_batch)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        current_q_selected = current_q.gather(1, actions_tensor).squeeze(1)
        
        with torch.no_grad():
            next_q = self.model(next_input1_batch, next_input2_batch, next_input3_batch)
            max_next_q = next_q.max(1)[0]
            targets = torch.tensor(rewards) + self.gamma * max_next_q * (1 - torch.tensor(dones))
        
        loss = F.mse_loss(current_q_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
    
    def save(self, name):
        torch.save(self.model.state_dict(), name)
    
def train_traffic_controller(num_episodes=920, batch_size=32):
    from collections import defaultdict

    phase_tracker = PhaseTracker()
    
    nodes = 4
    edges = 4
    lanes = 3
    
    sumoObject = SumoClass(nodes, edges, lanes)
    options = sumoObject.get_options()
    sumoBinary = checkBinary('sumo')  # Luôn chạy SUMO không giao diện
    
    traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--start", "--tripinfo-output", "tripinfo.xml","--quit-on-end"])
    sumoObject.generate_routefile()
    tlsObject = TLSClass()
    
    edges = sumoObject.get_edges()
    lanes = sumoObject.get_lanes()
    traci.close()

    episode_rewards = []
    episode_co2_emissions = []
    episode_queue_lengths = []
    episode_waiting_times = []
    episode_speeds = []
    episode_throughputs = []
    episode_durations = []  # Theo dõi thời gian trung bình của pha
    episode_transitions = []  # Theo dõi thời gian trung bình của pha chuyển tiếp

    for episode in range(num_episodes):
        print(f"\n===== Bắt đầu episode {episode + 1}/{num_episodes} =====")
        stepz = 0
        episode_reward = 0
        queue_length = 0
        waiting_time = 0
        co2_emission = 0
        avg_speed = 0
        throughput = 0
        
        # Theo dõi số lượng và thời gian của pha và chuyển tiếp
        phase_durations = []
        transition_durations = []
        
        phase_counts = defaultdict(int)
        duration_counts = defaultdict(int)
        
        episode_metrics = []
        traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--start", "--tripinfo-output", f"tripdatad/tripinfo_ep{episode+1}.xml","--quit-on-end"])
        
        # Khởi tạo trạng thái đèn giao thông
        traci.trafficlight.setPhase("0", 0)
        traci.trafficlight.setPhaseDuration("0", 200)  # Đặt thời gian đủ dài để không tự động chuyển pha
        
        while traci.simulation.getMinExpectedNumber() > 0 and stepz < 10000:
            traci.simulationStep()
            state = sumoObject.get_state()
            action = tlsObject.act(state)
            
            # Lấy pha và thời gian từ hành động
            phase, duration = tlsObject.get_phase_and_duration(action)
            
            # Theo dõi phân bố
            phase_counts[phase] += 1
            duration_counts[duration] += 1
            
            # Lấy pha hiện tại
            current_phase = int(np.argmax(state[2].reshape(8,)))
            
            # Tính toán các chỉ số giao thông hiện tại
            reward, step_co2, step_queue, step_wait, step_speed, step_throughput = calculate_reward(traci)
            
            # Cập nhật thống kê giao thông
            tlsObject.update_traffic_stats(step_speed, step_queue)
            
            # Chỉ xét các pha chính (không xét pha đèn vàng 6, 7)
            if current_phase in tlsObject.phases:
                if current_phase == phase:
                    traci.trafficlight.setPhaseDuration("0", duration + 1)  # +1 cho bước hiện tại
                    
                    # Log lựa chọn pha
                    phase_tracker.log_selection(
                        episode=episode+1,
                        step=stepz,
                        current_phase=current_phase,
                        selected_phase=phase,
                        duration=duration
                    )
                    
                    # Thực hiện pha trong thời gian được chọn
                    for i in range(duration):
                        stepz += 1
                        traci.trafficlight.setPhase('0', phase)
                        traci.simulationStep()
                        
                        # Tính reward sau mỗi bước
                        step_reward, step_co2, step_queue, step_wait, step_speed, step_throughput = calculate_reward(traci)
                        episode_reward += step_reward
                        co2_emission += step_co2
                        queue_length += step_queue
                        waiting_time += step_wait
                        avg_speed += step_speed
                        throughput += step_throughput
                        episode_metrics.append([stepz, step_co2, step_queue, step_wait, step_speed, step_throughput, step_reward])
                    
                    # Lưu thời gian pha
                    phase_durations.append(duration)
                
                else:
                    # Cần chuyển sang pha mới
                    # Xác định pha đèn vàng phù hợp (6 hoặc 7) dựa trên pha hiện tại và pha mục tiêu
                    yellow_phase = 6  # Mặc định
                    
                    # Logic chọn pha đèn vàng dựa trên chuyển đổi pha
                    if (current_phase == 0 and phase in [1, 3, 5]) or \
                       (current_phase == 1 and phase in [0, 2, 4]) or \
                       (current_phase == 2 and phase in [1, 3, 5]) or \
                       (current_phase == 3 and phase in [0, 2, 4]) or \
                       (current_phase == 4 and phase in [1, 3, 5]) or \
                       (current_phase == 5 and phase in [0, 2, 4]):
                        yellow_phase = 7
                    else:
                        yellow_phase = 6
                    
                    # Tính toán thời gian chuyển tiếp 
                    transition_time = calculate_transition_time(tlsObject.last_avg_speed, tlsObject.last_queue_length)
                    transition_durations.append(transition_time)
                    
                    # Log lựa chọn pha
                    phase_tracker.log_selection(
                        episode=episode+1,
                        step=stepz,
                        current_phase=current_phase,
                        selected_phase=phase,
                        duration=duration,
                        transition_phase=yellow_phase,
                        transition_time=transition_time
                    )
                    
                    # Thực hiện pha đèn vàng
                    for i in range(transition_time):
                        stepz += 1
                        traci.trafficlight.setPhase('0', yellow_phase)
                        traci.simulationStep()
                        
                        # Tính reward với trọng số thấp hơn cho pha chuyển tiếp
                        step_reward, step_co2, step_queue, step_wait, step_speed, step_throughput = calculate_reward(traci)
                        episode_reward += step_reward * 0.5  # Giảm 50% reward cho pha chuyển tiếp
                        co2_emission += step_co2
                        queue_length += step_queue
                        waiting_time += step_wait
                        avg_speed += step_speed
                        throughput += step_throughput
                        episode_metrics.append([stepz, step_co2, step_queue, step_wait, step_speed, step_throughput, step_reward * 0.5])
                    
                    # Sau khi hoàn thành chuyển tiếp, thực hiện pha mới
                    traci.trafficlight.setPhaseDuration("0", duration + 1)
                    for i in range(duration):
                        stepz += 1
                        traci.trafficlight.setPhase('0', phase)
                        traci.simulationStep()
                        
                        # Tính reward sau mỗi bước
                        step_reward, step_co2, step_queue, step_wait, step_speed, step_throughput = calculate_reward(traci)
                        episode_reward += step_reward
                        co2_emission += step_co2
                        queue_length += step_queue
                        waiting_time += step_wait
                        avg_speed += step_speed
                        throughput += step_throughput
                        episode_metrics.append([stepz, step_co2, step_queue, step_wait, step_speed, step_throughput, step_reward])
                    
                    # Lưu thời gian pha
                    phase_durations.append(duration)
            
            # Lấy trạng thái mới
            new_state = sumoObject.get_state()
            
            # Tính toán reward cuối cùng cho hành động này
            reward, current_co2, current_queue, current_wait, current_speed, current_throughput = calculate_reward(traci)
            co2_emission += current_co2
            queue_length += current_queue
            waiting_time += current_wait
            avg_speed += current_speed
            throughput += current_throughput
            tlsObject.remember(state, action, reward, new_state, False)
            
            # Huấn luyện mô hình
            if len(tlsObject.memory) > batch_size:
                loss = tlsObject.replay(batch_size)
                if loss and (stepz % 100 == 0):
                    print(f"Bước {stepz}, Loss: {loss:.4f}")

        # Tính toán chỉ số trung bình cho episode này
        avg_co2_emission = co2_emission / max(1, stepz)
        avg_queue_length = queue_length / max(1, stepz)
        avg_waiting_time = waiting_time / max(1, stepz)
        avg_speed_per_step = avg_speed / max(1, stepz)
        avg_throughput = throughput / max(1, stepz)
        avg_phase_duration = sum(phase_durations) / max(1, len(phase_durations)) if phase_durations else 0
        avg_transition_duration = sum(transition_durations) / max(1, len(transition_durations)) if transition_durations else 0
        
        # Lưu chỉ số episode
        episode_rewards.append(episode_reward)
        episode_co2_emissions.append(avg_co2_emission)
        episode_queue_lengths.append(avg_queue_length)
        episode_waiting_times.append(avg_waiting_time)
        episode_speeds.append(avg_speed_per_step)
        episode_throughputs.append(avg_throughput)
        episode_durations.append(avg_phase_duration)
        episode_transitions.append(avg_transition_duration)
        
        # In kết quả
        print(f"\n===== Kết thúc episode {episode + 1} =====")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Average CO2 emission: {avg_co2_emission:.2f}")
        print(f"  Average queue length: {avg_queue_length:.2f}")
        print(f"  Average waiting time: {avg_waiting_time:.2f}")
        print(f"  Average speed: {avg_speed_per_step:.2f}")
        print(f"  Average throughput: {avg_throughput:.2f}")
        print(f"  Average phase duration: {avg_phase_duration:.2f}")
        print(f"  Average transition time: {avg_transition_duration:.2f}")
        
        # In phân bố pha đèn
        print("\n  Phân bố pha đèn:")
        total_phases = sum(phase_counts.values())
        for phase, count in sorted(phase_counts.items()):
            percent = (count / total_phases) * 100
            print(f"    Pha {phase}: {count} lần ({percent:.1f}%)")
        
        # In phân bố thời gian pha
        print("\n  Phân bố thời gian pha:")
        total_durations = sum(duration_counts.values())
        for duration, count in sorted(duration_counts.items()):
            percent = (count / total_durations) * 100
            print(f"    {duration} bước: {count} lần ({percent:.1f}%)")
        
        # Đặt cờ done cho trải nghiệm cuối cùng trong bộ nhớ
        if tlsObject.memory:
            mem = tlsObject.memory[-1]
            del tlsObject.memory[-1]
            tlsObject.memory.append((mem[0], mem[1], reward, mem[3], True))
        
        # Giảm epsilon cho khám phá
        tlsObject.decay_epsilon()
        print(f"  Epsilon hiện tại: {tlsObject.epsilon:.4f}")
        
        # Đóng SUMO
        traci.close(wait=False)
        
        # Lưu mô hình sau mỗi 50 episode
        if (episode + 1) % 50 == 0 or episode == num_episodes - 1:
            model_path = f'model_checkpoint_ep{episode+1}.pth'
            tlsObject.save(model_path)
            print(f"  Đã lưu mô hình tại: {model_path}")

    print("\n===== Huấn luyện hoàn tất! =====")

    # Lưu mô hình cuối cùng
    tlsObject.save('flexible_tls_model.pth')
    
    with open('flexible_tls_metrics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward', 'Avg_Co2', 'Avg_Queue', 'Avg_Wait', 'Avg_Speed', 
                         'Avg_Throughput', 'Avg_Phase_Duration', 'Avg_Transition_Time'])
        for i in range(num_episodes):
            writer.writerow([i, episode_rewards[i], episode_co2_emissions[i], episode_queue_lengths[i], 
                           episode_waiting_times[i], episode_speeds[i], episode_throughputs[i],
                           episode_durations[i], episode_transitions[i]])
    
    print("Đã lưu số liệu vào flexible_tls_metrics.csv")
    sys.stdout.flush()

if __name__ == "__main__":
    from collections import defaultdict
    train_traffic_controller(num_episodes=920, batch_size=32)