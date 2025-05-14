import csv
import datetime
import os
from collections import defaultdict

import numpy as np
import traci
from sumolib import checkBinary

from sumoClass import SumoClass


class FixedTimeEvaluator:
    """
    Đánh giá hiệu suất của bộ điều khiển đèn giao thông fixed-time
    sử dụng cấu hình gốc từ file SUMO
    """
    
    def __init__(self, output_dir="fixed_time_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Định nghĩa cấu trúc pha theo file cấu hình
        self.phase_durations = [31, 31, 15, 15, 10, 10, 4, 4]
        
        # Lưu trữ thống kê giao thông
        self.last_avg_speed = 0
        self.last_queue_length = 0
    
    def calculate_metrics(self, traci):
        # Thu thập metrics giao thông hiện tại
        queue_length = (traci.edge.getLastStepHaltingNumber('1i') + 
                        traci.edge.getLastStepHaltingNumber('2i') + 
                        traci.edge.getLastStepHaltingNumber('3i') + 
                        traci.edge.getLastStepHaltingNumber('4i'))
        
        co2_emission = (traci.edge.getFuelConsumption('1i') + 
                       traci.edge.getFuelConsumption('2i') + 
                       traci.edge.getFuelConsumption('3i') + 
                       traci.edge.getFuelConsumption('4i')) / 1000
        
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
        
        # Lưu trữ để sử dụng sau này
        self.last_avg_speed = avg_speed
        self.last_queue_length = queue_length
        
        # Tính reward
        co2_penalty = -0.25 * co2_emission
        queue_penalty = -0.4 * queue_length
        waiting_penalty = -0.45 * waiting_time
        speed_reward = 0.5 * avg_speed
        throughput_reward = 0.2 * throughput
        
        reward = co2_penalty + queue_penalty + waiting_penalty + speed_reward + throughput_reward
        
        # Phạt nặng cho tình trạng kẹt xe
        if queue_length > 35 and avg_speed < 1.0:
            reward -= 50
        
        return reward, co2_emission, queue_length, waiting_time, avg_speed, throughput
    
    def run_simulation(self, episode, seed=None):
        # Khởi tạo biến theo dõi
        stepz = 0
        total_reward = 0
        co2_emission = 0
        queue_length = 0
        waiting_time = 0
        avg_speed = 0
        throughput = 0
        
        # Khởi tạo SumoClass
        sumoObject = SumoClass(4, 4, 3)
        sumoBinary = checkBinary('sumo')
        
        # Đặt seed nếu có
        if seed is not None:
            np.random.seed(seed)
            
        # Tạo route file
        sumoObject.generate_routefile()
        
        # Khởi động mô phỏng
        traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--start", 
                    "--tripinfo-output", f"tripdatad/tripinfo_fixed_time_ep{episode}.xml", 
                    "--quit-on-end", "--random"])
        
        # Lặp cho đến khi hết xe hoặc đạt giới hạn bước
        while traci.simulation.getMinExpectedNumber() > 0 and stepz < 10000:
            # Thực hiện bước mô phỏng
            traci.simulationStep()
            stepz += 1
            
            # Tính toán các chỉ số giao thông hiện tại
            step_reward, step_co2, step_queue, step_wait, step_speed, step_throughput = self.calculate_metrics(traci)
            
            # Cập nhật tổng metrics
            total_reward += step_reward
            co2_emission += step_co2
            queue_length += step_queue
            waiting_time += step_wait
            avg_speed += step_speed
            throughput += step_throughput
        
        # Đóng traci
        traci.close(wait=False)
        
        # Tính toán các chỉ số trung bình
        avg_co2_emission = co2_emission / max(1, stepz)
        avg_queue_length = queue_length / max(1, stepz)
        avg_waiting_time = waiting_time / max(1, stepz)
        avg_speed_per_step = avg_speed / max(1, stepz)
        avg_throughput = throughput / max(1, stepz)
        
        # Kết quả
        return {
            'steps': stepz,
            'total_reward': total_reward,
            'avg_reward': total_reward / max(1, stepz),
            'avg_co2': avg_co2_emission,
            'avg_queue': avg_queue_length,
            'avg_waiting': avg_waiting_time,
            'avg_speed': avg_speed_per_step,
            'avg_throughput': avg_throughput
        }
    
    def evaluate(self, num_episodes=20):
        # Khởi tạo kết quả
        all_metrics = {
            'total_reward': [],
            'avg_reward': [],
            'avg_co2': [],
            'avg_queue': [],
            'avg_waiting': [],
            'avg_speed': [],
            'avg_throughput': []
        }
        
        # Tạo file log metrics
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = os.path.join(self.output_dir, f"fixed_time_metrics_{timestamp}.csv")
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Episode', 'Steps', 'Total_Reward', 'Avg_Reward', 'CO2', 'Queue', 'Wait', 'Speed', 'Throughput'
            ])
        
        # Đánh giá qua nhiều episode
        for i in range(num_episodes):
            print(f"Đánh giá Fixed-time Controller - Episode {i+1}/{num_episodes}")
            
            # Chạy mô phỏng
            results = self.run_simulation(i+1)
            
            # Lưu metrics
            for key in all_metrics:
                all_metrics[key].append(results[key])
            
            # Ghi metrics vào file
            with open(metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    i+1,
                    results['steps'],
                    results['total_reward'],
                    results['avg_reward'],
                    results['avg_co2'],
                    results['avg_queue'],
                    results['avg_waiting'],
                    results['avg_speed'],
                    results['avg_throughput']
                ])
            
            # In kết quả cho episode này
            print(f"  Total Reward: {results['total_reward']:.2f}, Avg Reward: {results['avg_reward']:.4f}")
            print(f"  CO2: {results['avg_co2']:.4f}, Queue: {results['avg_queue']:.2f}, Wait: {results['avg_waiting']:.2f}")
            print(f"  Speed: {results['avg_speed']:.2f}, Throughput: {results['avg_throughput']:.2f}")
        
        # Tính toán chỉ số trung bình tổng thể
        overall_avg = {metric: np.mean(values) for metric, values in all_metrics.items()}
        overall_std = {metric: np.std(values) for metric, values in all_metrics.items()}
        
        # In kết quả tổng thể
        summary = "\n===== KET QUA DANH GIA FIXED-TIME CONTROLLER =====\n"
        summary += f"So episodes: {num_episodes}\n"
        summary += f"Total Reward: {overall_avg['total_reward']:.2f} ± {overall_std['total_reward']:.2f}\n"
        summary += f"Avg Reward: {overall_avg['avg_reward']:.4f} ± {overall_std['avg_reward']:.4f}\n"
        summary += f"CO2 Emission: {overall_avg['avg_co2']:.4f} ± {overall_std['avg_co2']:.4f}\n"
        summary += f"Queue Length: {overall_avg['avg_queue']:.2f} ± {overall_std['avg_queue']:.2f}\n"
        summary += f"Waiting Time: {overall_avg['avg_waiting']:.2f} ± {overall_std['avg_waiting']:.2f}\n"
        summary += f"Average Speed: {overall_avg['avg_speed']:.2f} ± {overall_std['avg_speed']:.2f}\n"
        summary += f"Throughput: {overall_avg['avg_throughput']:.2f} ± {overall_std['avg_throughput']:.2f}\n"
        
        # In ra màn hình
        print(summary)
        
        # Lưu kết quả vào file txt
        summary_file = os.path.join(self.output_dir, f"fixed_time_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"Kết quả chi tiết đã được lưu vào: {metrics_file}")
        print(f"Tóm tắt kết quả đã được lưu vào: {summary_file}")


def main():
    # Các hằng số được đặt trực tiếp
    NUM_EPISODES = 100
    OUTPUT_DIR = 'fixed_time_results'
    
    # Khởi tạo evaluator và đánh giá
    evaluator = FixedTimeEvaluator(output_dir=OUTPUT_DIR)
    evaluator.evaluate(num_episodes=NUM_EPISODES)


if __name__ == "__main__":
    main()