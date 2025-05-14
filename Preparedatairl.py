import os
import pickle
import shutil
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def filter_gridlock_data(input_h5, output_h5, output_meta=None, output_csv=None, plot_stats=True):
    """
    Lọc dữ liệu gridlock từ file H5 đã thu thập
    
    Args:
        input_h5: Đường dẫn đến file H5
        output_h5: Đường dẫn đến file H5 đã lọc 
        output_meta: Đường dẫn đến file meta mới (optional)
        output_csv: Đường dẫn đến file CSV metrics mới (optional)
        plot_stats: Có vẽ biểu đồ thống kê không
    
    Returns:
        output_h5: Đường dẫn đến file đã lọc
    """
    print(f"Đang lọc dữ liệu gridlock từ {input_h5}...")
    
    # Tạo thư mục chứa file đầu ra nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    
    # Cấu hình điều kiện phát hiện gridlock (đã tinh chỉnh)
    WAITING_THRESHOLD = 120        # Thời gian chờ > 120s (tăng lên)
    LOW_THROUGHPUT = 2.0           # Throughput < 2.0 (giữ nguyên)
    HIGH_THROUGHPUT = 35.0         # Throughput cao (thêm mới)
    ACCEPTABLE_WAITING = 150       # Mức waiting có thể chấp nhận với throughput cao (thêm mới)
    WAITING_MISMATCH = 60          # Thời gian chờ > 60s nhưng queue < 2 (tăng lên)
    REWARD_THRESHOLD = -700        # Reward quá thấp (điều chỉnh)
    QUEUE_THRESHOLD = 2.0          # Queue < 2.0 (giữ nguyên)
    SAME_PHASE_COUNT = 6           # Số lần lặp lại phase liên tiếp (tăng lên)
    GRIDLOCK_RATIO = 0.25          # Tỷ lệ transition bị gridlock để loại bỏ episode (giữ nguyên)
    MIN_VALID_RATIO = 0.7          # Tỷ lệ tối thiểu transition hợp lệ để giữ lại episode (giữ nguyên)
    
    # Thống kê
    total_episodes = 0
    kept_episodes = 0
    total_transitions = 0
    kept_transitions = 0
    episode_stats = []
    
    # Hàm kiểm tra gridlock cho một transition
    def is_gridlock(waiting_time, throughput, queue, reward):
        """
        Kiểm tra một transition có dấu hiệu gridlock không
        """
        # Thời gian chờ quá cao
        waiting_too_high = waiting_time > WAITING_THRESHOLD
        
        # Throughput thấp nhưng thời gian chờ cao
        throughput_waiting_mismatch = (throughput < LOW_THROUGHPUT and 
                                     waiting_time > WAITING_MISMATCH)
        
        # Queue thấp nhưng thời gian chờ cao (bất thường)
        queue_waiting_mismatch = (queue < QUEUE_THRESHOLD and 
                                waiting_time > WAITING_MISMATCH and
                                waiting_time > 80)  # Thêm ngưỡng tối thiểu
        
        # Reward quá thấp
        reward_too_low = reward < REWARD_THRESHOLD
        
        # THÊM MỚI: Loại trừ trường hợp có throughput cao với waiting time không quá cao
        high_throughput_exception = (throughput > HIGH_THROUGHPUT and 
                                  waiting_time < ACCEPTABLE_WAITING)
        
        # Phát hiện gridlock, trừ trường hợp ngoại lệ throughput cao
        result = (waiting_too_high or throughput_waiting_mismatch or 
               queue_waiting_mismatch or reward_too_low)
        
        # Nếu có exception và không phải waiting quá cao, bỏ qua
        if high_throughput_exception and not (waiting_time > WAITING_THRESHOLD + 50):
            result = False
            
        return result
    
    with h5py.File(input_h5, 'r') as src, h5py.File(output_h5, 'w') as dst:
        # Sao chép metadata
        if 'metadata' in src:
            src.copy('metadata', dst)
            # Sau này cập nhật lại num_episodes
        
        # Lấy danh sách các episode
        episode_keys = [k for k in src.keys() if k.startswith('episode_')]
        total_episodes = len(episode_keys)
        
        # Biến đếm episode mới
        new_episode_idx = 1
        
        # Xử lý từng episode
        for ep_key in episode_keys:
            ep_group = src[ep_key]
            
            # Thu thập các metrics và thông tin cần thiết
            metrics = ep_group['metrics'][:]
            rewards = ep_group['rewards'][:]
            action_phases = ep_group['action_phases'][:]
            
            # Tổng số transition trong episode này
            ep_transitions = len(metrics)
            total_transitions += ep_transitions
            
            print(f"\nĐang phân tích {ep_key} ({ep_transitions} transitions)...")
            
            # Danh sách các transition có vấn đề
            problematic_indices = []
            
            # Kiểm tra các transition riêng lẻ
            for i in range(len(metrics)):
                # Trích xuất các metrics
                co2 = metrics[i][0]
                queue = metrics[i][1]
                waiting_time = metrics[i][2]
                speed = metrics[i][3]
                throughput = metrics[i][4]
                reward = rewards[i]
                
                # Kiểm tra các điều kiện gridlock với hàm mới
                if is_gridlock(waiting_time, throughput, queue, reward):
                    problematic_indices.append(i)
                    if len(problematic_indices) <= 5:  # Chỉ in ra 5 mục đầu tiên để tránh quá nhiều output
                        print(f"  - Transition {i}: Gridlock phát hiện! [reward={reward:.2f}, waiting={waiting_time:.2f}, throughput={throughput:.2f}]")
            
            # Kiểm tra phase lặp lại liên tục
            if len(action_phases) >= SAME_PHASE_COUNT:
                for i in range(len(action_phases) - SAME_PHASE_COUNT + 1):
                    phases = action_phases[i:i+SAME_PHASE_COUNT]
                    if len(set(phases)) == 1:  # Cùng một phase được lặp lại nhiều lần
                        # Chỉ đánh dấu là gridlock nếu throughput không cao hoặc waiting time cao
                        # Lấy trung bình thời gian chờ và throughput cho đoạn phase lặp lại
                        avg_waiting = np.mean([metrics[j][2] for j in range(i, i+SAME_PHASE_COUNT)])
                        avg_throughput = np.mean([metrics[j][4] for j in range(i, i+SAME_PHASE_COUNT)])
                        
                        if avg_throughput < HIGH_THROUGHPUT or avg_waiting > ACCEPTABLE_WAITING:
                            for j in range(i, i+SAME_PHASE_COUNT):
                                if j not in problematic_indices:
                                    problematic_indices.append(j)
                            if len(problematic_indices) <= 10:
                                print(f"  - Transitions {i}-{i+SAME_PHASE_COUNT-1}: Phase {phases[0]} lặp lại liên tục")
            
            # Loại bỏ các indices trùng lặp và sắp xếp
            problematic_indices = sorted(set(problematic_indices))
            
            # Kiểm tra tỷ lệ transition có vấn đề
            gridlock_ratio = len(problematic_indices) / len(metrics)
            
            # Quyết định giữ hay loại bỏ episode
            if gridlock_ratio > GRIDLOCK_RATIO:
                print(f"  → Loại bỏ {ep_key}: {len(problematic_indices)}/{len(metrics)} transitions bị gridlock ({gridlock_ratio*100:.1f}%)")
                continue
            
            # Xác định các transition hợp lệ
            valid_indices = [i for i in range(len(metrics)) if i not in problematic_indices]
            
            # Kiểm tra số lượng transition hợp lệ còn lại
            if len(valid_indices) < len(metrics) * MIN_VALID_RATIO:
                print(f"  → Loại bỏ {ep_key}: Quá ít transition hợp lệ còn lại ({len(valid_indices)}/{len(metrics)})")
                continue
            
            print(f"  → Giữ lại {ep_key} với {len(valid_indices)}/{len(metrics)} transitions hợp lệ")
            kept_transitions += len(valid_indices)
            
            # Tạo nhóm mới cho episode này với chỉ số được đánh số lại
            new_ep_key = f'episode_{new_episode_idx}'
            new_ep_group = dst.create_group(new_ep_key)
            new_episode_idx += 1
            kept_episodes += 1
            
            # Cập nhật thuộc tính
            for attr_name, attr_value in ep_group.attrs.items():
                if attr_name == 'num_transitions':
                    new_ep_group.attrs[attr_name] = len(valid_indices)
                else:
                    new_ep_group.attrs[attr_name] = attr_value
            
            # Sao chép các dataset, chỉ giữ lại những transition hợp lệ
            for name in ep_group.keys():
                if name not in ['states', 'next_states']:
                    data = ep_group[name][:]
                    new_data = data[valid_indices]
                    new_ep_group.create_dataset(name, data=new_data)
            
            # Xử lý state và next_state
            if 'states' in ep_group and 'next_states' in ep_group:
                state_group = new_ep_group.create_group('states')
                next_state_group = new_ep_group.create_group('next_states')
                
                for new_idx, orig_idx in enumerate(valid_indices):
                    # Sao chép state
                    orig_state_key = f'transition_{orig_idx}'
                    if orig_state_key in ep_group['states']:
                        orig_state_group = ep_group['states'][orig_state_key]
                        new_state_group = state_group.create_group(f'transition_{new_idx}')
                        
                        for key in orig_state_group.keys():
                            new_state_group.create_dataset(key, data=orig_state_group[key][:])
                    
                    # Sao chép next_state
                    if orig_state_key in ep_group['next_states']:
                        orig_next_state_group = ep_group['next_states'][orig_state_key]
                        new_next_state_group = next_state_group.create_group(f'transition_{new_idx}')
                        
                        for key in orig_next_state_group.keys():
                            new_next_state_group.create_dataset(key, data=orig_next_state_group[key][:])
            
            # Thu thập thống kê về episode này để báo cáo
            episode_stats.append({
                'original_id': int(ep_key.split('_')[1]),
                'new_id': new_episode_idx - 1,
                'original_transitions': len(metrics),
                'kept_transitions': len(valid_indices),
                'gridlock_count': len(problematic_indices)
            })
        
        # Cập nhật metadata
        if 'metadata' in dst:
            dst['metadata'].attrs['num_episodes'] = kept_episodes
    
    # Xuất thống kê
    print("\n===== Thống kê lọc dữ liệu =====")
    print(f"Tổng số episode ban đầu: {total_episodes}")
    print(f"Số episode giữ lại: {kept_episodes} ({kept_episodes/total_episodes*100:.1f}%)")
    print(f"Tổng số transition ban đầu: {total_transitions}")
    print(f"Số transition giữ lại: {kept_transitions} ({kept_transitions/total_transitions*100:.1f}%)")
    
    # Cập nhật file meta nếu được chỉ định
    if output_meta:
        input_meta = None
        
        # Tìm file meta tương ứng với file H5 gốc
        possible_meta = Path(input_h5).parent / f"{Path(input_h5).stem.replace('data', 'meta')}.pkl"
        if os.path.exists(possible_meta):
            input_meta = possible_meta
        
        if input_meta and os.path.exists(input_meta):
            # Đọc dữ liệu meta
            with open(input_meta, 'rb') as f:
                meta_data = pickle.load(f)
            
            # Cập nhật meta data
            meta_data['num_episodes'] = kept_episodes
            
            # Lọc episode_stats
            if 'episode_stats' in meta_data:
                new_episode_stats = []
                for stat in meta_data['episode_stats']:
                    ep_id = stat.get('episode', 0)
                    for es in episode_stats:
                        if es['original_id'] == ep_id:
                            stat['transitions'] = es['kept_transitions']
                            new_episode_stats.append(stat)
                            break
                
                meta_data['episode_stats'] = new_episode_stats
            
            # Lưu meta data mới
            with open(output_meta, 'wb') as f:
                pickle.dump(meta_data, f)
            
            print(f"Đã cập nhật metadata: {output_meta}")
        else:
            print(f"Không tìm thấy file meta tương ứng với {input_h5}")
    
    # Cập nhật file CSV nếu được chỉ định
    if output_csv:
        input_csv = None
        
        # Tìm file CSV tương ứng với file H5 gốc
        possible_csv = Path(input_h5).parent / f"{Path(input_h5).stem.replace('data', 'metrics')}.csv"
        if os.path.exists(possible_csv):
            input_csv = possible_csv
        
        if input_csv and os.path.exists(input_csv):
            # Đọc file CSV
            df = pd.read_csv(input_csv)
            
            # Tạo dictionary ánh xạ episode gốc sang episode mới
            episode_mapping = {stat['original_id']: stat['new_id'] for stat in episode_stats}
            
            # Tạo danh sách các hàng cần giữ lại
            rows_to_keep = []
            
            for ep_id in set(df['Episode']):
                # Nếu episode này được giữ lại
                if ep_id in episode_mapping:
                    # Lấy các transition hợp lệ cho episode này
                    kept_indices = []
                    for stat in episode_stats:
                        if stat['original_id'] == ep_id:
                            # Đọc từ file H5 để lấy danh sách transition hợp lệ
                            with h5py.File(output_h5, 'r') as f:
                                new_ep_key = f'episode_{stat["new_id"]}'
                                if new_ep_key in f:
                                    features = f[new_ep_key]['features'][:]
                                    kept_indices = list(range(1, len(features) + 1))
                    
                    # Lọc các hàng thuộc episode này và transition nằm trong kept_indices
                    ep_rows = df[df['Episode'] == ep_id]
                    for idx, row in ep_rows.iterrows():
                        if row['Transition'] in kept_indices:
                            rows_to_keep.append(idx)
            
            # Lọc dataframe
            df_filtered = df.loc[rows_to_keep].copy()
            
            # Cập nhật lại số thứ tự Episode theo ánh xạ mới
            df_filtered['Episode'] = df_filtered['Episode'].map(lambda x: episode_mapping.get(x, x))
            
            # Cập nhật lại số thứ tự Transition - cần reset cho mỗi episode
            for ep_id in set(df_filtered['Episode']):
                mask = df_filtered['Episode'] == ep_id
                df_filtered.loc[mask, 'Transition'] = range(1, sum(mask) + 1)
            
            # Lưu file CSV mới
            df_filtered.to_csv(output_csv, index=False)
            print(f"Đã lưu file CSV đã lọc: {output_csv}")
        else:
            print(f"Không tìm thấy file CSV tương ứng với {input_h5}")
    
    # Vẽ biểu đồ thống kê nếu được yêu cầu
    if plot_stats and len(episode_stats) > 0:
        plt.figure(figsize=(12, 8))
        
        # Biểu đồ số lượng transition giữ lại cho mỗi episode
        plt.subplot(2, 1, 1)
        original_counts = [stat['original_transitions'] for stat in episode_stats]
        kept_counts = [stat['kept_transitions'] for stat in episode_stats]
        episode_ids = [stat['original_id'] for stat in episode_stats]
        
        x = range(len(episode_ids))
        plt.bar(x, original_counts, width=0.4, label='Original', alpha=0.7)
        plt.bar([i + 0.4 for i in x], kept_counts, width=0.4, label='Kept', alpha=0.7)
        plt.xlabel('Original Episode ID')
        plt.ylabel('Number of Transitions')
        plt.title('Transitions Before vs After Filtering')
        plt.xticks([i + 0.2 for i in x], episode_ids, rotation=90 if len(episode_ids) > 20 else 0)
        plt.legend()
        
        # Biểu đồ số lượng transition bị loại bỏ cho mỗi episode
        plt.subplot(2, 1, 2)
        gridlock_counts = [stat['gridlock_count'] for stat in episode_stats]
        gridlock_ratios = [stat['gridlock_count'] / stat['original_transitions'] * 100 for stat in episode_stats]
        
        plt.bar(x, gridlock_ratios, width=0.6)
        for i, count in enumerate(gridlock_counts):
            plt.text(i, gridlock_ratios[i] + 1, str(count), ha='center')
        
        plt.xlabel('Original Episode ID')
        plt.ylabel('Gridlock Percentage (%)')
        plt.title('Percentage of Gridlock Transitions per Episode')
        plt.xticks(x, episode_ids, rotation=90 if len(episode_ids) > 20 else 0)
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        plot_path = os.path.join(os.path.dirname(output_h5), 'gridlock_filtering_stats.png')
        plt.savefig(plot_path)
        print(f"Đã lưu biểu đồ thống kê tại: {plot_path}")
    
    print(f"\nĐã hoàn thành lọc dữ liệu. File kết quả: {output_h5}")
    return output_h5

# Hàm thuận tiện để sửa đổi file H5 hiện có mà không mất file gốc
def create_filtered_copy(input_h5, suffix="_filtered"):
    """
    Tạo bản sao lọc của file H5 với các tham số mặc định
    
    Args:
        input_h5: Đường dẫn đến file H5 gốc
        suffix: Hậu tố thêm vào tên file
        
    Returns:
        str: Đường dẫn đến file đã lọc
    """
    # Tạo tên file đầu ra
    base_path = os.path.splitext(input_h5)[0]
    output_h5 = f"{base_path}{suffix}.h5"
    output_meta = f"{base_path.replace('data', 'meta')}{suffix}.pkl"
    output_csv = f"{base_path.replace('data', 'metrics')}{suffix}.csv"
    
    # Lọc dữ liệu
    return filter_gridlock_data(
        input_h5=input_h5,
        output_h5=output_h5,
        output_meta=output_meta,
        output_csv=output_csv,
        plot_stats=True
    )

# Khi chạy trực tiếp


# Ví dụ sử dụng:
if __name__ == "__main__":
    # Đường dẫn input và output
    input_h5 = "maxent_irl_data/maxent_irl_data_20250421_010947.h5"
    output_h5 = "maxent_irl_data/maxent_irl_data_20250421_010947_filtered.h5"
    output_meta = "maxent_irl_data/maxent_irl_meta_20250421_010947_filtered.pkl"
    output_csv = "maxent_irl_data/maxent_irl_metrics_20250421_010947_filtered.csv"
    
    # Lọc dữ liệu
    filtered_h5 = filter_gridlock_data(
        input_h5=input_h5,
        output_h5=output_h5,
        output_meta=output_meta,
        output_csv=output_csv,
        plot_stats=True
    )