Dsoft-Team
Hùng Nguyễn
Kiên Bùi
Nam Dong

Inverse Reinforcement Learning cho Điều khiển Đèn Giao thông Thích ứng
Dự án này triển khai Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) để học hàm reward từ agent điều khiển đèn giao thông đã huấn luyện trước bằng Deep Q-Network (DQN). Sau đó, sử dụng hàm reward đã học để huấn luyện một agent điều khiển mới.
Tổng quan Dự án
Dự án sử dụng mô phỏng SUMO (Simulation of Urban MObility) để tái tạo một giao lộ với các luồng giao thông thực tế. Mục tiêu chính là:

Huấn luyện một agent DQN ban đầu để điều khiển thời gian và pha đèn giao thông
Thu thập dữ liệu từ agent đã huấn luyện
Học hàm reward ngầm bằng MaxEnt IRL
Huấn luyện lại agent DQN với hàm reward đã học
Đánh giá hiệu suất của agent mới

Cấu trúc Code
Dự án bao gồm các thành phần chính sau:
1. traindqnphaseduration.py
Module chính để huấn luyện DQN agent điều khiển thời gian và pha đèn giao thông.

DQN: Mạng nơ-ron sâu cho học Q-value
TLSClass: Lớp quản lý agent điều khiển đèn giao thông
PhaseTracker: Theo dõi lựa chọn pha đèn
calculate_reward: Tính toán reward dựa trên các chỉ số giao thông
calculate_transition_time: Tính thời gian chuyển tiếp giữa các pha

2. datacollector.py
Thu thập dữ liệu từ mô hình DQN đã huấn luyện để sử dụng cho MaxEnt IRL.

MaxEntIRLDataCollector: Thu thập quỹ đạo và chỉ số giao thông từ agent đã huấn luyện
extract_features: Trích xuất đặc trưng từ trạng thái và chỉ số giao thông
collect_maxent_irl_data: Chức năng chính để thu thập dữ liệu

3. preparedatairl.py
Tiền xử lý dữ liệu đã thu thập để loại bỏ các quỹ đạo gridlock và tình huống bất thường.

filter_gridlock_data: Lọc dữ liệu gridlock dựa trên các ngưỡng cấu hình
create_filtered_copy: Tiện ích để tạo bản sao đã lọc của tệp H5 hiện có

4. DeepMAXIRL.py
Triển khai thuật toán Maximum Entropy IRL để học hàm reward từ dữ liệu đã thu thập.

MaxEntIRL: Lớp chính triển khai thuật toán MaxEnt IRL
load_trajectories_from_h5: Đọc quỹ đạo từ tệp H5
extract_feature_names: Tạo tên đặc trưng từ vector đặc trưng
analyze_reward_patterns: Phân tích mẫu reward từ mô hình đã học
train_and_evaluate_maxent_irl: Hàm chính để huấn luyện và đánh giá mô hình

5. trainagentdqn.py
Huấn luyện lại DQN agent với hàm reward đã học từ MaxEnt IRL.

IRLRewardCalculator: Tính toán reward dựa trên mô hình IRL đã học
train_traffic_controller_with_irl: Huấn luyện controller sử dụng hàm reward từ IRL

6. recollect_datacollector_IRLagent.py
Thu thập dữ liệu từ agent đã huấn luyện lại với hàm reward IRL.

IRL_MaxEntDataCollector: Thu thập dữ liệu từ agent với hàm reward IRL
collect_irl_maxent_data: Chức năng chính để thu thập dữ liệu

7. sumoClass.py
Lớp tiện ích để tương tác với môi trường mô phỏng SUMO.

SumoClass: Quản lý môi trường SUMO và cung cấp các phương thức truy cập trạng thái

Quy trình sử dụng
1. Huấn luyện mô hình DQN ban đầu
python traindqnphaseduration.py
2. Thu thập dữ liệu từ DQN đã huấn luyện
python datacollector.py
3. Lọc dữ liệu để loại bỏ gridlock
python preparedatairl.py
4. Huấn luyện mô hình MaxEnt IRL
python DeepMAXIRL.py
5. Huấn luyện lại DQN với hàm reward IRL
python trainagentdqn.py
6. Thu thập dữ liệu từ mô hình được huấn luyện với reward IRL
python recollect_datacollector_IRLagent.py