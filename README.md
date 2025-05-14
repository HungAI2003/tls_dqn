# tls_dqn
Traffic Light Smart Controller With Deep Q-Learning
Dsoft-Team
Hung Nguyen
Kien Bui
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
