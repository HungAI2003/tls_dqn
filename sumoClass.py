from __future__ import absolute_import, print_function

import optparse
import os
import random
import sys

import numpy as np
import torch
import traci


class SumoClass:
    
    def __init__(self, nodes, edges, lanes):
        self.nodes = nodes
        self.edges = edges
        self.lanes = lanes
        self.main_phases = [0, 1, 2, 3, 4, 5]  # Các pha chính
        self.yellow_phases = [6, 7]  # Các pha đèn vàng
        self.all_phases = self.main_phases + self.yellow_phases

    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options
    
    def get_edges(self):
        edges = traci.edge.getIDList()
        edges = np.asarray(edges)
        return edges[0:7:2]

    def get_lanes(self):
        lanes = []
        edges = self.get_edges()
        for edge in edges:
            for i in range(self.lanes):
                lanes.append(edge + '_' + str(i))
        return lanes
    
    def generate_routefile(self):
        N = 3600  # Giữ N = 3600 (60 phút mô phỏng)

        pWE = 1. / 18
        pEW = 1. / 18
        pNS = 1. / 18
        pSN = 1. / 18

        pWN = 1. / 35
        pWS = 1. / 30
        pEN = 1. / 30
        pES = 1. / 35

        pNW = 1. / 35
        pSW = 1. / 30
        pNE = 1. / 30
        pSE = 1. / 35

        with open("data/cross.rou.xml", "w") as routes:
            print("""<routes>
                <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
                <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="20" guiShape="bus"/>
                
                <route id="right" edges="51o 1i 2o 52i" />
                <route id="left" edges="52o 2i 1o 51i" />
                <route id="down" edges="54o 4i 3o 53i" />
                <route id="up" edges="53o 3i 4o 54i" />
                
                <route id="right_up" edges="51o 1i 4o 54i" />
                <route id="right_down" edges="51o 1i 3o 53i" />
                <route id="left_up" edges="52o 2i 4o 54i" />
                <route id="left_down" edges="52o 2i 3o 53i" />
                
                <route id="up_right" edges="53o 3i 2o 52i" />
                <route id="up_left" edges="53o 3i 1o 51i" />
                <route id="down_right" edges="54o 4i 2o 52i" />
                <route id="down_left" edges="54o 4i 1o 51i" />
                """, file=routes)
            vehNr = 0
            for i in range(N):
                if random.uniform(0, 1) < pWE:
                    print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pEW:
                    print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pNS:
                    print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pSN:
                    print('    <vehicle id="up_%i" type="typeNS" route="up" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                
                if random.uniform(0, 1) < pWN:
                    print('    <vehicle id="left_up_%i" type="typeWE" route="left_up" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pWS:
                    print('    <vehicle id="left_down_%i" type="typeWE" route="left_down" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pEN:
                    print('    <vehicle id="right_up_%i" type="typeWE" route="right_up" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pES:
                    print('    <vehicle id="right_down_%i" type="typeWE" route="right_down" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                
                if random.uniform(0, 1) < pSE:
                    print('    <vehicle id="down_right_%i" type="typeNS" route="down_right" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pSW:
                    print('    <vehicle id="down_left_%i" type="typeNS" route="down_left" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pNE:
                    print('    <vehicle id="up_right_%i" type="typeNS" route="up_right" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pNW:
                    print('    <vehicle id="up_left_%i" type="typeNS" route="up_left" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1

            print("</routes>", file=routes)
            print(f"Tổng số xe được tạo trong episode: {vehNr}")

    def get_state(self):
        # """
        # Lấy trạng thái hiện tại của hệ thống giao thông để cung cấp cho DQN.
        # Bao gồm: ma trận vị trí, ma trận vận tốc, và vector đèn giao thông.
        # """
        edges = self.get_edges()
        lanes = self.get_lanes()
        
        cellLength = 7
        offset = 11
        speedLimit = 20
        
        positionMatrix = []
        velocityMatrix = []
        
        for i in range(12):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(12):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)
        
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('1i')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('2i')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('3i')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('4i')
        
        junctionPosition = traci.junction.getPosition('0')[0]  
        
        # Cập nhật ma trận vị trí và vận tốc cho xe trên đường 1
        for v in vehicles_road1:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
            if ind < 12:
                positionMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        # Cập nhật ma trận vị trí và vận tốc cho xe trên đường 2
        for v in vehicles_road2:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
            if ind < 12:
                positionMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        junctionPosition = traci.junction.getPosition('0')[1]

        # Cập nhật ma trận vị trí và vận tốc cho xe trên đường 3
        for v in vehicles_road3:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
            if ind < 12:
                positionMatrix[8 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[8 - traci.vehicle.getLaneIndex(v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit
        
        # Cập nhật ma trận vị trí và vận tốc cho xe trên đường 4
        for v in vehicles_road4:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[1] + offset)) / cellLength)
            if ind < 12:
                positionMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        # Lấy thông tin về pha đèn hiện tại
        phase = traci.trafficlight.getPhase('0')
        
        # Chuẩn bị vector one-hot encoding cho pha đèn (8 pha: 6 pha chính + 2 pha vàng)
        light = [0] * 8
        
        # Kiểm tra pha có hợp lệ không
        if 0 <= phase < 8:
            light[phase] = 1
        else:
            # Nếu không hợp lệ, mặc định pha 0
            light[0] = 1
            print(f"Cảnh báo: Pha đèn không hợp lệ: {phase}. Gán pha 0.")

        # Định dạng lại ma trận vị trí
        position = np.array(positionMatrix)
        position = position.reshape(1, 12, 12, 1)
        
        # Định dạng lại ma trận vận tốc
        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 12, 12, 1)
        
        # Định dạng lại vector đèn
        lights = np.array(light)
        lights = lights.reshape(1, 8, 1)

        return [position, velocity, lights]
    
    def get_current_phase(self):

        return traci.trafficlight.getPhase('0')
    
    def set_phase(self, phase_index, duration=None):

        if 0 <= phase_index < 8:
            traci.trafficlight.setPhase('0', phase_index)
            
            if duration is not None:
                traci.trafficlight.setPhaseDuration('0', duration)
            
            return True
        else:
            print(f"Lỗi: Pha đèn {phase_index} không hợp lệ. Phải nằm trong khoảng 0-7.")
            return False
    
    def get_traffic_metrics(self):

        # Tính tốc độ trung bình
        avg_speed = (traci.edge.getLastStepMeanSpeed('1i') + 
                    traci.edge.getLastStepMeanSpeed('2i') + 
                    traci.edge.getLastStepMeanSpeed('3i') + 
                    traci.edge.getLastStepMeanSpeed('4i')) / 4.0
        
        # Tính khí thải CO2
        co2_emission = (traci.edge.getFuelConsumption('1i') + 
                       traci.edge.getFuelConsumption('2i') + 
                       traci.edge.getFuelConsumption('3i') + 
                       traci.edge.getFuelConsumption('4i')) / 1000.0
        
        # Đếm số xe đang chờ
        waiting_vehicles = (traci.edge.getLastStepHaltingNumber('1i') + 
                           traci.edge.getLastStepHaltingNumber('2i') + 
                           traci.edge.getLastStepHaltingNumber('3i') + 
                           traci.edge.getLastStepHaltingNumber('4i'))
        
        # Chiều dài hàng đợi (giống waiting_vehicles trong trường hợp này)
        queue_length = waiting_vehicles
        
        # Số lượng xe đang tham gia giao thông
        throughput = (traci.edge.getLastStepVehicleNumber('1i') + 
                     traci.edge.getLastStepVehicleNumber('2i') + 
                     traci.edge.getLastStepVehicleNumber('3i') + 
                     traci.edge.getLastStepVehicleNumber('4i'))
        
        # Thời gian chờ đợi tích lũy
        waiting_time = (traci.edge.getWaitingTime('1i') + 
                       traci.edge.getWaitingTime('2i') + 
                       traci.edge.getWaitingTime('3i') + 
                       traci.edge.getWaitingTime('4i'))
        
        return {
            'avg_speed': avg_speed,
            'co2_emission': co2_emission,
            'waiting_vehicles': waiting_vehicles,
            'queue_length': queue_length,
            'throughput': throughput,
            'waiting_time': waiting_time
        }
    
    def get_state_reward(self):

        metrics = self.get_traffic_metrics()
        return [
            metrics['avg_speed'], 
            metrics['co2_emission'], 
            metrics['waiting_vehicles'], 
            metrics['queue_length'], 
            metrics['throughput']
        ]