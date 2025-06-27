import gym
import numpy as np
from typing import Dict, List

from lasvsim_openapi.client import Client
from lasvsim_openapi.http_client import HttpConfig
from lasvsim_openapi.simulator import Simulator, SimulatorConfig
from lasvsim_openapi.simulator_model import Point, ReferenceLine, StepCode

def world_to_local_batch(A, theta, B_set):
    """
    将多个点从世界坐标转换到以 A 为原点、theta 为朝向的局部坐标系。

    参数:
        A: Tuple[float, float] - 点 A 的坐标 (Ax, Ay)
        theta: float - A 的朝向角（弧度）
        B_set: np.ndarray or List[List[float]] - 点集，形状为 (N, 2)

    返回:
        np.ndarray - 局部坐标系下的点集，形状为 (N, 2)
    """
    B_set = np.asarray(B_set)  # shape (N, 2)
    dxdy = B_set - A           # 平移：世界坐标 → A 原点

    # 旋转矩阵（-theta）
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot_matrix = np.array([
        [cos_theta, sin_theta],
        [-sin_theta, cos_theta]
    ])  # shape (2, 2)

    local_coords = dxdy @ rot_matrix.T  # shape (N, 2)

    return local_coords


def interpolate_refine(refline: List[Point], interval: float = 0.5, length: float=400) -> np.ndarray:
    """
    对参考线进行固定间隔的插值,长度不足时,重复最后一个点
    """
    xys = []
    for p in refline:
        xys.append([p.x, p.y])
        
    refline = np.asarray(xys)  # shape (N, 2)

    # Step 1: 计算累计长度
    deltas = np.diff(refline, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cum_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    total_length = cum_lengths[-1]

    if total_length == 0:
        return np.tile(refline[0], (length, 1))  # 特殊情况：全为同一个点

    # Step 2: 按 interval 插值采样（最多 length 个点）
    desired_lengths = np.arange(0, total_length + interval, interval)
    if len(desired_lengths) > length:
        desired_lengths = desired_lengths[:length]

    # Step 3: 插值
    interp_x = np.interp(desired_lengths, cum_lengths, refline[:, 0])
    interp_y = np.interp(desired_lengths, cum_lengths, refline[:, 1])
    res = np.stack((interp_x, interp_y), axis=1)

    # Step 4: 不足 length 时重复最后一个点
    if len(res) < length:
        padding = np.tile(res[-1], (length - len(res), 1))
        res = np.vstack((res, padding))

    return res

paths = [["sg8_lk0", "sg8_lk0-sg5_lk0", "sg5_lk0"], 
         ["sg8_lk0", "sg8_lk0-sg1_lk0", "sg1_lk0"], 
         ["sg8_lk0", "sg8_lk0-sg3_lk0", "sg3_lk0"],
         ["sg2_lk0", "sg2_lk0-sg3_lk0", "sg3_lk0"], 
         ["sg2_lk0", "sg2_lk0-sg5_lk0", "sg5_lk0"], 
         ["sg2_lk0", "sg2_lk0-sg7_lk0", "sg7_lk0"], 
         ["sg6_lk0", "sg6_lk0-sg1_lk0", "sg1_lk0"], 
         ["sg6_lk0", "sg6_lk0-sg3_lk0", "sg3_lk0"], 
         ["sg6_lk0", "sg6_lk0-sg7_lk0", "sg7_lk0"],
         ["sg4_lk0", "sg4_lk0-sg1_lk0", "sg1_lk0"], 
         ["sg4_lk0", "sg4_lk0-sg5_lk0", "sg5_lk0"], 
         ["sg4_lk0", "sg4_lk0-sg7_lk0", "sg7_lk0"],]

MAX_FRONT_STEER = 35.0*3.14/180.0
K = 2.625

def rad_range_pi(x: float) -> float:
    while x <= -np.pi or x > np.pi:
        x = x - np.sign(x) * np.pi * 2
    return x

def point_to_line_distance(x, y, x1, y1, x2, y2):
    """
    点到直线 AB 的垂直距离
    """
    P, A, B = np.array([x, y]), np.array([x1, y1]), np.array([x2, y2])
    AB = B - A
    AP = P - A

    if np.dot(AB,AB) == 0:
        return np.linalg.norm(AP)
    
    distance = np.abs(np.cross(AB, AP)) / np.linalg.norm(AB)
    return distance

class LASVSimEnv(gym.Env):
    def __init__(self, token, endpoint, task_id, record_id, target_speed=15):
        self._client = Client(
            HttpConfig(
                token=token,
                endpoint=endpoint),
            )
        self._simulator: Simulator = self.init_simulator(task_id, record_id)
        self._vehicle: str = self.pick_vehicle()
        self._state = self.update_state()
        self._target_speed = target_speed
        self._observation = None
        self._reference_lines = self.get_reflines()

    def init_simulator(self, task_id:int, record_id:int) -> Simulator:
        task_record = self._client.process_task.copy_record(task_id=task_id, record_id=record_id)
        sim_config = SimulatorConfig(scen_id=task_record.scen_id, scen_ver=task_record.scen_ver, sim_record_id=task_record.sim_record_id)
        simulator = self._client.init_simulator_from_config(sim_config=sim_config)
        return simulator

    def pick_vehicle(self) -> str:
        res = self._simulator.get_vehicle_id_list()
        vehicles = res.list
        return np.random.choice(vehicles)

    def update_state(self) -> np.ndarray:
        pos_res = self._simulator.get_vehicle_position(vehicle_id_list=[self._vehicle])
        pos = pos_res.position_dict[self._vehicle]
        imu_res = self._simulator.get_vehicle_moving_info(vehicle_id_list=[self._vehicle])
        imu = imu_res.moving_info_dict[self._vehicle]
        control_res = self._simulator.get_vehicle_control_info(vehicle_id_list=[self._vehicle])
        control = control_res.control_info_dict[self._vehicle]
        return np.array([pos.point.x, pos.point.y, pos.phi, imu.u, control.ste_wheel, control.lon_acc])

    def get_reward(self) -> float:
        x, y, phi, u, _, _ = self._state

        # speed reward
        speed_reward = -abs(u - self._target_speed)

        # collision reward

        reward = speed_reward

        return reward

    def get_reflines(self) -> List[ReferenceLine]:
        resp = self._simulator.get_vehicle_reference_lines(vehicle_id=self._vehicle)
        if resp == None:
            return []
        
        return resp.reference_lines

    def judge_done(self) -> bool:
        pos_res = self._simulator.get_vehicle_position(vehicle_id_list=[self._vehicle])
        if pos_res == None:
            return True
        pos = pos_res.position_dict[self._vehicle]
        if pos.lane_id == "" and pos.junction_id == "":
            return True

        return False

    def reset(self):
        nav_path = paths[np.random.randint(len(paths))]
        reset_vehicles = [{"vehicle_id":self._vehicle, "link_path":nav_path, "s_range":[50, 300]}]
        self._simulator.reset(reset_traffic_flow=False, reset_vehicle=reset_vehicles, reset_env_ptcs=None)
        self._state = self.update_state()

    def step(self, refline_idx, action):
        reflines = self._reference_lines
        x, y, phi, u, delta, acc = self._state

        if refline_idx >= len(reflines):
            return self.state, self.reference_lines, 0.0, True, self.info

        track_refline = reflines[refline_idx]
        if len(track_refline.points) < 2:
            return self.state, self.reference_lines, 0.0, True, self.info

        dis_to_line = point_to_line_distance(x, y, track_refline.points[0].x, track_refline.points[0].y, track_refline.points[1].x, track_refline.points[1].y)
        
        ste_wheel = np.clip(action[0], -MAX_FRONT_STEER, MAX_FRONT_STEER)
        lon_accel = np.clip(action[1], -8, 4.2)

        self._simulator.set_vehicle_control_info(vehicle_id=self._vehicle, ste_wheel=ste_wheel, lon_acc=lon_accel)
        step_resp = self._simulator.step()
        if step_resp.code == StepCode.FINISHED:
            return self.state, self.reference_lines, 100.0, True, self.info
        elif step_resp.code == StepCode.FAILED:
            return self.state, self.reference_lines, -200.0, True, self.info

        self._state = self.update_state()
        self._reference_lines = self.get_reflines()

        x1, y1, phi1, u1, _, _ = self._state
        dis_to_line1 = point_to_line_distance(x1, y1, track_refline.points[0].x, track_refline.points[0].y, track_refline.points[1].x, track_refline.points[1].y)

        delta_max = min(MAX_FRONT_STEER, K/(u+0.001))

        # speed reward
        speed_reward = -abs(u1 - self._target_speed)

        # distance reward
        distance_reward = dis_to_line - dis_to_line1

        # action reward
        action_reward = -abs(ste_wheel) / (MAX_FRONT_STEER*2) - abs(lon_accel) / (8+4.2)
        if abs(action[0]) > delta_max:
            action_reward = action_reward + (delta_max - abs(action[0])) * 2
        action_reward = action_reward - abs(action[0] - delta) / MAX_FRONT_STEER
        if action[1] > 4.2:
            action_reward = action_reward + (4.2 - action[1]) * 2
        elif action[1] < -8:
            action_reward = action_reward + (action[1] + 8) * 2
        else:
            action_reward = action_reward - abs(action[1] - acc) / 8.0

        # phi reward
        phi_reward = 0.0
        if dis_to_line1 < 1:
            line_phi = np.arctan2(track_refline.points[1].y - track_refline.points[0].y, track_refline.points[1].x - track_refline.points[0].x)
            phi_diff = rad_range_pi(phi - line_phi)
            phi_diff1 = rad_range_pi(phi1 - line_phi)
            phi_reward = abs(phi_diff) - abs(phi_diff1)

        # step reward
        step_reward = 1.0
        
        reward = speed_reward + action_reward + step_reward + distance_reward + phi_reward

        return self.state, self.reference_lines, reward, False, self.info


    def stop(self):
        self._simulator.stop()

    def get_constraint(self) -> np.ndarray:
        # TODO: implement this, get constraint from self.stub
        return np.random.uniform(low=0, high=0, size=(1,))

    @property
    def info(self) -> Dict:
        return {
            "state": self._state,
            "constraint": self.get_constraint(),
        }

    @property
    def state(self) -> np.ndarray:
        return np.array(self._state)

    @property
    def reference_lines(self) -> np.ndarray:
        x, y, phi, _, _, _ = self._state
        refline_obs = []
        for line in self._reference_lines:
            tmp_line = interpolate_refine(line.points)
            tmp_line = world_to_local_batch([x, y], phi, tmp_line)
            refline_obs.append(tmp_line)

        return np.array(refline_obs)