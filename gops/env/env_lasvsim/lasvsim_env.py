import gym
import numpy as np
from typing import Dict

from lasvsim_openapi.client import Client
from lasvsim_openapi.http_client import HttpConfig
from lasvsim_openapi.simulator import Simulator, SimulatorConfig

class LASVSimEnv(gym.Env):
    def __init__(self, token, endpoint, task_id, record_id, target_speed=15):
        self._client = Client(
            HttpConfig(
                token=token,
                endpoint=endpoint),
            )
        self._simulator = self.init_simulator(task_id, record_id)
        self._vehicle = self.pick_vehicle()
        self._state = self.update_state()
        self._target_speed = target_speed

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
        return np.array([pos.point.x, pos.point.y, pos.phi, imu.u, imu.v, imu.w])

    def judge_done(self) -> bool:
        pass

    def reset(self):
        self._simulator.reset()
        self.update_state()

    def step(self, action):
        return self.env.step(action)

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