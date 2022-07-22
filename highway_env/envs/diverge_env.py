import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class DivergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.
    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
            "partition_length":[150, 40, 200]
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.config["lane_change_reward"],
                         1: 0,
                         2: self.config["lane_change_reward"],
                         3: 0,
                         4: 0}
        reward = self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * self.vehicle.lane_index[2] / 1 \
            + self.config["high_speed_reward"] * self.vehicle.speed_index / (self.vehicle.target_speeds.size - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
                reward += self.config["merging_speed_reward"] * \
                          (vehicle.target_speed - vehicle.speed) / vehicle.target_speed

        for vehicle in self.road.vehicles:
            if vehicle.position == ("a","b",100) and not isinstance(vehicle, ControlledVehicle):
                vehicle.target_lane_index=("b","c",np.random.randint(2))
                

        return utils.lmap(action_reward[action] + reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 325)

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = self.config["partition_length"]  # Before, converging, merge, after

        n_lanes = 4
        n_div = 2

        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        lanes = []

        lanes.append([StraightLane([0, 0], [ends[0], 0], line_types=[c,s])])
        for i in range(1,n_lanes-1):
            lanes.append([StraightLane([0, i*StraightLane.DEFAULT_WIDTH], [ends[0], i*StraightLane.DEFAULT_WIDTH], line_types=[n,s])])
        lanes.append([StraightLane([0, (n_lanes-1)*StraightLane.DEFAULT_WIDTH], [ends[0], (n_lanes-1)*StraightLane.DEFAULT_WIDTH], line_types=[n,c])])
            
        amplitude = 2.75

        if n_div == 1:
            lanes[0].append(SineLane(lanes[0][0].position(ends[0], -amplitude), lanes[0][0].position(sum(ends[:2]), -amplitude),
                    amplitude, np.pi / (ends[0]), np.pi / 2, line_types=[c, c],forbidden=True))
            lanes[1].append(SineLane(lanes[1][0].position(ends[0], amplitude), lanes[1][0].position(sum(ends[:2]), amplitude),
                    -amplitude, np.pi / (ends[0]), np.pi / 2, line_types=[c, c],forbidden=True))
        else:
            lanes[0].append(SineLane(lanes[0][0].position(ends[0], -amplitude), lanes[0][0].position(sum(ends[:2]), -amplitude),
                    amplitude, np.pi / (ends[0]), np.pi / 2, line_types=[c, s],forbidden=True))
            for i in range(1,n_div-1):
                lanes[i].append(SineLane(lanes[i][0].position(ends[0], -amplitude), lanes[i][0].position(sum(ends[:2]), -amplitude),
                    amplitude, np.pi / (ends[0]), np.pi / 2, line_types=[n, s],forbidden=False))
            lanes[n_div-1].append(SineLane(lanes[n_div-1][0].position(ends[0], -amplitude), lanes[n_div-1][0].position(sum(ends[:2]), -amplitude),
                    amplitude, np.pi / (ends[0]), np.pi / 2, line_types=[n, c],forbidden=True))
            
            amplitude*=-1
            lanes[n_div].append(SineLane(lanes[n_div][0].position(ends[0], -amplitude), lanes[n_div][0].position(sum(ends[:2]), -amplitude),
                    amplitude, np.pi / (ends[0]), np.pi / 2, line_types=[c, s],forbidden=True))
            for i in range(n_div+1,n_lanes-1):
                lanes[i].append(SineLane(lanes[i][0].position(ends[0], -amplitude), lanes[i][0].position(sum(ends[:2]), -amplitude),
                    amplitude, np.pi / (ends[0]), np.pi / 2, line_types=[n, s],forbidden=False))
            lanes[n_lanes-1].append(SineLane(lanes[n_lanes-1][0].position(ends[0], -amplitude), lanes[n_lanes-1][0].position(sum(ends[:2]), -amplitude),
                    amplitude, np.pi / (ends[0]), np.pi / 2, line_types=[n, c],forbidden=True))
            
        
        for lane in lanes:
            lane.append(StraightLane(lane[1].position(ends[1], 0), lane[1].position(ends[1], 0) + [ends[2], 0],
                           line_types=[c, c],forbidden=True))

        for l in lanes:
            net.add_lane("a", "b", l[0])
            net.add_lane("b", "c", l[1])
            net.add_lane("c", "d", l[2])

   
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
#        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("a", "b", 1)).position(30, 0),
                                                     speed=30, target_lane_index=("b","c",0))
        
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(90, 0), speed=29,target_lane_index=("b","c",np.random.randint(2))))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(40, 0), speed=30.5,target_lane_index=("b","c",np.random.randint(2))))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31,target_lane_index=("b","c",np.random.randint(2))))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5,target_lane_index=("b","c",np.random.randint(2))))

        self.vehicle = ego_vehicle


register(
    id='diverge-v0.03',
    entry_point='highway_env.envs:DivergeEnv',
)
