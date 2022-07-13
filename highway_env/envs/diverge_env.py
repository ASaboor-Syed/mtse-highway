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

        return utils.lmap(action_reward[action] + reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

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
        ends = [75, 40, 40]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        straight_lanes = []
        for i in range(len(line_type)):
            lane_parts = []
            lane_parts.append(StraightLane([0, i*StraightLane.DEFAULT_WIDTH], [sum(ends[:2]), i*StraightLane.DEFAULT_WIDTH], line_types=line_type[i]))
            lane_parts.append(StraightLane([sum(ends[:2]), i*StraightLane.DEFAULT_WIDTH], [sum(ends[:3]), i*StraightLane.DEFAULT_WIDTH], line_types=line_type[i]))
            straight_lanes.append(lane_parts)
            
        n_diverging = 1
        amplitude = 2.50
        
        net.add_lane("a", "b", straight_lanes[0][0])
        net.add_lane("b", "c", straight_lanes[0][1])
        net.add_lane("c", "d", SineLane(straight_lanes[0][1].position(ends[2], -amplitude), straight_lanes[0][1].position(sum(ends), -amplitude),
                    amplitude, np.pi / (ends[1]), np.pi / 2, line_types=[c, c], forbidden=True))
        
                     
        net.add_lane("a", "b", straight_lanes[1][0])
        net.add_lane("b", "c", straight_lanes[1][1])
        net.add_lane("c", "d", SineLane(straight_lanes[1][1].position(ends[2], amplitude), straight_lanes[1][1].position(sum(ends), amplitude),
                       -amplitude, np.pi / (ends[1]), np.pi / 2, line_types=[c, c], forbidden=True))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, np.pi / (ends[1]), np.pi / 4, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("a", "b", 1)).position(30, 0),
                                                     speed=30)
        
        ego_vehicle.configure({"target_lane_index":np.randint(0,2)})
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_vehicles_type.configure({"target_lane_index":np.randint(0,2)})
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        other_vehicles_type.configure({"target_lane_index":np.randint(0,2)})
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        other_vehicles_type.configure({"target_lane_index":np.randint(0,2)})
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))

        merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20)
        merging_v.target_speed = 30
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle


register(
    id='diverge-v0.03',
    entry_point='highway_env.envs:DivergeEnv',
)
