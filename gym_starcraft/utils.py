import math
import numpy as np

DISTANCE_FACTOR = 16

def get_degree(x1, y1, x2, y2):
    radians = math.atan2(y2 - y1, x2 - x1)
    return math.degrees(radians)


def get_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def get_position(degree, distance, x1, y1):
    theta = math.pi / 2 - math.radians(degree)
    return x1 + distance * math.sin(theta), y1 + distance * math.cos(theta)


def print_progress(episodes, wins):
    print "Episodes: %4d | Wins: %4d | WinRate: %1.3f" % (
        episodes, wins, wins / (episodes + 1E-6))


def get_observation(obs_shape, myself, enemy):
	obs = np.zeros(obs_shape)

	if myself is not None or enemy is None:
		obs[0] = 0.0
		obs[1] = enemy.health
		obs[2] = enemy.groundCD
		obs[3] = enemy.groundRange # / DISTANCE_FACTOR - 1
		obs[4] = get_degree(myself.x, -myself.y, enemy.x, -enemy.y) / 180
		obs[5] = get_distance(myself.x, -myself.y, enemy.x, -enemy.y) # / DISTANCE_FACTOR - 1
		obs[6] = 0.0
	else:
		obs[6] = 1.0

	return obs

def get_weakest_G(units_table):
    min_total_hp = 1E30
    weakest_uid = -1
    for uid, ut in units_table.iteritems():
        if ut is None:
            continue
        tmp_hp = ut.health + ut.shield
        if tmp_hp < min_total_hp:
            min_total_hp = tmp_hp
            weakest_uid = uid
    return weakest_uid

def get_weakest(units_dict):
    min_total_hp = 1E30
    weakest_uid = -1
    for uid, item in units_dict.items():
        # print uid
        # print item
        if item[0] < 1:
            continue
        tmp_hp = item[1]
        if tmp_hp < min_total_hp:
            min_total_hp = tmp_hp
            weakest_uid = uid
    return weakest_uid

def get_closest(units_dict):
    min_distance = 1E30
    closest_uid = -1
    for uid, item in units_dict.items():
        if item[0] < 1:
            continue
        tmp_dis = item[5]
        if tmp_dis < min_distance:
            min_distance = tmp_dis
            closest_uid = uid
    return closest_uid