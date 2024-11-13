import json
import math
import numpy as np
from collections import defaultdict

filename_sample_data = './data/nuscenes/v1.0-trainval/sample_data.json'
with open(filename_sample_data, 'r') as file:
    sample_data = json.load(file)

grouped_sample_data = defaultdict(list)
for item in sample_data:
    grouped_sample_data[item['sample_token']].append(item)

filename_ego_pose = './data/nuscenes/v1.0-trainval/ego_pose.json'
with open(filename_ego_pose, 'r') as file:
    ego_poses = {item['token']: item for item in json.load(file)}

for sample_token, samples in grouped_sample_data.items():
    for sample in samples:
        sample['ego_pose'] = ego_poses.get(sample['ego_pose_token'], "No corresponding ego_pose found")
    grouped_sample_data[sample_token] = sorted(samples, key=lambda x: x['timestamp'])

filtered_grouped_sample_data = defaultdict(list)
for sample_token, samples in grouped_sample_data.items():
    filtered_samples = [
        {
            'token': sample['token'],
            'sample_token': sample['sample_token'],
            'ego_pose_token': sample['ego_pose_token'],
            'timestamp': sample['timestamp'],
            'ego_pose': sample['ego_pose']
        }
        for sample in samples
        if sample['filename'].startswith('sweeps/LIDAR_TOP') or sample['filename'].startswith('samples/LIDAR_TOP')
    ]
    if filtered_samples:
        filtered_grouped_sample_data[sample_token] = filtered_samples


def quaternion_to_yaw(quaternion):
    w, x, y, z = quaternion
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


motion_datas = {}
for sample_token, sample_data in filtered_grouped_sample_data.items():
    speeds, accelerations, angular_speeds, yaw_angle_degreess = {}, {}, {}, {}
    for i, current in enumerate(sample_data):
        if i == 0:
            speeds[current['timestamp']] = 0
            angular_speeds[current['timestamp']] = 0
            accelerations[current['timestamp']] = 0
            continue

        previous = sample_data[i - 1]
        current_translation = current['ego_pose'].get('translation', [0, 0, 0])
        previous_translation = previous['ego_pose'].get('translation', [0, 0, 0])
        current_rotation = current['ego_pose'].get('rotation', [1, 0, 0, 0])
        previous_rotation = previous['ego_pose'].get('rotation', [1, 0, 0, 0])

        displacement = [current_translation[j] - previous_translation[j] for j in range(3)]
        distance = np.linalg.norm(displacement)
        time_diff = (current['timestamp'] - previous['timestamp']) * 1e-6
        speed = distance / time_diff if time_diff > 0 else 0
        speeds[current['timestamp']] = speed

        dot_product = sum(c * p for c, p in zip(current_rotation, previous_rotation))
        dot_product = max(-1, min(1, dot_product))
        angle_diff = math.acos(dot_product)
        angular_speed = angle_diff / time_diff if time_diff > 0 else 0
        angular_speeds[current['timestamp']] = angular_speed

        if i >= 2:
            speed_pre = speeds.get(previous['timestamp'], 0)
            accelerations[current['timestamp']] = (speed - speed_pre) / (time_diff * 10) if time_diff > 0 else 0

        yaw_angle = quaternion_to_yaw(current_rotation)
        yaw_angle_degreess[current['timestamp']] = np.degrees(yaw_angle)

    if len(sample_data) > 1:
        next_time = sample_data[1]
        speeds[sample_data[0]['timestamp']] = speeds[next_time['timestamp']]
        yaw_angle_degreess[sample_data[0]['timestamp']] = yaw_angle_degreess[next_time['timestamp']]

    motion_data = [
        {
            'timestamp': item['timestamp'],
            'translation': item['ego_pose'].get('translation'),
            'rotation': item['ego_pose'].get('rotation'),
            'speed': speeds.get(item['timestamp'], 0),
            'acceleration': accelerations.get(item['timestamp'], 0),
            'angular_speed': angular_speeds.get(item['timestamp'], 0),
            'yaw_angle_degrees': yaw_angle_degreess.get(item['timestamp'], 0)
        }
        for item in sample_data
    ]
    motion_datas[sample_token] = motion_data


def get_ego_pose(sample_token):
    sample_data = motion_datas.get(sample_token)
    if not sample_data:
        return None

    timestamps = []
    motion_data = []

    for item in sample_data:
        timestamps.append(item['timestamp'])
        ex_data = [item['speed'], item['acceleration'], item['yaw_angle_degrees']]
        motion_data.append(ex_data)

    stamp2arrays = {}
    for u, time in enumerate(timestamps):
        ego_array = motion_data[:u + 1]
        if len(ego_array) < 6:
            ego_array = [ego_array[0]] * (6 - len(ego_array)) + ego_array
        stamp2arrays[time] = ego_array

    return stamp2arrays


token2ego = {token: get_ego_pose(token) for token in motion_datas}

with open('./data/info/ego_pose_data.json', 'w') as output_file:
    json.dump(token2ego, output_file, indent=4)
