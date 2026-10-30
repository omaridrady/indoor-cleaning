#!/usr/bin/env python3
"""Apply slam_toolbox's map -> odom corrections to a motion bag (used by slam_offline.sh).

    python3 slam_apply.py <motion bag> <slam output bag> <corrected bag out>

Each scan's pose becomes map -> laser = (latest map -> odom at that time) * (odom -> laser).
The corrected bag has the usual /scan, /lidar_odom, /lidar_delta, /tf (now map -> laser).
The last /map message is also saved next to it as <out>_map.npy / .pgm / .yaml
(the .pgm + .yaml pair is the standard ROS map format).
"""
import bisect
import math
import os
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message

from lidar_motion_recorder import inv, make_messages, open_bag, rigid, se2, stamp_ns, xyyaw


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def read(bag, topics):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    while reader.has_next():
        topic, data, t = reader.read_next()
        yield topic, deserialize_message(data, types[topic]), t


def main():
    src, slam_bag, out = (os.path.abspath(os.path.expanduser(a)) for a in sys.argv[1:4])

    # map -> odom corrections over (message) time, and the final map.
    corr_t, corr_T, last_map = [], [], None
    for topic, msg, _ in read(slam_bag, ['/tf', '/map']):
        if topic == '/map':
            last_map = msg
            continue
        for tr in msg.transforms:
            if tr.header.frame_id == 'map' and tr.child_frame_id == 'odom':
                v = tr.transform.translation
                corr_t.append(stamp_ns(tr))
                corr_T.append(se2(v.x, v.y, yaw_of(tr.transform.rotation)))
    if not corr_t:
        sys.exit('No map -> odom transforms in the SLAM output; did slam_toolbox run?')
    order = np.argsort(corr_t)
    corr_t = [corr_t[i] for i in order]
    corr_T = [corr_T[i] for i in order]
    print(f'{len(corr_t)} map->odom corrections; final one: '
          f'x={corr_T[-1][0, 2]:+.2f} y={corr_T[-1][1, 2]:+.2f} '
          f'yaw={math.degrees(xyyaw(corr_T[-1])[2]):+.1f}deg')

    # Pair each scan with its odom pose (same stamp in the motion bag).
    odom = {}
    scans = []
    for topic, msg, _ in read(src, ['/scan', '/lidar_odom']):
        if topic == '/scan':
            scans.append(msg)
        else:
            p = msg.pose.pose
            odom[stamp_ns(msg)] = se2(p.position.x, p.position.y, yaw_of(p.orientation))

    writer = open_bag(out)
    prev_pose, prev_t, path_len = None, None, 0.0
    for msg in scans:
        t = stamp_ns(msg)
        if t not in odom:
            continue
        k = bisect.bisect_right(corr_t, t) - 1
        correction = corr_T[max(k, 0)]  # before SLAM's first correction, use its first
        pose = rigid(correction @ odom[t])
        delta = rigid(inv(prev_pose) @ pose) if prev_pose is not None else np.eye(3)
        dt = (t - prev_t) * 1e-9 if prev_t else 0.0
        messages = make_messages(msg, pose, delta, dt)
        messages[1][1].header.frame_id = 'map'
        messages[3][1].transforms[0].header.frame_id = 'map'
        for topic, m in messages:
            writer.write(topic, serialize_message(m), t)
        path_len += math.hypot(*xyyaw(delta)[:2])
        prev_pose, prev_t = pose, t
    if hasattr(writer, 'close'):
        writer.close()
    del writer
    x, y, yaw = xyyaw(prev_pose)
    print(f'Corrected bag -> {out} | {len(scans)} scans | path {path_len:.2f} m | '
          f'final pose x={x:+.3f} y={y:+.3f} yaw={math.degrees(yaw):+.1f}deg')

    if last_map is not None:
        info = last_map.info
        grid = np.asarray(last_map.data, dtype=np.int8).reshape(info.height, info.width)
        np.save(out + '_map.npy', grid)
        img = np.full(grid.shape, 205, dtype=np.uint8)   # unknown = grey
        img[(grid >= 0) & (grid < 50)] = 254             # free = white
        img[grid >= 50] = 0                              # occupied = black
        with open(out + '_map.pgm', 'wb') as f:
            f.write(f'P5 {info.width} {info.height} 255\n'.encode())
            f.write(np.flipud(img).tobytes())
        o = info.origin.position
        with open(out + '_map.yaml', 'w') as f:
            f.write(f'image: {os.path.basename(out)}_map.pgm\nresolution: {info.resolution}\n'
                    f'origin: [{o.x}, {o.y}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\n'
                    f'free_thresh: 0.25\n')
        print(f'Map {info.width}x{info.height} @ {info.resolution} m -> {out}_map.pgm/.npy/.yaml')


if __name__ == '__main__':
    main()
