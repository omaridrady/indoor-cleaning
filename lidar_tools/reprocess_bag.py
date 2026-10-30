#!/usr/bin/env python3
"""Recompute the motion of a saved recording from its raw /scan messages.

Writes a new bag with the same topics as lidar_motion_recorder.py (/scan, /lidar_odom,
/lidar_delta, /tf), using the current tracking code. Use it to repair a recording
made before a tracking fix, or to try different settings without driving again:

    python3 reprocess_bag.py ~/lidar_bags/run4 ~/lidar_bags/run4_fixed
    python3 reprocess_bag.py ~/lidar_bags/run4 ~/lidar_bags/run4_try --max-corr 0.4
"""
import argparse
import math
import os

import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import LaserScan

from lidar_motion_recorder import (Tracker, add_tracking_args, make_messages, open_bag,
                                   stamp_ns, xyyaw)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', help='recorded bag directory to read /scan from')
    parser.add_argument('out', help='new bag directory to create')
    add_tracking_args(parser)
    args = parser.parse_args()
    src = os.path.abspath(os.path.expanduser(args.bag.rstrip('/')))
    out = os.path.abspath(os.path.expanduser(args.out.rstrip('/')))

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=src, storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/scan']))
    writer = open_bag(out)
    tracker = Tracker(args)

    scans, path_len, last_t_ns = 0, 0.0, None
    while reader.has_next():
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, LaserScan)
        t_ns = stamp_ns(msg)
        dt = (t_ns - last_t_ns) * 1e-9 if last_t_ns else 0.0
        last_t_ns = t_ns
        delta, _ = tracker.update(msg, dt)
        path_len += math.hypot(*xyyaw(delta)[:2])
        for topic, m in make_messages(msg, tracker.pose, delta, dt):
            writer.write(topic, serialize_message(m), t_ns)
        scans += 1
    if hasattr(writer, 'close'):
        writer.close()
    del writer

    x, y, yaw = xyyaw(tracker.pose)
    print(f'Reprocessed {scans} scans -> {out} | path {path_len:.2f} m | '
          f'final pose x={x:+.3f} y={y:+.3f} yaw={math.degrees(yaw):+.1f}deg | '
          f'lost {tracker.lost}')


if __name__ == '__main__':
    main()
