#!/usr/bin/env python3
"""Split a bag from lidar_motion_recorder.py into one CSV file per topic.

    python3 export_topics.py ~/lidar_bags/run2              # writes ~/lidar_bags/run2_csv/
    python3 export_topics.py ~/lidar_bags/run2 --out mydir

A rosbag keeps every topic inside its single .mcap file; this pulls them out:
    scan.csv   /scan         t, angle_min, angle_increment, range_min, range_max, r0 ... rN
                             (one row per scan; empty cell = no return at that angle)
    pose.csv   /lidar_odom   t, x, y, yaw_deg, vx, vy, wz_deg_s   (pose since start)
    delta.csv  /lidar_delta  t, dx, dy, dyaw_deg, distance, direction_deg  (vs previous scan)
    tf.csv     /tf           t, parent, child, x, y, z, qx, qy, qz, qw, yaw_deg
All share the same t (seconds), so rows line up across files.
"""
import argparse
import csv
import math
import os

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def yaw_deg(q):
    return math.degrees(math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


def f(v, nd=6):
    return '' if v is None or not math.isfinite(v) else round(v, nd)


def scan_row(t, m):
    return [t, f(m.angle_min), f(m.angle_increment), f(m.range_min), f(m.range_max)] + \
           [f(r, 4) for r in m.ranges]


def pose_row(t, m):
    p, tw = m.pose.pose.position, m.twist.twist
    return [t, f(p.x), f(p.y), f(yaw_deg(m.pose.pose.orientation), 3),
            f(tw.linear.x), f(tw.linear.y), f(math.degrees(tw.angular.z), 3)]


def delta_row(t, m):
    dx, dy = m.transform.translation.x, m.transform.translation.y
    dist = math.hypot(dx, dy)
    return [t, f(dx), f(dy), f(yaw_deg(m.transform.rotation), 4), f(dist),
            f(math.degrees(math.atan2(dy, dx)), 2) if dist > 1e-4 else '']


def tf_rows(t, m):
    for tr in m.transforms:
        v, q = tr.transform.translation, tr.transform.rotation
        yield [t, tr.header.frame_id, tr.child_frame_id, f(v.x), f(v.y), f(v.z),
               f(q.x), f(q.y), f(q.z), f(q.w), f(yaw_deg(q), 3)]


# topic -> (file, header, row function returning a list of rows)
EXPORTS = {
    '/scan': ('scan.csv', None, lambda t, m: [scan_row(t, m)]),
    '/lidar_odom': ('pose.csv', ['t', 'x', 'y', 'yaw_deg', 'vx', 'vy', 'wz_deg_s'],
                    lambda t, m: [pose_row(t, m)]),
    '/lidar_delta': ('delta.csv', ['t', 'dx', 'dy', 'dyaw_deg', 'distance', 'direction_deg'],
                     lambda t, m: [delta_row(t, m)]),
    '/tf': ('tf.csv', ['t', 'parent', 'child', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'yaw_deg'],
            lambda t, m: list(tf_rows(t, m))),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag')
    parser.add_argument('--out', help='output directory (default <bag>_csv)')
    args = parser.parse_args()
    bag = os.path.abspath(os.path.expanduser(args.bag.rstrip('/')))
    out_dir = os.path.expanduser(args.out or bag + '_csv')
    os.makedirs(out_dir, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}

    files, writers, counts = {}, {}, {}
    try:
        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            if topic not in EXPORTS:
                continue
            name, header, rows = EXPORTS[topic]
            msg = deserialize_message(data, types[topic])
            if topic not in writers:
                files[topic] = open(os.path.join(out_dir, name), 'w', newline='')
                writers[topic] = csv.writer(files[topic])
                if header is None:  # scan: one column per beam, sized from the first scan
                    header = ['t', 'angle_min', 'angle_increment', 'range_min', 'range_max'] + \
                             [f'r{i}' for i in range(len(msg.ranges))]
                writers[topic].writerow(header)
                counts[topic] = 0
            for row in rows(round(t_ns * 1e-9, 6), msg):
                writers[topic].writerow(row)
                counts[topic] += 1
    finally:
        for fh in files.values():
            fh.close()

    for topic, n in counts.items():
        print(f'{EXPORTS[topic][0]:10s} {n:6d} rows  ({topic})')
    print(f'Wrote to {out_dir}')


if __name__ == '__main__':
    main()
