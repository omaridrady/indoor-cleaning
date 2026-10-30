#!/usr/bin/env python3
"""Read a bag made by lidar_motion_recorder.py as {reading, orientation} records.

    python3 read_motion_bag.py ~/lidar_bags/motion_<time>                   # summary + sample
    python3 read_motion_bag.py ~/lidar_bags/motion_<time> --json run.jsonl  # export every scan

Each record (one JSON object per line with --json):
    {"t": seconds,
     "reading":     {"angle_min", "angle_increment", "range_min", "range_max",
                     "ranges": [m, ... null where there was no return]},
     "orientation": {"x": m, "y": m, "yaw_deg": deg},           # pose since recording started
     "delta":       {"dx": m, "dy": m, "dyaw_deg": deg,         # change since previous scan,
                     "distance": m, "direction_deg": deg}}      # in the previous scan's frame
"""
import argparse
import json
import math

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

NEEDED = ('/scan', '/lidar_odom', '/lidar_delta')


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def to_record(t_ns, scan, odom, step):
    p = odom.pose.pose.position
    dx, dy = step.transform.translation.x, step.transform.translation.y
    dist = math.hypot(dx, dy)
    return {
        't': round(t_ns * 1e-9, 6),
        'reading': {
            'angle_min': scan.angle_min,
            'angle_increment': scan.angle_increment,
            'range_min': scan.range_min,
            'range_max': scan.range_max,
            'ranges': [round(r, 4) if math.isfinite(r) else None for r in scan.ranges],
        },
        'orientation': {
            'x': round(p.x, 4),
            'y': round(p.y, 4),
            'yaw_deg': round(math.degrees(yaw_of(odom.pose.pose.orientation)), 2),
        },
        'delta': {
            'dx': round(dx, 5),
            'dy': round(dy, 5),
            'dyaw_deg': round(math.degrees(yaw_of(step.transform.rotation)), 3),
            'distance': round(dist, 5),
            'direction_deg': round(math.degrees(math.atan2(dy, dx)), 1) if dist > 1e-4 else None,
        },
    }


def read_records(bag):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    pending = {}  # timestamp -> {topic: msg}; the recorder writes all topics with the scan's stamp
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in NEEDED:
            continue
        group = pending.setdefault(t_ns, {})
        group[topic] = deserialize_message(data, types[topic])
        if len(group) == len(NEEDED):
            del pending[t_ns]
            yield to_record(t_ns, *(group[n] for n in NEEDED))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', help='bag directory (contains metadata.yaml and a .mcap file)')
    parser.add_argument('--json', help='write every record to this JSON Lines file')
    args = parser.parse_args()

    out = open(args.json, 'w') if args.json else None
    first = last = None
    count, path_len = 0, 0.0
    for rec in read_records(args.bag):
        if out:
            out.write(json.dumps(rec, separators=(',', ':')) + '\n')
        if count < 3:
            sample = dict(rec, reading=f'<{len(rec["reading"]["ranges"])} ranges>')
            print(json.dumps(sample))
        first = first or rec
        last = rec
        count += 1
        path_len += rec['delta']['distance']
    if out:
        out.close()

    if not count:
        print('No complete records found.')
        return
    o = last['orientation']
    print(f'\n{count} scans over {last["t"] - first["t"]:.1f} s | path {path_len:.3f} m | '
          f'final pose x={o["x"]:+.3f} y={o["y"]:+.3f} yaw={o["yaw_deg"]:+.1f}deg')
    if out:
        print(f'Wrote {count} records to {args.json}')


if __name__ == '__main__':
    main()
