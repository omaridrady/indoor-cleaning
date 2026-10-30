#!/usr/bin/env python3
"""Estimate how the RPLIDAR C1 moves by matching each scan against earlier ones,
and record every scan together with its pose into one rosbag2 (MCAP) file.

For every /scan message the bag gets these, all with the scan's timestamp:
  /scan         sensor_msgs/LaserScan            the raw reading
  /lidar_odom   nav_msgs/Odometry                pose (x, y, yaw) since recording started
  /lidar_delta  geometry_msgs/TransformStamped   motion since the previous scan, in the
                                                 previous scan's frame (+x = lidar forward)
  /tf           tf2_msgs/TFMessage               odom -> laser, so `ros2 bag play` works in RViz

Motion comes from point-to-line ICP scan matching, so it assumes the surroundings
are mostly static. Needs the lidar driver running (publishing /scan):

    python3 lidar_motion_recorder.py                 # bag goes to ~/lidar_bags/motion_<time>
    python3 lidar_motion_recorder.py --out ~/run1
"""
import argparse
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
from scipy.spatial import cKDTree

import rclpy
import rosbag2_py
from geometry_msgs.msg import Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

TOPICS = [
    ('/scan', 'sensor_msgs/msg/LaserScan'),
    ('/lidar_odom', 'nav_msgs/msg/Odometry'),
    ('/lidar_delta', 'geometry_msgs/msg/TransformStamped'),
    ('/tf', 'tf2_msgs/msg/TFMessage'),
]

MIN_POINTS = 60          # fewer valid points than this: don't try to match
MIN_INLIER_RATIO = 0.5   # share of points that must land near the reference scan
DEGENERATE_DAMPING = 0.02  # see icp(); 0 = plain ICP
MAX_SPEED = 1.5          # m/s; a match implying more than this is rejected as a slip
MAX_TURN_RATE = 720.0    # deg/s; same for rotation (carts do spin in place at 150+ deg/s)


# ---------- 2D rigid transforms as 3x3 homogeneous matrices ----------

def se2(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def xyyaw(T):
    return T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0])


def rigid(T):
    """Snap back to an exact rotation + translation. Without this, float error in
    chained products feeds through the motion guess and grows every scan."""
    return se2(*xyyaw(T))


def inv(T):
    out = np.eye(3)
    out[:2, :2] = T[:2, :2].T
    out[:2, 2] = -T[:2, :2].T @ T[:2, 2]
    return out


def yaw_quat(yaw):
    return Quaternion(z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


# ---------- scan matching ----------

def scan_to_points(msg, min_range):
    r = np.asarray(msg.ranges, dtype=np.float64)
    a = msg.angle_min + np.arange(r.size) * msg.angle_increment
    ok = np.isfinite(r) & (r >= max(msg.range_min, min_range)) & (r <= msg.range_max)
    return np.column_stack((r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])))


class Reference:
    """A scan prepared for matching: KD-tree plus a surface normal per point."""

    def __init__(self, pts, k=6, max_spread=0.3):
        self.pts = pts
        self.tree = cKDTree(pts)
        dist, idx = self.tree.query(pts, k=k)
        nb = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
        evals, evecs = np.linalg.eigh(np.einsum('nki,nkj->nij', nb, nb) / k)
        self.normals = evecs[:, :, 0]  # direction of least spread
        # Trust a normal only where the neighbours form a short, straight segment.
        self.good = (dist[:, -1] < max_spread) & (evals[:, 0] < 0.1 * evals[:, 1])


def coarse_rotation(src, ref, T, max_corr, span_deg=30.0, step_deg=5.0):
    """Try rotating the guess T about the lidar by ±span_deg; return the candidate with the
    most points landing near the reference. Fast spins (15-25 deg per scan) start ICP
    too far off for it to lock on by itself."""
    offsets = np.arange(-span_deg, span_deg + 1e-9, step_deg)
    best_T, best_score = T, -1
    for d in sorted(offsets, key=abs):  # nearest to the guess first, so ties keep it
        cand = T @ se2(0.0, 0.0, math.radians(d))
        p = src @ cand[:2, :2].T + cand[:2, 2]
        dist, _ = ref.tree.query(p, distance_upper_bound=max_corr)
        score = np.isfinite(dist).sum()
        if score > best_score:
            best_T, best_score = cand, score
    return best_T


def range_rotation(prev, cur, inc, max_deg=60.0):
    """Rotation (rad) between two same-size scans, from the circular shift that best lines
    up their range arrays. Ignores translation, so it is only a fallback for when ICP fails
    (typically mid-spin, where the cart barely moves)."""
    n = len(cur)
    max_shift = int(math.radians(max_deg) / inc)
    shifts = np.arange(-max_shift, max_shift + 1)
    idx = (np.arange(n)[None, :] + shifts[:, None]) % n
    with np.errstate(all='ignore'):
        err = np.nanmedian(np.abs(cur[idx] - prev[None, :]), axis=1)
    if np.all(np.isnan(err)):
        return None
    return -shifts[np.nanargmin(err)] * inc


def icp(src, ref, T, max_corr, iters=30, damping=DEGENERATE_DAMPING):
    """Point-to-line ICP starting from guess T. Returns (T_ref_src, inlier_ratio, rmse).

    In a corridor the walls only constrain sideways position and heading, so the
    along-corridor direction of H is near zero and plain ICP slides or jumps there.
    Damping by a fraction of H's largest eigenvalue leaves well-measured directions
    alone but keeps poorly measured ones at the guess (the constant-velocity prediction).
    """
    ratio, rmse = 0.0, math.inf
    for _ in range(iters):
        p = src @ T[:2, :2].T + T[:2, 2]
        d, j = ref.tree.query(p, distance_upper_bound=max_corr)
        near = np.isfinite(d)
        ratio = near.mean()
        m = near.copy()
        m[near] = ref.good[j[near]]
        if m.sum() < MIN_POINTS // 2:
            return T, 0.0, math.inf
        pm, q, n = p[m], ref.pts[j[m]], ref.normals[j[m]]
        r = np.einsum('ij,ij->i', n, pm - q)  # signed point-to-line distance
        J = np.column_stack((n[:, 0], n[:, 1], n[:, 1] * pm[:, 0] - n[:, 0] * pm[:, 1]))
        w = 1.0 / (1.0 + (r / 0.05) ** 2)     # Cauchy weights tame outliers
        H = J.T @ (J * w[:, None])
        H += (damping * np.linalg.eigvalsh(H)[-1] + 1e-6) * np.eye(3)
        step = -np.linalg.solve(H, J.T @ (w * r))
        T = se2(*step) @ T
        rmse = math.sqrt(np.mean(r ** 2))
        if abs(step[0]) + abs(step[1]) < 1e-5 and abs(step[2]) < 1e-5:
            break
    return T, ratio, rmse


# ---------- recorder node ----------

def open_bag(path):
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    for i, (name, type_) in enumerate(TOPICS):
        writer.create_topic(rosbag2_py.TopicMetadata(
            id=i, name=name, type=type_, serialization_format='cdr'))
    return writer


class Tracker:
    """Scan-matching odometry: feed it scans in order, read .pose (odom -> laser)."""

    def __init__(self, args):
        self.args = args
        self.pose = np.eye(3)        # odom -> laser at the latest scan
        self.last_delta = np.eye(3)  # motion over the last scan; next match's initial guess
        self.ref = None              # keyframe scan that new scans are matched against
        self.ref_pose = None
        self.lost = 0
        self.win_ratio = []          # match ratios since the caller last cleared it
        self.prev_ranges = None      # previous scan's ranges, for the range_rotation fallback

    def set_keyframe(self, pts):
        self.ref = Reference(pts)
        self.ref_pose = self.pose.copy()

    def match(self, pts, predicted):
        """Set self.pose by matching pts to the keyframe. Returns False if the match failed."""
        if len(pts) < MIN_POINTS:
            return False
        if self.ref is None:  # first scan defines the origin
            self.set_keyframe(pts)
            return True

        guess = coarse_rotation(pts, self.ref, rigid(inv(self.ref_pose) @ predicted),
                                self.args.max_corr)
        T_ref_cur, ratio, _ = icp(pts, self.ref, guess, self.args.max_corr,
                                  damping=self.args.damping)
        self.win_ratio.append(ratio)
        if ratio < MIN_INLIER_RATIO:
            return False

        self.pose = rigid(self.ref_pose @ T_ref_cur)
        # Match against a fixed keyframe until we've moved enough; matching every scan
        # to the one before would let small errors pile up even while standing still.
        x, y, yaw = xyyaw(T_ref_cur)
        if (math.hypot(x, y) > self.args.keyframe_dist
                or abs(yaw) > math.radians(self.args.keyframe_deg)):
            self.set_keyframe(pts)
        return True

    def update(self, msg, dt):
        """Track one LaserScan; dt = seconds since the previous scan (0 for the first).
        Returns (delta since previous scan, ok)."""
        pts = scan_to_points(msg, self.args.min_range)
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        ranges[~np.isfinite(ranges) | (ranges < self.args.min_range)] = np.nan
        prev_ranges, self.prev_ranges = self.prev_ranges, ranges
        prev = self.pose
        predicted = rigid(prev @ self.last_delta)  # constant velocity
        ok = self.match(pts, predicted)
        if ok and dt > 0:
            dx, dy, dyaw = xyyaw(inv(prev) @ self.pose)
            dt = max(dt, 0.05)
            if (math.hypot(dx, dy) > MAX_SPEED * dt
                    or abs(math.degrees(dyaw)) > MAX_TURN_RATE * dt):
                ok = False  # faster than the cart can move: the match slipped
        if ok:
            delta = rigid(inv(prev) @ self.pose)
            self.last_delta = delta
        else:
            # Match failed: take the rotation from the range arrays (coasting on the last
            # spin rate overshoots when a spin slows down), keep half the last translation,
            # and restart matching from this scan.
            self.lost += 1
            dx, dy, dyaw = xyyaw(self.last_delta)
            if prev_ranges is not None and len(prev_ranges) == len(ranges):
                measured = range_rotation(prev_ranges, ranges, msg.angle_increment)
                if measured is not None:
                    dyaw = measured
            delta = se2(dx / 2, dy / 2, dyaw)
            self.pose = rigid(prev @ delta)
            self.last_delta = delta
            if len(pts) >= MIN_POINTS:
                self.set_keyframe(pts)
        return delta, ok


def make_messages(msg, pose, delta, dt):
    """(topic, message) pairs written for one scan, all stamped like the scan."""
    frame = msg.header.frame_id or 'laser'
    x, y, yaw = xyyaw(pose)
    dx, dy, dyaw = xyyaw(delta)

    odom = Odometry()
    odom.header.stamp = msg.header.stamp
    odom.header.frame_id = 'odom'
    odom.child_frame_id = frame
    odom.pose.pose.position.x, odom.pose.pose.position.y = x, y
    odom.pose.pose.orientation = yaw_quat(yaw)
    if dt > 0:
        odom.twist.twist.linear.x, odom.twist.twist.linear.y = dx / dt, dy / dt
        odom.twist.twist.angular.z = dyaw / dt

    step = TransformStamped()
    step.header.stamp = msg.header.stamp
    step.header.frame_id = frame + '_prev'
    step.child_frame_id = frame
    step.transform.translation.x, step.transform.translation.y = dx, dy
    step.transform.rotation = yaw_quat(dyaw)

    tf = TransformStamped()
    tf.header = odom.header
    tf.child_frame_id = frame
    tf.transform.translation.x, tf.transform.translation.y = x, y
    tf.transform.rotation = odom.pose.pose.orientation

    return [('/scan', msg), ('/lidar_odom', odom),
            ('/lidar_delta', step), ('/tf', TFMessage(transforms=[tf]))]


def stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


class MotionRecorder(Node):
    def __init__(self, args):
        super().__init__('lidar_motion_recorder')
        self.args = args
        self.writer = open_bag(args.out)
        self.tracker = Tracker(args)
        self.tf_pub = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/lidar_odom', 10)
        self.create_subscription(LaserScan, args.topic, self.on_scan, qos_profile_sensor_data)
        self.create_timer(1.0, self.report)

        self.last_t_ns = None
        self.scans = 0
        self.path_len = 0.0
        self.report_pose = np.eye(3)
        self.win_ms = []
        self.get_logger().info(f'Recording to {args.out}')

    def on_scan(self, msg):
        t_start = time.monotonic()
        t_ns = stamp_ns(msg)
        dt = (t_ns - self.last_t_ns) * 1e-9 if self.last_t_ns else 0.0
        self.last_t_ns = t_ns
        delta, _ = self.tracker.update(msg, dt)
        self.path_len += math.hypot(*xyyaw(delta)[:2])

        messages = make_messages(msg, self.tracker.pose, delta, dt)
        for topic, m in messages:
            self.writer.write(topic, serialize_message(m), t_ns)
        self.odom_pub.publish(messages[1][1])
        self.tf_pub.sendTransform(messages[3][1].transforms[0])

        self.scans += 1
        self.win_ms.append((time.monotonic() - t_start) * 1000.0)

    def report(self):
        if not self.win_ms:
            self.get_logger().warn(f'No scans on {self.args.topic} in the last second')
            return
        pose, ratios = self.tracker.pose, self.tracker.win_ratio
        x, y, yaw = xyyaw(pose)
        mx, my, myaw = xyyaw(inv(self.report_pose) @ pose)
        dist = math.hypot(mx, my)
        heading = f'toward {math.degrees(math.atan2(my, mx)):+4.0f}deg' if dist > 0.005 else '(still)'
        match = f'{100 * np.mean(ratios):3.0f}%' if ratios else '  - '
        self.get_logger().info(
            f'{len(self.win_ms):2d} Hz | pose x={x:+.3f} y={y:+.3f} yaw={math.degrees(yaw):+6.1f}deg | '
            f'last 1s: moved {dist:.3f}m {heading}, turned {math.degrees(myaw):+5.1f}deg | '
            f'match {match} | {np.mean(self.win_ms):4.1f} ms/scan | lost {self.tracker.lost}')
        self.report_pose = pose.copy()
        self.tracker.win_ratio, self.win_ms = [], []

    def close(self):
        if self.writer is None:
            return
        if hasattr(self.writer, 'close'):
            self.writer.close()
        self.writer = None  # older rosbag2_py closes the bag on destruction
        x, y, yaw = xyyaw(self.tracker.pose)
        # print, not the ROS logger: logging may already be shut down at this point.
        print(f'Saved {self.scans} scans to {self.args.out} | path {self.path_len:.2f} m | '
              f'final pose x={x:+.3f} y={y:+.3f} yaw={math.degrees(yaw):+.1f}deg | '
              f'lost {self.tracker.lost}', flush=True)


def add_tracking_args(parser):
    parser.add_argument('--min-range', type=float, default=0.15,
                        help='ignore returns closer than this (cable, mount), m')
    parser.add_argument('--max-corr', type=float, default=0.3,
                        help='max distance for a point to pair with the reference scan, m')
    parser.add_argument('--keyframe-dist', type=float, default=0.10,
                        help='start a new reference scan after moving this far, m')
    parser.add_argument('--keyframe-deg', type=float, default=5.0,
                        help='start a new reference scan after turning this much, deg')
    parser.add_argument('--damping', type=float, default=DEGENERATE_DAMPING,
                        help='corridor damping, share of the strongest ICP direction (0 = off)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='',
                        help='bag directory to create (default ~/lidar_bags/motion_<time>)')
    parser.add_argument('--topic', default='/scan')
    add_tracking_args(parser)
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])
    if not args.out:
        args.out = os.path.expanduser(
            f'~/lidar_bags/motion_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    args.out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rclpy.init()
    node = MotionRecorder(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # A scan that arrives mid-shutdown fails to publish ("context is invalid").
        if rclpy.ok():
            raise
    finally:
        node.close()  # writes the bag's summary and metadata.yaml
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
