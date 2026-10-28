import math
import numpy as np

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


# ============================================================
# CONFIGURATION
# ============================================================

BAG_PATH = r"D:\indoor mapping\2011-08-09-11-17-42.bag"

POSE_TOPICS = [
    "/robot_pose_ekf/odom_combined",
    "/base_odometry/odom",
    "/base_odometry/odometer",
    "/base_odometry/state",
    "/tf",
   "/map "
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_get(obj, attribute, default=None):
    """
    Safely get an attribute from a ROS message.
    """
    try:
        return getattr(obj, attribute)
    except Exception:
        return default


def extract_position(msg):
    """
    Try to extract x, y, z from common ROS message structures.

    Supports structures such as:

        msg.pose.pose.position

        msg.pose.position

        msg.position

        msg.transform.translation

        msg.translation
    """

    candidates = []

    # nav_msgs/Odometry
    try:
        candidates.append(
            msg.pose.pose.position
        )
    except Exception:
        pass

    # geometry_msgs/PoseStamped
    try:
        candidates.append(
            msg.pose.position
        )
    except Exception:
        pass

    # geometry_msgs/Pose
    try:
        candidates.append(
            msg.position
        )
    except Exception:
        pass

    # TransformStamped
    try:
        candidates.append(
            msg.transform.translation
        )
    except Exception:
        pass

    # Transform
    try:
        candidates.append(
            msg.translation
        )
    except Exception:
        pass

    for position in candidates:

        try:

            x = float(position.x)
            y = float(position.y)
            z = float(position.z)

            return x, y, z

        except Exception:
            continue

    return None


def extract_velocity(msg):
    """
    Try to extract linear velocity from common
    nav_msgs/Odometry structures.
    """

    candidates = []

    try:
        candidates.append(
            msg.twist.twist.linear
        )
    except Exception:
        pass

    try:
        candidates.append(
            msg.twist.linear
        )
    except Exception:
        pass

    for velocity in candidates:

        try:

            vx = float(velocity.x)
            vy = float(velocity.y)
            vz = float(velocity.z)

            return vx, vy, vz

        except Exception:
            continue

    return None


def quaternion_to_yaw(q):
    """
    Convert quaternion to yaw if available.
    """

    try:

        x = float(q.x)
        y = float(q.y)
        z = float(q.z)
        w = float(q.w)

        sin_yaw = 2.0 * (
            w * z + x * y
        )

        cos_yaw = 1.0 - 2.0 * (
            y * y + z * z
        )

        return math.atan2(
            sin_yaw,
            cos_yaw
        )

    except Exception:
        return None


def extract_yaw(msg):
    """
    Try common ROS pose/orientation structures.
    """

    candidates = []

    try:
        candidates.append(
            msg.pose.pose.orientation
        )
    except Exception:
        pass

    try:
        candidates.append(
            msg.pose.orientation
        )
    except Exception:
        pass

    try:
        candidates.append(
            msg.orientation
        )
    except Exception:
        pass

    try:
        candidates.append(
            msg.transform.rotation
        )
    except Exception:
        pass

    try:
        candidates.append(
            msg.rotation
        )
    except Exception:
        pass

    for q in candidates:

        yaw = quaternion_to_yaw(q)

        if yaw is not None:
            return yaw

    return None


def timestamp_seconds(timestamp_ns):
    return timestamp_ns / 1e9


# ============================================================
# ANALYZE NORMAL POSE TOPIC
# ============================================================

def analyze_pose_topic(
    reader,
    typestore,
    topic
):

    print()
    print("============================================================")
    print("TOPIC:", topic)
    print("============================================================")

    connections = [
        c
        for c in reader.connections
        if c.topic == topic
    ]

    if not connections:

        print("Topic not found.")

        return None

    connection = connections[0]

    print(
        "Message type:",
        connection.msgtype
    )

    print(
        "Connections:",
        len(connections)
    )

    positions = []
    velocities = []
    yaws = []
    timestamps = []

    message_count = 0

    # --------------------------------------------------------
    # Read messages
    # --------------------------------------------------------

    for connection, timestamp, rawdata in reader.messages(
        connections=connections
    ):

        message_count += 1

        try:

            msg = typestore.deserialize_cdr(
                rawdata,
                connection.msgtype
            )

        except Exception:

            try:

                msg = typestore.deserialize_ros1(
                    rawdata,
                    connection.msgtype
                )

            except Exception:

                continue

        # Position

        position = extract_position(msg)

        if position is not None:

            positions.append(
                position
            )

            timestamps.append(
                timestamp
            )

        # Velocity

        velocity = extract_velocity(msg)

        if velocity is not None:

            velocities.append(
                velocity
            )

        # Yaw

        yaw = extract_yaw(msg)

        if yaw is not None:

            yaws.append(
                yaw
            )

    print()
    print(
        "Messages:",
        message_count
    )

    print(
        "Messages with position:",
        len(positions)
    )

    print(
        "Messages with velocity:",
        len(velocities)
    )

    print(
        "Messages with orientation:",
        len(yaws)
    )

    # --------------------------------------------------------
    # No position
    # --------------------------------------------------------

    if len(positions) < 2:

        print()
        print(
            "Could not obtain enough position data "
            "from this topic."
        )

        return {
            "topic": topic,
            "message_count": message_count,
            "position_count": len(positions),
            "moving": None
        }

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    positions = np.asarray(
        positions,
        dtype=float
    )

    timestamps = np.asarray(
        timestamps,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Time span
    # --------------------------------------------------------

    duration = (
        timestamps[-1]
        -
        timestamps[0]
    ) / 1e9

    # --------------------------------------------------------
    # XY displacement between consecutive messages
    # --------------------------------------------------------

    dx = np.diff(
        positions[:, 0]
    )

    dy = np.diff(
        positions[:, 1]
    )

    step_distances = np.hypot(
        dx,
        dy
    )

    total_path_length = np.sum(
        step_distances
    )

    start_x = positions[0, 0]
    start_y = positions[0, 1]

    end_x = positions[-1, 0]
    end_y = positions[-1, 1]

    net_displacement = math.hypot(
        end_x - start_x,
        end_y - start_y
    )

    # Maximum distance from starting point

    distance_from_start = np.hypot(
        positions[:, 0] - start_x,
        positions[:, 1] - start_y
    )

    maximum_displacement = np.max(
        distance_from_start
    )

    # --------------------------------------------------------
    # Average speed based on path
    # --------------------------------------------------------

    if duration > 0:

        average_speed = (
            total_path_length
            /
            duration
        )

    else:

        average_speed = 0.0

    # --------------------------------------------------------
    # Velocity information
    # --------------------------------------------------------

    maximum_velocity = None

    if len(velocities) > 0:

        velocities = np.asarray(
            velocities,
            dtype=float
        )

        velocity_magnitudes = np.linalg.norm(
            velocities[:, :2],
            axis=1
        )

        maximum_velocity = np.max(
            velocity_magnitudes
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("TIME / MOTION")
    print("----------------------------")

    print(
        f"Duration:              {duration:.2f} s"
    )

    print(
        f"Start position:        "
        f"({start_x:.3f}, {start_y:.3f})"
    )

    print(
        f"End position:          "
        f"({end_x:.3f}, {end_y:.3f})"
    )

    print(
        f"Net displacement:      "
        f"{net_displacement:.3f} m"
    )

    print(
        f"Maximum displacement:  "
        f"{maximum_displacement:.3f} m"
    )

    print(
        f"Total path length:     "
        f"{total_path_length:.3f} m"
    )

    print(
        f"Average path speed:    "
        f"{average_speed:.3f} m/s"
    )

    if maximum_velocity is not None:

        print(
            f"Maximum reported speed:"
            f" {maximum_velocity:.3f} m/s"
        )

    # --------------------------------------------------------
    # Stationary decision
    # --------------------------------------------------------

    # We use multiple indicators instead of only
    # looking at the first and last pose.

    if (
        maximum_displacement < 0.10
        and
        total_path_length < 0.50
    ):

        moving = False

    elif (
        maximum_displacement > 1.0
        or
        total_path_length > 2.0
    ):

        moving = True

    else:

        moving = None

    print()

    if moving is True:

        print(
            "RESULT: ROBOT APPEARS TO BE MOVING."
        )

    elif moving is False:

        print(
            "RESULT: ROBOT APPEARS TO BE STATIONARY."
        )

    else:

        print(
            "RESULT: MOTION IS UNCERTAIN."
        )

    return {
        "topic": topic,
        "message_count": message_count,
        "position_count": len(positions),
        "duration": duration,
        "net_displacement": net_displacement,
        "maximum_displacement": maximum_displacement,
        "total_path_length": total_path_length,
        "average_speed": average_speed,
        "moving": moving
    }


# ============================================================
# SPECIAL ANALYSIS FOR /tf
# ============================================================

def analyze_tf_topic(
    reader,
    typestore,
    topic="/tf"
):

    print()
    print("============================================================")
    print("TF ANALYSIS")
    print("============================================================")

    connections = [
        c
        for c in reader.connections
        if c.topic == topic
    ]

    if not connections:

        print("TF topic not found.")

        return

    print(
        "TF connections:",
        len(connections)
    )

    frame_pairs = {}

    total_messages = 0

    for connection, timestamp, rawdata in reader.messages(
        connections=connections
    ):

        total_messages += 1

        try:

            msg = typestore.deserialize_ros1(
                rawdata,
                connection.msgtype
            )

        except Exception:

            continue

        # TF is usually tf2_msgs/TFMessage
        try:

            transforms = msg.transforms

        except Exception:

            continue

        for transform in transforms:

            try:

                parent = (
                    transform.header.frame_id
                )

                child = (
                    transform.child_frame_id
                )

                key = (
                    parent,
                    child
                )

                frame_pairs[key] = (
                    frame_pairs.get(
                        key,
                        0
                    )
                    +
                    1
                )

            except Exception:

                continue

    print()
    print(
        "TF messages:",
        total_messages
    )

    print()
    print(
        "Frame relationships found:"
    )

    sorted_pairs = sorted(
        frame_pairs.items(),
        key=lambda item:
        item[1],
        reverse=True
    )

    for (parent, child), count in sorted_pairs[:30]:

        print(
            f"  {parent} -> {child}"
            f"   ({count} transforms)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("============================================================")
    print("ROS BAG ROBOT MOTION DIAGNOSTIC")
    print("============================================================")

    print()
    print(
        "Bag:",
        BAG_PATH
    )

    typestore = get_typestore(
        Stores.ROS1_NOETIC
    )

    results = []

    with Reader(BAG_PATH) as reader:

        print()
        print(
            "Bag opened successfully."
        )

        print()
        print(
            "Available topics:"
        )

        for connection in reader.connections:

            print(
                f"  {connection.topic:40s}"
                f" {connection.msgtype}"
            )

        # ----------------------------------------------------
        # Analyze likely pose topics
        # ----------------------------------------------------

        for topic in POSE_TOPICS:

            if topic == "/tf":

                continue

            result = analyze_pose_topic(
                reader,
                typestore,
                topic
            )

            if result is not None:

                results.append(
                    result
                )

        # ----------------------------------------------------
        # Analyze TF separately
        # ----------------------------------------------------

        analyze_tf_topic(
            reader,
            typestore,
            "/tf"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("============================================================")
    print("FINAL SUMMARY")
    print("============================================================")

    print()

    if not results:

        print(
            "No usable pose topics were found."
        )

        return

    for result in results:

        print(
            f"{result['topic']}"
        )

        print(
            f"    Messages: "
            f"{result['message_count']}"
        )

        print(
            f"    Position messages: "
            f"{result['position_count']}"
        )

        if result.get(
            "duration"
        ) is not None:

            print(
                f"    Duration: "
                f"{result['duration']:.2f} s"
            )

            print(
                f"    Net displacement: "
                f"{result['net_displacement']:.3f} m"
            )

            print(
                f"    Maximum displacement: "
                f"{result['maximum_displacement']:.3f} m"
            )

            print(
                f"    Total path: "
                f"{result['total_path_length']:.3f} m"
            )

            print(
                f"    Average speed: "
                f"{result['average_speed']:.3f} m/s"
            )

            if result["moving"] is True:

                print(
                    "    STATUS: MOVING"
                )

            elif result["moving"] is False:

                print(
                    "    STATUS: STATIONARY"
                )

            else:

                print(
                    "    STATUS: UNCERTAIN"
                )

        print()


if __name__ == "__main__":
    main()