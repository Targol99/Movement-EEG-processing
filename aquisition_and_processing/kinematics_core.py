"""
kinematics_core.py
==================
Pure kinematics math. No hardware SDK, no plotting, no I/O side effects.

This module is MAC-agnostic: every dictionary is keyed by a joint's unique
*name* (e.g. "Shoulder", "Elbow", "Wrist"), and parents are referenced by name
too. The mapping from a physical sensor (MAC) to a joint name is decided at
runtime by the engine and never reaches this file.

Frames (see kinematics_math.md):
  W = World FLU,  S = sensor body,  E = expected sensor frame for a joint's
  neutral pose. The sensor reports orientation relative to its earth frame, but
  that earth frame cancels in the calibration delta, so it never appears here.

A rotation R_{A<-B} maps coordinates from frame B into frame A: v_A = R @ v_B.
"""

import numpy as np


# ------------------------------------------------------------------ rotations
def rpy_to_rotation_matrix_xyz(roll_deg, pitch_deg, yaw_deg):
    """
    Roll/pitch/yaw -> rotation matrix using fixed-axis (extrinsic) X -> Y -> Z,
    which equals R = Rz @ Ry @ Rx (standard Movella/Xsens aerospace convention).
    If your angles are meant as INTRINSIC X -> Y -> Z, return R_x @ R_y @ R_z.
    """
    r, p, y = np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return R_z @ (R_y @ R_x)


def matrix_to_rpy_xyz(R):
    """Inverse of rpy_to_rotation_matrix_xyz (ZYX extraction). Degrees. Display only."""
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    pitch = np.degrees(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def rotation_between_vectors(a, b):
    """
    Smallest rotation R mapping unit(a) onto unit(b): R @ a_hat == b_hat.
    Pure geometry, used by the renderer to aim a box along a bone. Handles the
    parallel and antiparallel degenerate cases.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3)
    a, b = a / na, b / nb
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        # antiparallel: 180 deg about any axis perpendicular to a
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


# ------------------------------------------------------- startup-built matrices
def compute_frame_change_matrices(state_dict, sorted_joints):
    """R_frame_change = R_{E<-W} for each joint, from its JSON `ang` (deg)."""
    frame_changes = {}
    for joint in sorted_joints:
        ang = state_dict[joint]["ang"]
        frame_changes[joint] = rpy_to_rotation_matrix_xyz(ang[0], ang[1], ang[2])

    return frame_changes

def get_frame_change_matrices(state_dict, sorted_joints):
    """R_frame_change = R_{E<-W} for each joint, from its JSON `ang` (deg)."""
    frame_changes = {}
    for joint in sorted_joints:
        rot_mat = np.asarray(state_dict[joint]["rot_mat"])
        frame_changes[joint] = rot_mat
    return frame_changes


# ----------------------------------------------------- sensor -> world (tare)
def sensor_to_world_calibrated(R_frame_change, R_calib, R_sensor_live):
    """
    Calibrated (tare) mapping. Returns the segment's rotation relative to its
    captured neutral pose, in World FLU:
        D = R_frame_change.T @ (R_calib.T @ R_sensor_live) @ R_frame_change
    Identity at the calibration instant.
    """
    R_delta_body = R_calib.T @ R_sensor_live
    return R_frame_change.T @ R_delta_body @ R_frame_change


def build_world_rotations_calibrated(frame_change, calibration, live_states, sorted_joints):
    """Per-joint world rotations for every active joint (all dicts keyed by joint name)."""
    return {
        joint: sensor_to_world_calibrated(frame_change[joint], calibration[joint], live_states[joint])
        for joint in sorted_joints
    }


def build_intrinsic_joint_rotations(frame_change, calibration, live_states,
                                    sorted_joints, active_joints, parent_dict):
    active_joints = set(active_joints)
    # Each joint's own world delta (identical to build_world_rotations_calibrated)
    Wd = {j: frame_change[j].T @ (calibration[j].T @ live_states[j]) @ frame_change[j]
          for j in sorted_joints}
    intrinsic = {}
    for joint in sorted_joints:
        F = frame_change[joint]
        parent = parent_dict.get(joint)
        if parent is None or parent not in active_joints:
            rel = Wd[joint]                     # root: absolute world delta
        else:
            rel = Wd[parent].T @ Wd[joint]      # child: parent-relative IN WORLD
        intrinsic[joint] = F @ rel @ F.T        # express in the joint's expected frame
    return intrinsic
 
 
def build_world_rotations_from_intrinsic(frame_change, intrinsic_rotations,
                                         sorted_joints, active_joints,
                                         parent_dict):
    """
    Reconstruct world rotations from intrinsic joint rotations (the inverse of
    build_intrinsic_joint_rotations, composed down the chain).
 
    Output convention:
        world_rotations[j] = S_j @ S0_j.T
    i.e. each segment's world-frame rotation away from its NEUTRAL
    (calibration) pose -- identity at calibration. This is exactly the
    quantity the endpoint-position code consumes
    (current_offset = W[parent] @ neutral_offset), and the fixed sensor
    mount cancels inside it.
 
    Derivation. With M_j = frame_change[j] @ calibration[j]:
 
        child:  D_c = M_c @ (Q0.T @ Q) @ M_c.T
                    = F_c S0_c (S0_c.T S0_p S_p.T S_c) S0_c.T F_c.T
        =>      S_c = S_p S0_p.T S0_c S0_c.T F_c.T D_c F_c S0_c
        =>      W_c = S_c S0_c.T = W_p @ F_c.T @ D_c @ F_c
 
        root:   D_r = M_r @ (S0.T S) @ M_r.T  =>  W_r = F_r.T @ D_r @ F_r
 
    The calibration matrices cancel completely: only frame_change and the
    intrinsic rotations are needed.
 
    Recursion (parent-before-child order required):
        W[j] = (W[parent] if active parent else I) @ F[j].T @ D[j] @ F[j]
 
    Returns:
        dict joint_name -> np.ndarray (3x3 rotation matrix)
    """
    active_joints = set(active_joints)
    world_rotations = {}
 
    for joint in sorted_joints:
        F = frame_change[joint]
        # This joint's contribution, re-expressed from its expected frame
        # into world axes
        local_world = F.T @ intrinsic_rotations[joint] @ F
 
        parent = parent_dict.get(joint)
        if parent is None or parent not in active_joints:
            world_rotations[joint] = local_world
        else:
            world_rotations[joint] = world_rotations[parent] @ local_world
 
    return world_rotations
# ------------------------------------------------------------------ topology
def parse_topology(config, active_joints):
    """
    Returns (state_dict, sorted_joints, endpoints).

    active_joints: the set/iterable of joint names that have a sensor mapped to
                   them this session. Joints in the config but not mapped are
                   simply ignored.

    sorted_joints: active joints in parent-before-child order.
    endpoints:     dict name -> {"parent": joint_name, "pos": [...]}. Optional
                   nodes with no sensor and no children; world position computed
                   from the parent. Endpoints whose parent is not an active joint
                   are dropped.
    """
    active_joints = set(active_joints)
    state_dict = config.get("state", {})
    endpoints_raw = config.get("endpoints", {}) or {}

    # Only keep joints that exist in the config AND were mapped to a sensor.
    active = {name: info for name, info in state_dict.items() if name in active_joints}

    ordered, visited = [], set()
    while len(ordered) < len(active):
        progress = False
        for name, info in active.items():
            if name in visited:
                continue
            parent = info.get("parent")
            # Root if it has no parent, or its parent is not an active joint.
            if parent is None or parent in visited or parent not in active:
                ordered.append(name)
                visited.add(name)
                progress = True
        if not progress:
            raise ValueError("[CRITICAL] Loop detected in configuration hierarchy!")

    endpoints = {name: ep for name, ep in endpoints_raw.items() if ep.get("parent") in active}
    return state_dict, ordered, endpoints


# --------------------------------------------------------- forward kinematics
def run_forward_kinematics_absolute(state_dict, sorted_joints, active_joints, world_rotations, endpoints=None):
    """
    Propagate world positions through the joint tree, then place any endpoints.
    All joint dicts are keyed by joint name.

    Returns:
        joint_positions:    dict joint_name -> np.array([x, y, z])
        endpoint_positions: dict name       -> np.array([x, y, z])

    A child's neutral link offset (world-frame difference of JSON `pos`) is swung
    by the PARENT's world rotation. Endpoints follow the same rule, rigidly fixed
    to their parent segment, and contribute no rotation of their own.
    """
    active_joints = set(active_joints)
    joint_positions = {}
    for joint in sorted_joints:
        info = state_dict[joint]
        parent = info.get("parent")
        if parent is None or parent not in active_joints:
            joint_positions[joint] = np.array(info["pos"], dtype=float)
        else:
            offset = np.array(info["pos"], float) - np.array(state_dict[parent]["pos"], float)
            joint_positions[joint] = joint_positions[parent] + world_rotations[parent] @ offset

    endpoint_positions = {}
    if endpoints:
        for name, ep in endpoints.items():
            parent = ep.get("parent")
            ep_pos = np.array(ep["pos"], float)
            if parent in joint_positions and parent in world_rotations:
                offset = ep_pos - np.array(state_dict[parent]["pos"], float)
                endpoint_positions[name] = joint_positions[parent] + world_rotations[parent] @ offset
            else:
                endpoint_positions[name] = ep_pos

    return joint_positions, endpoint_positions


def run_forward_kinematics_intrinsic(state_dict, sorted_joints, active_joints,
                                     intrinsic_rotations, world_rotations=None,
                                     endpoints=None):
    """
    Return per-joint intrinsic rotations (parent-independent, vs calibration,
    in each joint's expected frame) plus optional endpoint positions.
 
    NOTE: intrinsic rotations alone cannot place points in the world -- they
    deliberately discard parent motion. Endpoint positions therefore still
    require absolute world rotations (pass `world_rotations`, e.g. from
    build_world_rotations_calibrated).
 
    Returns:
        joint_rotations:    dict joint -> 3x3 intrinsic rotation
        endpoint_positions: dict name  -> np.array([x, y, z])
    """
    active_joints = set(active_joints)
    joint_rotations = intrinsic_rotations  # already computed with proper framing
 
    endpoint_positions = {}
    if endpoints and world_rotations:
        joint_positions = {}
        for jnt in sorted_joints:
            info = state_dict[jnt]
            par = info.get("parent")
            if par is None or par not in active_joints:
                joint_positions[jnt] = np.array(info["pos"], dtype=float)
            else:
                offset = (np.array(info["pos"], float)
                          - np.array(state_dict[par]["pos"], float))
                joint_positions[jnt] = (joint_positions[par]
                                        + world_rotations[par] @ offset)
 
        for name, ep in endpoints.items():
            par = ep.get("parent")
            ep_pos = np.array(ep["pos"], float)
            if par in joint_positions and par in world_rotations:
                offset = ep_pos - np.array(state_dict[par]["pos"], float)
                endpoint_positions[name] = (joint_positions[par]
                                            + world_rotations[par] @ offset)
            else:
                endpoint_positions[name] = ep_pos
 
    return joint_rotations, endpoint_positions
 

def link_length(pos_a, pos_b):
    """Convenience invariant check: Euclidean distance between two world points."""
    return float(np.linalg.norm(np.asarray(pos_a, float) - np.asarray(pos_b, float)))



if __name__ == "__main__":
    import json
    from pathlib import Path

    config_path = Path(__file__).parent / "2_arms.json"
    with open(config_path) as f:
        config = json.load(f)

    print("CONFIG PARSE TEST")
    state_dict, sorted_joints, endpoints = parse_topology(config, active_joints={"LeftShoulder", "LeftElbow", "RightShoulder", "RightElbow"})
    print("state_dict:", state_dict)
    print("sorted_joints:", sorted_joints)
    print("endpoints:", endpoints)

    print("\nangle -> rotation matrix")
    frame_changes = compute_frame_change_matrices(state_dict, sorted_joints)
    for joint in sorted_joints:
        print(f"{joint}:")
        print(np.array2string(frame_changes[joint], precision=2, suppress_small=True))
    print("\nrotation matrix")
    frame_changes = get_frame_change_matrices(state_dict, sorted_joints)
    for joint in sorted_joints:
        print(f"{joint}:")
        print(np.array2string(frame_changes[joint], precision=2, suppress_small=True))
