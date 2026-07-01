"""
playback_csv.py
===============
Playback CSV recordings with skeleton visualization.

Supports both old CSV format (world angles only) and new format (world + intrinsic angles).
If both world and intrinsic angles are available, displays them side by side.

CSV format compatibility:
  Old: {joint}_{axis}_deg, {joint}_{axis}_vel_dps, {joint}_{axis}_acc_dps2
  New: {joint}_world_{axis}_deg, {joint}_intrinsic_{axis}_deg, ...

Configuration:
  CSV_PATH: Path to the CSV recording file
  CONFIG_PATH: Path to the JSON configuration file
"""

import csv
import json
import time
import numpy as np
from pathlib import Path

import aquisition_and_processing.kinematics_core as kc
from visualisation.visualizer import SkeletonVisualizer, DualSkeletonVisualizer

# ==============================================================================
# CONFIGURATION - SET THESE
# ==============================================================================
CSV_PATH = "recordings/recording_20260616_135504_imu.csv"
CONFIG_PATH = Path(__file__).parent / "configs" / "kinematics_config_arm.json"
PLAYBACK_SPEED = 1.0  # 1.0 = real-time, >1.0 = faster than real-time
LOG_ANGLES = True  # If True, saves angles to CSV for debugging 

# ===================================================================s===========
# CSV READER
# ==============================================================================
class CSVPlayer:
    """Load and playback CSV recording data."""
    
    AXES = ("roll", "pitch", "yaw")
    
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.rows = []
        self.timestamps = []
        self.has_world = False
        self.has_intrinsic = False
        self.joints = set()
        self._load_csv()
    
    def _load_csv(self):
        """Load CSV and detect format."""
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)
        
        if not self.rows:
            raise ValueError("CSV is empty")
        
        # Detect format and available joints
        first_row = self.rows[0]
        for key in first_row.keys():
            # Parse joint name from column key
            if "_world_roll_deg" in key:
                joint = key.replace("_world_roll_deg", "")
                self.joints.add(joint)
                self.has_world = True
            elif "_intrinsic_roll_deg" in key:
                joint = key.replace("_intrinsic_roll_deg", "")
                self.joints.add(joint)
                self.has_intrinsic = True
            elif "_roll_deg" in key and "_vel_dps" not in key and "_acc_dps2" not in key:
                # Old format
                joint = key.replace("_roll_deg", "")
                self.joints.add(joint)
                self.has_world = True
        
        # Extract timestamps
        for row in self.rows:
            self.timestamps.append(float(row["time_s"]))
        
        print(f"[CSV] Loaded {len(self.rows)} rows from {self.csv_path.name}")
        print(f"[CSV] Format: world={self.has_world}, intrinsic={self.has_intrinsic}")
        print(f"[CSV] Joints: {sorted(self.joints)}")
    
    def _get_angles_from_row(self, row, angle_type="world"):
        """Extract RPY angles from a CSV row. angle_type: 'world' or 'intrinsic'."""
        angles = {}
        
        for joint in self.joints:
            rpy = []
            for axis in self.AXES:
                if angle_type == "world":
                    # Try new format first
                    key = f"{joint}_world_{axis}_deg"
                    if key not in row:
                        # Fall back to old format
                        key = f"{joint}_{axis}_deg"
                    val = row.get(key)
                    if val is not None:
                        rpy.append(float(val))
                    else:
                        rpy.append(0.0)
                elif angle_type == "intrinsic":
                    key = f"{joint}_intrinsic_{axis}_deg"
                    val = row.get(key)
                    if val is not None:
                        rpy.append(float(val))
                    else:
                        rpy.append(0.0)
            
            if rpy:
                angles[joint] = tuple(rpy)
        
        return angles
    
    def get_frame(self, row_idx, angle_type="world"):
        """Get frame data for a given row index, using specified angle type."""
        if row_idx < 0 or row_idx >= len(self.rows):
            return None
        
        row = self.rows[row_idx]
        angles = self._get_angles_from_row(row, angle_type)
        return angles
    
    def get_timestamp(self, row_idx):
        """Get timestamp for a row."""
        if row_idx < 0 or row_idx >= len(self.rows):
            return None
        return self.timestamps[row_idx]
    
    def __len__(self):
        return len(self.rows)


# ==============================================================================
# FRAME BUILDER (angles -> skeleton frame)
# ==============================================================================
def build_frame_from_angles(state_dict, sorted_joints, active_joints, endpoints, angles):
    """
    Build a skeleton frame from angle data.
    
    Args:
        angles: dict joint_name -> (roll_deg, pitch_deg, yaw_deg)
    
    Returns:
        frame dict for visualizer
    """
    world_rot = {}
    for joint in sorted_joints:
        if joint in angles:
            r, p, y = angles[joint]
            world_rot[joint] = kc.rpy_to_rotation_matrix_xyz(r, p, y)
        else:
            world_rot[joint] = np.eye(3)
    
    joint_pos, endpoint_pos = kc.run_forward_kinematics_absolute(
        state_dict, sorted_joints, active_joints, world_rot, endpoints)
    
    joints = {
        name: {
            "name": name,
            "parent": state_dict[name].get("parent"),
            "pos": joint_pos[name],
            "R": world_rot[name],
        }
        for name in sorted_joints
    }
    eps = {
        ep_name: {
            "parent": endpoints[ep_name]["parent"],
            "pos": endpoint_pos[ep_name],
        }
        for ep_name in endpoint_pos
    }
    return {"joints": joints, "endpoints": eps}


def build_frame_from_intrinsic_angles(state_dict, sorted_joints, active_joints, endpoints,
                                      frame_change, parent_dict, intrinsic_angles):
    """
    Build a skeleton frame from intrinsic angles by converting to world rotations first.
    
    Args:
        intrinsic_angles: dict joint_name -> (roll_deg, pitch_deg, yaw_deg)
        frame_change: dict joint_name -> frame change rotation matrix (from config)
        parent_dict: dict joint_name -> parent_name
    
    Returns:
        frame dict for visualizer (with world-space positions)
    """
    # Convert intrinsic angles to intrinsic rotation matrices
    intrinsic_rot = {}
    for joint in sorted_joints:
        if joint in intrinsic_angles:
            r, p, y = intrinsic_angles[joint]
            intrinsic_rot[joint] = kc.rpy_to_rotation_matrix_xyz(r, p, y)
        else:
            intrinsic_rot[joint] = np.eye(3)
    
    # Convert intrinsic rotations to world rotations via the kinematics core.
    # This is the required path: each intrinsic (expected-frame) rotation is
    # lifted into world axes and chained down the tree by
    # build_world_rotations_from_intrinsic, which returns each segment's world
    # rotation away from neutral (identity at calibration) -- exactly the
    # convention run_forward_kinematics_absolute consumes for positioning, and
    # the same convention the world path produces. Guard frame_change with
    # identity so a joint that is present in the CSV but missing a frame-change
    # matrix in the config cannot raise inside the core function.
    fc = {j: np.asarray(frame_change.get(j, np.eye(3)), float) for j in sorted_joints}
    world_rot = kc.build_world_rotations_from_intrinsic(
        fc, intrinsic_rot, sorted_joints, active_joints, parent_dict)
    
    # Build frame using world rotations (same as absolute kinematics)
    joint_pos, endpoint_pos = kc.run_forward_kinematics_absolute(
        state_dict, sorted_joints, active_joints, world_rot, endpoints)
    
    joints = {
        name: {
            "name": name,
            "parent": state_dict[name].get("parent"),
            "pos": joint_pos[name],
            "R": world_rot[name],
        }
        for name in sorted_joints
    }
    eps = {
        ep_name: {
            "parent": endpoints[ep_name]["parent"],
            "pos": endpoint_pos[ep_name],
        }
        for ep_name in endpoint_pos
    }
    return {"joints": joints, "endpoints": eps}


# ==============================================================================
# LOGGING
# ==============================================================================
def write_angle_comparison_csv(log_path, player, state_dict, sorted_joints, active_joints,
                               frame_change, parent_dict):
    """
    Write a CSV comparing world angles from CSV with world angles computed from intrinsic.
    Only includes computed columns if intrinsic angles exist in the CSV.
    
    Args:
        log_path: path to write the log CSV file
        player: CSVPlayer instance
        state_dict, sorted_joints, active_joints: kinematics topology
        frame_change, parent_dict: kinematics matrices
    """
    import csv
    
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, "w", newline="") as f:
        fieldnames = ["row_idx", "time_s"]
        
        # Add world angles from CSV
        for joint in sorted_joints:
            for axis in ("roll", "pitch", "yaw"):
                fieldnames.append(f"{joint}_world_{axis}_deg")
        
        # Only add computed columns if intrinsic angles exist in the CSV
        if player.has_intrinsic:
            for joint in sorted_joints:
                for axis in ("roll", "pitch", "yaw"):
                    fieldnames.append(f"{joint}_world_from_intrinsic_{axis}_deg")
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each row
        for row_idx in range(len(player)):
            row_data = {"row_idx": row_idx, "time_s": f"{player.get_timestamp(row_idx):.6f}"}
            
            # Get world angles from CSV
            world_angles_csv = player.get_frame(row_idx, "world")
            for joint in sorted_joints:
                if joint in world_angles_csv:
                    r, p, y = world_angles_csv[joint]
                    row_data[f"{joint}_world_roll_deg"] = f"{r:.4f}"
                    row_data[f"{joint}_world_pitch_deg"] = f"{p:.4f}"
                    row_data[f"{joint}_world_yaw_deg"] = f"{y:.4f}"
                else:
                    row_data[f"{joint}_world_roll_deg"] = "N/A"
                    row_data[f"{joint}_world_pitch_deg"] = "N/A"
                    row_data[f"{joint}_world_yaw_deg"] = "N/A"
            
            # Get world angles computed from intrinsic (only if intrinsic angles exist)
            if player.has_intrinsic:
                intrinsic_angles = player.get_frame(row_idx, "intrinsic")
                
                # Convert intrinsic angles to intrinsic rotation matrices
                intrinsic_rot = {}
                for joint in sorted_joints:
                    if joint in intrinsic_angles:
                        r, p, y = intrinsic_angles[joint]
                        intrinsic_rot[joint] = kc.rpy_to_rotation_matrix_xyz(r, p, y)
                    else:
                        intrinsic_rot[joint] = np.eye(3)
                
                # Convert to world rotations
                fc = {j: np.asarray(frame_change.get(j, np.eye(3)), float) for j in sorted_joints}
                world_rot = kc.build_world_rotations_from_intrinsic(
                    fc, intrinsic_rot, sorted_joints, active_joints, parent_dict)
                
                # Extract angles from world rotations
                for joint in sorted_joints:
                    if joint in world_rot:
                        r, p, y = kc.matrix_to_rpy_xyz(world_rot[joint])
                        row_data[f"{joint}_world_from_intrinsic_roll_deg"] = f"{r:.4f}"
                        row_data[f"{joint}_world_from_intrinsic_pitch_deg"] = f"{p:.4f}"
                        row_data[f"{joint}_world_from_intrinsic_yaw_deg"] = f"{y:.4f}"
                    else:
                        row_data[f"{joint}_world_from_intrinsic_roll_deg"] = "N/A"
                        row_data[f"{joint}_world_from_intrinsic_pitch_deg"] = "N/A"
                        row_data[f"{joint}_world_from_intrinsic_yaw_deg"] = "N/A"
            
            writer.writerow(row_data)
    
    print(f"[logging] Wrote angle comparison to {log_path}")


# ==============================================================================
# PLAYBACK CONTROLLER
# ==============================================================================
def playback_csv(csv_path, config_path, max_playback_speed=1.0):
    """
    Load CSV and config, then play back with visualization.
    
    Args:
        csv_path: path to CSV recording
        config_path: path to JSON config
        max_playback_speed: playback speed multiplier (1.0 = real-time)
    """
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Parse topology
    active_joints = set()  # We'll determine from CSV joints
    
    # Load CSV
    player = CSVPlayer(csv_path)
    active_joints = player.joints
    
    # Parse kinematics
    state_dict, sorted_joints, endpoints = kc.parse_topology(config, active_joints)
    
    # Get frame change matrices and parent dict for intrinsic conversion
    frame_change = kc.get_frame_change_matrices(state_dict, sorted_joints)
    parent_dict = {j: state_dict[j].get("parent") for j in sorted_joints}
    
    if not sorted_joints:
        print("[ERROR] No active joints found. Check CSV column names and config.")
        return
    
    print(f"[playback] Active joints: {sorted_joints}")
    
    # Playback state. Advancement is driven purely by absolute wall-clock time
    # (see _advance_to_walltime) so it is idempotent. The dual visualizer ticks
    # two windows on independent timers, and both call should_close() each tick;
    # an incremental "subtract elapsed time" scheme would be advanced twice per
    # frame and let the two windows desync (and race on row_idx). Computing the
    # target row from total elapsed wall time instead means repeated calls -- from
    # one window or two -- converge to the same row.
    row_idx = [0]            # current row, shared (list for nested-scope mutation)
    is_playing = [True]
    wall_start = [None]      # perf_counter() captured at first advance
    ts_start = player.get_timestamp(0)
    
    def get_frame_world():
        """Get current frame for world skeleton."""
        if row_idx[0] < len(player):
            return build_frame_from_angles(
                state_dict, sorted_joints, active_joints, endpoints,
                player.get_frame(row_idx[0], "world")
            )
        return None
    
    def get_frame_intrinsic():
        """Get current frame from intrinsic angles (converted to world via build_world_rotations_from_intrinsic)."""
        if row_idx[0] < len(player):
            return build_frame_from_intrinsic_angles(
                state_dict, sorted_joints, active_joints, endpoints,
                frame_change, parent_dict,
                player.get_frame(row_idx[0], "intrinsic")
            )
        return None
    
    def _advance_to_walltime():
        """Set row_idx from elapsed wall time. Idempotent and safe to call from
        multiple visualizer timers: each call recomputes the same target row
        from the shared wall clock, so the world and intrinsic windows stay in
        lock-step instead of double-stepping or drifting apart."""
        if not is_playing[0]:
            return
        if wall_start[0] is None:
            wall_start[0] = time.perf_counter()
        target_ts = ts_start + (time.perf_counter() - wall_start[0]) * max_playback_speed
        i = row_idx[0]
        n = len(player)
        while i < n - 1 and player.get_timestamp(i + 1) <= target_ts:
            i += 1
        row_idx[0] = i

    def should_close():
        """Advance playback and report completion."""
        _advance_to_walltime()
        return row_idx[0] >= len(player) - 1
    
    # Start visualization
    print(f"\n[playback] Starting playback...")
    print(f"           Total rows: {len(player)}")
    print(f"           Duration: {player.get_timestamp(len(player) - 1):.2f}s")
    print(f"           Playback speed: {max_playback_speed}x\n")
    
    if player.has_world and player.has_intrinsic:
        print("[playback] Displaying both world and intrinsic angles (side by side)\n")
        # NOTE: DualSkeletonVisualizer (in visualizer.py) launches two
        # BackgroundPlotter windows on separate threads. The wall-clock playback
        # timing below keeps the two windows synchronized, but the underlying
        # two-Qt-event-loops-on-non-main-threads design is itself fragile on some
        # platforms. Fixing that requires touching visualizer.py.
        visualizer = DualSkeletonVisualizer(
            axis_length=12.0,
            title="CSV Playback"
        )
        visualizer.run(get_frame_world, get_frame_intrinsic, should_close)
    elif player.has_world:
        print("[playback] Displaying world angles\n")
        visualizer = SkeletonVisualizer(
            axis_length=12.0,
            title="CSV Playback - World Angles"
        )
        visualizer.run(get_frame_world, should_close)
    else:
        print("[playback] Displaying intrinsic angles\n")
        visualizer = SkeletonVisualizer(
            axis_length=12.0,
            title="CSV Playback - Intrinsic Angles"
        )
        visualizer.run(get_frame_intrinsic, should_close)
    
    # Write angle comparison log if enabled
    if LOG_ANGLES:
        log_filename = f"angle_comparison_{Path(csv_path).stem}.csv"
        log_path = Path(__file__).parent / "logs" / log_filename
        write_angle_comparison_csv(log_path, player, state_dict, sorted_joints, active_joints,
                                   frame_change, parent_dict)
    
    print("[playback] Finished.")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    playback_csv(CSV_PATH, CONFIG_PATH, PLAYBACK_SPEED)