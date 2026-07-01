# Motion Tracking System with EEG Integration

A real-time motion capture system that combines Movella DOT IMU sensors for body tracking with EEG data collection, featuring live 3D visualization and CSV logging.

---

## 🚀 Quick Start: How to Run an Experiment

### Prerequisites
- **Hardware**: Movella DOT IMUs (connected via Bluetooth) and an EEG stream (LSL)
- **Python Packages**: See requirements in the code (numpy, scipy, pyvista, pyvistaqt, pylsl)
- **Config Files**: JSON files in the workspace root (e.g., `2_arms.json`, `kinematics_config.json`)

### Step-by-Step Workflow

#### **Step 1: Start the Program**
```bash
python kinematics_engine_live_write.py
```
Or run without visualization (headless mode):
```bash
python kinematics_engine_live_write.py --no-viz
```

The program will:
- Connect to all Movella DOT sensors (up to 4)
- Attempt to connect to the EEG stream via LSL
- Wait for you to proceed to the next step

#### **Step 2: Select Configuration & Map Sensors**
The system will prompt you to:
1. **Choose a config file** - Pick a JSON file defining your skeleton structure (e.g., `2_arms.json`)
2. **Map each joint to an IMU** - The program shows connected IMUs by index; you assign one to each joint in your skeleton

Example output:
```
Available configs:
  [0] 2_arms.json
  [1] kinematics_config.json
  [q] Quit
Select config index: 0

Connected IMUs:
  [0] 00:0D:EF:12:AB:CD
  [1] 00:0D:EF:34:EF:01

Map each joint to a connected IMU (press Enter to skip a joint).
  IMU index for joint 'RightShoulder' (Enter to skip): 0
  IMU index for joint 'RightElbow' (Enter to skip): 1
```

#### **Step 3: Calibrate (Press 'R')**
- Place the subject in a **neutral pose** (e.g., arms at sides, relaxed)
- Press **'R'** in the console to capture this pose as the reference (tare)
- The system will now compute all rotations *relative* to this neutral pose
- 3D visualization begins (if enabled)

#### **Step 4: Record Data with 'W'**
After calibration, press **'W'** to start CSV recording:
- A new file `recordings/recording_YYYYMMDD_HHMMSS.csv` is created
- The system logs:
  - **Time** (in seconds, t=0 when recording starts)
  - **Per-joint kinematics**: roll/pitch/yaw (degrees), angular velocity (deg/s), angular acceleration (deg/s²)
  - **EEG channels**: 8 EEG + 1 reference electrode
  - **Optional markers**: Event timestamps for synchronization

Press **'W'** again to stop recording.

---

## 📊 CSV Output Format

Each row contains:

| Column | Format | Description |
|--------|--------|-------------|
| `time_s` | `float` | Seconds since recording start (t=0) |
| `marker` | `int` or `""` | Marker ID (1, 2, 3...) or empty string for periodic rows |
| **Per Joint (e.g., `RightShoulder_roll_deg`)** | | |
| `{joint}_roll_deg` | `float` | Roll angle (degrees) |
| `{joint}_roll_vel_dps` | `float` | Roll angular velocity (deg/s) |
| `{joint}_roll_acc_dps2` | `float` | Roll angular acceleration (deg/s²) |
| `{joint}_pitch_deg` | `float` | Pitch angle |
| `{joint}_pitch_vel_dps` | `float` | Pitch angular velocity |
| `{joint}_pitch_acc_dps2` | `float` | Pitch angular acceleration |
| `{joint}_yaw_deg` | `float` | Yaw angle |
| `{joint}_yaw_vel_dps` | `float` | Yaw angular velocity |
| `{joint}_yaw_acc_dps2` | `float` | Yaw angular acceleration |
| **EEG Channels** | | |
| `EEG_F7`, `EEG_C3`, `EEG_PZ`, ... | `float` | EEG sample values from each electrode |

**Example row:**
```
0.050,, 45.2345, 0.1234, 0.0012, -10.5678, -0.0567, 0.0001, ..., 12.34, 45.67, -23.45, ...
```

---

## 🎮 Hotkeys (Type in Console)

| Key | Action | Description |
|-----|--------|-------------|
| **R** | Re-calibrate | Captures current pose as new neutral reference |
| **W** | Toggle Recording | Starts/stops CSV logging to `recordings/` |
| **T** | Add Marker | Writes exact-time marker row (synchronized event) |
| **A** | Reconfigure | Returns to config selection (sensors stay connected) |
| **Q** | Quit | Exits the program |

---

## 🏗️ System Architecture

### Thread Organization

The system uses **4 independent threads** that communicate via thread-safe buffers:

```
┌─────────────────────────────────────────────────────────────────┐
│                     MAIN THREAD (UI/Viz)                         │
│                                                                   │
│  • PyVista 3D Visualizer (blocked until window closes)          │
│  • Reads latest frame from FrameBuffer                          │
│  • Renders skeleton in real-time                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   PRODUCER THREAD (Kinematics)                  │
│                                                                   │
│  • Reads raw IMU orientations from MovellaBuffer                │
│  • Applies calibration + forward kinematics                     │
│  • Publishes computed frame to FrameBuffer                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    CSV WRITER THREAD (Logging)                  │
│                                                                   │
│  • Periodically reads from FrameBuffer + EEGBuffer              │
│  • Computes finite-difference velocities/accelerations          │
│  • Writes timestamped rows to CSV file                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               EEG & MOVELLA THREADS (Acquisition)               │
│                                                                   │
│  • EEG Thread: Connects to LSL stream, buffers latest sample    │
│  • Movella Thread: Connects to IMUs, streams orientations       │
│  • Both run independently; data lives in EEGBuffer/MovellaBuffer│
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
HARDWARE (IMUs + EEG)
        ↓
    [Acquisition Threads]
        ↓
    ┌───────────────────────────┐
    │   MovellaBuffer           │  ← Latest orientation per MAC
    │   (Thread-safe, R/W lock) │
    │   EEGBuffer               │  ← Latest EEG sample
    │   (Thread-safe, deque)    │
    └───────────────────────────┘
        ↓ (read by Producer)
    [Producer Thread]
        ↓ (reads from buffers, applies kinematics)
    ┌───────────────────────────┐
    │   FrameBuffer             │  ← Computed frame snapshot
    │   (Single-slot, R/W lock) │     {joints: ..., endpoints: ...}
    └───────────────────────────┘
        ↓ (read by multiple consumers)
        ├→ [CSV Writer] → CSV file
        ├→ [Visualizer] → PyVista window
        └→ [Console Debug]
```

---

## 📦 Key Components

### 1. **eeg_engine.py** — EEG Data Acquisition
```
┌─────────────────────┐
│   EEG Stream (LSL)  │
└──────────┬──────────┘
           ↓
    [eeg_loop thread]
           ↓
    ┌──────────────────┐
    │   EEGBuffer      │  ← deque(maxlen=512)
    │  (Thread-safe)   │  ← Latest: (timestamp, sample)
    └──────────────────┘
```

**How it works:**
- **`EEGBuffer`**: Thread-safe deque that stores up to 512 timestamped EEG samples
- **`eeg_loop()`**: Background thread that continuously polls the LSL stream and adds samples
- **`get_latest()`**: Returns the most recent EEG sample or `None`
- **`get_window()`**: Returns all 512 samples (used for feature extraction)

### 2. **movella_engine.py** — IMU Data Acquisition
```
┌──────────────────────────┐
│  Movella DOT IMUs (BLE)  │
└──────────┬───────────────┘
           ↓
    [movella_loop thread]  ← owns ALL Movella SDK I/O
           ↓
    ┌────────────────────────────┐
    │   MovellaBuffer            │  ← {mac: (timestamp, R 3x3)}
    │  (Thread-safe, R/W lock)   │
    │  _connected_macs           │
    │  _ready event              │
    └────────────────────────────┘
```

**How it works:**
- **`initialize_hardware_handler()`**: Initializes the Movella PC SDK
- **`connect_and_configure_imus()`**: Scans, connects, and starts measurement on up to 4 sensors
- **`capture_raw_sensor_orientations()`**: Polls packets and converts Euler angles → rotation matrices
- **`movella_loop()`**: Continuously streams and stores orientation matrices keyed by sensor MAC address

**Key Detail**: Orientations are stored as **3×3 rotation matrices** (not Euler angles), so downstream code stays MAC- and SDK-agnostic.

### 3. **kinematics_core.py** — Pure Math (No Hardware)
The mathematical core: rotations, kinematics, topology parsing.

**Main Functions:**

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `rpy_to_rotation_matrix_xyz()` | roll, pitch, yaw (deg) | 3×3 matrix | Convert Euler angles to rotation matrix |
| `matrix_to_rpy_xyz()` | 3×3 matrix | roll, pitch, yaw (deg) | Inverse: rotation matrix to Euler angles |
| `sensor_to_world_calibrated()` | R_frame_change, R_calib, R_sensor_live | 3×3 matrix | Tared rotation (calibrated mode) |
| `parse_topology()` | config, active_joints | (state_dict, sorted_joints, endpoints) | Parse skeleton structure from JSON |
| `run_forward_kinematics()` | state_dict, world_rotations | {joint: position} | Compute world positions from rotations |

**Calibration (Taring):**
```
    D = R_frame_change.T @ (R_calib.T @ R_sensor_live) @ R_frame_change
```
- At calibration instant (t=0): `R_sensor_live == R_calib`, so `D = I` (identity)
- Later: `D` represents rotation relative to the neutral pose

### 4. **kinematics_engine_live_write.py** — Main Orchestrator
Coordinates all threads, handles user input, and manages the tracking session lifecycle.

**Key Classes:**

#### **`FrameBuffer`** — Single-slot frame cache
```python
class FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
    
    def set(self, frame):  # Written by Producer
        with self._lock:
            self._frame = frame
    
    def get(self):  # Read by Visualizer, CSV Writer
        with self._lock:
            return self._frame
```

Frame structure:
```python
{
  "joints": {
    "RightShoulder": {
      "name": "RightShoulder",
      "parent": None,
      "pos": [x, y, z],          # World position (meters)
      "R": [[3x3 rotation]]      # World rotation matrix
    },
    ...
  },
  "endpoints": {
    "RightFingertip": {
      "parent": "RightElbow",
      "pos": [x, y, z]           # Computed from parent + offset
    }
  }
}
```

#### **`CsvWriter`** — Timestamped logging
Runs on its own thread; periodically reads from `FrameBuffer` + `EEGBuffer`, computes finite-difference derivatives, writes to CSV.

**Key Method:**
```python
def _maybe_write_locked(self, now):
    # If period has elapsed:
    # 1. Extract joint angles (roll/pitch/yaw) from latest frame
    # 2. Compute velocities and accelerations (finite differences)
    # 3. Fetch latest EEG sample
    # 4. Write row: [time, marker, angles, vels, accs, EEG, ...]
```

**Velocity Computation:**
```
vel[axis] = (angle[now] - angle[prev]) / (time[now] - time[prev])
```
Using `_wrap180()` to handle 360° wraparound discontinuities.

#### **`engine_loop()`** — Producer thread
Continuously updates `live_states` from `MovellaBuffer`, applies calibration, runs forward kinematics, publishes frame.

**Per-frame workflow:**
```python
1. raw = movella_buffer.get_orientations()           # {mac: R}
2. samples = samples_by_joint(raw, joint_to_mac)    # {joint: R}
3. live_states = update_live_states(..., samples)    # Update latest per joint
4. frame = compute_frame(...)                        # Apply kinematics
5. frame_buffer.set(frame)                           # Publish
6. [check for hotkey input: R/A/Q/W/T]
```

### 5. **visualizer.py** — 3D Skeleton Rendering
Uses PyVista (VTK-based) to render:
- **Joints**: Colored spheres
- **Bones**: Solid boxes connecting parent–child joints
- **Axes**: RGB triads showing each joint's orientation
- **Endpoints**: Optional non-joint nodes (e.g., fingertips)

**Key Method:**
```python
def _update(self, frame):
    # For each joint: set actor's 4x4 transform matrix M = [R|t; 0|1]
    # where R is the joint's rotation, t is its world position
    # Geometry is built ONCE; every frame only updates transforms
```

### 6. **eeg_processing.py** — Signal Processing
Provides EEG feature extraction (optional):
- **`preprocess_channel()`**: Remove DC, bandpass (1–40 Hz), notch (50 Hz)
- **`band_power()`**: Compute power in frequency bands (theta, alpha, beta)
- **`extract_features_multichannel()`**: Apply to all 8 channels

---

## 🔧 Configuration Files (JSON)

### Structure
```json
{
  "state": {
    "JointName": {
      "parent": null,                           // null if root, else parent joint name
      "pos": [x, y, z],                         // World position (meters)
      "ang": [roll, pitch, yaw],                // Euler angles (degrees) for frame change
      "rot_mat": [[...3x3...]]                  // Pre-computed rotation matrix (optional)
    },
    ...
  },
  "endpoints": {
    "EndpointName": {
      "parent": "JointName",
      "pos": [x, y, z]                          // Fixed offset from parent
    }
  }
}
```

### Example: `2_arms.json`
```json
{
  "state": {
    "RightShoulder": {
      "parent": null,
      "pos": [0.0, 0.0, 0.0],
      "ang": [-90.0, 0.0, -90.0],
      "rot_mat": [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    },
    "RightElbow": {
      "parent": "RightShoulder",
      "pos": [0.0, 0.0, -30.0],
      "ang": [-90.0, 0.0, -90.0],
      "rot_mat": [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    },
    ...
  },
  "endpoints": {
    "RightFingertip": {
      "parent": "RightElbow",
      "pos": [0.0, 0.0, -65.0]
    }
  }
}
```

---

## 🔄 How Data Flows Through the System: A Complete Example

### Scenario: Record a 10-second experiment

1. **User starts program** → `main()` launches acquisition threads
   - EEG thread connects to LSL stream, starts buffering samples
   - Movella thread connects to 2 IMUs, starts streaming orientations

2. **User selects config** → Chooses `2_arms.json` (4 joints: shoulders + elbows)

3. **User maps sensors** → `RightShoulder ← IMU[0]`, `RightElbow ← IMU[1]`

4. **User presses 'R'** → Calibration snapshot
   ```python
   calibration = {
     "RightShoulder": R_sensor[IMU0],  # 3x3 matrix at t=0
     "RightElbow": R_sensor[IMU1],
     ...
   }
   ```

5. **Program enters tracking loop**
   - **t=0.0s**: User presses 'W' → CSV file opens
   - **Every 0.05s** (default): CSV writer polls buffers
     ```
     Frame @ t=0.050:
       RightShoulder world rotation = R_frame_change.T @ (R_calib.T @ R_sensor_live) @ R_frame_change
       Position = [0, 0, 0] (root)
       Angle = matrix_to_rpy_xyz(R_world) = [5.2°, -1.3°, 0.8°]
       Velocity = (5.2 - prev_angle) / 0.05 = [0.0 deg/s, 0.0, 0.0]
       EEG sample = latest from EEGBuffer
     
     CSV Row: 0.050, , 5.2000, 0.0000, 0.0000, -1.3000, 0.0000, 0.0000, ...
     ```

6. **t=5.0s**: User presses 'T' → Marker written
   ```
   CSV Row: 5.000, 1, [joint data], [EEG], [...]
   ```

7. **t=10.0s**: User presses 'W' → Recording stops
   - File is flushed and closed
   - `recordings/recording_20260612_143022.csv` ready for analysis

---

## 📈 Understanding the Math

### Rotation Matrices & Calibration

**Fixed-Axis (Extrinsic) X→Y→Z Convention:**
```
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
```

**Calibrated Rotation (Tare):**
At the neutral pose, the sensor reads `R_sensor_live == R_calib`. The calibrated rotation is:
```
R_world = R_frame_change.T @ (R_calib.T @ R_sensor_live) @ R_frame_change
```
- `R_frame_change`: User-defined frame offset (from JSON `ang`)
- `R_calib.T @ R_sensor_live`: Delta from calibration instant
- Result: `R_world = I` at calibration; deviations show motion relative to neutral pose

### Forward Kinematics

Given parent-child relationships and per-joint rotations:
```
joint_pos[child] = joint_pos[parent] + R_world[parent] @ offset
```
- `offset = child.pos - parent.pos` (from JSON)
- Endpoints follow the same rule (but don't rotate)

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **No IMUs detected** | Ensure Movella DOT devices are powered and in pairing mode. Check Bluetooth drivers. |
| **EEG stream not connecting** | Verify LSL stream is running (e.g., from acquisition software). Check stream name matches code expectations. |
| **CSV file is empty or sparse** | Ensure calibration was completed ('R' pressed). Check that frame data is being updated by watching console Hz counter. |
| **Visualization stutters** | Reduce CSV logging period or disable visualization (`--no-viz`). |
| **Wrong joint mapping** | Re-calibrate ('R') after correcting the mapping via 'A' reconfigure. |
| **Angles jump at 180°** | This is normal; finite-difference derivatives wrap angles to ±180° to avoid discontinuities. |

---

## 📝 Tips for Experiments

1. **Calibration is critical**: Ensure subject is in a *consistent* neutral pose when pressing 'R'
2. **Use markers ('T')**: Mark event onsets (stimulus, movement start, etc.) for synchronization
3. **Check Hz counter**: Monitor console output to ensure the producer loop is running at ~30 Hz
4. **Backup recordings**: The `recordings/` folder is not git-tracked; back up important CSVs
5. **Log period tradeoff**: 
   - Smaller period → higher temporal resolution, larger files
   - Larger period → coarser data, smaller files

---

## 📚 References

- **kinematics_math.md** (if available): Detailed frame and rotation math
- **Movella DOT PC SDK**: [https://www.movella.com/support/software-documentation](https://www.movella.com/support/software-documentation)
- **Lab Streaming Layer (LSL)**: [https://github.com/sccn/labstreaminglayer](https://github.com/sccn/labstreaminglayer)

---

## 📄 License & Attribution

This system integrates:
- **Movella DOT PC SDK** (proprietary)
- **Lab Streaming Layer** (BSD license)
- **PyVista** (MIT license)
- **SciPy** (BSD license)

See individual package licenses for details.

