# Kinematics Engine — Math Reference

This document specifies every transformation used by the IMU kinematics engine,
from raw sensor angles to World-FLU joint positions.

---

## 1. Notation and conventions

Rotation matrices are written so that they map **coordinates** from one frame to
another:

$$
\mathbf{v}_A = R_{A \leftarrow B}\,\mathbf{v}_B
$$

i.e. $R_{A\leftarrow B}$ takes a vector expressed in frame $B$ and returns the same
physical vector expressed in frame $A$. Consequences used throughout:

- $R_{A\leftarrow B}^{-1} = R_{A\leftarrow B}^{T} = R_{B\leftarrow A}$ (rotations are orthonormal).
- Composition chains by matching subscripts: $R_{A\leftarrow C} = R_{A\leftarrow B}\,R_{B\leftarrow C}$.

### Frames

| Symbol | Frame | Description |
|--------|-------|-------------|
| $W$ | World FLU | Global reference. **All JSON `pos` values live here.** |
| $G$ | Sensor global inertial | Each DOT's own gravity/heading reference. Per-sensor, not shared. |
| $S$ | Sensor body | The physical sensor at the current instant. |
| $E$ | Expected sensor | Where the sensor body is *supposed* to point in the neutral pose. |

### Euler angle convention

The sensor reports roll/pitch/yaw, and the JSON stores per-joint `ang`. Both are
interpreted as **fixed-axis (extrinsic) rotations applied in the order X → Y → Z**.
That sequence is algebraically identical to the intrinsic Z → Y → X sequence, so the
matrix is:

$$
R(\text{roll}, \text{pitch}, \text{yaw}) = R_z(\text{yaw})\,R_y(\text{pitch})\,R_x(\text{roll})
$$

with the elementary matrices

$$
R_x =
\begin{bmatrix} 1 & 0 & 0 \\ 0 & c_r & -s_r \\ 0 & s_r & c_r \end{bmatrix},\quad
R_y =
\begin{bmatrix} c_p & 0 & s_p \\ 0 & 1 & 0 \\ -s_p & 0 & c_p \end{bmatrix},\quad
R_z =
\begin{bmatrix} c_y & -s_y & 0 \\ s_y & c_y & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

where $c_\bullet = \cos$, $s_\bullet = \sin$. This is the standard Movella/Xsens
aerospace RPY convention. (If the JSON angles are instead meant as **intrinsic**
X → Y → Z, the matrix becomes $R_x R_y R_z$ — this is the only convention switch in
the codebase.)

---

## 2. The two per-node matrices

Each sensor (keyed by MAC address) gets two rotation matrices.

### 2.1 Frame-change matrix — built once at startup

From the JSON `ang = [roll, pitch, yaw]`:

$$
R_{\text{frame}} \;=\; R_{E\leftarrow W} \;=\; R(\text{ang})
$$

It maps the World FLU frame to the expected sensor frame for that joint in its
neutral mounting pose. It is structural and never changes during a run.

### 2.2 Calibration matrix — captured the instant you press **R**

The raw orientation the sensor reports is the body frame expressed in its own global
frame:

$$
R_{\text{calib}} \;=\; R_{G\leftarrow S_0}
$$

where $S_0$ is the sensor body at the calibration instant. It captures the combined
sensor mounting offset and the sensor's arbitrary heading reference.

**Calibration assumption:** at the moment of taring, the subject is in the neutral
pose, so the actual body frame equals the expected frame, $S_0 = E$. Hence
$R_{\text{calib}} = R_{G\leftarrow E}$.

---

## 3. Mapping a live reading to World FLU

The live reading is $R_{\text{live}} = R_{G\leftarrow S_1}$, the body frame at the
current instant expressed in the sensor's global frame.

We want $D$, the segment's rotation **relative to its neutral pose, expressed in
World FLU**. This is the quantity the forward kinematics consumes, and it must equal
identity at the calibration instant.

### 3.1 Step 1 — remove the mounting offset

$$
\Delta_{\text{body}} \;=\; R_{\text{calib}}^{T}\,R_{\text{live}}
\;=\; R_{S_0\leftarrow G}\,R_{G\leftarrow S_1}
\;=\; R_{S_0\leftarrow S_1}
$$

This is the rotation of the body away from its calibration pose, expressed in the
calibration body frame. It is identity when $R_{\text{live}} = R_{\text{calib}}$.

### 3.2 Step 2 — change of basis into World FLU

$\Delta_{\text{body}}$ is expressed in the expected-sensor frame $E$ (since $S_0 = E$).
Re-expressing a **rotation** in a different frame is a similarity transform
(a conjugation), not a single multiplication:

$$
D \;=\; R_{W\leftarrow E}\,\Delta_{\text{body}}\,R_{E\leftarrow W}
\;=\; R_{\text{frame}}^{T}\,\big(R_{\text{calib}}^{T}\,R_{\text{live}}\big)\,R_{\text{frame}}
$$

> **This is the core formula.** The sandwich $C^{T}(\cdot)C$ is mandatory. A single
> left-multiply $R_{\text{frame}}^{T}\,\Delta$ is *not* a change of basis and does not
> return identity at calibration — that was the original bug.

### 3.3 Identity-at-calibration check

At calibration $R_{\text{live}} = R_{\text{calib}}$, so $R_{\text{calib}}^{T}R_{\text{live}} = I$ and

$$
D \;=\; R_{\text{frame}}^{T}\,I\,R_{\text{frame}} \;=\; I \quad\checkmark
$$

so the neutral pose reproduces the configured skeleton exactly.

### 3.4 Equivalent global-frame form (sanity check)

Conjugating the **global-frame** delta $\Delta_{\text{global}} = R_{\text{live}}R_{\text{calib}}^{T}$
by the true world-from-global rotation $R_{W\leftarrow G} = R_{\text{frame}}^{T}R_{\text{calib}}^{T}$
gives the **same matrix**:

$$
R_{W\leftarrow G}\,\Delta_{\text{global}}\,R_{W\leftarrow G}^{T}
= R_{\text{frame}}^{T}\,R_{\text{calib}}^{T}\,R_{\text{live}}\,R_{\text{frame}}
= D
$$

The two derivations agreeing is a useful consistency check.

---

## 4. Forward kinematics

Let $D_{\text{mac}}$ be the World-FLU delta from §3 for each node, and let
$\mathbf{p}^{0}_{\text{mac}}$ be the JSON `pos` (already in World FLU, neutral pose).

### 4.1 Neutral link offset

For a child with parent, the rigid bone vector in the neutral pose is the World-FLU
difference of the two stored positions:

$$
\mathbf{o}_{\text{mac}} \;=\; \mathbf{p}^{0}_{\text{mac}} - \mathbf{p}^{0}_{\text{parent}}
$$

### 4.2 Position propagation

Walk the tree parent-first. Roots sit at their configured position; children swing
their neutral offset by the **parent's** current World-FLU rotation:

$$
\mathbf{p}_{\text{root}} = \mathbf{p}^{0}_{\text{root}},
\qquad
\mathbf{p}_{\text{mac}} = \mathbf{p}_{\text{parent}} + D_{\text{parent}}\,\mathbf{o}_{\text{mac}}
$$

Because $D = I$ at calibration, every $\mathbf{p}_{\text{mac}} = \mathbf{p}^{0}_{\text{mac}}$
in the neutral pose. As segments rotate, the rigid offsets follow the parent's
measured orientation.

> Note: $D$ is each segment's own world-referenced rotation (the sensor measures it
> absolutely), so positioning a child from the *parent's* $D$ is correct — there is no
> need to compose rotations down the chain.

---

## 5. Display extraction (monitoring only)

To print human-readable angles, $D$ is decomposed back into roll/pitch/yaw using the
inverse of the §1 convention (ZYX extraction):

$$
\text{roll} = \operatorname{atan2}(D_{21}, D_{22}),\quad
\text{pitch} = -\arcsin(D_{20}),\quad
\text{yaw} = \operatorname{atan2}(D_{10}, D_{00})
$$

$D_{20}$ is clamped to $[-1, 1]$ before $\arcsin$ to avoid domain errors near gimbal
lock. These angles are for display only and do not feed back into the pipeline.

---

## 6. End-to-end pipeline per frame

1. **Read** raw Euler → $R_{\text{live, mac}} = R(\text{roll}, \text{pitch}, \text{yaw})$ (§1).
2. **Compose** $D_{\text{mac}} = R_{\text{frame}}^{T}\,(R_{\text{calib}}^{T}R_{\text{live}})\,R_{\text{frame}}$ (§3).
3. **Propagate** positions via $\mathbf{p}_{\text{mac}} = \mathbf{p}_{\text{parent}} + D_{\text{parent}}\mathbf{o}_{\text{mac}}$ (§4).
4. **Extract** display angles from $D_{\text{mac}}$ (§5).

$R_{\text{frame}}$ is fixed at startup; $R_{\text{calib}}$ is refreshed on each **R** press.

---

## 7. Assumptions and failure modes

- **Single-pose calibration.** The neutral pose at tare must match the configured
  `ang` for every segment. Any physical mounting error at that instant is baked into
  $R_{\text{calib}}$ and cannot be detected from one pose.
- **Per-sensor global frames.** Each DOT has its own heading $G$. The composition
  cancels it through the calibration term, so no cross-sensor heading sync is required
  — provided calibration is taken with all sensors stationary in the neutral pose.
- **Gimbal lock** affects only the §5 display angles ($D$ itself stays well-defined).
- **Convention coupling.** §1 and §5 are exact inverses. If you change one, change the
  other.