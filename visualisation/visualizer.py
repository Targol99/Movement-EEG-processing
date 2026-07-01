"""
visualizer.py
=============
Live 3D skeleton view built on PyVista (VTK). Sensor joints are spheres, bones
are solid parallelepipeds (long thin boxes), endpoints are distinct spheres tied
to their parent by a box, and each joint carries an RGB axis triad (also boxes)
showing its orientation.

Separation of concerns:
  - No Movella SDK here.
  - No kinematics here. The only math used is kinematics_core.rotation_between_vectors,
    a generic geometry helper, to aim a bone box along its segment.

Geometry is built ONCE from the first frame's topology; every subsequent frame
only updates each actor's 4x4 user_matrix, so nothing is rebuilt at runtime.

Frame contract (keys are joint NAMES, not MACs):
    frame = {
      "joints":    { name: {"name": str, "parent": name|None, "pos": (3,), "R": (3,3)} },
      "endpoints": { name: {"parent": name, "pos": (3,)} }
    }

Dependencies:  pip install pyvista pyvistaqt pyqt5
"""

import time
import numpy as np
import pyvista as pv

import aquisition_and_processing.kinematics_core as kc

# Colors
JOINT_COLOR = "#1f77b4"
ENDPOINT_COLOR = "#ff7f0e"
BONE_COLOR = "#50545a"
ENDPOINT_BONE_COLOR = "#e3c9a8"
AXIS_X = "#d62728"
AXIS_Y = "#2ca02c"
AXIS_Z = "#3b6fd6"
BACKGROUND = "#a0a4ae"


def _affine(R=None, t=(0.0, 0.0, 0.0), s=(1.0, 1.0, 1.0)):
    """Build a 4x4 transform M with M[:3,:3] = R @ diag(s) and M[:3,3] = t."""
    if R is None:
        R = np.eye(3)
    M = np.eye(4)
    M[:3, :3] = R @ np.diag(s)
    M[:3, 3] = np.asarray(t, float)
    return M


def _set_matrix(actor, M):
    """Set an actor's user matrix, with a VTK fallback for older PyVista."""
    try:
        actor.user_matrix = M
    except Exception:
        import vtk
        vm = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vm.SetElement(i, j, float(M[i, j]))
        actor.SetUserMatrix(vm)


class SkeletonVisualizer:
    def __init__(self, axis_length=12.0, joint_radius=2.6, endpoint_radius=2.2,
                 bone_width=1.6, triad_width=0.7, interval_ms=33, title="Live Skeleton"):
        self.axis_length = axis_length
        self.joint_radius = joint_radius
        self.endpoint_radius = endpoint_radius
        self.bone_width = bone_width
        self.triad_width = triad_width
        self.interval_ms = interval_ms
        self.title = title

        self.plotter = None
        self.get_frame = None
        self.should_close = None        # optional callable -> bool; closes window when True
        self._closing = False
        self._loc = None                # subplot (row, col) when sharing a plotter; None = single
        self._name_prefix = ""          # actor-name prefix, prevents collisions across subplots
        self.joint_actors = {}
        self.bone_actors = {}
        self.triad_actors = {}
        self.endpoint_actors = {}
        self.endpoint_bone_actors = {}
        self._reach = axis_length

    # ------------------------------------------------------------ base meshes
    def _unit_bone(self):
        # Box spanning z in [0,1], cross-section bone_width. Scaled to length per frame.
        w = self.bone_width / 2.0
        return pv.Box(bounds=(-w, w, -w, w, 0.0, 1.0))

    def _axis_box(self, which):
        # Thin box from 0..axis_length along the given local axis.
        t = self.triad_width / 2.0
        L = self.axis_length
        if which == "x":
            return pv.Box(bounds=(0.0, L, -t, t, -t, t))
        if which == "y":
            return pv.Box(bounds=(-t, t, 0.0, L, -t, t))
        return pv.Box(bounds=(-t, t, -t, t, 0.0, L))

    # ----------------------------------------------------------------- build
    def _build(self, frame):
        self._activate()
        p = self._name_prefix
        joints = frame["joints"]
        endpoints = frame["endpoints"]

        for key, j in joints.items():
            self.joint_actors[key] = self.plotter.add_mesh(
                pv.Sphere(radius=self.joint_radius), color=JOINT_COLOR,
                smooth_shading=True, name=f"{p}joint_{key}")

            triad = []
            for which, color in (("x", AXIS_X), ("y", AXIS_Y), ("z", AXIS_Z)):
                triad.append(self.plotter.add_mesh(
                    self._axis_box(which), color=color, name=f"{p}triad_{key}_{which}"))
            self.triad_actors[key] = triad

            if j.get("parent") in joints:
                self.bone_actors[key] = self.plotter.add_mesh(
                    self._unit_bone(), color=BONE_COLOR, smooth_shading=True,
                    name=f"{p}bone_{key}")

        for name, e in endpoints.items():
            self.endpoint_actors[name] = self.plotter.add_mesh(
                pv.Sphere(radius=self.endpoint_radius), color=ENDPOINT_COLOR,
                smooth_shading=True, name=f"{p}ep_{name}")
            if e.get("parent") in joints:
                self.endpoint_bone_actors[name] = self.plotter.add_mesh(
                    self._unit_bone(), color=ENDPOINT_BONE_COLOR, smooth_shading=True,
                    name=f"{p}epbone_{name}")

    # --------------------------------------------------------------- updates
    def _bone_matrix(self, a, b):
        d = np.asarray(b, float) - np.asarray(a, float)
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            return _affine(t=a, s=(1.0, 1.0, 1e-6))
        R = kc.rotation_between_vectors((0.0, 0.0, 1.0), d / L)
        return _affine(R=R, t=a, s=(1.0, 1.0, L))

    def _update(self, frame):
        joints = frame["joints"]
        endpoints = frame["endpoints"]
        reach = self._reach

        for key, j in joints.items():
            pos = np.asarray(j["pos"], float)
            R = np.asarray(j["R"], float)
            reach = max(reach, float(np.max(np.abs(pos))))
            _set_matrix(self.joint_actors[key], _affine(t=pos))
            M = _affine(R=R, t=pos)
            for ta in self.triad_actors[key]:
                _set_matrix(ta, M)
            if key in self.bone_actors:
                parent_pos = np.asarray(joints[j["parent"]]["pos"], float)
                _set_matrix(self.bone_actors[key], self._bone_matrix(parent_pos, pos))

        for name, e in endpoints.items():
            pos = np.asarray(e["pos"], float)
            reach = max(reach, float(np.max(np.abs(pos))))
            _set_matrix(self.endpoint_actors[name], _affine(t=pos))
            if name in self.endpoint_bone_actors:
                parent_pos = np.asarray(joints[e["parent"]]["pos"], float)
                _set_matrix(self.endpoint_bone_actors[name], self._bone_matrix(parent_pos, pos))

        self._reach = reach
        self.plotter.render()

    # ------------------------------------------------------------------- run
    def _tick(self):
        # Honor an external request to close the window (e.g. user pressed 'A').
        if self.should_close is not None and self.should_close() and not self._closing:
            self._closing = True
            try:
                self.plotter.close()
                self.plotter.app.quit()
            except Exception:
                pass
            return
        frame = self.get_frame()
        if frame:
            self._update(frame)

    # --------------------------------------------------- shared-plotter setup
    def _attach(self, plotter, loc=None, name_prefix=""):
        """Bind this visualizer to a (possibly shared) plotter and subplot."""
        self.plotter = plotter
        self._loc = loc
        self._name_prefix = name_prefix
        self._closing = False

    def _activate(self):
        """Make this visualizer's subplot the active renderer (no-op if single)."""
        if self._loc is not None:
            self.plotter.subplot(*self._loc)

    def _setup_scene(self, get_frame, should_close=None):
        """Build geometry and frame the camera WITHOUT entering an event loop.
        Shared by the standalone run() and DualSkeletonVisualizer, so every
        plotter is created and driven on the main thread under one QApplication."""
        self.get_frame = get_frame
        self.should_close = should_close

        self._activate()
        self.plotter.add_axes()  # orientation marker in this viewport's corner

        # Wait for the first frame, then build geometry once.
        frame = None
        while frame is None:
            frame = get_frame()
            time.sleep(0.01)
        self._build(frame)
        self._update(frame)

        # Frame the camera on the world origin for this viewport.
        self._activate()
        d = max(self._reach * 2.2, 1.0)
        self.plotter.camera.focal_point = (0.0, 0.0, 0.0)
        self.plotter.camera.position = (d, d, d * 0.7)
        self.plotter.camera.up = (0.0, 0.0, 1.0)

    # ------------------------------------------------------------------- run
    def run(self, get_frame, should_close=None):
        """
        Block on a live, interactive window (single skeleton).
          get_frame:    returns the latest frame or None.
          should_close: optional callable -> bool; when it returns True the
                        window closes and run() returns (used to reconfigure
                        without disconnecting the hardware).
        """
        from pyvistaqt import BackgroundPlotter

        plotter = BackgroundPlotter(title=self.title, window_size=(960, 840))
        self._attach(plotter)  # loc=None, prefix="" -> single default renderer
        plotter.set_background(BACKGROUND)

        self._setup_scene(get_frame, should_close)

        plotter.add_callback(self._tick, interval=int(self.interval_ms))
        plotter.app.exec_()


# ==============================================================================
# DUAL VISUALIZER (side-by-side)
# ==============================================================================
class DualSkeletonVisualizer:
    """Display two skeletons side by side (world vs intrinsic playback)."""
    
    def __init__(self, axis_length=12.0, title="Skeleton Playback"):
        self.axis_length = axis_length
        self.title = title
        
        self.viz_world = SkeletonVisualizer(axis_length=axis_length, title=f"{title} - World")
        self.viz_intrinsic = SkeletonVisualizer(axis_length=axis_length, title=f"{title} - Intrinsic")
        
        self.get_frame_world = None
        self.get_frame_intrinsic = None
        self.should_close = None
    
    def run(self, get_frame_world, get_frame_intrinsic, should_close=None):
        """
        Show both skeletons side by side in ONE window (two linked viewports),
        driven by a single QApplication on the main thread.

        The previous version launched two BackgroundPlotter windows on separate
        threads; Qt requires its event loop on the main thread, so on most
        platforms neither window appeared (the symptom you saw). Here a single
        plotter with shape=(1, 2) hosts world (left) and intrinsic (right); one
        timer ticks both, and should_close() is polled exactly once per tick so
        playback advances at the right rate and the two views stay frame-synced.

        Args:
            get_frame_world:     callable -> frame (world skeleton)
            get_frame_intrinsic: callable -> frame (intrinsic skeleton)
            should_close:        optional callable -> bool
        """
        from pyvistaqt import BackgroundPlotter

        self.get_frame_world = get_frame_world
        self.get_frame_intrinsic = get_frame_intrinsic
        self.should_close = should_close

        plotter = BackgroundPlotter(title=self.title, shape=(1, 2),
                                    window_size=(1680, 840))
        plotter.set_background(BACKGROUND)

        # Distinct actor-name prefixes are required: both skeletons share joint
        # names (e.g. "LeftElbow"), so without prefixes the right viewport's
        # actors would collide with the left's in the plotter name registry.
        self.viz_world._attach(plotter, loc=(0, 0), name_prefix="world_")
        self.viz_intrinsic._attach(plotter, loc=(0, 1), name_prefix="intrinsic_")

        # Build both scenes on the main thread (no event loop yet).
        self.viz_world._setup_scene(get_frame_world)        # closing handled below
        self.viz_intrinsic._setup_scene(get_frame_intrinsic)

        # Link the two viewports so orbiting/zooming one moves both -- makes the
        # world-vs-intrinsic comparison meaningful -- and frame the shared camera.
        try:
            plotter.link_views()
            d = max(self.viz_world._reach, self.viz_intrinsic._reach, 1.0) * 2.2
            plotter.subplot(0, 0)
            plotter.camera.focal_point = (0.0, 0.0, 0.0)
            plotter.camera.position = (d, d, d * 0.7)
            plotter.camera.up = (0.0, 0.0, 1.0)
        except Exception:
            pass

        closing = [False]

        def tick_both():
            if closing[0]:
                return
            # Poll the playback clock ONCE per tick so both viewports show the
            # same row and playback runs at the intended speed.
            if should_close is not None and should_close():
                closing[0] = True
                try:
                    plotter.close()
                    plotter.app.quit()
                except Exception:
                    pass
                return
            fw = get_frame_world()
            if fw:
                self.viz_world._update(fw)
            fi = get_frame_intrinsic()
            if fi:
                self.viz_intrinsic._update(fi)

        plotter.add_callback(tick_both, interval=int(self.viz_world.interval_ms))
        plotter.app.exec_()

# ===================================================================== demo
def _demo():
    """Synthetic motion so the renderer can be checked without hardware."""
    state_dict = {
        "Shoulder": {"parent": None,       "pos": [0.0, 0.0, 0.0],   "ang": [0, 0, 0]},
        "Elbow":    {"parent": "Shoulder", "pos": [0.0, 0.0, -30.0], "ang": [0, 0, 0]},
        "Wrist":    {"parent": "Elbow",    "pos": [0.0, 0.0, -55.0], "ang": [0, 0, 0]},
    }
    endpoints = {"Fingertip": {"parent": "Wrist", "pos": [0.0, 0.0, -75.0]}}
    joints = ["Shoulder", "Elbow", "Wrist"]
    t0 = time.time()

    def get_frame():
        t = time.time() - t0
        Rs = kc.rpy_to_rotation_matrix_xyz(25.0 * np.sin(0.8 * t), 0.0, 0.0)
        Re = kc.rpy_to_rotation_matrix_xyz(0.0, 35.0 * np.sin(1.2 * t), 0.0)
        Rw = kc.rpy_to_rotation_matrix_xyz(0.0, 45.0 * np.sin(1.5 * t), 0.0)
        wr = {"Shoulder": Rs, "Elbow": Re, "Wrist": Rw}
        jp, ep = kc.run_forward_kinematics_absolute(state_dict, joints, joints, wr, endpoints)
        joint_frame = {n: {"name": n, "parent": state_dict[n]["parent"],
                           "pos": jp[n], "R": wr[n]} for n in joints}
        eps = {n: {"parent": endpoints[n]["parent"], "pos": ep[n]} for n in ep}
        return {"joints": joint_frame, "endpoints": eps}

    print("Demo: synthetic shoulder roll + wrist pitch. Close the window to exit.")
    SkeletonVisualizer(title="Skeleton Visualizer - DEMO").run(get_frame)


if __name__ == "__main__":
    _demo()
