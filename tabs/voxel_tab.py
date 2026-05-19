import numpy as np
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QSpinBox, QLabel, QPushButton, QCheckBox
)
import pyqtgraph.opengl as gl
from data_pipeline import DataPipeline


class VoxelTab(QWidget):
    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline

        # State variables
        self.voxel_list = []
        self.mesh = None
        self.is_playing = False
        self.camera_initialized = False

        # Playback timer
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(100)  # 10 FPS to prevent CPU hogging
        self.play_timer.timeout.connect(self.next_frame)

        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Initialize Viewport
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(255, 255, 255, 255)
        self.view.opts['distance'] = 150
        self.view.setCameraPosition(azimuth=135, elevation=30)
        layout.addWidget(self.view, stretch=1)

        # 2. Setup Mesh Item
        self.base_color = np.array([0, 0.8, 0.8, 1.0], dtype=float)  # Slightly brighter base cyan
        self.core_color = np.array([0.8, 1.0, 1.0, 1.0], dtype=float)  # Soft bright cyan/white for the "lit" top

        self.mesh = gl.GLMeshItem(
            color=self.base_color,
            edgeColor=(0, 0, 0, 0.5),  # Slightly softer edges so it's not a heavy black grid
            drawEdges=False,
            glOptions='opaque',
            shader=None,
            smooth=False
        )
        self.mesh.setVisible(False)
        self.view.addItem(self.mesh)

        # A semi-transparent magenta plane to contrast with the cyan voxels
        self.plane_mesh = gl.GLMeshItem(
            color=np.array([1.0, 0.0, 0.5, 0.4], dtype=float),
            glOptions='translucent',
            shader=None,
            drawEdges=True,
            edgeColor=(1.0, 0.0, 0.5, 0.8)
        )
        self.plane_mesh.setVisible(False)
        self.view.addItem(self.plane_mesh)


        # 3. Playback Controls (Bottom Bar)

        bottom_layout = QHBoxLayout()
        controls_layout = QVBoxLayout()
        play_layout = QHBoxLayout()

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setFixedWidth(80)
        self.btn_play.setEnabled(False)

        # --- NEW: Speed/Skip SpinBox ---
        self.lbl_speed = QLabel("Skip:")
        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(1, 100)  # Give it a generous upper limit
        self.spin_speed.setValue(10)  # Default to 10 as requested
        self.spin_speed.setToolTip("Number of frames to skip per tick during playback")
        self.spin_speed.setEnabled(False)

        self.lbl_frame = QLabel("Frame:")

        self.spin_frame = QSpinBox()
        self.spin_frame.setEnabled(False)

        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setTickPosition(QSlider.TicksBelow)
        self.slider_frame.setTickInterval(0)
        self.slider_frame.setMinimumHeight(25)
        self.slider_frame.setEnabled(False)

        self.chk_plane = QCheckBox("Show Cross-Section Plane")
        self.chk_plane.setChecked(True)

        play_layout.addWidget(self.btn_play)
        play_layout.addWidget(self.lbl_speed)
        play_layout.addWidget(self.spin_speed)
        play_layout.addWidget(self.lbl_frame)
        play_layout.addWidget(self.spin_frame)
        controls_layout.addLayout(play_layout)
        controls_layout.addWidget(self.chk_plane)

        bottom_layout.addLayout(controls_layout)
        bottom_layout.addWidget(self.slider_frame)

        layout.addLayout(bottom_layout)



    def connect_signals(self):
        self.pipeline.voxels_available.connect(self.on_voxels_received)

        # Sync slider and spinbox
        self.slider_frame.valueChanged.connect(self.spin_frame.setValue)
        self.spin_frame.valueChanged.connect(self.slider_frame.setValue)

        # Trigger render when value changes
        self.slider_frame.valueChanged.connect(self.render_frame)

        # Play button toggle
        self.btn_play.clicked.connect(self.toggle_playback)

        self.chk_plane.toggled.connect(self.toggle_plane_visibility)

    @Slot(bool)
    def toggle_plane_visibility(self, checked: bool):
        """Safely toggles the plane, ignoring the click if no data is loaded."""
        if not self.voxel_list:
            self.plane_mesh.setVisible(False)
            return
        self.plane_mesh.setVisible(checked)

    @Slot(list)
    def on_voxels_received(self, voxel_list: list):
        """Stores the list of 3D arrays and initializes the UI controls."""
        if not voxel_list:
            return

        self.slider_frame.setTickInterval(int(len(voxel_list)/(1+np.max(self.pipeline.data["cycle"][:-1]))))

        self.voxel_list = voxel_list
        max_idx = len(self.voxel_list) - 1

        # Configure controls
        self.slider_frame.setRange(0, max_idx)
        self.spin_frame.setRange(0, max_idx)

        self.slider_frame.setEnabled(True)
        self.spin_frame.setEnabled(True)
        self.btn_play.setEnabled(True)

        # Enable our new speed spinbox and ensure its max doesn't exceed the total frame count
        self.spin_speed.setEnabled(True)
        self.spin_speed.setMaximum(max(1, max_idx))

        # Reset state
        self.camera_initialized = False
        self.slider_frame.setValue(0)

        # Force a render of the first frame
        self.render_frame(0)

    @Slot()
    def toggle_playback(self):
        """Starts or stops the auto-advancing timer."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("⏸ Pause")
            self.play_timer.start()
        else:
            self.btn_play.setText("▶ Play")
            self.play_timer.stop()

    @Slot()
    def next_frame(self):
        """Advances to the next frame, looping back to the start if necessary."""
        current = self.slider_frame.value()
        max_val = self.slider_frame.maximum()
        step = self.spin_speed.value()

        # Calculate next value. If it exceeds max_val, modulo handles the clean loop back to the start!
        next_val = (current + step) % (max_val + 1)

        self.slider_frame.setValue(next_val)

    @Slot(int)
    def render_frame(self, index: int):
        if not self.voxel_list or index < 0 or index >= len(self.voxel_list):
            self.mesh.setVisible(False)
            self.plane_mesh.setVisible(False)
            return

        data = self.voxel_list[index]

        if not np.any(data):
            self.mesh.setVisible(False)
            self.plane_mesh.setVisible(False)
            return

        self.mesh.setVisible(True)
        self.plane_mesh.setVisible(self.chk_plane.isChecked())

        # 1. UNSHARED VERTICES: 6 faces * 4 vertices = 24 vertices per voxel
        # This guarantees perfectly sharp, flat normals for crisp block rendering
        cube_verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Z- (Bottom)
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # Z+ (Top)
            [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],  # Y- (Front)
            [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1],  # Y+ (Back)
            [0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1],  # X- (Left)
            [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]  # X+ (Right)
        ], dtype=float)

        cube_faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Z-
            [4, 5, 6], [4, 6, 7],  # Z+
            [8, 9, 10], [8, 10, 11],  # Y-
            [12, 13, 14], [12, 14, 15],  # Y+
            [16, 17, 18], [16, 18, 19],  # X-
            [20, 21, 22], [20, 22, 23]  # X+
        ], dtype=int)

        # 2. BAKE LIGHTING MULTIPLIERS (Minecraft Style)
        # Top = 100% bright, Front/Back = 70%, Left/Right = 50%, Bottom = 30%
        face_mults = np.array([
            0.3, 0.3, 0.3, 0.3,  # Bottom vertices
            1.0, 1.0, 1.0, 1.0,  # Top vertices
            0.6, 0.6, 0.6, 0.6,  # Front vertices
            0.6, 0.6, 0.6, 0.6,  # Back vertices
            0.8, 0.8, 0.8, 0.8,  # Left vertices
            0.8, 0.8, 0.8, 0.8  # Right vertices
        ])

        z, y, x = np.where(data == 1)
        positions = np.column_stack([x, y, z])
        num_voxels = len(positions)

        # Broadcast positions and faces to the 24-vertex format
        all_verts = (positions[:, np.newaxis, :] + cube_verts[np.newaxis, :, :]).reshape(num_voxels * 24, 3)
        indices = np.arange(0, num_voxels * 24, 24)
        all_faces = (indices[:, np.newaxis, np.newaxis] + cube_faces[np.newaxis, :, :]).reshape(num_voxels * 12, 3)

        # 3. CALCULATE HEIGHT GRADIENT
        z_positions = positions[:, 2]
        min_z = z_positions.min()
        max_z = z_positions.max()

        if max_z > min_z:
            intensity = (z_positions - min_z) / (max_z - min_z)
            intensity = 0.3 + (intensity * 0.7)  # Scale from 0.3 to 1.0

            # Create base color for each voxel based on its Z height
            voxel_colors = intensity[:, np.newaxis] * self.core_color + (
                        1.0 - intensity[:, np.newaxis]) * self.base_color
        else:
            voxel_colors = np.tile(self.base_color, (num_voxels, 1))

        # 4. APPLY BAKED LIGHTING TO FACES
        # Repeat the voxel color 24 times (once for each vertex)
        expanded_colors = np.repeat(voxel_colors[:, np.newaxis, :], 24, axis=1)

        # Multiply the RGB channels by our fake lighting multipliers
        expanded_colors[:, :, 0:3] *= face_mults[np.newaxis, :, np.newaxis]

        # Flatten colors to match the vertex array
        final_colors = expanded_colors.reshape(num_voxels * 24, 4)

        # Send it to PyQtGraph using vertexColors
        self.mesh.setMeshData(vertexes=all_verts, faces=all_faces, vertexColors=final_colors)

        # Get the bounding box of the voxel object
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0) + 1  # +1 accounts for the 1-unit width of the cube

        # Calculate the exact center along the Y-axis (stretch direction)
        mid_y = (min_pos[1] + max_pos[1]) / 2.0

        # Add a slight margin so the plane visually extends just past the object
        margin = 3
        p_x_min = min_pos[0] - margin
        p_x_max = max_pos[0] + margin
        p_z_min = min_pos[2] - margin
        p_z_max = max_pos[2] + margin

        # 4 vertices for an X-Z plane intersecting at mid_y
        plane_verts = np.array([
            [p_x_min, mid_y, p_z_min],
            [p_x_max, mid_y, p_z_min],
            [p_x_max, mid_y, p_z_max],
            [p_x_min, mid_y, p_z_max]
        ], dtype=float)

        # Double-sided faces so the plane is visible from both the front and back
        plane_faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Front
            [0, 2, 1], [0, 3, 2]  # Back
        ], dtype=int)

        self.plane_mesh.setMeshData(vertexes=plane_verts, faces=plane_faces)

        # Auto-frame camera
        if num_voxels > 1 and not self.camera_initialized:
            center = np.array(data.shape) / 2.0
            self.view.pan(center[0], center[1], center[2])
            self.view.opts['distance'] = np.linalg.norm(data.shape) * 1.5
            self.camera_initialized = True