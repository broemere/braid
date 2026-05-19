import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
from data_pipeline import DataPipeline


class MechanicsTab(QWidget):
    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        pg.setConfigOptions(antialias=True)
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)

        # --- 1. Hysteresis Loop (Stress vs. Stretch) ---
        self.plot_hysteresis = self.plot_widget.addPlot(title="True Stress vs. Axial Stretch")
        self.plot_hysteresis.setLabel('bottom', 'Stretch Ratio X (Loading)')
        self.plot_hysteresis.setLabel('left', 'True Stress', units='kPa')
        self.plot_hysteresis.showGrid(x=True, y=True, alpha=0.3)
        self.plot_hysteresis.addLegend(offset=(10, 10))  # Add a legend!

        # Mean Stress (Solid)
        self.curve_hysteresis = self.plot_hysteresis.plot(
            name="Mean Stress", pen=pg.mkPen(color='#ff007f', width=2)
        )
        # Max Stress (Dashed)
        self.curve_hysteresis_max = self.plot_hysteresis.plot(
            name="Max Stress", pen=pg.mkPen(color='#ffaa00', width=2, style=Qt.DashLine)
        )

        # --- 2. Energy Dissipation (Bar Graph) ---
        self.plot_energy = self.plot_widget.addPlot(title="Energy Dissipation per Cycle")
        self.plot_energy.setLabel('bottom', 'Cycle Number')
        self.plot_energy.setLabel('left', 'Dissipated Energy', units='mJ/mm³')
        self.bar_energy = pg.BarGraphItem(x=[], height=[], width=0.6, brush='#ffaa00')
        self.plot_energy.addItem(self.bar_energy)

        self.plot_widget.nextRow()

        # --- 3. Orthogonal Stretch Trajectory ---
        self.plot_phase = self.plot_widget.addPlot(title="Stretch Trajectory (Z vs. X)")
        self.plot_phase.setLabel('bottom', 'Stretch Ratio X (Loading)')
        self.plot_phase.setLabel('left', 'Stretch Ratio Z (Thinning)')
        self.plot_phase.showGrid(x=True, y=True, alpha=0.3)
        self.plot_phase.addLegend(offset=(-10, 10))

        self.curve_ideal = self.plot_phase.plot(
            name="Ideal Incompressible", pen=pg.mkPen(color='#888888', width=2, style=Qt.DashLine)
        )
        self.curve_ideal.setZValue(-1)

        self.curve_phase = self.plot_phase.plot(
            name="Actual Stretch", pen=pg.mkPen(color='#00d2ff', width=2)
        )

        # --- 4. Volumetric Ratio (J) ---
        self.plot_volume = self.plot_widget.addPlot(title="Volumetric Ratio (J = V/V₀)")
        self.plot_volume.setLabel('bottom', 'Time', units='s')
        self.plot_volume.setLabel('left', "Volumetric Ratio (J)")
        self.plot_volume.showGrid(x=True, y=True, alpha=0.3)

        # Hard baseline at 1.0 (Perfect Incompressibility)
        self.line_j_ideal = pg.InfiniteLine(pos=1.0, angle=0, pen=pg.mkPen(color='#888888', width=2, style=Qt.DashLine))
        self.plot_volume.addItem(self.line_j_ideal)

        self.curve_volume = self.plot_volume.plot(pen=pg.mkPen(color='#00ff00', width=2))
        self.cycle_lines = []

    def connect_signals(self):
        self.pipeline.mechanics_available.connect(self.on_mechanics_received)

    @Slot(dict)
    def on_mechanics_received(self, data: dict):
        print("Populating Mechanics Tab...")

        time_s = np.array(data.get('time_s', []))
        cycle_parsing = data.get('cycle_parsing', {})

        true_stress = np.array(data.get('true_stress_kpa', []))
        true_stress_max = np.array(data.get('true_stress_max_kpa', []))  # Pull the new max stress

        stretch_x_opt = np.array(data.get('stretch_x_opt', []))
        stretch_z = np.array(data.get('stretch_z', []))

        volumetric_ratio = np.array(data.get('volumetric_ratio', []))  # Pull J instead of raw volume
        energy_dissipated = np.array(data.get('energy_dissipated', []))

        # 1. Update Hysteresis Loop
        if len(stretch_x_opt) > 0 and len(stretch_x_opt) == len(true_stress):
            self.curve_hysteresis.setData(stretch_x_opt, true_stress)

        if len(stretch_x_opt) > 0 and len(stretch_x_opt) == len(true_stress_max):
            self.curve_hysteresis_max.setData(stretch_x_opt, true_stress_max)

        # 2. Update Energy Dissipation
        if len(energy_dissipated) > 0:
            cycle_numbers = np.arange(1, len(energy_dissipated) + 1)
            self.bar_energy.setOpts(x=cycle_numbers, height=energy_dissipated)
            x_ticks = [(int(i), str(int(i))) for i in cycle_numbers]
            self.plot_energy.getAxis('bottom').setTicks([x_ticks])

        # 3. Update Stretch Trajectory Loop
        if len(stretch_x_opt) > 0 and len(stretch_x_opt) == len(stretch_z):
            self.curve_phase.setData(stretch_x_opt, stretch_z)

            # --- CORRECTED IDEAL INCOMPRESSIBLE MATH ---
            # Generate a smooth array of X stretches from 1.0 to the max experienced stretch
            max_x = np.max(stretch_x_opt)
            ideal_x = np.linspace(1.0, max(1.01, max_x), 100)
            # Apply the lambda_z = 1 / sqrt(lambda_x) formula
            ideal_y = 1.0 / np.sqrt(ideal_x)

            self.curve_ideal.setData(ideal_x, ideal_y)

        # 4. Update Volumetric Ratio
        if len(volumetric_ratio) > 0 and len(volumetric_ratio) == len(time_s):
            self.curve_volume.setData(time_s, volumetric_ratio)

            # Auto-scale the Y-axis to focus around 1.0 (e.g., 0.8 to 1.2)
            # This makes deviations from incompressibility glaringly obvious
            v_min, v_max = volumetric_ratio.min(), volumetric_ratio.max()
            margin = max(0.05, abs(v_max - 1.0), abs(1.0 - v_min)) * 1.2
            self.plot_volume.setYRange(1.0 - margin, 1.0 + margin)

            # Redraw Cycle Lines
            for line in self.cycle_lines:
                self.plot_volume.removeItem(line)
            self.cycle_lines.clear()

            for c_num, c_data in cycle_parsing.items():
                start_idx = c_data['full_idx'][0]
                if start_idx < len(time_s):
                    v_line = pg.InfiniteLine(
                        pos=time_s[start_idx], angle=90, movable=False,
                        pen=pg.mkPen(color='#555555', width=1.5, style=Qt.DashLine)
                    )
                    self.plot_volume.addItem(v_line)
                    self.cycle_lines.append(v_line)