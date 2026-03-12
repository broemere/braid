import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
from data_pipeline import DataPipeline


class MechanicsTab(QWidget):
    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline

        # Enable anti-aliasing for smooth curves
        pg.setConfigOptions(antialias=True)

        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create a GraphicsLayoutWidget to hold a 2x2 grid of plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)

        # --- 1. Hysteresis Loop (Stress vs. Stretch) ---
        self.plot_hysteresis = self.plot_widget.addPlot(title="True Stress vs. Axial Stretch")
        self.plot_hysteresis.setLabel('bottom', 'Stretch Ratio X (Loading)')
        self.plot_hysteresis.setLabel('left', 'True Stress', units='kPa')
        self.plot_hysteresis.showGrid(x=True, y=True, alpha=0.3)
        self.curve_hysteresis = self.plot_hysteresis.plot(pen=pg.mkPen(color='#ff007f', width=2))

        # --- 2. Energy Dissipation (Bar Graph) ---
        self.plot_energy = self.plot_widget.addPlot(title="Energy Dissipation per Cycle")
        self.plot_energy.setLabel('bottom', 'Cycle Number')
        self.plot_energy.setLabel('left', 'Dissipated Energy', units='mJ/mm³')

        # Initialize an empty BarGraphItem
        self.bar_energy = pg.BarGraphItem(x=[], height=[], width=0.6, brush='#ffaa00')
        self.plot_energy.addItem(self.bar_energy)


        # Move to the next row in the 2x2 grid
        self.plot_widget.nextRow()

        # --- 3. Orthogonal Stretch Trajectory ---
        self.plot_phase = self.plot_widget.addPlot(title="Stretch Trajectory (And Ideal Incompressible)")
        self.plot_phase.setLabel('bottom', 'Stretch Ratio X (Loading)')
        self.plot_phase.setLabel('left', 'Stretch Ratio Z (Thinning)')
        self.plot_phase.showGrid(x=True, y=True, alpha=0.3)
        self.curve_ideal = self.plot_phase.plot(pen=pg.mkPen(color='#888888', width=2, style=Qt.DashLine))
        self.curve_ideal.setZValue(-1)
        self.curve_phase = self.plot_phase.plot(pen=pg.mkPen(color='#00d2ff', width=2))

        # --- 4. Dynamic Volume (Stadium Assumption) ---
        self.plot_volume = self.plot_widget.addPlot(title="Dynamic Volume (Rounded Rectangle)")
        self.plot_volume.setLabel('bottom', 'Time', units='s')
        # Hardcode the label and turn off PyQtGraph's automatic SI prefixes
        self.plot_volume.setLabel('left', "Volume (mm³)")
        self.plot_volume.showGrid(x=True, y=True, alpha=0.3)
        self.curve_volume = self.plot_volume.plot(pen=pg.mkPen(color='#00ff00', width=2))
        # List to hold our dynamic cycle boundary lines
        self.cycle_lines = []

    def connect_signals(self):
        # Pipeline -> UI
        self.pipeline.mechanics_available.connect(self.on_mechanics_received)

    @Slot(dict)
    def on_mechanics_received(self, data: dict):
        """
        Populates the mechanics dashboard using strict XYZ coordinate keys.
        """
        print("Populating Mechanics Tab...")

        # Extract arrays safely from the new pipeline keys
        time_s = np.array(data.get('time_s', []))
        cycle_parsing = data.get('cycle_parsing', {})
        true_stress = np.array(data.get('true_stress_kpa', []))

        # Pulling the smooth mechanical stretch for the X-axis
        stretch_x_mech = np.array(data.get('stretch_x_mech', []))
        stretch_x_opt = np.array(data.get('stretch_x_opt', []))
        stretch_z = np.array(data.get('stretch_z', []))
        geom = self.pipeline.geometry_data
        dim_x = np.array(geom.get('dim_x', []))
        dim_y = np.array(geom.get('dim_y', []))
        dim_z = np.array(geom.get('dim_z', []))
        energy_dissipated = np.array(data.get('energy_dissipated', []))

        # 1. Update Hysteresis Loop (Stress vs. Stretch X)
        if len(stretch_x_opt) > 0 and len(stretch_x_opt) == len(true_stress):
            self.curve_hysteresis.setData(stretch_x_opt, true_stress)

        # 2. Update Energy Dissipation (Bar Graph)
        if len(energy_dissipated) > 0:
            cycle_numbers = np.arange(1, len(energy_dissipated) + 1)
            self.bar_energy.setOpts(x=cycle_numbers, height=energy_dissipated)

        # 3. Update Stretch Trajectory Loop (Stretch Z vs. Stretch X)
        if len(stretch_x_opt) > 0 and len(stretch_x_opt) == len(stretch_z):
            self.curve_phase.setData(stretch_x_opt, stretch_z)
            max_x = np.max(stretch_x_opt)
            ideal_x = np.array([1.0, max_x])
            ideal_y = -1.0 * (ideal_x - 1.0) + 1.0
            self.curve_ideal.setData(ideal_x, ideal_y)

            # 4. Update Dynamic Volume (Volume vs. Time)
            # Ensure we have matching geometry arrays
            if len(dim_x) > 0 and len(dim_x) == len(time_s):
                # Calculate Stadium Area in the X-Z plane
                area_xz = (np.pi * (dim_z / 2.0) ** 2) + ((dim_x - dim_z) * dim_z)

                # Calculate Volume (Area * Y) in mm^3
                volume_mm3 = area_xz * dim_y

                # 1. Update the live data using mm3 directly
                self.curve_volume.setData(time_s, volume_mm3)

                # 3. Anchor the Y-axis so the noise isn't blown out of proportion
                # Let's show a range from 80% of V0 up to 105% of V0
                self.plot_volume.setYRange(volume_mm3.min() * 0.95, volume_mm3.max() * 1.05)

                # First, clear out any old lines from a previous test
                for line in self.cycle_lines:
                    self.plot_volume.removeItem(line)
                self.cycle_lines.clear()

                # Next, iterate through the cycles and drop a line at the start of each
                for c_num, c_data in cycle_parsing.items():
                    start_idx = c_data['full_idx'][0]

                    # Failsafe to ensure the index exists in our time array
                    if start_idx < len(time_s):
                        cycle_start_time = time_s[start_idx]

                        # Create a subtle, dark gray dashed line
                        v_line = pg.InfiniteLine(
                            pos=cycle_start_time,
                            angle=90,
                            movable=False,
                            pen=pg.mkPen(color='#555555', width=1.5, style=Qt.DashLine)
                        )

                        self.plot_volume.addItem(v_line)
                        self.cycle_lines.append(v_line)