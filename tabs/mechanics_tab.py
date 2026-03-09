import numpy as np
from PySide6.QtCore import Slot
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

        # --- 1. Kinematic Phase Loop ---
        self.plot_phase = self.plot_widget.addPlot(title="Kinematic Phase Loop")
        self.plot_phase.setLabel('bottom', 'Stretch Ratio (Width)')
        self.plot_phase.setLabel('left', 'Stretch Ratio (Thickness)')
        # Add a grid for easier visual phase-lag tracking
        self.plot_phase.showGrid(x=True, y=True, alpha=0.3)
        self.curve_phase = self.plot_phase.plot(pen=pg.mkPen(color='#00d2ff', width=2))

        # --- 2. Hysteresis Loop (Stress vs. Stretch) ---
        self.plot_hysteresis = self.plot_widget.addPlot(title="True Stress vs. Stretch")
        self.plot_hysteresis.setLabel('bottom', 'Stretch Ratio (Width)')
        self.plot_hysteresis.setLabel('left', 'True Stress', units='kPa')
        self.plot_hysteresis.showGrid(x=True, y=True, alpha=0.3)
        self.curve_hysteresis = self.plot_hysteresis.plot(pen=pg.mkPen(color='#ff007f', width=2))

        # Move to the next row in the 2x2 grid
        self.plot_widget.nextRow()

        # --- 3. Energy Dissipation (Bar Graph) ---
        self.plot_energy = self.plot_widget.addPlot(title="Energy Dissipation per Cycle")
        self.plot_energy.setLabel('bottom', 'Cycle Number')
        self.plot_energy.setLabel('left', 'Dissipated Energy', units='mJ/mm³')

        # Initialize an empty BarGraphItem
        self.bar_energy = pg.BarGraphItem(x=[], height=[], width=0.6, brush='#ffaa00')
        self.plot_energy.addItem(self.bar_energy)

        # --- 4. Dynamic Poisson's Ratio ---
        self.plot_poisson = self.plot_widget.addPlot(title="Dynamic Poisson's Ratio")
        self.plot_poisson.setLabel('bottom', 'Time', units='s')
        self.plot_poisson.setLabel('left', "Poisson's Ratio (ν)")
        self.plot_poisson.showGrid(x=True, y=True, alpha=0.3)
        self.curve_poisson = self.plot_poisson.plot(pen=pg.mkPen(color='#00ff00', width=2))

    def connect_signals(self):
        # Pipeline -> UI
        self.pipeline.mechanics_available.connect(self.on_mechanics_received)

    @Slot(dict)
    def on_mechanics_received(self, data: dict):
        """
        Populates the mechanics dashboard when new cyclic data is calculated.
        """
        print("Populating Mechanics Tab...")

        # Extract arrays safely
        time_s = np.array(data.get('time_s', []))
        true_stress = np.array(data.get('true_stress_kpa', []))
        stretch_w = np.array(data.get('stretch_w', []))
        stretch_t = np.array(data.get('stretch_t', []))
        poissons_ratio = np.array(data.get('poissons_ratio', []))
        energy_dissipated = np.array(data.get('energy_dissipated', []))

        # 1. Update Kinematic Phase Loop (Stretch T vs. Stretch W)
        if len(stretch_w) > 0 and len(stretch_w) == len(stretch_t):
            self.curve_phase.setData(stretch_w, stretch_t)

        # 2. Update Hysteresis Loop (Stress vs. Stretch W)
        if len(stretch_w) > 0 and len(stretch_w) == len(true_stress):
            self.curve_hysteresis.setData(stretch_w, true_stress)

        # 3. Update Energy Dissipation (Bar Graph)
        if len(energy_dissipated) > 0:
            # Create an X-axis array of integers [1, 2, 3, 4, 5...] for the cycle numbers
            cycle_numbers = np.arange(1, len(energy_dissipated) + 1)
            self.bar_energy.setOpts(x=cycle_numbers, height=energy_dissipated)

        # 4. Update Poisson's Ratio (Poisson vs. Time)
        if len(time_s) > 0 and len(time_s) == len(poissons_ratio):
            self.curve_poisson.setData(time_s, poissons_ratio)