import logging
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QDoubleSpinBox, QLabel, QTableWidget, QTableWidgetItem, QHeaderView
)
import pyqtgraph as pg
from data_pipeline import DataPipeline

log = logging.getLogger(__name__)


class LastPullTab(QWidget):
    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        pg.setConfigOptions(antialias=True)
        self.init_ui()
        self.connect_signals()

        # Force an initial update in case data is already loaded in the pipeline
        self.request_update()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # --- 1. Control Panel (Left) ---
        controls_layout = QVBoxLayout()
        # Fixed: Using Unicode X₀ instead of LaTeX $X_0$
        group_box = QGroupBox("Reference Length (X₀) Configuration")
        form_layout = QFormLayout(group_box)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Last Cycle Start",
            "Global Start (Cycle 0)",
            "Manual Length",
            "Preload Force Threshold"
        ])

        self.spin_manual_len = QDoubleSpinBox()
        self.spin_manual_len.setRange(0.001, 1000.0)
        self.spin_manual_len.setDecimals(3)
        self.spin_manual_len.setSuffix(" mm")
        self.spin_manual_len.setEnabled(False)

        self.spin_preload = QDoubleSpinBox()
        self.spin_preload.setRange(0.0, 10000.0)
        self.spin_preload.setDecimals(2)
        self.spin_preload.setSuffix(" mN")
        self.spin_preload.setEnabled(False)

        self.lbl_applied_x0 = QLabel("-- mm")
        self.lbl_applied_x0.setStyleSheet("font-weight: bold; color: #00d2ff;")

        form_layout.addRow("Configuration Mode:", self.combo_mode)
        form_layout.addRow("Manual Length:", self.spin_manual_len)
        form_layout.addRow("Preload Threshold:", self.spin_preload)
        # Fixed: Using Unicode X₀
        form_layout.addRow("Applied X₀:", self.lbl_applied_x0)

        controls_layout.addWidget(group_box)

        # --- NEW: Linearized Stiffness Parameters Table ---
        self.stiffness_group = QGroupBox("Linearized Stiffness Parameters")
        stiff_layout = QVBoxLayout(self.stiffness_group)

        self.table_stiff = QTableWidget(5, 2)
        self.table_stiff.setHorizontalHeaderLabels(["Initial (Low Stress)", "Final (High Stress)"])
        self.table_stiff.setVerticalHeaderLabels(
            ["Fitted Points (n)", "Slope (E) [kPa]", "Final λ", "Final σ [kPa]", "Intersection"])

        # Make the table cleanly fill the layout
        self.table_stiff.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_stiff.setEditTriggers(QTableWidget.NoEditTriggers)  # Read-only
        self.table_stiff.setSpan(4, 0, 1, 2)  # Span the intersection row across both columns

        stiff_layout.addWidget(self.table_stiff)

        cutoff_layout = QFormLayout()
        self.spin_cutoff = QDoubleSpinBox()
        self.spin_cutoff.setRange(0.0, 100.0)
        self.spin_cutoff.setDecimals(3)
        self.spin_cutoff.setSingleStep(0.05)
        self.spin_cutoff.setToolTip("Set to 0.0 to auto-calculate the middle index.")

        # Use Unicode λ for the label
        cutoff_layout.addRow("Curve Bend Location (λ):", self.spin_cutoff)
        stiff_layout.addLayout(cutoff_layout)

        controls_layout.addWidget(self.stiffness_group)
        controls_layout.addStretch()

        layout.addLayout(controls_layout, stretch=1)

        # --- 2. Plot Widget (Right) ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget, stretch=3)

        self.plot_pull = self.plot_widget.addPlot(title="Last Pull: True Stress vs. Stretch")
        # Fixed: Using Unicode λ
        self.plot_pull.setLabel('bottom', 'Stretch Ratio X (λ)')
        self.plot_pull.setLabel('left', 'True Stress', units='kPa')
        self.plot_pull.showGrid(x=True, y=True, alpha=0.3)

        self.curve_pull = self.plot_pull.plot(
            name="Last Pull",
            pen=pg.mkPen(color='#ffaa00', width=2.5)
        )

        # --- NEW: Graph Elements for Linear Fits ---
        self.curve_line0 = self.plot_pull.plot(pen=pg.mkPen(color='#00d2ff', width=2, style=Qt.DashLine))
        self.curve_line1 = self.plot_pull.plot(pen=pg.mkPen(color='#ff007f', width=2, style=Qt.DashLine))

        # Large white dot for the intersection
        self.scatter_intersect = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 255))
        self.plot_pull.addItem(self.scatter_intersect)

    def connect_signals(self):
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)

        self.combo_mode.currentIndexChanged.connect(self.request_update)
        self.spin_manual_len.valueChanged.connect(self.request_update)
        self.spin_preload.valueChanged.connect(self.request_update)
        self.spin_cutoff.valueChanged.connect(self.request_update)

        self.pipeline.last_pull_available.connect(self.on_last_pull_received)
        self.pipeline.mechanics_available.connect(lambda _: self.request_update())

    @Slot()
    def on_mode_changed(self):
        mode_text = self.combo_mode.currentText()
        self.spin_manual_len.setEnabled(mode_text == "Manual Length")
        self.spin_preload.setEnabled(mode_text == "Preload Force Threshold")

    @Slot()
    def request_update(self):
        mode_text = self.combo_mode.currentText()
        log.info(f"LastPullTab requesting update. Mode selected: {mode_text}")

        mode_map = {
            "Global Start (Cycle 0)": "global_start",
            "Last Cycle Start": "cycle_start",
            "Manual Length": "manual",
            "Preload Force Threshold": "preload"
        }

        mode_key = mode_map.get(mode_text, "cycle_start")
        manual_len = self.spin_manual_len.value()
        preload = self.spin_preload.value()
        cutoff = self.spin_cutoff.value()

        self.pipeline.calculate_last_pull(
            ref_mode=mode_key,
            manual_length=manual_len,
            preload_force=preload,
            cutoff_stretch = cutoff
        )

    @Slot(dict)
    def on_last_pull_received(self, data: dict):
        stretch = np.array(data.get('stretch', []))
        stress = np.array(data.get('stress', []))
        ref_length = data.get('ref_length', 0.0)
        stiff = data.get('stiffness')
        applied_cutoff = data.get('applied_cutoff', 0.0)

        # --- NEW: Safely update the spinbox without triggering a recalculation loop ---
        self.spin_cutoff.blockSignals(True)
        self.spin_cutoff.setValue(applied_cutoff)
        self.spin_cutoff.blockSignals(False)

        log.info(f"LastPullTab received data payload. Size: {len(stretch)} points. X0: {ref_length}")

        if len(stretch) > 0 and len(stretch) == len(stress):
            self.curve_pull.setData(stretch, stress)
            self.lbl_applied_x0.setText(f"{ref_length:.3f} mm")

            # --- NEW: Smart X-Axis Anchoring ---
            min_stretch = np.min(stretch)
            max_stretch = np.max(stretch)

            # Anchor left side to 1.0, UNLESS the actual data goes lower
            # (e.g., a user manually enters an X0 that is larger than the starting length)
            x_anchor = min(1.0, min_stretch)

            # Create a ~2% visual padding so the graph line doesn't scrape the bounding box walls
            x_range = max_stretch - x_anchor
            padding = x_range * 0.02 if x_range > 0 else 0.05

            # Manually set X-axis limits
            self.plot_pull.setXRange(x_anchor - padding, max_stretch + padding, padding=0)

            # Let pyqtgraph continue to auto-scale the Y-axis freely based on the new slice
            self.plot_pull.enableAutoRange(axis=pg.ViewBox.YAxis)

            # --- NEW: Populate Analysis Table and Fit Lines ---
            if stiff:
                init = stiff['initial']
                term = stiff['terminal']
                inter = stiff['intersect']

                # Table Population
                self.table_stiff.setItem(0, 0, QTableWidgetItem(str(init['n'])))
                self.table_stiff.setItem(1, 0, QTableWidgetItem(f"{init['E']:.2f}"))
                self.table_stiff.setItem(2, 0, QTableWidgetItem(f"{init['lambda']:.3f}"))
                self.table_stiff.setItem(3, 0, QTableWidgetItem(f"{init['sigma']:.2f}"))

                self.table_stiff.setItem(0, 1, QTableWidgetItem(str(term['n'])))
                self.table_stiff.setItem(1, 1, QTableWidgetItem(f"{term['E']:.2f}"))
                self.table_stiff.setItem(2, 1, QTableWidgetItem(f"{term['lambda']:.3f}"))
                self.table_stiff.setItem(3, 1, QTableWidgetItem(f"{term['sigma']:.2f}"))

                # Intersection row
                intersect_text = f"λm: {inter['lambda']:.3f}   |   σm: {inter['sigma']:.2f} kPa" if not np.isnan(
                    inter['lambda']) else "No Intersection"
                inter_item = QTableWidgetItem(intersect_text)
                inter_item.setTextAlignment(Qt.AlignCenter)
                self.table_stiff.setItem(4, 0, inter_item)

                # Render the lines across the boundaries of the plotted stretch data
                x_bounds_init = np.array([x_anchor, max_stretch])
                l0_y = init['E'] * x_bounds_init + init['b']
                self.curve_line0.setData(x_bounds_init, l0_y)

                # 2. Terminal Line: Prevent crossing below y = 0
                if term['E'] > 1e-6:  # Prevent division by zero if slope is totally flat
                    x_zero_crossing = -term['b'] / term['E']
                    # Start at the zero crossing, unless the zero crossing is somehow off the left edge of the graph
                    x_start_term = max(x_anchor, x_zero_crossing)
                else:
                    x_start_term = x_anchor

                x_bounds_term = np.array([x_start_term, max_stretch])
                l1_y = term['E'] * x_bounds_term + term['b']
                self.curve_line1.setData(x_bounds_term, l1_y)

                # Render Intersection scatter point
                if not np.isnan(inter['lambda']):
                    self.scatter_intersect.setData([inter['lambda']], [inter['sigma']])
                else:
                    self.scatter_intersect.setData([], [])

        else:
            log.warning("LastPullTab received empty or mismatched data arrays. Clearing plot.")
            self.curve_pull.setData([], [])
            self.curve_line0.setData([], [])
            self.curve_line1.setData([], [])
            self.scatter_intersect.setData([], [])
            self.lbl_applied_x0.setText("-- mm")