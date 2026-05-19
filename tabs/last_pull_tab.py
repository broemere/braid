import logging
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QDoubleSpinBox, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QPushButton, QSlider, QSpinBox, QCheckBox
)
from PySide6.QtGui import QDoubleValidator, QColor
import pyqtgraph as pg
from data_pipeline import DataPipeline

log = logging.getLogger(__name__)


class LastPullTab(QWidget):
    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.current_stretch = np.array([])
        pg.setConfigOptions(antialias=True)
        self.init_ui()
        self.connect_signals()

        # Force an initial update in case data is already loaded in the pipeline
        self.request_update()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        top_split_layout = QHBoxLayout()

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
        self.table_stiff.setHorizontalHeaderLabels(["Low Stress", "High Stress"])
        self.table_stiff.setVerticalHeaderLabels(
            ["Fitted Points (n)", "Slope (E) [kPa]", "Stop λ", "Stop σ [kPa]", "Intersection"])

        # Make the table cleanly fill the layout
        self.table_stiff.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_stiff.setEditTriggers(QTableWidget.NoEditTriggers)  # Read-only
        self.table_stiff.setSpan(4, 0, 1, 2)  # Span the intersection row across both columns

        stiff_layout.addWidget(self.table_stiff)

        cutoff_layout = QHBoxLayout()  # Changed to HBox to put the button next to the input

        lbl_cutoff = QLabel("Region Split Cutoff (λ):")

        # Replace SpinBox with a LineEdit and a Validator
        self.edit_cutoff = QLineEdit()
        self.edit_cutoff.setValidator(QDoubleValidator(0.0, 100.0, 3))
        self.edit_cutoff.setPlaceholderText("e.g. 1.050 (0 to auto-calculate)")
        self.edit_cutoff.setToolTip("Set to 0 or leave blank to auto-calculate the middle index.")

        self.btn_apply_cutoff = QPushButton("Apply")

        self.show_linear_cb = QCheckBox("Show Fits")
        self.show_linear_cb.setChecked(False)  # Disabled by default

        stiff_layout.addWidget(lbl_cutoff)
        cutoff_layout.addWidget(self.edit_cutoff)
        cutoff_layout.addWidget(self.btn_apply_cutoff)
        cutoff_layout.addWidget(self.show_linear_cb)

        stiff_layout.addLayout(cutoff_layout)

        controls_layout.addWidget(self.stiffness_group)

        controls_layout.addStretch()

        top_split_layout.addLayout(controls_layout, stretch=1)
        self.plot_widget = pg.GraphicsLayoutWidget()
        top_split_layout.addWidget(self.plot_widget, stretch=3)

        # Add the completed split layout to the main root layout
        main_layout.addLayout(top_split_layout, stretch=1)

        #layout.addLayout(controls_layout, stretch=1)

        # --- Bottom Section: Full-Width Spline Smoothing Tool ---
        self.spline_group = QGroupBox("Spline Smoothing")
        spline_layout = QHBoxLayout(self.spline_group)

        spline_layout.addWidget(QLabel("Smoothing Factor (s):"))

        self.s_slider = QSlider(Qt.Horizontal)
        self.s_slider.setMinimum(0)
        self.s_slider.setMaximum(100000)
        self.s_slider.setValue(50)
        self.s_slider.setTickPosition(QSlider.TicksBelow)
        # Adding stretch=1 allows the slider to greedily consume all remaining horizontal space
        spline_layout.addWidget(self.s_slider, stretch=1)

        self.s_spinbox = QSpinBox()
        self.s_spinbox.setMinimum(0)
        self.s_spinbox.setMaximum(100000)
        self.s_spinbox.setValue(50)
        self.s_spinbox.setFixedWidth(80)
        spline_layout.addWidget(self.s_spinbox)

        self.show_spline_cb = QCheckBox("Show Smoothed Spline")
        self.show_spline_cb.setChecked(True)
        spline_layout.addWidget(self.show_spline_cb)

        # Pin the spline group to the bottom of the main layout
        main_layout.addWidget(self.spline_group)

        # --- 2. Plot Widget (Right) ---
        #self.plot_widget = pg.GraphicsLayoutWidget()
        #ayout.addWidget(self.plot_widget, stretch=3)

        self.plot_pull = self.plot_widget.addPlot(title="Last Pull: True Stress vs. Stretch")
        # Fixed: Using Unicode λ
        self.plot_pull.setLabel('bottom', 'Stretch Ratio X (λ)')
        self.plot_pull.setLabel('left', 'True Stress', units='kPa')
        self.plot_pull.showGrid(x=True, y=True, alpha=0.3)

        self.curve_pull = self.plot_pull.plot(
            name="Raw Data",
            pen=pg.mkPen(color='#ffaa00', width=1.5)
        )
        self.curve_pull.setZValue(1)

        # Spline Curve
        self.curve_spline = self.plot_pull.plot(
            name="B-Spline",
            pen=pg.mkPen(color='#E74C3C', width=2.5)
        )
        self.curve_spline.setZValue(2)

        # Linear Fit Curves
        self.curve_line0 = self.plot_pull.plot(pen=pg.mkPen(color='#00d2ff', width=2, style=Qt.DashLine))
        self.curve_line1 = self.plot_pull.plot(pen=pg.mkPen(color='#ff007f', width=2, style=Qt.DashLine))
        self.scatter_intersect = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 255))
        self.plot_pull.addItem(self.scatter_intersect)
        self.curve_line0.setZValue(3)
        self.curve_line1.setZValue(3)
        self.scatter_intersect.setZValue(4)

        self.curve_line0.setVisible(False)
        self.curve_line1.setVisible(False)
        self.scatter_intersect.setVisible(False)

    def connect_signals(self):
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)

        self.combo_mode.currentIndexChanged.connect(self.request_update)
        self.spin_manual_len.valueChanged.connect(self.request_update)
        self.spin_preload.valueChanged.connect(self.request_update)
        self.btn_apply_cutoff.clicked.connect(self.request_update)
        self.edit_cutoff.returnPressed.connect(self.request_update)
        self.show_linear_cb.stateChanged.connect(self.toggle_linear_fits)

        self.s_slider.valueChanged.connect(self._slider_value_changed)
        self.s_spinbox.valueChanged.connect(self._spinbox_value_changed)
        self.show_spline_cb.stateChanged.connect(self.run_spline_transform)

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
        cutoff_text = self.edit_cutoff.text().strip()
        try:
            cutoff = float(cutoff_text) if cutoff_text else 0.0
        except ValueError:
            cutoff = 0.0

        self.pipeline.calculate_last_pull(
            ref_mode=mode_key,
            manual_length=manual_len,
            preload_force=preload,
            cutoff_stretch = cutoff
        )

    @Slot()
    def toggle_linear_fits(self):
        """Toggles the visibility of the linearized stiffness fits and intersection point."""
        is_visible = self.show_linear_cb.isChecked()
        self.curve_line0.setVisible(is_visible)
        self.curve_line1.setVisible(is_visible)
        self.scatter_intersect.setVisible(is_visible)

    def _slider_value_changed(self, value: int):
        self.s_spinbox.blockSignals(True)
        self.s_spinbox.setValue(value)
        self.s_spinbox.blockSignals(False)
        self.run_spline_transform()

    def _spinbox_value_changed(self, value: int):
        self.s_slider.blockSignals(True)
        self.s_slider.setValue(value)
        self.s_slider.blockSignals(False)
        self.run_spline_transform()

    def run_spline_transform(self):
        """Fetches the smoothed Y values from the pipeline and plots them."""
        show_spline = self.show_spline_cb.isChecked()
        self.curve_spline.setVisible(show_spline)

        if not show_spline or len(self.current_stretch) == 0:
            return

        s_value = self.s_slider.value()

        # Ask pipeline for smoothed Y data
        spline_y_values = self.pipeline.calculate_spline(s_value)

        if spline_y_values is not None and len(spline_y_values) == len(self.current_stretch):
            self.curve_spline.setData(self.current_stretch, spline_y_values)
        else:
            self.curve_spline.setData([], [])

    @Slot(dict)
    def on_last_pull_received(self, data: dict):
        stretch = np.array(data.get('stretch', []))
        stress = np.array(data.get('stress', []))
        ref_length = data.get('ref_length', 0.0)
        stiff = data.get('stiffness')
        applied_cutoff = data.get('applied_cutoff', 0.0)

        self.current_stretch = stretch

        # --- NEW: Safely update the spinbox without triggering a recalculation loop ---
        self.edit_cutoff.blockSignals(True)
        # Only update the box with the pipeline's value if the user left it blank/0,
        # otherwise leave their typed text alone so it doesn't jarringly reformat on them.
        current_text = self.edit_cutoff.text().strip()
        if not current_text or (current_text.replace('.', '', 1).isdigit() and float(current_text) == 0.0):
            self.edit_cutoff.setText(f"{applied_cutoff:.3f}")
        self.edit_cutoff.blockSignals(False)

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

            self.s_slider.blockSignals(True)
            self.s_spinbox.blockSignals(True)

            max_s = len(stretch) * 2  # Arbitrary scaling so the slider has good range
            self.s_slider.setMaximum(max_s)
            self.s_spinbox.setMaximum(max_s)
            self.s_slider.setTickInterval(max_s // 10)

            self.s_slider.blockSignals(False)
            self.s_spinbox.blockSignals(False)

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

            self.run_spline_transform()

        else:
            log.warning("LastPullTab received empty or mismatched data arrays. Clearing plot.")
            self.curve_pull.setData([], [])
            self.curve_spline.setData([], [])
            self.curve_line0.setData([], [])
            self.curve_line1.setData([], [])
            self.scatter_intersect.setData([], [])
            self.lbl_applied_x0.setText("-- mm")