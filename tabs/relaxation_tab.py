import numpy as np
from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout
import pyqtgraph as pg
from data_pipeline import DataPipeline


class RelaxationTab(QWidget):
    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        pg.setConfigOptions(antialias=True)
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # --- 1. Plotting Area (Top) ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget, stretch=3)

        # --- True Stress vs Time ---
        self.plot_stress = self.plot_widget.addPlot(title="True Stress Relaxation")
        self.plot_stress.setLabel('left', 'True Stress', units='kPa')
        self.plot_stress.showGrid(x=True, y=True, alpha=0.3)

        self.legend_stress = self.plot_stress.addLegend(offset=(-10, 10), movable=True)

        self.curve_stress_data = self.plot_stress.plot(pen=pg.mkPen(color='#ff007f', width=2), name="True Stress")
        self.curve_stress_fit = self.plot_stress.plot(pen=pg.mkPen(color='#ffffff', width=2, style=Qt.DashLine), name = "Fitted Model")

        # FIX: Add the missing vertical marker for the Stress plot
        self.v_line_stress = pg.InfiniteLine(angle=90, movable=False,
                                             pen=pg.mkPen(color='#aaaaaa', width=1.5, style=Qt.DashLine))
        self.plot_stress.addItem(self.v_line_stress)

        # NEW: Horizontal baseline for the Equilibrium Stress Asymptote
        self.h_line_inf = pg.InfiniteLine(angle=0, movable=False,
                                          pen=pg.mkPen(color='#888888', width=1.5, style=Qt.DashLine))
        self.plot_stress.addItem(self.h_line_inf)
        self.h_line_inf.setZValue(-1)

        self.plot_widget.nextRow()

        # --- Kinematic Creep (Z-Thickness vs Time) ---
        self.plot_creep = self.plot_widget.addPlot(title="Creep (Transverse Thinning)")
        self.plot_creep.setLabel('bottom', 'Time', units='s')
        self.plot_creep.setLabel('left', 'Z-Thickness', units='m')
        self.plot_creep.showGrid(x=True, y=True, alpha=0.3)
        self.curve_creep = self.plot_creep.plot(pen=pg.mkPen(color='#00d2ff', width=2))

        # Vertical marker for the Creep plot
        self.v_line_creep = pg.InfiniteLine(angle=90, movable=False,
                                            pen=pg.mkPen(color='#aaaaaa', width=1.5, style=Qt.DashLine))
        self.plot_creep.addItem(self.v_line_creep)

        # NEW: Horizontal baseline for the Initial Thickness (Z0)
        self.h_line_z0 = pg.InfiniteLine(angle=0, movable=False,
                                         pen=pg.mkPen(color='#888888', width=1.5, style=Qt.DashLine))
        self.plot_creep.addItem(self.h_line_z0)
        self.h_line_z0.setZValue(-1)

        # Link X axes for synchronized zooming
        self.plot_creep.setXLink(self.plot_stress)
        # --- 2. Metrics Dashboard (Bottom) ---
        # Create a horizontal layout to hold our two side-by-side categories
        dashboard_layout = QHBoxLayout()

        # --- Group A: Empirical Mechanics (Graph Derived) ---
        empirical_group = QGroupBox("Empirical Mechanics (Graph Derived)")
        empirical_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; }")
        emp_grid = QGridLayout()
        empirical_group.setLayout(emp_grid)

        self.lbl_peak = QLabel("Peak Stress (σ_peak): -- kPa")
        self.lbl_e_inst = QLabel("Step Modulus (E_step): -- kPa")
        self.lbl_percent = QLabel("Total Relaxation: -- %")

        # 2x2 Grid for the raw physical metrics
        emp_grid.addWidget(self.lbl_peak, 0, 0)
        emp_grid.addWidget(self.lbl_e_inst, 0, 1)
        emp_grid.addWidget(self.lbl_percent, 1, 0, 1, 2, Qt.AlignCenter)
        #emp_grid.addWidget(self.lbl_e_inf, 1, 1)

        # --- Group B: Viscoelastic Model (Fitted Parameters) ---
        model_group = QGroupBox("Generalized Maxwell 3-Arm Model (Least Squares Fit)")
        model_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; }")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)

        # The Equation Label (Using HTML for sub/superscripts)
        eq_text = "<b>Equation:</b> σ(t) = σ<sub>∞</sub> + σ<sub>1</sub>e<sup>-t/τ<sub>1</sub></sup> + σ<sub>2</sub>e<sup>-t/τ<sub>2</sub></sup>"
        self.lbl_equation = QLabel(eq_text)
        self.lbl_equation.setAlignment(Qt.AlignCenter)
        self.lbl_equation.setStyleSheet(
            "font-size: 15px; padding: 5px; color: #ff007f;")  # Matches the stress curve
        model_layout.addWidget(self.lbl_equation)

        model_grid = QGridLayout()
        model_layout.addLayout(model_grid)

        self.lbl_inf = QLabel("Equilibrium Stress (σ_inf): -- kPa")
        self.lbl_e_inf = QLabel("Relaxed Modulus (E_inf): -- kPa")
        self.lbl_visc1 = QLabel("Fast Visc. Drop (σ_1): -- kPa")
        self.lbl_tau1 = QLabel("Fast Time Const (τ_1): -- s")
        self.lbl_visc2 = QLabel("Slow Visc. Drop (σ_2): -- kPa")
        self.lbl_tau2 = QLabel("Slow Time Const (τ_2): -- s")

        # Layout the model parameters symmetrically
        model_grid.addWidget(self.lbl_inf, 0, 0) # 1, 2, Qt.AlignCenter)  # Spans the top row
        model_grid.addWidget(self.lbl_e_inf, 0, 1)
        model_grid.addWidget(self.lbl_visc1, 1, 0)
        model_grid.addWidget(self.lbl_tau1, 1, 1)
        model_grid.addWidget(self.lbl_visc2, 2, 0)
        model_grid.addWidget(self.lbl_tau2, 2, 1)

        # Apply consistent styling to all dynamic data labels
        labels = [self.lbl_peak, self.lbl_e_inst, self.lbl_e_inf, self.lbl_percent,
                  self.lbl_inf, self.lbl_visc1, self.lbl_tau1, self.lbl_visc2, self.lbl_tau2]
        for lbl in labels:
            lbl.setStyleSheet("font-size: 14px; padding: 5px;")

        # Add both groups to the horizontal dashboard layout
        dashboard_layout.addWidget(empirical_group)
        dashboard_layout.addWidget(model_group)

        # Add the dashboard to the main tab layout
        layout.addLayout(dashboard_layout, stretch=1)
    def connect_signals(self):
        self.pipeline.relaxation_available.connect(self.on_relaxation_received)

    @Slot(dict)
    def on_relaxation_received(self, data: dict):
        print("Populating Relaxation Tab...")

        # 1. Update Plots
        time_s = np.array(data['time_s'])
        self.curve_stress_data.setData(time_s, data['stress_kpa'])
        self.curve_stress_fit.setData(data['hold_time_raw'], data['fitted_stress'])
        self.curve_creep.setData(time_s, np.array(data['dim_z']) * 1e-3)

        # 2. Update Vertical Markers (Hold Start)
        hold_start_idx = data['peak_stress_idx']
        if 0 <= hold_start_idx < len(time_s):
            hold_time = time_s[hold_start_idx]
            self.v_line_stress.setValue(hold_time)
            self.v_line_creep.setValue(hold_time)

        # 3. NEW: Update Horizontal Baselines
        m = data['metrics']
        self.h_line_inf.setValue(m['sigma_inf'])  # Equilibrium Stress

        # Grab the initial resting thickness at t=0
        z0 = np.array(data['dim_z'])[hold_start_idx] * 1e-3
        self.h_line_z0.setValue(z0)

        # 4. Update Dashboard Text
        self.lbl_peak.setText(f"Peak Stress (σ_peak): {m['peak_stress']:.2f} kPa")
        self.lbl_inf.setText(f"Equilibrium Stress (σ_inf): {m['sigma_inf']:.2f} kPa")

        self.lbl_visc1.setText(f"Fast Visc. Drop (σ_1): {m['sigma_1']:.2f} kPa")
        self.lbl_tau1.setText(f"Fast Time Const (τ_1): {m['tau_1']:.2f} s")

        self.lbl_visc2.setText(f"Slow Visc. Drop (σ_2): {m['sigma_2']:.2f} kPa")
        self.lbl_tau2.setText(f"Slow Time Const (τ_2): {m['tau_2']:.2f} s")

        self.lbl_e_inst.setText(f"Inst. Modulus (E_inst): {m['e_inst']:.2f} kPa")
        self.lbl_e_inf.setText(f"Relaxed Modulus (E_inf): {m['e_inf']:.2f} kPa")
        self.lbl_percent.setText(f"Total Relaxation: {m['percent_relax']:.1f} %")