import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QButtonGroup, QFrame, QLabel
from data_pipeline import DataPipeline
from config import PLOT_COLORS


class PlotTab(QWidget):

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)

        self.pipeline = pipeline
        self.data = np.array([])
        self.unique_cycles = []
        self.plot_data_items = []

        self._init_ui()
        self.connect_signals()
        self.update_plot()

    def connect_signals(self):
        self.pipeline.data_available.connect(self.on_new_data_received)
        self.plot_selection_group.buttonClicked.connect(self._on_plot_selection_changed)
        self.cycle_selection_group.buttonClicked.connect(self._on_cycle_selection_changed)
        self.cycle_selection_group.buttonClicked.connect(self.update_plot)
        self.plot_selection_group.buttonClicked.connect(self.update_plot)

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Controls Panel ---
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(20)
        plot_selection_widget = self._create_plot_selection_controls()  # Time v Force, Time v Distance, Distance v Force
        controls_layout.addWidget(plot_selection_widget)
        self.cycle_selection_widget = self._create_cycle_selection_container()  # All Cycles, Cycle 0, ..., Last Cycle
        controls_layout.addWidget(self.cycle_selection_widget)
        controls_layout.addStretch()  # Push buttons to the top

        # --- Plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.addLegend()
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_item.getAxis('left').setTextPen('w')
        self.plot_item.getAxis('bottom').setTextPen('w')

        # --- Assemble Main Layout ---
        controls_container = QWidget()
        controls_container.setLayout(controls_layout)
        controls_container.setFixedWidth(200)
        main_layout.addWidget(controls_container)
        main_layout.addWidget(self.plot_widget)

    def _create_plot_selection_controls(self):
        """Creates the static buttons for choosing which data to plot."""
        self.plot_selection_group = QButtonGroup(self)
        buttons_config = [
            ("Time vs. Force", ["time_s", "force", "Time (s)", "Force (mN)", "Force vs. Time"]),
            ("Time vs. Distance", ["time_s", "distance", "Time (s)", "Distance (mm)", "Distance vs. Time"]),
            ("Distance vs. Force", ["distance", "force", "Distance (mm)", "Force (mN)", "Force vs. Distance"]),
        ]
        widget = self._create_control_group_widget("Plot Data", buttons_config, self.plot_selection_group)
        return widget

    def _create_control_group_widget(self, title, buttons_config, button_group):
        """Helper to create a styled group of buttons."""
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(container)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        layout.addWidget(title_label)
        button_group.setExclusive(True)
        for text, data_key in buttons_config:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setProperty("data_key", data_key)
            layout.addWidget(btn)
            button_group.addButton(btn)
        return container

    def _create_cycle_selection_container(self):
        """Creates the container for the cycle buttons. Buttons are added later."""
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(container)
        title_label = QLabel("Filter by Cycle")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        layout.addWidget(title_label)

        # This layout will hold the buttons when they are created
        self.cycle_buttons_layout = QVBoxLayout()
        layout.addLayout(self.cycle_buttons_layout)
        layout.addStretch()
        self.cycle_selection_group = QButtonGroup(self)
        self.cycle_selection_group.setExclusive(True)
        return container

    def _rebuild_cycle_buttons(self):
        """Clears and rebuilds the cycle filter buttons based on current data."""
        # Clear old buttons from layout
        while self.cycle_buttons_layout.count():
            child = self.cycle_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Clear old buttons from button group
        for button in self.cycle_selection_group.buttons():
            self.cycle_selection_group.removeButton(button)

        if not len(self.unique_cycles):
            # If there are no cycles, maybe show a label.
            info_label = QLabel("No cycle data loaded.")
            self.cycle_buttons_layout.addWidget(info_label)
            return

        # Rebuild config based on the new data
        buttons_config = [("All Cycles", -1)]
        buttons_config.extend([(f"Cycle {c}", c) for c in self.unique_cycles])
        buttons_config.append(("Last Cycle", -2))

        # Add new buttons
        for text, data_key in buttons_config:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setProperty("data_key", data_key)
            self.cycle_buttons_layout.addWidget(btn)
            self.cycle_selection_group.addButton(btn)

    def update_plot(self):
        """Core function to update the plot based on current button selections."""

        # --- 0. Clear all previous plot items ---
        for item in self.plot_data_items:
            self.plot_item.removeItem(item)
        self.plot_data_items = []

        # --- 1. Get Selected Plot Type ---
        checked_plot_btn = self.plot_selection_group.checkedButton()
        if not checked_plot_btn or self.data.size == 0:
            return

        plot_props = checked_plot_btn.property("data_key")
        x_key, y_key, x_label, y_label, title = plot_props

        # --- 2. Get Selected Cycle and Slice Data ---
        checked_cycle_btn = self.cycle_selection_group.checkedButton()
        if not checked_cycle_btn:
            return

        cycle_to_plot = checked_cycle_btn.property("data_key")
        cycle_text = checked_cycle_btn.text()  # Get button text for title

        if cycle_to_plot == -1:  # 'All'
            sliced_data = self.data
        elif cycle_to_plot == -2:  # 'Last'
            if not self.unique_cycles.size:
                return
            last_cycle = self.unique_cycles[-1]
            mask = self.data["cycle"] == last_cycle
            sliced_data = self.data[mask]
        else:  # Specific cycle number
            mask = self.data["cycle"] == cycle_to_plot
            sliced_data = self.data[mask]

        if sliced_data.size == 0:
            return  # Nothing to plot

        # --- 3. Plot Sliced Data (one line per cycle) ---

        # Find the unique cycles *within the data slice*
        # This handles all cases:
        # - "All Cycles" -> [1, 2, 3, ...]
        # - "Last Cycle" -> [5] (if 5 is last)
        # - "Cycle 3"    -> [3]
        cycles_in_slice = np.unique(sliced_data["cycle"])

        for i, cycle_num in enumerate(cycles_in_slice):
            # Get color, wrapping around the list using modulo
            color = PLOT_COLORS[int(cycle_num) % len(PLOT_COLORS)]

            # Get data for *this specific cycle* from the slice
            cycle_mask = sliced_data["cycle"] == cycle_num
            cycle_data = sliced_data[cycle_mask]

            x_data = cycle_data[x_key]
            y_data = cycle_data[y_key]

            # Set name for the legend
            name = f"Cycle {int(cycle_num)}"

            # Plot this cycle's data and add the item to our list
            pen = {"color": color, "width": 2}
            plot = self.plot_item.plot(x_data, y_data, pen=pen, name=name)
            self.plot_data_items.append(plot)

        # --- 4. Update Plot Labels and Title ---
        bottom_axis = self.plot_item.getAxis('bottom')
        left_axis = self.plot_item.getAxis('left')

        bottom_axis.setLabel(text=x_label, color='#ffffff', font_size='14pt')
        left_axis.setLabel(text=y_label, color='#ffffff', font_size='14pt')

        # Combine the base title with the selected cycle's text
        full_title = f"{title} - {cycle_text}"
        self.plot_item.setTitle(full_title, color='#ffffff', size='16pt')

    @Slot(QPushButton)
    def _on_plot_selection_changed(self, button: QPushButton):
        """Called when a plot selection button is clicked. Updates the pipeline."""
        if button:
            self.pipeline.set_plot_selection(button.text())

    @Slot(QPushButton)
    def _on_cycle_selection_changed(self, button: QPushButton):
        """Called when a cycle selection button is clicked. Updates the pipeline."""
        if button:
            self.pipeline.set_cycle_selection(button.text())


    @Slot(dict)
    def on_new_data_received(self, data: dict):
        """
        Slot to receive new data, convert it, rebuild UI components, and update the plot.
        """
        print(f"Data received{data.keys()}")
        if not data or "cycle" not in data:
            print("PlotTab received invalid or empty data.")
            self.data = np.array([])
            self.unique_cycles = []
        else:
            print("PlotTab received new data.")
            # Convert incoming dictionary to a structured numpy array for easier handling
            try:
                dtype = [(key, 'f8' if key != 'cycle' else 'i4') for key in data.keys()]
                records = list(zip(*data.values()))
                self.data = np.array(records, dtype=dtype)
                self.unique_cycles = np.unique(self.data["cycle"])
            except Exception as e:
                print(f"Could not convert data dictionary to structured numpy array: {e}")
                self.data = np.array([])
                self.unique_cycles = []

        # Rebuild the dynamic parts of the UI
        self._rebuild_cycle_buttons()

        saved_plot_text = self.pipeline.plot_selection
        plot_btn_found = False
        for button in self.plot_selection_group.buttons():
            if button.text() == saved_plot_text:
                button.setChecked(True)
                plot_btn_found = True
                break
        # Fallback if saved button (e.g., from a future version) isn't found
        if not plot_btn_found and self.plot_selection_group.buttons():
            self.plot_selection_group.buttons()[0].setChecked(True)

        saved_cycle_text = self.pipeline.cycle_selection
        cycle_btn_found = False
        for button in self.cycle_selection_group.buttons():
            if button.text() == saved_cycle_text:
                button.setChecked(True)
                cycle_btn_found = True
                break

        # Fallback if saved cycle (e.g., "Cycle 5") isn't in the new data
        if not cycle_btn_found and self.cycle_selection_group.buttons():
            default_button = self.cycle_selection_group.buttons()[0]  # "All Cycles"
            default_button.setChecked(True)

        # Trigger the plot update
        print(self.data)
        self.update_plot()
