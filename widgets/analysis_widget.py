import os
import logging
from PySide6.QtCore import Slot, Signal, QEvent, QSettings
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFrame, QTabWidget, QMessageBox, QApplication, QLineEdit, \
    QHBoxLayout, QLabel, QSizePolicy
from data_pipeline import DataPipeline
from processing.task_manager import TaskManager
from widgets.file_picker import FilePickerWidget
from processing.data_loader import get_system_username
from tabs.plot_tab import PlotTab
from tabs.scale_tab import ScaleTab
from tabs.roi_tab import ROITab
from tabs.seed_tab import SeedTab
from tabs.thresh_tab import ThreshTab
from tabs.geometry_tab import GeometryTab
from tabs.mechanics_tab import MechanicsTab
from tabs.export_tab import ExportTab


log = logging.getLogger(__name__)

class AnalysisWidget(QWidget):
    """
    A widget that encapsulates a single, self-contained analysis session.
    It manages its own DataPipeline and the UI components related to it.
    """
    tab_name_requested = Signal(str)

    def __init__(self, task_manager: TaskManager, settings: QSettings, parent=None):
        """
        Initializes the session widget.

        Args:
            task_manager (TaskManager): The shared TaskManager instance from MainWindow.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.task_manager = task_manager
        self.settings = settings
        self.pipeline = DataPipeline(self)
        self.pipeline.task_manager = self.task_manager

        self.init_ui()
        self.connect_signals()
        QApplication.instance().installEventFilter(self)
        self._load_settings_into_pipeline()
        log.info("AnalysisWidget created.")

    def init_ui(self):
        """Initializes the user interface for this session."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0) # Use the full space of the tab

        # File selection widgets for this session
        self.file_pickers = FilePickerWidget()
        self.file_pickers.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.file_pickers, 1)

        # Text field for author name
        author_widget = QWidget()
        author_widget.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Preferred
        )
        author_layout = QHBoxLayout(author_widget)
        author_label = QLabel("Author:")
        author_layout.addWidget(author_label)
        self.tab_author = QLineEdit()
        self.tab_author.setPlaceholderText("Enter name...")
        username = get_system_username()
        self.pipeline.on_author_changed(username)
        self.tab_author.setText(username)
        self.tab_author.setClearButtonEnabled(True)
        self.tab_author.textEdited.connect(self.pipeline.on_author_changed)
        author_layout.addWidget(self.tab_author)

        top_layout.addWidget(author_widget, 0)

        main_layout.addLayout(top_layout)

        # A visual separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(sep)

        # The tab widget for all the analysis steps (Plot, Scale, etc.)
        self.analysis_tabs = QTabWidget()
        main_layout.addWidget(self.analysis_tabs)

        self.init_analysis_tabs()


    def eventFilter(self, watched, event):
        """
        Filters events for the entire application. Used here to clear focus
        from the new_tab_name_input when the user clicks elsewhere.
        """
        if event.type() == QEvent.MouseButtonPress:
            # Check if the input field currently has focus
            if self.tab_author.hasFocus():
                # Find out which widget was actually clicked on
                clicked_widget = QApplication.widgetAt(event.globalPos())

                # If the click was outside the line edit and its children (like the clear button), then unfocus it.
                if clicked_widget and clicked_widget != self.tab_author and not self.tab_author.isAncestorOf(
                        clicked_widget):
                    self.tab_author.clearFocus()

        # Pass the event on to the parent class for default processing
        return super().eventFilter(watched, event)

    def init_analysis_tabs(self):
        """Creates and adds all the analysis-specific tabs."""
        self.plot_tab = PlotTab(self.pipeline)
        self.analysis_tabs.addTab(self.plot_tab, "📈 Plot")
        self.scale_tab = ScaleTab(self.pipeline)
        self.analysis_tabs.addTab(self.scale_tab, "📏 Scale")
        self.roi_tab = ROITab(self.pipeline)
        self.analysis_tabs.addTab(self.roi_tab, "🎯 ROI")
        self.seed_tab = SeedTab(self.pipeline)
        self.analysis_tabs.addTab(self.seed_tab, "🌱 Seed")
        self.thresh_tab = ThreshTab(self.pipeline)
        self.analysis_tabs.addTab(self.thresh_tab, "🏁 Threshold")
        self.geometry_tab = GeometryTab(self.pipeline)
        self.analysis_tabs.addTab(self.geometry_tab, "📊 Geometry")
        self.mechanics_tab = MechanicsTab(self.pipeline)
        self.analysis_tabs.addTab(self.mechanics_tab, "🧮 Mechanics")
        self.export_tab = ExportTab(self.pipeline)
        self.analysis_tabs.addTab(self.export_tab, "📦 Export")


    def connect_signals(self):
        """Connects signals from widgets to the appropriate slots in this session."""
        self.file_pickers.video_selected.connect(self.on_file_selected)
        self.pipeline.plot_selection_changed.connect(self._save_plot_selection)
        self.pipeline.cycle_selection_changed.connect(self._save_cycle_selection)
        self.pipeline.known_length_changed.connect(self._save_known_length)
        self.pipeline.scale_is_manual_changed.connect(self._save_scale_is_manual)
        self.pipeline.manual_conversion_factor_changed.connect(self._save_manual_conversion_factor)
        #self.pipeline.scale_changed.connect(self._save_scale)

    def _load_settings_into_pipeline(self):
        """Reads values from QSettings and populates the pipeline."""
        log.info("Loading persistent settings into data pipeline...")
        plot_selection = self.settings.value("plot/plot_selection", defaultValue="Time vs. Force", type=str)
        self.pipeline.set_plot_selection(plot_selection)
        cycle_selection = self.settings.value("plot/cycle_selection", defaultValue="All Cycles", type=str)
        self.pipeline.set_cycle_selection(cycle_selection)
        # Use the pipeline's setter method. This will also emit the signal
        # that the ScaleTab is listening to, automatically updating the UI.
        known_length = self.settings.value("scale/known_length", defaultValue=0.0, type=float)
        self.pipeline.set_known_length(known_length)
        manual_factor = self.settings.value("scale/manual_factor", defaultValue=0.0, type=float)
        self.pipeline.set_manual_conversion_factor(manual_factor)
        is_manual = self.settings.value("scale/is_manual", defaultValue=False, type=bool)
        self.pipeline.set_scale_is_manual(is_manual)



    @Slot(str)
    def on_file_selected(self, path: str):
        """Handles the selection of a video file for this session."""
        log.info(f"Session received video file path: {path}")
        self._handle_video_load(path)

    def _handle_video_load(self, path: str):
        """Single source of truth for loading a video file within this session."""
        log.info(f"Loading Video {path} for session.")
        self.pipeline.load_video_file(path)
        self.file_pickers.set_video_label(path)
        file_name = os.path.basename(path)
        base_name, _ = os.path.splitext(file_name)
        self.tab_name_requested.emit(base_name.strip("_video").strip("recording_"))

    # --- SLOTS FOR SAVING ---

    @Slot(str)
    def _save_plot_selection(self, selection_text: str):
        """Saves the selected plot to settings."""
        self.settings.setValue("plot/plot_selection", selection_text)

    @Slot(str)
    def _save_cycle_selection(self, selection_text: str):
        """Saves the selected cycle filter to settings."""
        self.settings.setValue("plot/cycle_selection", selection_text)

    @Slot(float)
    def _save_known_length(self, length: float):
        """Saves the known_length to settings."""
        self.settings.setValue("scale/known_length", length)

    @Slot(bool)
    def _save_scale_is_manual(self, is_manual: bool):
        """Saves the manual mode state to settings."""
        self.settings.setValue("scale/is_manual", is_manual)

    @Slot(float)
    def _save_manual_conversion_factor(self, factor: float):
        """Saves the manual conversion factor to settings."""
        self.settings.setValue("scale/manual_factor", factor)

