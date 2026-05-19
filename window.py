import logging
from PySide6.QtCore import QSettings, Slot, Qt, QSize
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QMessageBox, QPushButton, QMenu
from config import APP_NAME
from processing.task_manager import TaskManager
from widgets.status_bar import StatusBarWidget
from widgets.analysis_widget import AnalysisWidget
from widgets.circle_widget import make_circle_icon, get_color
from widgets.error_bus import bus


log = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """
    The main application window. It manages the overall UI shell, including
    the 'supertabs' for different analysis sessions, the status bar, and the
    shared TaskManager.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.settings = QSettings()
        self._restore_window()

        # These are the globally shared components
        self.task_manager = TaskManager()
        self.status_bar = StatusBarWidget()

        self.init_ui()
        self.connect_global_signals()

        # Start with a single analysis session by default
        self.add_new_super_tab()
        log.info("MainWindow initialized with one session.")

    def init_ui(self):
        """Initializes the main window's UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        # Remove extra spacing between widgets
        main_layout.setSpacing(0)

        # --- Main QTabWidget for sessions ---
        self.super_tabs = QTabWidget()
        self.super_tabs.setIconSize(QSize(10, 10))
        self.super_tabs.setTabsClosable(True)
        self.super_tabs.setMovable(True)
        self.super_tabs.setTabBarAutoHide(False)
        main_layout.addWidget(self.super_tabs)

        # Menu button (left corner)
        self.menu_button = QPushButton("☰ File")
        self.menu_button.setToolTip("File Menu")
        self.file_menu = QMenu(self)
        self.menu_button.setMenu(self.file_menu)
        self.super_tabs.setCornerWidget(self.menu_button, Qt.TopLeftCorner)

        # Add tab button (right corner)
        self.add_tab_button = QPushButton("+")
        self.add_tab_button.setToolTip("Open a new analysis session")
        self.super_tabs.setCornerWidget(self.add_tab_button, Qt.TopRightCorner)

        # --- Global status bar ---
        main_layout.addWidget(self.status_bar)

    def connect_global_signals(self):
        """Connects signals for globally shared components."""

        # Create menu actions
        new_action = QAction("New Session", self)
        save_action = QAction("Save Session", self)
        load_action = QAction("Load Session", self)
        self.file_menu.addAction(new_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(save_action)
        self.file_menu.addAction(load_action)

        # Connect actions to slots
        new_action.triggered.connect(self.add_new_super_tab)
        save_action.triggered.connect(self.on_save_session)
        load_action.triggered.connect(self.on_load_session)

        # Connect the "add tab" button to its slot
        self.add_tab_button.clicked.connect(self.add_new_super_tab)

        # Connect the close request for a supertab
        self.super_tabs.tabCloseRequested.connect(self.on_super_tab_close_requested)

        # Connect the shared TaskManager to the shared StatusBar
        self.task_manager.status_updated.connect(self.status_bar.update_status)
        self.task_manager.progress_updated.connect(self.status_bar.update_progress)
        self.task_manager.batch_finished.connect(self.status_bar.batch_finished)
        self.task_manager.error_occurred.connect(self.show_error_dialog)

        # Connect the cancel button from the status bar to the task manager
        self.status_bar.cancel_clicked.connect(self.task_manager.cancel_batch)

        # Connect error bus to dialog
        bus.user_error_details.connect(lambda exc, tb: self.show_error_dialog((exc, tb)))

    @Slot()
    def add_new_super_tab(self, unfocus=False):
        """Creates a new AnalysisSessionWidget and adds it as a new 'supertab'."""
        session_widget = AnalysisWidget(self.task_manager, self.settings)
        session_widget.tab_name_requested.connect(self.on_tab_name_change_requested)

        tab_name = f"Analysis {self.super_tabs.count() + 1}"
        index = self.super_tabs.addTab(session_widget, tab_name)
        if not unfocus:
            self.super_tabs.setCurrentIndex(index)
        log.info(f"Added new super tab: '{tab_name}'")
        return index

    @Slot(str)
    def on_tab_name_change_requested(self, new_name: str):
        """
        Sets the tab text for the widget that emitted the signal.
        This slot is connected when a new AnalysisSessionWidget is created.
        """
        sender_widget = self.sender()
        if sender_widget:
            index = self.super_tabs.indexOf(sender_widget)
            if index != -1:
                self.super_tabs.setTabText(index, new_name)
                icon = make_circle_icon(get_color(new_name), diameter=14)  # blue dot example
                self.super_tabs.setTabIcon(index, icon)
                log.info(f"Renamed tab at index {index} to '{new_name}'.")

    @Slot(int)
    def on_super_tab_close_requested(self, index: int):
        """Handles the request to close a 'supertab'."""
        widget_to_close = self.super_tabs.widget(index)
        if widget_to_close:
            # You could add a confirmation dialog here if needed
            # e.g., if self.confirm_close():
            log.info(f"Closing super tab at index {index}.")
            self.super_tabs.removeTab(index)
            widget_to_close.deleteLater()  # Ensure proper memory cleanup
        if self.super_tabs.count() == 0:
            self.add_new_super_tab()

    def closeEvent(self, event):
        """Saves window geometry upon closing the application."""
        self.settings.setValue("windowGeometry", self.saveGeometry())
        log.info("Saving window geometry and closing application.")
        super().closeEvent(event)

    def _restore_window(self):
        """Restores the window's size and position from the last session."""
        geometry = self.settings.value("windowGeometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.resize(1300, 750)  # Fallback to a default size

    @Slot(str)
    def on_update_available(self, new_version):
        """
        Slot called only if the UpdateChecker finds a newer version.
        """
        log.info(f"Update available: {new_version}")

        # Construct a friendly message with a link
        repo_url = "https://github.com/broemere/proper/releases/latest"
        msg = (
            f"A new version of {APP_NAME} is available!<br><br>"
            f"Current version: <b>v{APP_VERSION}</b><br>"
            f"New version: <b>{new_version}</b><br><br>"
            f"Click <a href='{REPO_URL}'>here</a> to view the release page."
        )

        self.show_info_dialog("Update Available", msg)

    @Slot(tuple)
    def show_error_dialog(self, err_tb):
        """Displays a modal dialog for critical errors from background tasks."""
        exc, tb_str = err_tb
        is_user_warning = hasattr(exc, "hint")
        if is_user_warning:
            log.warning(f"Displaying user warning: {exc}")
        else:
            log.error(f"Displaying critical error: {exc}\nTraceback: {tb_str}")
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning if is_user_warning else QMessageBox.Critical)
        title = "Action Required" if is_user_warning else "An Error Occurred"
        msg_box.setWindowTitle(title)
        msg_box.setText(str(exc))
        msg_box.setInformativeText(getattr(exc, "hint", f"Please check '{APP_NAME.lower()}.log' for details."))
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

    def show_info_dialog(self, title: str, message: str):
        """Displays a modal informational dialog with a standard icon and rich text."""
        log.info(f"Displaying info dialog: {title}")
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setTextFormat(Qt.RichText)  # parse HTML
        msg_box.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg_box.setText(f"<b>{title}</b>")
        msg_box.setInformativeText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()
