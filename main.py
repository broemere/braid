import multiprocessing
import sys


def run_application():
    """Import and start the GUI only in the primary application process."""
    from window import MainWindow
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPalette, QPixmap, QIcon
    from PySide6.QtWidgets import QApplication, QSplashScreen
    from config import APP_NAME, APP_VERSION, ORG
    from processing.resource_loader import setup_logging, load_icon

    log = setup_logging()
    log.info("Application starting...")

    app = QApplication(sys.argv)
    app.setOrganizationName(ORG)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon(load_icon())) # Set program Icon
    app.setStyle('Fusion')

    # Splash screen
    splash_pix = QPixmap(400, 200)
    splash_pix.fill(app.palette().color(QPalette.Window))
    splash = QSplashScreen(splash_pix)
    splash.showMessage(f"{APP_NAME} Loading...\n\n v{APP_VERSION}", Qt.AlignCenter | Qt.AlignCenter, app.palette().color(QPalette.Text))
    splash.show()

    app.processEvents()
    win = MainWindow()
    win.show()
    splash.finish(win)    # Close splash when ready

    return app.exec()


def bootstrap(application_runner=None):
    """Route frozen child processes before any GUI imports or initialization."""
    multiprocessing.freeze_support()
    runner = application_runner or run_application
    return runner()


if __name__ == '__main__':
    sys.exit(bootstrap())
