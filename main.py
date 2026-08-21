import multiprocessing
import sys


def extract_open_request(argv):
    """Remove the generic ``--open`` request before Qt parses its arguments."""
    arguments = list(argv)
    if not arguments:
        return None, []
    cleaned = [arguments[0]]
    open_path = None
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--open":
            if index + 1 < len(arguments):
                open_path = arguments[index + 1]
                index += 2
            else:
                index += 1
            continue
        if argument.startswith("--open="):
            open_path = argument.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    return open_path, cleaned


def run_application():
    """Import and start the GUI only in the primary application process."""
    from window import MainWindow
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QPalette, QPixmap, QIcon
    from PySide6.QtWidgets import QApplication, QSplashScreen
    from config import APP_NAME, APP_VERSION, ORG
    from processing.resource_loader import setup_logging, load_icon

    log = setup_logging()
    log.info("Application starting...")

    open_path, qt_arguments = extract_open_request(sys.argv)
    app = QApplication(qt_arguments)
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
    if open_path:
        QTimer.singleShot(0, lambda: win.open_video_path(open_path))

    return app.exec()


def bootstrap(application_runner=None):
    """Route frozen child processes before any GUI imports or initialization."""
    multiprocessing.freeze_support()
    runner = application_runner or run_application
    return runner()


if __name__ == '__main__':
    sys.exit(bootstrap())
