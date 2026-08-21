import multiprocessing
import sys


INSTANCE_SERVER_NAME = "TykockiLab.BRAID.Application"


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


def build_launch_request(open_path):
    """Describe what this invocation wants the primary process to do."""

    if open_path:
        return {"action": "open", "path": open_path}
    return {"action": "new_window"}


def run_application():
    """Import and start the GUI only in the primary application process."""
    from PySide6.QtWidgets import QApplication
    from config import APP_NAME, APP_VERSION, ORG
    from instance_router import LocalApplicationRouter

    open_path, qt_arguments = extract_open_request(sys.argv)
    launch_request = build_launch_request(open_path)
    app = QApplication(qt_arguments)
    app.setOrganizationName(ORG)
    app.setApplicationName(APP_NAME)

    if LocalApplicationRouter.forward_request(
            INSTANCE_SERVER_NAME, launch_request
    ):
        return 0

    router = LocalApplicationRouter(INSTANCE_SERVER_NAME, app)
    router_available = router.listen()
    if not router_available:
        # Another process may have won the startup race after our first probe.
        if LocalApplicationRouter.forward_request(
                INSTANCE_SERVER_NAME, launch_request, timeout_ms=1500
        ):
            return 0
        # Unix-domain socket files can survive an unclean exit. Windows named
        # pipes are released by the OS, so removal is neither needed nor safe.
        router_available = router.listen(remove_stale_endpoint=True)

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QPalette, QPixmap, QIcon
    from PySide6.QtWidgets import QSplashScreen
    from application_controller import ApplicationWindowController
    from processing.resource_loader import setup_logging, load_icon
    from window import MainWindow

    log = setup_logging()
    log.info("Application starting...")
    if not router_available:
        log.warning(
            "Local application routing is unavailable; this window will run "
            "without launch forwarding."
        )

    app.setWindowIcon(QIcon(load_icon())) # Set program Icon
    app.setStyle('Fusion')

    controller = ApplicationWindowController(MainWindow)
    pending_requests = []
    active_request_handler = [None]

    def dispatch_request(request):
        handler = active_request_handler[0]
        if handler is None:
            pending_requests.append(request)
        else:
            handler(request)

    if router_available:
        # Requests can arrive while the splash screen is being painted. Queue
        # them until the initial window and its blank analysis tab exist.
        router.request_received.connect(dispatch_request)

    # Splash screen
    splash_pix = QPixmap(400, 200)
    splash_pix.fill(app.palette().color(QPalette.Window))
    splash = QSplashScreen(splash_pix)
    splash.showMessage(f"{APP_NAME} Loading...\n\n v{APP_VERSION}", Qt.AlignCenter | Qt.AlignCenter, app.palette().color(QPalette.Text))
    splash.show()

    app.processEvents()
    win = controller.create_window()
    splash.finish(win)    # Close splash when ready

    def finish_startup():
        if open_path:
            win.open_video_path(open_path)
        active_request_handler[0] = controller.handle_request
        queued_requests = list(pending_requests)
        pending_requests.clear()
        for request in queued_requests:
            controller.handle_request(request)

    QTimer.singleShot(0, finish_startup)

    return app.exec()


def bootstrap(application_runner=None):
    """Route frozen child processes before any GUI imports or initialization."""
    multiprocessing.freeze_support()
    runner = application_runner or run_application
    return runner()


if __name__ == '__main__':
    sys.exit(bootstrap())
