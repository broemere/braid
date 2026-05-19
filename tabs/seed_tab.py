from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QWidget, QBoxLayout, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QButtonGroup, QRadioButton
from widgets.seed_widget import SeedDrawingLabel
from widgets.error_bus import user_error  # Adjust path if necessary based on your file structure

class SeedTab(QWidget):
    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.editors = []  # Stores our two SeedDrawingLabel instances
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Toolbar (Tools) ---
        tool_layout = QHBoxLayout()

        # Radio Buttons for Tool Selection
        self.btn_group = QButtonGroup(self)
        self.rb_rect = QRadioButton("Rectangle")
        self.rb_ellipse = QRadioButton("Ellipse")
        self.rb_rect.setChecked(True)

        self.btn_group.addButton(self.rb_rect)
        self.btn_group.addButton(self.rb_ellipse)

        tool_layout.addWidget(QLabel("Draw Tool:"))
        tool_layout.addWidget(self.rb_rect)
        tool_layout.addWidget(self.rb_ellipse)
        tool_layout.addStretch()

        tool_layout.addStretch()

        # Flip Layout Button
        self.btn_flip = QPushButton("Flip Layout")
        self.btn_flip.clicked.connect(self.toggle_layout)
        tool_layout.addWidget(self.btn_flip)

        main_layout.addLayout(tool_layout)

        # --- Image Editors Row ---
        #editors_layout = QHBoxLayout()
        self.editors_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom)

        # We need 4 editors
        for i in range(4):
            v_box = QVBoxLayout()

            # Custom Editor Widget
            editor = SeedDrawingLabel()
            # Needed for Ctrl+Z to work on hover
            editor.setFocusPolicy(Qt.StrongFocus)
            self.editors.append(editor)
            v_box.addWidget(editor)

            # Undo Button
            btn_undo = QPushButton("Undo / Clear")
            # Connect using closure to capture index
            btn_undo.clicked.connect(lambda checked=False, e=editor: e.undo())
            v_box.addWidget(btn_undo)

            self.editors_layout.addLayout(v_box, stretch=1)

        main_layout.addLayout(self.editors_layout, stretch=1)

    def connect_signals(self):
        # 1. Tool Selection Changes
        self.btn_group.buttonToggled.connect(self._on_tool_change)

        # 2. Pipeline -> UI (Receive Images)
        # Assuming pipeline emits list of QPixmaps via `images_ready` or similar
        self.pipeline.cropped_images_ready.connect(self.update_displays)

        # 3. UI -> Pipeline (Send Shape Data)
        for i, editor in enumerate(self.editors):
            # Use lambda to pass the image index 'i' along with the data
            editor.shape_drawn.connect(
                lambda s_type, data, idx=i: self._on_shape_drawn(idx, s_type, data)
            )

    @Slot()
    def toggle_layout(self):
        """Swaps the editor layout direction between horizontal and vertical."""
        current_dir = self.editors_layout.direction()

        if current_dir == QBoxLayout.Direction.LeftToRight:
            self.editors_layout.setDirection(QBoxLayout.Direction.TopToBottom)
            self.pipeline.layout = "vertical"
        else:
            self.editors_layout.setDirection(QBoxLayout.Direction.LeftToRight)
            self.pipeline.layout = "horizontal"

    def _on_tool_change(self):
        tool = 'rect' if self.rb_rect.isChecked() else 'ellipse'
        for editor in self.editors:
            editor.set_tool(tool)

    def _on_shape_drawn(self, index, shape_type, data):
        """Passes the drawn shape data back to the pipeline after validating pairs."""
        print(f"Shape drawn on Image {index}: {shape_type}, {data}")

        # --- Validation Logic ---
        # We only validate if the user actually drew a shape (shape_type is not None)
        # and if we have all 4 editors initialized.
        if shape_type is not None and len(self.editors) == 4:

            # Fetch current shapes directly from the widgets
            shape0 = self.editors[0].drawn_shape_type
            shape1 = self.editors[1].drawn_shape_type
            shape2 = self.editors[2].drawn_shape_type
            shape3 = self.editors[3].drawn_shape_type

            # Check Pair 1 (Index 0 and 2 / Editors 1 and 3)
            if index in (0, 2) and shape0 and shape2 and shape0 != shape2:
                user_error(
                    "Invalid Seed Shape Match",
                    f"Editor 1 and Editor 3 must use the same shape type.\n\n"
                    f"Editor 1 has: '{shape0}'\n"
                    f"Editor 3 has: '{shape2}'\n\n"
                    "Your last drawing has been undone."
                )
                self.editors[index].undo()  # Reject the bad draw
                return  # Do not pass data to pipeline

            # Check Pair 2 (Index 1 and 3 / Editors 2 and 4)
            if index in (1, 3) and shape1 and shape3 and shape1 != shape3:
                user_error(
                    "Invalid Seed Shape Match",
                    f"Editor 2 and Editor 4 must use the same shape type.\n\n"
                    f"Editor 2 has: '{shape1}'\n"
                    f"Editor 4 has: '{shape3}'\n\n"
                    "Your last drawing has been undone."
                )
                self.editors[index].undo()  # Reject the bad draw
                return  # Do not pass data to pipeline

        # If validation passes (or if it was an undo action), send to pipeline
        self.pipeline.receive_seed_shape(index, shape_type, data)

    @Slot(list)
    def update_displays(self, pixmaps):
        """
        Receives [QPixmap, QPixmap, QPixmap, QPixmap]
        corresponding to MinROI1, MinROI2, MaxROI1, MaxROI2.
        """
        for i, editor in enumerate(self.editors):
            if i < len(pixmaps):
                editor.set_pixmap(pixmaps[i])
                # Enable the editor now that it has an image
                editor.setEnabled(True)
            else:
                # Clear editor if we have fewer crops than slots
                editor.set_pixmap(None)
                editor.setEnabled(False)  # Disable if empty