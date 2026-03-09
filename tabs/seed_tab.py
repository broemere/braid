from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QButtonGroup, QRadioButton
from widgets.seed_widget import SeedDrawingLabel


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

        main_layout.addLayout(tool_layout)

        # --- Image Editors Row ---
        editors_layout = QHBoxLayout()

        # We need 2 editors
        for i in range(4):
            v_box = QVBoxLayout()

            # Label Title
            lbl_title = QLabel(f"Image {i + 1} Seed")
            lbl_title.setAlignment(Qt.AlignCenter)
            v_box.addWidget(lbl_title)

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

            editors_layout.addLayout(v_box, stretch=1)

        main_layout.addLayout(editors_layout, stretch=1)

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

    def _on_tool_change(self):
        tool = 'rect' if self.rb_rect.isChecked() else 'ellipse'
        for editor in self.editors:
            editor.set_tool(tool)

    def _on_shape_drawn(self, index, shape_type, data):
        """Passes the drawn shape data back to the pipeline."""
        print(f"Shape drawn on Image {index}: {shape_type}, {data}")
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