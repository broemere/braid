import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot
#from processing.data_transform import zero_data, smooth_data, label_image, create_visual_from_labels, convert_numpy, restore_numpy, n_closest_numbers
from processing.data_loader import frame_loader, geometry_worker
from processing.data_transform import auto_thresh
from config import APP_VERSION
from widgets.error_bus import user_error
import cv2
from processing.data_transform import numpy_to_qpixmap

import numpy as np
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
from skimage.draw import disk, ellipse, rectangle
from scipy.signal import savgol_filter


log = logging.getLogger(__name__)

KERNEL_SIZE = 7
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

class DataPipeline(QObject):
    # --- SIGNALS ---
    data_available = Signal(dict)

    # Plot inputs
    plot_selection_changed = Signal(str)
    cycle_selection_changed = Signal(str)

    # Scale inputs
    known_length_changed = Signal(float)
    pixel_length_changed = Signal(float)
    scale_is_manual_changed = Signal(bool)
    manual_conversion_factor_changed = Signal(float)
    conversion_factor_changed = Signal(float)

    # Frames
    left_image_changed = Signal(np.ndarray)
    roi_min_image_loaded = Signal(np.ndarray)
    roi_max_image_loaded = Signal(np.ndarray)

    rois_ready = Signal(list)
    cropped_images_ready = Signal(list)
    seed_masks_ready = Signal(dict)
    threshed_images_ready = Signal(list)

    dimension_images_ready = Signal(np.ndarray, np.ndarray)

    geometry_available = Signal(dict)
    mechanics_available = Signal(dict)


    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_manager = None
        self.VERSION = APP_VERSION
        self.author = None
        self.video_formats = [".tiff", ".tif", ".avi", ".mkv"]

        self.platform = 'mac' if sys.platform == 'darwin' else 'win'
        self.ctrl_key = "Cmd" if self.platform == "mac" else "Ctrl"

        self.video = None
        self.frame_count = 0
        self.start = 0
        self.frame_count = 0

        # PLOT TAB
        self.plot_selection: str = "Time vs. Force"
        self.cycle_selection: str = "All Cycles"

        # SCALE TAB
        self.known_length = 0.0
        self.pixel_length = 0.0
        self.scale_is_manual = False
        self.manual_conversion_factor = 0.0
        self.conversion_factor = 0.0  # The final, authoritative value

        self.left_image: np.ndarray | None = None

        self.data = None
        self.frame_data = {}
        self.roi_data = {
            'min': [],
            'max': []
        }

        self.mu = 0
        self.lambda1 = 0
        self.gamma = 0

        self.width_roi_idx = 0
        self.thickness_roi_idx = 1


    def on_author_changed(self, new_author):
        self.author = new_author
        log.info(f"User: {self.author}")

    def load_video_file(self, file_path: str, index=None):
        self.video = file_path
        if index is None:
            index = self.start
        self.task_manager.queue_task(
            frame_loader,  # function
            self.video,  # file_path
            [index],  # frame_indices
            True,  # return frame_count
            on_result=self.initial_frame_loaded # Optional: a method in DataPipeline to handle the result
        )

    def initial_frame_loaded(self, result: dict):
        """
        Callback function executed when the frame_loader task is complete.
        'result' is the dictionary of NumPy arrays returned by frame_loader.
        """
        log.info(f"Received {len(result)-2} loaded frames from worker.")
        if not result:
            log.warning("Frame loader returned no frames.")
            return
        it = iter(result)
        first_frame_index = next(it)
        self.left_image = result[first_frame_index]
        if self.left_image is None:
            log.error("Did not receive left frame")
        self.left_image_changed.emit(self.left_image)
        self.frame_count = next(it)
        log.info(f"Frame count found: {self.frame_count}")
        log.info(f"Data keys: {result['data'].keys()}")
        log.info(f"Data found: {result['data']}")
        self.data = result["data"]
        self.data_available.emit(self.data)
        self.max_distance_index = np.argmax(self.data["distance"])
        self.min_distance_index = np.argmin(self.data["distance"])
        self.load_frames([self.max_distance_index, self.min_distance_index])

    def load_frames(self, index):
        self.task_manager.queue_task(
            frame_loader,  # function
            self.video,  # file_path
            index,  # frame_indices
            False,  # return frame_count
            on_result=self.frame_loaded # Optional: a method in DataPipeline to handle the result
        )

    def frame_loaded(self, result):
        for k, v in result.items():
            self.frame_data[k] = v
            if k == self.max_distance_index:
                print("loaded max roi frame")
                self.roi_max_image_loaded.emit(v)
            if k == self.min_distance_index:
                print('loaded min roi frame')
                self.roi_min_image_loaded.emit(v)


    ### region PLOT TAB

    @Slot(str)
    def set_plot_selection(self, selection_text: str):
        """Sets the active plot selection (e.g., 'Time vs. Force')"""
        if self.plot_selection != selection_text:
            self.plot_selection = selection_text
            self.plot_selection_changed.emit(self.plot_selection)

    @Slot(str)
    def set_cycle_selection(self, selection_text: str):
        """Sets the active cycle filter (e.g., 'All Cycles')"""
        if self.cycle_selection != selection_text:
            self.cycle_selection = selection_text
            self.cycle_selection_changed.emit(self.cycle_selection)

    # endregion

    ### region SCALE TAB

    def _recalculate_conversion_factor(self):
        """Central calculation. Called whenever an input changes."""
        new_factor = 0.0
        if self.scale_is_manual:
            new_factor = self.manual_conversion_factor
        elif self.known_length > 0 and self.pixel_length > 0:
            new_factor = self.pixel_length / self.known_length

        # Use the main setter to update the value and emit the signal
        self.set_conversion_factor(new_factor, force_update=True)

    def set_known_length(self, length: float):
        if self.known_length != length:
            self.known_length = length
            self.known_length_changed.emit(self.known_length)
            self._recalculate_conversion_factor()

    def set_pixel_length(self, length: float):
        if self.pixel_length != length:
            self.pixel_length = length
            self.pixel_length_changed.emit(self.pixel_length)
            self._recalculate_conversion_factor()

    def set_scale_is_manual(self, is_manual: bool):
        if self.scale_is_manual != is_manual:
            self.scale_is_manual = is_manual
            self.scale_is_manual_changed.emit(self.scale_is_manual)
            self._recalculate_conversion_factor()

    def set_manual_conversion_factor(self, factor: float):
        """This is called when the user types in the manual spinbox."""
        if self.manual_conversion_factor != factor:
            self.manual_conversion_factor = factor
            # Only recalculate if we are currently in manual mode
            self.manual_conversion_factor_changed.emit(self.manual_conversion_factor)
            if self.scale_is_manual:
                self._recalculate_conversion_factor()

    def set_conversion_factor(self, factor: float, force_update=False):
        """
        This is the final setter for the authoritative value.
        It's called by the recalculate method or can be set directly.
        """
        if self.conversion_factor != factor or force_update:
            self.conversion_factor = factor
            log.info(f"Conversion factor updated: {self.conversion_factor}")
            #self.data_version += 1
            self.conversion_factor_changed.emit(self.conversion_factor)

    # endregion


    ### region ROI TAB

    def _generate_crops_for_target(self, target):
        """Helper: Slices the arrays and stores them in the roi_data dictionary."""
        if not self.frame_data:
            return

        if target == 'min':
            image = self.frame_data.get(self.min_distance_index)  # Get full-size image
        else:
            image = self.frame_data.get(self.max_distance_index)

        if image is None:
            return

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply crops
        for roi_dict in self.roi_data[target]:
            rect = roi_dict["roi_rect"]
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

            if w > 0 and h > 0:
                crop = gray[y:y + h, x:x + w]
                if crop.size > 0:
                    roi_dict["crop_img"] = crop

    def receive_roi_data(self, roi_data, target):
        """
        Receives QRects from ROITab, initializes the nested structure,
        generates crops, and emits QPixmaps for the SeedTab.
        """
        print(f"{len(roi_data)} boxes for {target}: {roi_data}")

        # 1. Reset and populate the list of dictionaries for this target
        self.roi_data[target] = []
        for rect in roi_data:
            self.roi_data[target].append({
                "roi_rect": rect,
                "crop_img": None,
                "seed_shape_type": None,
                "seed_coords": None,
                "seed_mask": None
            })


        self.correlate_rois()

        # 2. Generate crops and store them directly inside the nested dict
        self._generate_crops_for_target(target)

        # 3. Gather all valid crops in order (Min first, then Max) to send to UI
        crops_pixmaps = []
        for t in ['min', 'max']:
            for roi_dict in self.roi_data.get(t, []):
                if roi_dict["crop_img"] is not None:
                    crops_pixmaps.append(numpy_to_qpixmap(roi_dict["crop_img"]))

        # 4. Emit the Pixmaps to the Seed Tab
        if crops_pixmaps:
            self.cropped_images_ready.emit(crops_pixmaps)

    # endregion

    ### region SEED TAB

    def receive_seed_shape(self, index, shape_type, data):
        """
        Called by UI when user finishes drawing. Maps the UI index back
        to the nested dictionary and stores the seed data/mask.
        """
        # 1. Map the flat UI index (0, 1, 2, 3) to target ('min' or 'max') and sub-index
        target = None
        sub_index = 0
        min_len = len(self.roi_data.get('min', []))
        max_len = len(self.roi_data.get('max', []))

        if index < min_len:
            target = 'min'
            sub_index = index
        elif index < min_len + max_len:
            target = 'max'
            sub_index = index - min_len
        else:
            return  # Index out of bounds (user clicked an empty canvas)

        # 2. Grab the specific dictionary for this ROI
        roi_dict = self.roi_data[target][sub_index]
        crop_img = roi_dict.get("crop_img")

        if crop_img is None:
            return

        # 3. Handle clearing the shape (Undo/Reset)
        if shape_type is None or not data:
            roi_dict["seed_shape_type"] = None
            roi_dict["seed_coords"] = None
            roi_dict["seed_mask"] = None
            print(f"Mask cleared for image {index} ({target} ROI {sub_index})")
            return

        # 4. Generate the Mask
        image_shape = crop_img.shape
        mask = np.zeros(image_shape, dtype=bool)

        if shape_type == 'rect':
            start = (data['y'], data['x'])
            end = (data['y'] + data['h'], data['x'] + data['w'])
            rr, cc = rectangle(start=start, end=end, shape=image_shape)
            mask[rr, cc] = True

        elif shape_type == 'ellipse':
            rr, cc = ellipse(data['center_y'], data['center_x'],
                             data['radius_y'], data['radius_x'], shape=image_shape)
            mask[rr, cc] = True

        # 5. Save everything into the unified dictionary
        roi_dict["seed_shape_type"] = shape_type
        roi_dict["seed_coords"] = data
        roi_dict["seed_mask"] = mask
        print(f"Mask created for image {index} ({target} ROI {sub_index}) with shape {shape_type}")

        # 6. Check if we have all required masks to emit
        # (Extracts non-None masks from the nested structure)
        valid_masks = [
            item["seed_mask"]
            for t in ['min', 'max']
            for item in self.roi_data.get(t, [])
            if item.get("seed_mask") is not None
        ]

        if len(valid_masks) == 4:
            self.seed_masks_ready.emit(valid_masks)

    # endregion

    def run_chan_vese(self, img, mu, gamma, lambda1, seed):
        image = img_as_float(img)
        if gamma != 1.0:
            image = image ** gamma

        cv_result = chan_vese(image,
                              mu=mu,
                              lambda1=lambda1,
                              lambda2=1.0,
                              tol=1e-3,
                              max_num_iter=100,
                              dt=0.5,
                              init_level_set=seed,
                              extended_output=True)

        final_mask = cv_result[0]
        # final_energies = cv_result[2]  # To check convergence
        raw_mask = final_mask.astype(np.uint8) * 255
        cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, KERNEL)

        return cleaned_mask

    def apply_threshold(self, smoothness_val: int, shadow_recovery_val: int):
        """
        Applies Chan-Vese segmentation to all valid crops using their paired seed masks.
        """
        processed_pixmaps = []

        # 1. Gather all valid ROI dictionaries that have both a crop and a mask
        valid_rois = []
        for target in ['min', 'max']:
            for roi_dict in self.roi_data.get(target, []):
                crop = roi_dict.get("crop_img")
                mask = roi_dict.get("seed_mask")

                if crop is not None and mask is not None:
                    valid_rois.append(roi_dict)

        # Ensure we have all 4 before doing heavy computation
        if len(valid_rois) != 4:
            print("Waiting for all 4 seed masks to be drawn...")
            return

        # 2. Process each valid ROI
        for roi_dict in valid_rois:
            img = roi_dict["crop_img"]
            seed_mask = roi_dict["seed_mask"]

            # Convert to float for Chan-Vese (skimage handles the 0.0-1.0 mapping)

            mu_norm = (smoothness_val - 1) / 99.0
            mu = 0.001 + (mu_norm ** 2) * (0.2 - 0.001)

            # 2. Calculate Coupled Shadow Parameters (Linear scale)
            # shadow_recovery_val is 0 to 100
            shadow_norm = shadow_recovery_val / 100.0

            # Map 0->100 to Gamma 1.0->0.5
            gamma_val = 1.0 - (shadow_norm * 0.5)

            # Map 0->100 to Lambda1 1.0->0.2
            lambda1_val = 1.0 - (shadow_norm * 0.8)

            cleaned_mask = self.run_chan_vese(img, mu, gamma_val, lambda1_val, seed_mask)

            # --- OVERLAY LOGIC ---
            # 1. Ensure the original crop is 3-channel (BGR) so we can apply color
            if len(img.shape) == 2:
                base_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                base_bgr = img.copy()

            # 2. Define the overlay color and transparency level
            overlay_color = (0, 255, 0)  # BGR format: Green (Change to 0,0,255 for Red)
            alpha = 0.4  # Transparency (0.0 is invisible, 1.0 is a solid block of color)

            # 3. Create a copy of the base image and paint the mask area solid green
            color_layer = base_bgr.copy()
            color_layer[cleaned_mask == 255] = overlay_color

            # 4. Blend the solid color layer with the original image
            blended_output = cv2.addWeighted(color_layer, alpha, base_bgr, 1 - alpha, 0)

            # 5. Convert to QPixmap (your helper already handles BGR -> RGB conversion)
            pix = numpy_to_qpixmap(blended_output)
            processed_pixmaps.append(pix)

            roi_dict["cv_img"] = cleaned_mask

        self.mu = mu
        self.lambda1 = lambda1_val
        self.gamma_val = gamma_val



            # 5. Emit back to the UI
        self.threshed_images_ready.emit(processed_pixmaps)

    def request_dimension_images(self):
        """
        Called by the GeometryTab when it opens or when dimensions are swapped.
        Fetches the 'min' state crops based on current index tracking and emits them.
        """
        min_rois = self.roi_data.get('min', [])
        if len(min_rois) >= 2:
            w_img = min_rois[self.width_roi_idx].get('cv_img')
            t_img = min_rois[self.thickness_roi_idx].get('cv_img')

            if w_img is not None and t_img is not None:
                self.dimension_images_ready.emit(w_img, t_img)

    def swap_dimensions(self):
        """
        Flips the state mapping for Width and Thickness, then updates the UI.
        """
        # Swap the tracking indices
        self.width_roi_idx, self.thickness_roi_idx = self.thickness_roi_idx, self.width_roi_idx
        print(f"Swapped Dimensions! Width is now ROI {self.width_roi_idx}, Thickness is ROI {self.thickness_roi_idx}")

        # Emit the new images to immediately update the visual reference in the UI
        self.request_dimension_images()
        if hasattr(self, 'first_segments'):
            self.calculate_dimensions()

        # NOTE: If we already have computed segmentation masks stored,
        # we will eventually call our math recalculation function right here!

    def correlate_rois(self):
        """
        Pairs the 'min' ROIs with the correct 'max' ROIs based on spatial overlap.
        Reorders the 'max' list in self.roi_data to perfectly match the 'min' list.
        """
        min_data = self.roi_data.get('min', [])
        max_data = self.roi_data.get('max', [])

        # We need exactly 2 ROIs in both states to perform a correlation
        if len(min_data) != 2 or len(max_data) != 2:
            print("Cannot correlate: Need exactly 2 ROIs in both min and max states.")
            return

        m0_rect = min_data[0]["roi_rect"]
        m1_rect = min_data[1]["roi_rect"]

        x0_rect = max_data[0]["roi_rect"]
        x1_rect = max_data[1]["roi_rect"]

        def get_overlap_area(rect1, rect2):
            """Helper to calculate the overlapping area of two QRects."""
            intersection = rect1.intersected(rect2)
            if intersection.isValid():
                return intersection.width() * intersection.height()
            return 0

        # --- Test Configuration A ---
        # Min 0 pairs with Max 0
        # Min 1 pairs with Max 1
        score_a = get_overlap_area(m0_rect, x0_rect) + get_overlap_area(m1_rect, x1_rect)

        # --- Test Configuration B ---
        # Min 0 pairs with Max 1
        # Min 1 pairs with Max 0
        score_b = get_overlap_area(m0_rect, x1_rect) + get_overlap_area(m1_rect, x0_rect)

        # --- Apply the Best Match ---
        if score_a >= score_b:
            print("Correlation: Min 0 -> Max 0, Min 1 -> Max 1")
            # They are already in the correct order, do nothing.
        else:
            print("Correlation: Min 0 -> Max 1, Min 1 -> Max 0")
            # Swap the max list so the indices match the min list
            self.roi_data['max'] = [max_data[1], max_data[0]]

    def get_interpolated_data(self, pct: float) -> list[dict]:
        """
        Calculates the interpolated ROI bounding boxes and seed shapes between min and max states.
        Returns a list of dictionaries structured identically to the base roi_data items.
        """
        if not self.roi_data.get("min") or not self.roi_data.get("max"):
            return []

        def lerp_int(start: float, end: float, pct: float) -> int:
            """Standard linear interpolation, correctly rounded to the nearest integer pixel."""
            return int(round(start + (end - start) * pct))

        def lerp_rect(r_min: QRect, r_max: QRect, pct: float) -> QRect:
            """Interpolates all four dimensions of a QRect."""
            x = lerp_int(r_min.x(), r_max.x(), pct)
            y = lerp_int(r_min.y(), r_max.y(), pct)
            w = lerp_int(r_min.width(), r_max.width(), pct)
            h = lerp_int(r_min.height(), r_max.height(), pct)
            return QRect(x, y, w, h)

        def lerp_seed(min_dict: dict, max_dict: dict, pct: float) -> tuple:
            """Interpolates the seed coordinates if the shape types match."""
            shape_type = min_dict.get("seed_shape_type")

            # Enforce shape matching (cannot interpolate a rect into an ellipse)
            if not shape_type or shape_type != max_dict.get("seed_shape_type"):
                return None, None

            c_min = min_dict.get("seed_coords")
            c_max = max_dict.get("seed_coords")

            if not c_min or not c_max:
                return None, None

            interp_coords = {}

            if shape_type == 'rect':
                interp_coords['x'] = lerp_int(c_min['x'], c_max['x'], pct)
                interp_coords['y'] = lerp_int(c_min['y'], c_max['y'], pct)
                interp_coords['w'] = lerp_int(c_min['w'], c_max['w'], pct)
                interp_coords['h'] = lerp_int(c_min['h'], c_max['h'], pct)

            elif shape_type == 'ellipse':
                interp_coords['center_x'] = lerp_int(c_min['center_x'], c_max['center_x'], pct)
                interp_coords['center_y'] = lerp_int(c_min['center_y'], c_max['center_y'], pct)
                interp_coords['radius_x'] = lerp_int(c_min['radius_x'], c_max['radius_x'], pct)
                interp_coords['radius_y'] = lerp_int(c_min['radius_y'], c_max['radius_y'], pct)

            return shape_type, interp_coords

        interpolated_rois = []

        # Iterate through the paired ROIs (assuming correlate_rois() was already run)
        for i in range(2):
            min_roi = self.roi_data["min"][i]
            max_roi = self.roi_data["max"][i]

            # 1. Interpolate the global bounding box
            interp_rect = lerp_rect(min_roi["roi_rect"], max_roi["roi_rect"], pct)

            # 2. Interpolate the local seed shape
            shape_type, interp_coords = lerp_seed(min_roi, max_roi, pct)

            # 3. Package it into the standard dictionary format
            interpolated_rois.append({
                "roi_rect": interp_rect,
                "seed_shape_type": shape_type,
                "seed_coords": interp_coords,
                "crop_img": None,  # To be filled when the specific frame is loaded
                "seed_mask": None  # To be generated from the coords
            })

        return interpolated_rois

    def get_geometry(self):
        """
        Packages current state data and kicks off the background calculation task.
        """
        if not self.video or self.data is None or "distance" not in self.data:
            print("Cannot calculate geometry: Data or video not loaded.")
            return

        if len(self.roi_data.get('min', [])) != 2 or len(self.roi_data.get('max', [])) != 2:
            print("Cannot calculate geometry: ROIs not fully established.")
            return

        distances = self.data["distance"]

        # Package a "snapshot" so the thread doesn't read live UI variables
        snapshot_config = {
            'file_path': self.video,
            'distances': distances,
            'min_dist': distances[self.min_distance_index],
            'max_dist': distances[self.max_distance_index],
            'roi_data': self.roi_data,  # QRects and dicts are safe to pass by reference if we only read them
            'conversion_factor': self.conversion_factor,
            'mu': getattr(self, 'mu', 0.05),  # Fallbacks in case apply_threshold wasn't run
            'gamma': getattr(self, 'gamma_val', 1.0),
            'lambda1': getattr(self, 'lambda1', 1.0)
        }

        # Queue the heavy-lifting task
        self.task_manager.queue_task(
            geometry_worker,  # The pure worker function
            snapshot_config,  # The configuration dict
            on_result=self._on_geometry_computed
        )

    def _on_geometry_computed(self, result: dict):
        """
        Callback triggered when the background task finishes successfully.
        """
        print("Geometry computed successfully. Calculating dimensions...")
        self.first_segments = result["first_masks"]
        self.second_segments = result["second_masks"]
        self.calculate_dimensions()

    def calculate_dimensions(self):
        """
        Calculates Width, Thickness, and Area from the stored segmentation masks.
        Respects the current state of width_roi_idx and thickness_roi_idx.
        Emits the final data payload to the GeometryTab.
        """
        if not hasattr(self, 'first_segments') or not hasattr(self, 'second_segments'):
            print("No segmentation data available to calculate dimensions.")
            return

        print("Calculating dimensions from stored masks...")

        frames = []
        widths = []
        thicknesses = []
        areas = []
        raw_widths = []
        raw_thicknesses = []

        # 1. Determine which mask list represents Width and which represents Thickness
        # self.width_roi_idx is either 0 or 1.
        if self.width_roi_idx == 0:
            width_masks_list = self.first_segments
            thickness_masks_list = self.second_segments
        else:
            width_masks_list = self.second_segments
            thickness_masks_list = self.first_segments

        num_frames = len(width_masks_list)

        # Ensure conversion factor is valid (fallback to 1.0 if not set)
        cf = self.conversion_factor if self.conversion_factor > 0 else 1.0

        for i in range(num_frames):
            frames.append(i)

            # --- Width Calculation ---
            w_mask = width_masks_list[i]['mask']
            if w_mask.size > 0:
                row_sums = np.sum(w_mask, axis=1)
                valid_rows = row_sums[row_sums > 0]
                raw_widths.append(np.mean(valid_rows) if len(valid_rows) > 0 else 0)
            else:
                raw_widths.append(0)

            # --- Thickness Calculation ---
            t_mask = thickness_masks_list[i]['mask']
            if t_mask.size > 0:
                col_sums = np.sum(t_mask, axis=0)
                valid_cols = col_sums[col_sums > 0]
                raw_thicknesses.append(np.mean(valid_cols) if len(valid_cols) > 0 else 0)
            else:
                raw_thicknesses.append(0)

            # 2. Outside the loop: Convert lists to 1D arrays and apply scaling
        w_mm = np.array(raw_widths) / cf
        t_mm = np.array(raw_thicknesses) / cf

        # 1. Calculate the Raw Stadium Area FIRST (maintains synchronous frame data)
        raw_areas = (np.pi * (t_mm / 2) ** 2) + ((w_mm - t_mm) * t_mm)

        # 2. Setup Smoothing Parameters
        calc_window = int(len(frames) * 0.05)
        window_length = max(5, calc_window if calc_window % 2 != 0 else calc_window + 1)

        # 3. Apply Smoothing to all three arrays independently
        if window_length > 3 and len(frames) >= window_length:
            smooth_w = savgol_filter(w_mm, window_length=window_length, polyorder=3)
            smooth_t = savgol_filter(t_mm, window_length=window_length, polyorder=3)
            smooth_area = savgol_filter(raw_areas, window_length=window_length, polyorder=3)
        else:
            smooth_w, smooth_t, smooth_area = w_mm, t_mm, raw_areas

        # 4. Package and Emit
        result = {
            'frames': frames,
            'width': smooth_w.tolist(),
            'thickness': smooth_t.tolist(),
            'area': smooth_area.tolist()
        }

        #
        # # 2. Iterate through all frames and calculate metrics
        # for i in range(num_frames):
        #     frames.append(i)
        #
        #     # --- Width Calculation (Horizontal Average) ---
        #     w_dict = width_masks_list[i]
        #     w_mask = w_dict['mask']
        #
        #     if w_mask.size > 0:
        #         # Sum across columns (axis=1) to count True pixels in each row
        #         row_sums = np.sum(w_mask, axis=1)
        #         valid_rows = row_sums[row_sums > 0]
        #         avg_width_px = np.mean(valid_rows) if len(valid_rows) > 0 else 0
        #     else:
        #         avg_width_px = 0
        #
        #     # --- Thickness Calculation (Vertical Average) ---
        #     t_dict = thickness_masks_list[i]
        #     t_mask = t_dict['mask']
        #
        #     if t_mask.size > 0:
        #         # Sum across rows (axis=0) to count True pixels in each column
        #         col_sums = np.sum(t_mask, axis=0)
        #         valid_cols = col_sums[col_sums > 0]
        #         avg_thickness_px = np.mean(valid_cols) if len(valid_cols) > 0 else 0
        #     else:
        #         avg_thickness_px = 0
        #
        #     # 3. Apply Pixel-to-MM Conversion and Calculate Area
        #     w_mm = np.array(avg_width_px) / cf
        #     t_mm = np.array(avg_thickness_px) / cf
        #     #area = w_mm * t_mm
        #
        #     # area = ( np.pi * (t_mm/2)**2 ) + ( (w_mm - t_mm)*t_mm)  # Stadium shape
        #     #
        #     # widths.append(w_mm)
        #     # thicknesses.append(t_mm)
        #     # areas.append(area)
        #
        #     # 2. Apply Savitzky-Golay Smoothing
        #     # window_length must be odd. 11 to 21 is usually great for 400-500 frames.
        #     # polyorder of 3 allows the curve to flex into peaks perfectly.
        #     window_length = min(21, len(frames) - (len(frames) % 2 == 0))  # Ensure odd number <= frame count
        #     if window_length > 3:
        #         smooth_w = savgol_filter(w_mm, window_length=window_length, polyorder=3)
        #         smooth_t = savgol_filter(t_mm, window_length=window_length, polyorder=3)
        #     else:
        #         smooth_w, smooth_t = w_mm, t_mm  # Fallback if too few frames
        #
        #     # 3. Calculate Stadium Area using the SMOOTHED arrays
        #     # area = (pi * r^2) + (rectangle_width * thickness)
        #     areas = (np.pi * (smooth_t / 2) ** 2) + ((smooth_w - smooth_t) * smooth_t)
        #
        # # 4. Package and Emit
        # result = {
        #     'frames': frames,
        #     'width': smooth_w.tolist(),
        #     'thickness': smooth_t.tolist(),
        #     'area': areas.tolist()
        # }

        # # 4. Package and Emit
        # result = {
        #     'frames': frames,
        #     'width': widths,
        #     'thickness': thicknesses,
        #     'area': areas
        # }

        self.geometry_available.emit(result)
        print("Dimensions calculated and emitted to UI.")
        # Save the payload so the mechanics function can use it
        self.geometry_data = result
        self.calculate_mechanics()

    def calculate_mechanics(self):
        """
        Derives True Stress, Strain, Poisson's Ratio, and Energy Dissipation
        by syncing the optical geometry data with the mechanical load cell data.
        """
        print("Calculating biomechanics...")
        if getattr(self, 'geometry_data', None) is None or self.data is None:
            print("Cannot calculate mechanics: Missing geometry or machine data.")
            return

        # 1. Gather Arrays
        geom = self.geometry_data
        width = np.array(geom['width'])
        thickness = np.array(geom['thickness'])
        area = np.array(geom['area'])

        force_mN = -np.array(self.data.get('force', []))
        distance = np.array(self.data.get('distance', []))
        cycle_flags = np.array(self.data.get('cycle', []))
        time_s = np.array(self.data.get('time_s', []))

        # Failsafe: Ensure arrays are perfectly aligned in length
        min_len = min(len(width), len(force_mN))
        if min_len == 0:
            return

        width, thickness, area = width[:min_len], thickness[:min_len], area[:min_len]
        force_mN, distance = force_mN[:min_len], distance[:min_len]
        cycle_flags, time_s = cycle_flags[:min_len], time_s[:min_len]

        # 2. Establish Rest State (t=0 Baselines)
        # Using the average of the first 3 frames to prevent a single noisy pixel from shifting the whole test
        W0 = np.mean(width[:3])
        T0 = np.mean(thickness[:3])
        A0 = np.mean(area[:3])

        # 3. Continuous Array Math
        # True Stress (kPa = mN / mm^2)
        true_stress_kpa = force_mN / area

        # Engineering Stress (kPa)
        eng_stress_kpa = force_mN / A0

        # Stretch Ratios (Lambda)
        stretch_w = width / W0
        stretch_t = thickness / T0

        # Logarithmic (True) Strain (epsilon = ln(lambda))
        # This replaces the old (stretch - 1.0) engineering strain
        true_strain_w = np.log(stretch_w)
        true_strain_t = np.log(stretch_t)

        # Engineering Strain (e = lambda - 1) - Kept for reference
        eng_strain_w = stretch_w - 1.0

        # Dynamic Poisson's Ratio (v = -e_transverse / e_axial)
        # Using True Strain for finite deformation accuracy
        poissons_ratio = np.where(np.abs(true_strain_w) > 0.005,
                                  -true_strain_t / true_strain_w,
                                  0.5)

        # 4. Parse Cycles & Calculate Hysteresis Energy
        unique_cycles = np.unique(cycle_flags)
        cycle_parsing = {}
        energy_dissipated = []

        for c in unique_cycles:
            # Create a boolean mask for the current cycle
            mask = (cycle_flags == c)
            idx = np.where(mask)[0]

            if len(idx) < 3:
                continue

            c_strain = true_strain_w[idx]
            c_stress = true_stress_kpa[idx]
            c_dist = distance[idx]

            # Find the physical turnaround point (Max stretch distance)
            peak_relative_idx = np.argmax(c_dist)
            peak_global_idx = idx[peak_relative_idx]

            # Calculate Volumetric Energy Dissipation (Area inside the hysteresis loop)
            # np.trapz integrates forwards (loading) and backwards (unloading) automatically
            dissipated_mJ_mm3 = np.trapz(c_stress, c_strain)
            energy_dissipated.append(dissipated_mJ_mm3)

            # Store the indices so the UI can easily color-code Loading vs. Unloading curves
            cycle_parsing[int(c)] = {
                'full_idx': idx.tolist(),
                'peak_idx': int(peak_global_idx),
                'loading_idx': idx[:peak_relative_idx + 1].tolist(),
                'unloading_idx': idx[peak_relative_idx:].tolist()
            }

        # 5. Package and Emit Payload
        mechanics_payload = {
            'time_s': time_s.tolist(),
            'true_stress_kpa': true_stress_kpa.tolist(),
            'eng_stress_kpa': eng_stress_kpa.tolist(),
            'strain_w': true_strain_w.tolist(),
            'strain_t': true_strain_t.tolist(),
            'stretch_w': stretch_w.tolist(),
            'stretch_t': stretch_t.tolist(),
            'poissons_ratio': poissons_ratio.tolist(),
            'energy_dissipated': energy_dissipated,
            'cycle_parsing': cycle_parsing
        }

        self.mechanics_available.emit(mechanics_payload)
        print("Biomechanics calculated and emitted!")