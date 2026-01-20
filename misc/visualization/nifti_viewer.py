#!/usr/bin/env python3
"""NIfTI 3D Viewer - Multi-modality medical image viewer.

A PyQt5 application for viewing 3D NIfTI files with:
- Auto-detection of modalities (bravo, seg, t1_pre, t1_gd, flair)
- Three orthogonal views (axial, coronal, sagittal)
- 3D volume rendering with PyVista
- Synchronized slice navigation
- Zoom/pan controls for 2D views
- Fullscreen 3D mode with double-click

Usage:
    python misc/nifti_viewer.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Configure module logger
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import nibabel as nib
except ImportError:
    print("Error: nibabel not installed. Run: pip install nibabel")
    sys.exit(1)

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QPushButton, QFileDialog, QSlider,
        QLabel, QGroupBox, QSplitter, QComboBox, QCheckBox
    )
    from PyQt5.QtCore import Qt
except ImportError:
    print("Error: PyQt5 not installed. Run: pip install PyQt5")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkPolyDataMapper, vtkActor
    from vtkmodules.vtkCommonCore import vtkObject
    # Suppress VTK warnings about texture size limitations
    vtkObject.GlobalWarningDisplayOff()
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    vtkRenderer = vtkPolyDataMapper = vtkActor = None  # type: ignore
    print("Warning: pyvista/pyvistaqt not installed. 3D view disabled.")
    print("Install with: pip install pyvista pyvistaqt")


# =============================================================================
# Constants
# =============================================================================

DOUBLE_CLICK_TIMEOUT = 0.3  # Seconds for double-click detection
PERFORMANCE_THRESHOLD = 10_000_000  # Voxels above which to subsample for performance
DEFAULT_SPLITTER_SIZES = [800, 500]  # Default width ratio for main splitter

# Preferred order for displaying modalities
MODALITY_ORDER = ['bravo', 'seg', 't1_gd', 't1_pre', 'flair']

# Window/level presets for different tissue types (min%, max%)
WINDOW_PRESETS = {
    "Full Range": (0, 100),
    "Brain": (0, 40),
    "Soft Tissue": (10, 50),
    "Bone": (50, 100),
    "Tumor": (5, 30),
}

# Color palette for dark theme
COLOR_BG_DARK = "#2b2b2b"
COLOR_BG_DARKER = "#1e1e1e"
COLOR_TEXT_DIM = "#888"
COLOR_TEXT_LIGHT = "#aaa"
COLOR_LABEL_DISABLED = "color: #888; font-size: 12px;"
COLOR_LABEL_BOLD = "color: #aaa; font-weight: bold;"

# Dark theme stylesheet for consistent UI appearance
DARK_THEME = """
    QMainWindow, QWidget {
        background-color: #2b2b2b;
        color: #ffffff;
    }
    QGroupBox {
        border: 1px solid #555;
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 8px;
        color: #ffffff;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
    QSlider::groove:horizontal {
        height: 6px;
        background: #555;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        width: 14px;
        margin: -4px 0;
        background: #888;
        border-radius: 7px;
    }
    QSlider::handle:horizontal:hover {
        background: #aaa;
    }
    QPushButton {
        background-color: #444;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 5px 15px;
        color: #ffffff;
    }
    QPushButton:hover {
        background-color: #555;
    }
    QPushButton:pressed {
        background-color: #333;
    }
    QComboBox {
        background-color: #444;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 3px 10px;
        color: #ffffff;
    }
    QComboBox:hover {
        background-color: #555;
    }
    QComboBox::drop-down {
        border: none;
    }
    QComboBox QAbstractItemView {
        background-color: #444;
        color: #ffffff;
        selection-background-color: #555;
    }
    QLabel {
        color: #ffffff;
    }
    QCheckBox {
        color: #ffffff;
        spacing: 8px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
    }
    QCheckBox::indicator:unchecked {
        background-color: #333;
        border: 2px solid #555;
        border-radius: 3px;
    }
    QCheckBox::indicator:unchecked:hover {
        border: 2px solid #888;
    }
    QCheckBox::indicator:checked {
        background-color: #4a9eff;
        border: 2px solid #4a9eff;
        border-radius: 3px;
    }
    QCheckBox::indicator:checked:hover {
        background-color: #6ab0ff;
        border: 2px solid #6ab0ff;
    }
"""


class SliceCanvas(FigureCanvas):
    """Matplotlib canvas for displaying a single slice."""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        self.fig = Figure(figsize=(4, 4), dpi=100)
        self.fig.patch.set_facecolor(COLOR_BG_DARK)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(COLOR_BG_DARKER)
        self.ax.set_title(title, color='white', fontsize=10)
        self.ax.tick_params(colors='white', labelsize=8)

        super().__init__(self.fig)
        self.setParent(parent)

        self.image_data: Optional[np.ndarray] = None
        self.overlay_data: Optional[np.ndarray] = None
        self.show_overlay: bool = False
        self.crosshair_h: Optional[int] = None  # Horizontal crosshair position
        self.crosshair_v: Optional[int] = None  # Vertical crosshair position
        self.img_plot: Optional[Any] = None
        self.axis: int = 0  # Which axis this canvas displays
        self.click_callback: Optional[Callable[[int, int, int], None]] = None

        # Zoom/pan state
        self._xlim: Optional[tuple] = None
        self._ylim: Optional[tuple] = None
        self._pan_start: Optional[tuple] = None
        self._pan_xlim: Optional[tuple] = None
        self._pan_ylim: Optional[tuple] = None

        # Colorbar
        self.colorbar = None
        self._needs_tight_layout: bool = True  # Track if tight_layout needs to be called

        # Connect mouse events
        self.mpl_connect('button_press_event', self._on_click)
        self.mpl_connect('scroll_event', self._on_scroll)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)

    def set_click_callback(self, callback) -> None:
        """Set callback for click events. Callback receives (axis, x, y)."""
        self.click_callback = callback

    def _on_click(self, event) -> None:
        """Handle mouse click on canvas."""
        if event.inaxes != self.ax or self.image_data is None:
            return

        # Double-click: reset zoom
        if event.dblclick:
            self.reset_zoom()
            return

        # Middle button: start pan
        if event.button == 2:
            self._pan_start = (event.xdata, event.ydata)
            self._pan_xlim = self.ax.get_xlim()
            self._pan_ylim = self.ax.get_ylim()
            return

        # Left click: navigate (existing behavior)
        if event.button == 1 and self.click_callback is not None:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            self.click_callback(self.axis, x, y)

    def _on_scroll(self, event) -> None:
        """Handle mouse scroll for zooming."""
        if event.inaxes != self.ax or self.image_data is None:
            return

        # Zoom factor
        zoom_factor = 1.2 if event.button == 'down' else 1 / 1.2

        # Get current limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Calculate new limits centered on cursor
        xdata, ydata = event.xdata, event.ydata
        new_width = (xlim[1] - xlim[0]) * zoom_factor
        new_height = (ylim[1] - ylim[0]) * zoom_factor

        # Keep cursor position fixed
        rel_x = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rel_y = (ydata - ylim[0]) / (ylim[1] - ylim[0])

        new_xlim = (xdata - rel_x * new_width, xdata + (1 - rel_x) * new_width)
        new_ylim = (ydata - rel_y * new_height, ydata + (1 - rel_y) * new_height)

        # Apply limits
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self._xlim = new_xlim
        self._ylim = new_ylim
        self.draw()

    def _on_motion(self, event) -> None:
        """Handle mouse motion for panning."""
        if self._pan_start is None or event.inaxes != self.ax:
            return

        # Calculate pan delta
        dx = self._pan_start[0] - event.xdata
        dy = self._pan_start[1] - event.ydata

        # Apply pan
        new_xlim = (self._pan_xlim[0] + dx, self._pan_xlim[1] + dx)
        new_ylim = (self._pan_ylim[0] + dy, self._pan_ylim[1] + dy)

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self._xlim = new_xlim
        self._ylim = new_ylim
        self.draw()

    def _on_release(self, event) -> None:
        """Handle mouse button release."""
        if event.button == 2:
            self._pan_start = None

    def reset_zoom(self) -> None:
        """Reset zoom to fit the entire image."""
        self._xlim = None
        self._ylim = None
        if self.image_data is not None:
            self.ax.autoscale()
            self.draw()

    def set_data(self, data: np.ndarray) -> None:
        """Set the 3D volume data."""
        self.image_data = data
        self._needs_tight_layout = True  # New data requires layout adjustment

    def set_overlay(self, data: Optional[np.ndarray], show: bool = True) -> None:
        """Set the overlay segmentation data.

        Args:
            data: 3D segmentation mask (or None to disable)
            show: Whether to show the overlay
        """
        self.overlay_data = data
        self.show_overlay = show

    def set_crosshairs(self, h_pos: Optional[int], v_pos: Optional[int]) -> None:
        """Set crosshair positions.

        Args:
            h_pos: Horizontal line position (y coordinate)
            v_pos: Vertical line position (x coordinate)
        """
        self.crosshair_h = h_pos
        self.crosshair_v = v_pos

    def update_slice(self, slice_idx: int, axis: int) -> None:
        """Update the displayed slice.

        Args:
            slice_idx: Index of the slice to display
            axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        """
        if self.image_data is None:
            return

        self.axis = axis  # Store for click handling

        # Remove existing colorbar before clearing
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        self.ax.clear()
        self.ax.set_facecolor(COLOR_BG_DARKER)

        # Get slice based on axis
        if axis == 0:  # Sagittal (side view)
            slice_data = self.image_data[slice_idx, :, :]
            overlay_slice = self.overlay_data[slice_idx, :, :] if self.overlay_data is not None else None
            self.ax.set_title('Sagittal (Side)', color='white', fontsize=10)
        elif axis == 1:  # Coronal (front view)
            slice_data = self.image_data[:, slice_idx, :]
            overlay_slice = self.overlay_data[:, slice_idx, :] if self.overlay_data is not None else None
            self.ax.set_title('Coronal (Front)', color='white', fontsize=10)
        else:  # Axial (top view)
            slice_data = self.image_data[:, :, slice_idx]
            overlay_slice = self.overlay_data[:, :, slice_idx] if self.overlay_data is not None else None
            self.ax.set_title('Axial (Top)', color='white', fontsize=10)

        # Display base image
        img = self.ax.imshow(slice_data.T, cmap='gray', origin='lower', aspect='equal')

        # Add colorbar
        self.colorbar = self.fig.colorbar(img, ax=self.ax, fraction=0.046, pad=0.04)
        self.colorbar.ax.tick_params(colors='white', labelsize=7)

        # Overlay segmentation if enabled
        if self.show_overlay and overlay_slice is not None:
            # Create masked array for overlay (only show where seg > 0)
            masked_overlay = np.ma.masked_where(overlay_slice.T <= 0, overlay_slice.T)
            self.ax.imshow(masked_overlay, cmap='Reds', origin='lower', aspect='equal',
                          alpha=0.5, vmin=0, vmax=overlay_slice.max() if overlay_slice.max() > 0 else 1)

        # Draw crosshairs
        if self.crosshair_h is not None:
            self.ax.axhline(y=self.crosshair_h, color='yellow', linewidth=0.8, alpha=0.7)
        if self.crosshair_v is not None:
            self.ax.axvline(x=self.crosshair_v, color='yellow', linewidth=0.8, alpha=0.7)

        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.set_xlabel(f'Slice {slice_idx}', color='white', fontsize=8)

        # Restore zoom/pan if set
        if self._xlim is not None:
            self.ax.set_xlim(self._xlim)
        if self._ylim is not None:
            self.ax.set_ylim(self._ylim)

        # Only call tight_layout when needed (first render or after new data)
        if self._needs_tight_layout:
            self.fig.tight_layout()
            self._needs_tight_layout = False
        self.draw()


class VolumeWidget(QWidget):
    """Widget for 3D volume rendering using PyVista."""

    # Available colormaps for volume rendering
    COLORMAPS = ["gray", "bone", "hot", "viridis", "magma", "plasma"]

    # Opacity presets - different visibility levels (all keep background transparent)
    OPACITY_PRESETS = {
        "soft": "sigmoid",  # Gentle S-curve
        "medium": "linear",  # Linear ramp
        "strong": "geom",  # Emphasizes high values, keeps low values transparent
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter: Optional[QtInteractor] = None

        # Store current state for re-rendering
        self.current_data: Optional[np.ndarray] = None
        self.current_seg_overlay: Optional[np.ndarray] = None
        self.is_segmentation: bool = False

        # Rendering parameters
        self.clim_min: float = 0.0
        self.clim_max: float = 1.0
        self.data_min: float = 0.0
        self.data_max: float = 1.0
        self.colormap: str = "gray"
        self.opacity_preset: str = "medium"
        self.shade_enabled: bool = False  # Disable shading by default (causes darkness)
        self.seg_opacity: float = 0.0  # Segmentation overlay opacity (0 = hidden)
        self.seg_always_visible: bool = False  # Render seg on top of everything
        self.bg_color: float = 0.0  # Background color (0.0 = black, 1.0 = white)
        self.overlay_renderer = None  # Track overlay renderer for cleanup
        self.seg_data_backup: Optional[np.ndarray] = None  # Store seg data for toggling

        # Fullscreen state
        self.is_fullscreen: bool = False
        self.fullscreen_window: Optional[QWidget] = None

        # For fullscreen modality switching
        self.modalities_list: List[str] = []
        self.current_modality_name: str = ""
        self.modality_change_callback: Optional[Callable[[str], None]] = None

        # For fullscreen patient navigation
        self.prev_patient_callback: Optional[Callable[[], None]] = None
        self.next_patient_callback: Optional[Callable[[], None]] = None
        self.patient_label_text: str = ""
        self.on_exit_fullscreen_callback: Optional[Callable[[], None]] = None

        # Fullscreen UI elements (set when entering fullscreen)
        self.fs_plotter: Optional[Any] = None
        self.fs_patient_label: Optional[QLabel] = None
        self.fs_modality_combo: Optional[QComboBox] = None
        self.fs_clim_min_slider: Optional[QSlider] = None
        self.fs_clim_max_slider: Optional[QSlider] = None
        self.fs_seg_opacity_slider: Optional[QSlider] = None
        self.fs_bg_slider: Optional[QSlider] = None

        if PYVISTA_AVAILABLE:
            self._init_plotter()
        else:
            label = QLabel("3D View Unavailable\n\nInstall pyvista:\npip install pyvista pyvistaqt")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(COLOR_LABEL_DISABLED)
            self._main_layout.addWidget(label)

    def _init_plotter(self) -> None:
        """Initialize the PyVista plotter."""
        self.plotter = QtInteractor(self)
        self.plotter.set_background((self.bg_color, self.bg_color, self.bg_color))
        self._main_layout.addWidget(self.plotter.interactor)

        # Add double-click observer for fullscreen toggle
        self._click_count = 0
        self._last_click_time = 0.0

        def on_left_button_press(obj, event):
            """Handle left button press for double-click detection to toggle fullscreen."""
            current_time = time.time()
            if current_time - self._last_click_time < DOUBLE_CLICK_TIMEOUT:
                self._click_count += 1
                if self._click_count >= 2:
                    self.toggle_fullscreen()
                    self._click_count = 0
            else:
                self._click_count = 1
            self._last_click_time = current_time

        self.plotter.interactor.AddObserver('LeftButtonPressEvent', on_left_button_press)

    def set_volume(
        self,
        data: np.ndarray,
        is_segmentation: bool = False,
        seg_overlay: Optional[np.ndarray] = None,
        seg_data_for_backup: Optional[np.ndarray] = None,
        reset_camera: bool = False,
        reset_clim: bool = False
    ) -> None:
        """Set the 3D volume to render.

        Args:
            data: 3D numpy array (base modality)
            is_segmentation: If True, render data as segmentation mask only
            seg_overlay: Optional segmentation mask to overlay on base volume
            seg_data_for_backup: Seg data to store for fullscreen toggle (even if overlay disabled)
            reset_camera: If True, reset camera to fit volume
            reset_clim: If True, reset contrast to full data range
        """
        # Store for re-rendering
        self.current_data = data.copy()
        self.current_seg_overlay = seg_overlay.copy() if seg_overlay is not None else None
        # Store seg data for fullscreen independent toggle (use backup param if provided)
        backup_source = seg_data_for_backup if seg_data_for_backup is not None else seg_overlay
        self.seg_data_backup = backup_source.copy() if backup_source is not None else None
        self.is_segmentation = is_segmentation

        # Update data range
        self.data_min = float(data.min())
        self.data_max = float(data.max())

        # Only reset clim if requested
        if reset_clim:
            self.clim_min = self.data_min
            self.clim_max = self.data_max

        self._render(reset_camera=reset_camera)

    def update_clim(self, clim_min: float, clim_max: float) -> None:
        """Update color/intensity limits (contrast adjustment).

        Args:
            clim_min: Minimum intensity value
            clim_max: Maximum intensity value
        """
        self.clim_min = clim_min
        self.clim_max = clim_max
        self._render()

    def update_colormap(self, cmap: str) -> None:
        """Update the colormap.

        Args:
            cmap: Colormap name (e.g., 'gray', 'bone', 'hot')
        """
        if cmap in self.COLORMAPS:
            self.colormap = cmap
            self._render()

    def update_opacity_preset(self, preset: str) -> None:
        """Update the opacity preset.

        Args:
            preset: Preset name ('soft', 'medium', 'strong')
        """
        if preset in self.OPACITY_PRESETS:
            self.opacity_preset = preset
            self._render()

    def update_shade(self, enabled: bool) -> None:
        """Update shading.

        Args:
            enabled: Whether to enable shading
        """
        self.shade_enabled = enabled
        self._render()

    def update_seg_opacity(self, opacity: float) -> None:
        """Update segmentation overlay opacity.

        Args:
            opacity: Opacity value (0.0 to 1.0)
        """
        self.seg_opacity = opacity
        self._render()

    def update_seg_always_visible(self, enabled: bool) -> None:
        """Update segmentation always-visible mode.

        Args:
            enabled: If True, segmentation renders on top of volume
        """
        self.seg_always_visible = enabled
        self._render()

    def update_bg_color(self, value: float) -> None:
        """Update background color.

        Args:
            value: Grayscale value (0.0 = black, 1.0 = white)
        """
        old_bg = self.bg_color
        self.bg_color = value
        color = (value, value, value)
        if self.plotter is not None:
            self.plotter.set_background(color)
        if self.fs_plotter is not None:
            self.fs_plotter.set_background(color)

        # Re-render if crossing the 0.5 threshold to update scalar bar text color
        crossed_threshold = (old_bg <= 0.5) != (value <= 0.5)
        if crossed_threshold and self.current_data is not None:
            self._render()
            if self.is_fullscreen:
                self._fs_render()

    def reset_camera(self) -> None:
        """Reset 3D camera to default view (isometric)."""
        if self.plotter is not None:
            self.plotter.reset_camera()
            self.plotter.view_isometric()

    def set_modalities(self, modalities: list, current: str, callback) -> None:
        """Set available modalities for fullscreen mode.

        Args:
            modalities: List of modality names
            current: Current modality name
            callback: Function to call when modality changes
        """
        self.modalities_list = modalities
        self.current_modality_name = current
        self.modality_change_callback = callback

    def set_nav_callbacks(self, prev_callback, next_callback) -> None:
        """Set patient navigation callbacks for fullscreen mode.

        Args:
            prev_callback: Function to call for previous patient
            next_callback: Function to call for next patient
        """
        self.prev_patient_callback = prev_callback
        self.next_patient_callback = next_callback

    def set_patient_label(self, text: str) -> None:
        """Set patient label text for fullscreen display."""
        self.patient_label_text = text
        if self.fs_patient_label is not None:
            self.fs_patient_label.setText(text)

    def mouseDoubleClickEvent(self, event) -> None:
        """Handle double-click to toggle fullscreen."""
        self.toggle_fullscreen()
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        """Handle ESC key to exit fullscreen."""
        if event.key() == Qt.Key_Escape and self.is_fullscreen:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    def toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode for the 3D view."""
        if not PYVISTA_AVAILABLE or self.plotter is None:
            return

        if self.is_fullscreen:
            # Exit fullscreen - restore to original parent
            if self.fullscreen_window is not None:
                self.fullscreen_window.close()
                self.fullscreen_window = None
            # Reset fullscreen UI elements
            self.fs_plotter = None
            self.fs_patient_label = None
            self.fs_modality_combo = None
            self.fs_clim_min_slider = None
            self.fs_clim_max_slider = None
            self.fs_seg_opacity_slider = None
            self.fs_bg_slider = None
            self.is_fullscreen = False
            # Sync normal view with current settings and re-render
            if self.on_exit_fullscreen_callback:
                self.on_exit_fullscreen_callback()
            self._render()
        else:
            # Enter fullscreen - create fullscreen window with splitter layout
            self.fullscreen_window = QWidget()
            self.fullscreen_window.setWindowTitle("3D View - Press ESC to exit")
            self.fullscreen_window.setStyleSheet(DARK_THEME)

            # Create horizontal layout with plotter on left, controls on right
            fs_layout = QHBoxLayout(self.fullscreen_window)
            fs_layout.setContentsMargins(0, 0, 0, 0)
            fs_layout.setSpacing(0)

            # Create a new plotter for fullscreen
            self.fs_plotter = QtInteractor(self.fullscreen_window)
            self.fs_plotter.set_background((self.bg_color, self.bg_color, self.bg_color))
            fs_layout.addWidget(self.fs_plotter.interactor, 1)  # Stretch factor 1

            # Create controls panel on the right side
            controls_container = QWidget()
            controls_container.setFixedWidth(280)
            controls_container.setStyleSheet(f"background-color: {COLOR_BG_DARK};")

            controls_group = QGroupBox("3D View Controls")
            controls_layout = QGridLayout(controls_group)
            controls_layout.setSpacing(6)

            # Patient label
            self.fs_patient_label = QLabel(self.patient_label_text or "No patient loaded")
            self.fs_patient_label.setStyleSheet(COLOR_LABEL_BOLD)
            controls_layout.addWidget(self.fs_patient_label, 0, 0, 1, 2)

            # Modality selector
            controls_layout.addWidget(QLabel("Modality:"), 1, 0)
            self.fs_modality_combo = QComboBox()
            if self.modalities_list:
                self.fs_modality_combo.addItems(self.modalities_list)
                self.fs_modality_combo.setCurrentText(self.current_modality_name)
            self.fs_modality_combo.currentTextChanged.connect(self._fs_on_modality_changed)
            controls_layout.addWidget(self.fs_modality_combo, 1, 1)

            # Contrast Min
            controls_layout.addWidget(QLabel("Contrast Min:"), 2, 0)
            self.fs_clim_min_slider = QSlider(Qt.Horizontal)
            self.fs_clim_min_slider.setRange(0, 1000)
            self.fs_clim_min_slider.setValue(int((self.clim_min - self.data_min) / (self.data_max - self.data_min) * 1000) if self.data_max > self.data_min else 0)
            self.fs_clim_min_slider.sliderReleased.connect(self._fs_on_clim_changed)
            controls_layout.addWidget(self.fs_clim_min_slider, 2, 1)

            # Contrast Max
            controls_layout.addWidget(QLabel("Contrast Max:"), 3, 0)
            self.fs_clim_max_slider = QSlider(Qt.Horizontal)
            self.fs_clim_max_slider.setRange(0, 1000)
            self.fs_clim_max_slider.setValue(int((self.clim_max - self.data_min) / (self.data_max - self.data_min) * 1000) if self.data_max > self.data_min else 1000)
            self.fs_clim_max_slider.sliderReleased.connect(self._fs_on_clim_changed)
            controls_layout.addWidget(self.fs_clim_max_slider, 3, 1)

            # Window preset
            controls_layout.addWidget(QLabel("Window Preset:"), 4, 0)
            self.fs_preset_combo = QComboBox()
            self.fs_preset_combo.addItems(list(WINDOW_PRESETS.keys()))
            self.fs_preset_combo.currentTextChanged.connect(self._fs_on_preset_changed)
            controls_layout.addWidget(self.fs_preset_combo, 4, 1)

            # Colormap
            controls_layout.addWidget(QLabel("Colormap:"), 5, 0)
            self.fs_cmap_combo = QComboBox()
            self.fs_cmap_combo.addItems(self.COLORMAPS)
            self.fs_cmap_combo.setCurrentText(self.colormap)
            self.fs_cmap_combo.currentTextChanged.connect(self._fs_on_colormap_changed)
            controls_layout.addWidget(self.fs_cmap_combo, 5, 1)

            # Opacity
            controls_layout.addWidget(QLabel("Opacity:"), 6, 0)
            self.fs_opacity_combo = QComboBox()
            self.fs_opacity_combo.addItems(list(self.OPACITY_PRESETS.keys()))
            self.fs_opacity_combo.setCurrentText(self.opacity_preset)
            self.fs_opacity_combo.currentTextChanged.connect(self._fs_on_opacity_changed)
            controls_layout.addWidget(self.fs_opacity_combo, 6, 1)

            # Shade toggle
            self.fs_shade_checkbox = QCheckBox("Enable Shading")
            self.fs_shade_checkbox.setChecked(self.shade_enabled)
            self.fs_shade_checkbox.stateChanged.connect(self._fs_on_shade_changed)
            controls_layout.addWidget(self.fs_shade_checkbox, 7, 0, 1, 2)

            # Seg opacity (0% = hidden, increase to show)
            controls_layout.addWidget(QLabel("Seg Opacity:"), 8, 0)
            self.fs_seg_opacity_slider = QSlider(Qt.Horizontal)
            self.fs_seg_opacity_slider.setRange(0, 100)
            self.fs_seg_opacity_slider.setValue(int(self.seg_opacity * 100))
            self.fs_seg_opacity_slider.sliderReleased.connect(self._fs_on_seg_opacity_changed)
            controls_layout.addWidget(self.fs_seg_opacity_slider, 8, 1)

            # Seg always visible checkbox
            self.fs_seg_always_visible_checkbox = QCheckBox("Seg Always Visible")
            self.fs_seg_always_visible_checkbox.setChecked(self.seg_always_visible)
            self.fs_seg_always_visible_checkbox.stateChanged.connect(self._fs_on_seg_always_visible_changed)
            controls_layout.addWidget(self.fs_seg_always_visible_checkbox, 9, 0, 1, 2)

            # Background color slider (0 = black, 100 = white)
            controls_layout.addWidget(QLabel("Background:"), 10, 0)
            self.fs_bg_slider = QSlider(Qt.Horizontal)
            self.fs_bg_slider.setRange(0, 100)
            self.fs_bg_slider.setValue(int(self.bg_color * 100))
            self.fs_bg_slider.sliderReleased.connect(self._fs_on_bg_changed)
            controls_layout.addWidget(self.fs_bg_slider, 10, 1)

            # Reset camera button
            fs_reset_btn = QPushButton("Reset Camera")
            fs_reset_btn.clicked.connect(self._fs_reset_camera)
            controls_layout.addWidget(fs_reset_btn, 11, 0, 1, 2)

            # Patient navigation
            nav_layout = QHBoxLayout()
            self.fs_prev_btn = QPushButton("< Prev")
            self.fs_prev_btn.clicked.connect(self._fs_prev_patient)
            nav_layout.addWidget(self.fs_prev_btn)
            self.fs_next_btn = QPushButton("Next >")
            self.fs_next_btn.clicked.connect(self._fs_next_patient)
            nav_layout.addWidget(self.fs_next_btn)
            controls_layout.addLayout(nav_layout, 12, 0, 1, 2)

            # Exit button
            fs_exit_btn = QPushButton("Exit Fullscreen (ESC)")
            fs_exit_btn.clicked.connect(self.toggle_fullscreen)
            controls_layout.addWidget(fs_exit_btn, 13, 0, 1, 2)

            container_layout = QVBoxLayout(controls_container)
            container_layout.addStretch()  # Push controls to bottom
            container_layout.addWidget(controls_group)

            # Add controls to the main layout
            fs_layout.addWidget(controls_container)

            # Copy the current scene to fullscreen plotter
            if self.current_data is not None:
                self._render_to_plotter(self.fs_plotter, fullscreen=True)

            # Handle ESC key in fullscreen window
            def on_key(obj, event):
                """Handle keyboard events in fullscreen mode (ESC to exit)."""
                if event == 'KeyPressEvent':
                    key = obj.GetKeySym()
                    if key == 'Escape':
                        self.toggle_fullscreen()

            self.fs_plotter.interactor.AddObserver('KeyPressEvent', on_key)

            self.fullscreen_window.showFullScreen()
            self.is_fullscreen = True

    def _render_to_plotter(self, target_plotter, fullscreen: bool = False) -> None:
        """Render current volume to a specific plotter."""
        if self.current_data is None:
            return

        data = self.current_data
        seg_overlay = self.current_seg_overlay

        # Subsample for performance if needed
        step = 2 if data.size > PERFORMANCE_THRESHOLD else 1
        if step > 1:
            data = data[::step, ::step, ::step]
            if seg_overlay is not None:
                seg_overlay = seg_overlay[::step, ::step, ::step]

        # Create base volume grid
        grid = pv.ImageData()
        grid.dimensions = data.shape
        grid.spacing = (step, step, step)
        grid.point_data["values"] = data.flatten(order="F")

        if self.is_segmentation:
            if data.max() > 0:
                contour = grid.contour([0.5])
                target_plotter.add_mesh(contour, color='red', opacity=0.7)
        else:
            opacity_func = self.OPACITY_PRESETS.get(self.opacity_preset, "linear")
            font_size = 16 if fullscreen else 8

            # Add light source for shading
            if self.shade_enabled:
                light = pv.Light(position=(1, 1, 1), light_type='headlight')
                target_plotter.add_light(light)

            # Choose scalar bar text color based on background brightness
            text_color = 'black' if self.bg_color > 0.5 else 'white'

            vol = target_plotter.add_volume(
                grid, scalars="values", cmap=self.colormap,
                opacity=opacity_func, clim=[self.clim_min, self.clim_max],
                shade=False,  # Set via property instead for better control
                scalar_bar_args={
                    'title': '',
                    'label_font_size': font_size,
                    'color': text_color,
                    'vertical': True,
                    'fmt': '%.0f',
                    'position_x': 0.92,
                    'position_y': 0.1,
                    'height': 0.8,
                    'width': 0.05,
                }
            )

            # Set volume property lighting if shading enabled
            if self.shade_enabled and vol is not None:
                try:
                    vol.prop.SetAmbient(0.8)
                    vol.prop.SetDiffuse(1.0)
                    vol.prop.SetSpecular(0.0)
                    vol.prop.ShadeOn()
                except (AttributeError, TypeError):
                    pass

            if seg_overlay is not None and seg_overlay.max() > 0:
                seg_grid = pv.ImageData()
                seg_grid.dimensions = seg_overlay.shape
                seg_grid.spacing = (step, step, step)
                seg_grid.point_data["values"] = seg_overlay.flatten(order="F")
                contour = seg_grid.contour([0.5])

                if self.seg_always_visible:
                    # Use overlay renderer for always-visible mode
                    overlay_renderer = vtkRenderer()
                    overlay_renderer.SetLayer(1)
                    overlay_renderer.InteractiveOff()
                    overlay_renderer.SetBackground(0, 0, 0)
                    overlay_renderer.SetBackgroundAlpha(0.0)
                    target_plotter.render_window.SetNumberOfLayers(2)
                    target_plotter.render_window.AddRenderer(overlay_renderer)
                    overlay_renderer.SetActiveCamera(target_plotter.renderer.GetActiveCamera())

                    mapper = vtkPolyDataMapper()
                    mapper.SetInputData(contour)
                    mapper.ScalarVisibilityOff()

                    actor = vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(1.0, 0.0, 0.0)
                    actor.GetProperty().SetOpacity(max(0.1, self.seg_opacity) if self.seg_opacity > 0 else 0.8)
                    actor.GetProperty().LightingOff()
                    overlay_renderer.AddActor(actor)
                else:
                    target_plotter.add_mesh(contour, color='red', opacity=self.seg_opacity)

        target_plotter.reset_camera()

    def _fs_render(self) -> None:
        """Re-render fullscreen view preserving camera."""
        if not self.fs_plotter:
            return
        # Save camera position
        cam_pos = self.fs_plotter.camera_position

        # Clean up any existing overlay renderers from fullscreen plotter
        render_window = self.fs_plotter.render_window
        renderers = render_window.GetRenderers()
        renderers.InitTraversal()
        to_remove = []
        for i in range(renderers.GetNumberOfItems()):
            ren = renderers.GetNextItem()
            if ren and ren.GetLayer() > 0:
                to_remove.append(ren)
        for ren in to_remove:
            render_window.RemoveRenderer(ren)
        render_window.SetNumberOfLayers(1)

        self.fs_plotter.clear()
        self._render_to_plotter(self.fs_plotter, fullscreen=True)
        # Restore camera position
        if cam_pos:
            self.fs_plotter.camera_position = cam_pos

    def _fs_on_clim_changed(self) -> None:
        """Handle fullscreen contrast slider change."""
        data_range = self.data_max - self.data_min
        self.clim_min = self.data_min + (self.fs_clim_min_slider.value() / 1000.0) * data_range
        self.clim_max = self.data_min + (self.fs_clim_max_slider.value() / 1000.0) * data_range
        if self.clim_min >= self.clim_max:
            self.clim_max = self.clim_min + 0.01 * data_range
        self._fs_render()

    def _fs_on_preset_changed(self, preset: str) -> None:
        """Handle fullscreen window preset change."""
        if preset not in WINDOW_PRESETS:
            return
        min_pct, max_pct = WINDOW_PRESETS[preset]
        # Update sliders
        self.fs_clim_min_slider.setValue(min_pct * 10)
        self.fs_clim_max_slider.setValue(max_pct * 10)
        # Update clim values
        data_range = self.data_max - self.data_min
        self.clim_min = self.data_min + (min_pct / 100.0) * data_range
        self.clim_max = self.data_min + (max_pct / 100.0) * data_range
        self._fs_render()

    def _fs_on_colormap_changed(self, cmap: str) -> None:
        """Handle fullscreen colormap change."""
        self.colormap = cmap
        self._fs_render()

    def _fs_on_opacity_changed(self, preset: str) -> None:
        """Handle fullscreen opacity change."""
        if preset in self.OPACITY_PRESETS:
            self.opacity_preset = preset
            self._fs_render()

    def _fs_on_shade_changed(self, state: int) -> None:
        """Handle fullscreen shade checkbox change."""
        self.shade_enabled = (state == Qt.Checked)
        self._fs_render()

    def _fs_on_seg_opacity_changed(self) -> None:
        """Handle fullscreen seg opacity slider change."""
        self.seg_opacity = self.fs_seg_opacity_slider.value() / 100.0
        self._fs_render()

    def _fs_on_seg_always_visible_changed(self, state: int) -> None:
        """Handle fullscreen seg always visible checkbox change."""
        self.seg_always_visible = (state == Qt.Checked)
        self._fs_render()

    def _fs_on_bg_changed(self) -> None:
        """Handle fullscreen background color slider change."""
        if self.fs_bg_slider is not None:
            value = self.fs_bg_slider.value() / 100.0
            self.update_bg_color(value)

    def _fs_on_modality_changed(self, modality: str) -> None:
        """Handle fullscreen modality change."""
        if self.modality_change_callback and modality:
            # Save current contrast percentages before modality change
            min_pct = self.fs_clim_min_slider.value() if self.fs_clim_min_slider is not None else 0
            max_pct = self.fs_clim_max_slider.value() if self.fs_clim_max_slider is not None else 1000

            self.current_modality_name = modality
            self.modality_change_callback(modality)

            # Restore contrast percentages (data range was updated by callback)
            if self.fs_clim_min_slider is not None:
                self.fs_clim_min_slider.setValue(min_pct)
                self.fs_clim_max_slider.setValue(max_pct)
                # Recalculate clim from preserved percentages
                if self.data_max > self.data_min:
                    data_range = self.data_max - self.data_min
                    self.clim_min = self.data_min + (min_pct / 1000.0) * data_range
                    self.clim_max = self.data_min + (max_pct / 1000.0) * data_range
            self._fs_render()

    def _fs_reset_camera(self) -> None:
        """Reset fullscreen camera."""
        if self.fs_plotter:
            self.fs_plotter.reset_camera()
            self.fs_plotter.view_isometric()

    def _fs_prev_patient(self) -> None:
        """Navigate to previous patient from fullscreen."""
        if self.prev_patient_callback:
            self.prev_patient_callback()
            self._update_fs_after_patient_change()

    def _fs_next_patient(self) -> None:
        """Navigate to next patient from fullscreen."""
        if self.next_patient_callback:
            self.next_patient_callback()
            self._update_fs_after_patient_change()

    def _update_fs_after_patient_change(self) -> None:
        """Update fullscreen controls after patient change."""
        # Update patient label
        if self.fs_patient_label is not None:
            self.fs_patient_label.setText(self.patient_label_text)
        # Update modality combo
        if self.fs_modality_combo is not None:
            self.fs_modality_combo.blockSignals(True)
            self.fs_modality_combo.clear()
            self.fs_modality_combo.addItems(self.modalities_list)
            self.fs_modality_combo.setCurrentText(self.current_modality_name)
            self.fs_modality_combo.blockSignals(False)
        # Recalculate clim from fullscreen slider percentages (data range changed with new patient)
        if self.fs_clim_min_slider is not None and self.data_max > self.data_min:
            min_pct = self.fs_clim_min_slider.value() / 1000.0
            max_pct = self.fs_clim_max_slider.value() / 1000.0
            data_range = self.data_max - self.data_min
            self.clim_min = self.data_min + min_pct * data_range
            self.clim_max = self.data_min + max_pct * data_range
        self._fs_render()

    def _render(self, reset_camera: bool = False) -> None:
        """Render the volume with current parameters.

        Args:
            reset_camera: If True, reset camera to fit volume. If False, preserve current view.
        """
        if not PYVISTA_AVAILABLE or self.plotter is None or self.current_data is None:
            return

        # Save camera state before clearing
        camera_position = self.plotter.camera_position if not reset_camera else None

        self.plotter.clear()

        data = self.current_data
        seg_overlay = self.current_seg_overlay

        # Subsample for performance if needed
        step = 2 if data.size > PERFORMANCE_THRESHOLD else 1

        if step > 1:
            data = data[::step, ::step, ::step]
            if seg_overlay is not None:
                seg_overlay = seg_overlay[::step, ::step, ::step]

        # Create base volume grid
        grid = pv.ImageData()
        grid.dimensions = data.shape
        grid.spacing = (step, step, step)
        grid.point_data["values"] = data.flatten(order="F")

        # Clean up any existing overlay renderer first (always do this)
        if self.overlay_renderer is not None:
            self.plotter.render_window.RemoveRenderer(self.overlay_renderer)
            self.overlay_renderer = None
            self.plotter.render_window.SetNumberOfLayers(1)

        if self.is_segmentation:
            # Render segmentation only as isosurface
            if data.max() > 0:
                contour = grid.contour([0.5])
                self.plotter.add_mesh(contour, color='red', opacity=0.7)
        else:
            # Get opacity function from preset
            opacity_func = self.OPACITY_PRESETS.get(self.opacity_preset, "linear")

            # Add light source for shading
            if self.shade_enabled:
                light = pv.Light(position=(1, 1, 1), light_type='headlight')
                self.plotter.add_light(light)

            # Choose scalar bar text color based on background brightness
            text_color = 'black' if self.bg_color > 0.5 else 'white'

            # Add base volume
            vol = self.plotter.add_volume(
                grid,
                scalars="values",
                cmap=self.colormap,
                opacity=opacity_func,
                clim=[self.clim_min, self.clim_max],
                shade=False,  # Set via property instead for better control
                scalar_bar_args={
                    'title': '',
                    'label_font_size': 8,
                    'color': text_color,
                    'vertical': True,
                    'fmt': '%.0f',
                    'position_x': 0.92,
                    'position_y': 0.1,
                    'height': 0.8,
                    'width': 0.05,
                }
            )

            # Set volume property lighting if shading enabled
            if self.shade_enabled and vol is not None:
                try:
                    vol.prop.SetAmbient(0.8)
                    vol.prop.SetDiffuse(1.0)
                    vol.prop.SetSpecular(0.0)
                    vol.prop.ShadeOn()
                except (AttributeError, TypeError):
                    pass

            # Add segmentation overlay if provided
            if seg_overlay is not None and seg_overlay.max() > 0:
                seg_grid = pv.ImageData()
                seg_grid.dimensions = seg_overlay.shape
                seg_grid.spacing = (step, step, step)
                seg_grid.point_data["values"] = seg_overlay.flatten(order="F")
                contour = seg_grid.contour([0.5])

                if self.seg_always_visible:
                    # Create a second renderer for always-on-top segmentation
                    self.overlay_renderer = vtkRenderer()
                    self.overlay_renderer.SetLayer(1)
                    self.overlay_renderer.InteractiveOff()
                    self.overlay_renderer.SetBackground(0, 0, 0)
                    self.overlay_renderer.SetBackgroundAlpha(0.0)

                    # Add overlay renderer to the render window
                    self.plotter.render_window.SetNumberOfLayers(2)
                    self.plotter.render_window.AddRenderer(self.overlay_renderer)

                    # Sync camera with main renderer
                    self.overlay_renderer.SetActiveCamera(self.plotter.renderer.GetActiveCamera())

                    # Create mapper and actor for segmentation
                    mapper = vtkPolyDataMapper()
                    mapper.SetInputData(contour)
                    mapper.ScalarVisibilityOff()  # Use actor color, not scalar mapping

                    actor = vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red
                    actor.GetProperty().SetOpacity(max(0.1, self.seg_opacity) if self.seg_opacity > 0 else 0.8)
                    actor.GetProperty().LightingOff()

                    self.overlay_renderer.AddActor(actor)
                else:
                    self.plotter.add_mesh(contour, color='red', opacity=self.seg_opacity)

        # Restore or reset camera
        if camera_position is not None:
            self.plotter.camera_position = camera_position
        else:
            self.plotter.reset_camera()


class NiftiViewer(QMainWindow):
    """Main window for NIfTI 3D viewer."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIfTI 3D Viewer")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(DARK_THEME)

        # Data storage
        self.modalities: Dict[str, np.ndarray] = {}
        self.current_modality: Optional[str] = None
        self.patient_dir: Optional[Path] = None

        # Patient navigation
        self.patient_list: list[Path] = []
        self.current_patient_idx: int = 0

        # Slice indices for each axis
        self.slice_indices = [0, 0, 0]  # sagittal, coronal, axial
        self.max_slices = [1, 1, 1]

        # Overlay state
        self.overlay_enabled: bool = True  # Always enabled, controlled by seg opacity

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top bar: patient selector and modality tabs
        top_bar = QHBoxLayout()

        # Browse button
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_patient)
        top_bar.addWidget(self.browse_btn)

        # Previous button
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self._prev_patient)
        self.prev_btn.setEnabled(False)
        top_bar.addWidget(self.prev_btn)

        # Next button
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self._next_patient)
        self.next_btn.setEnabled(False)
        top_bar.addWidget(self.next_btn)

        # Patient path label
        self.path_label = QLabel("No patient loaded")
        self.path_label.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        top_bar.addWidget(self.path_label, 1)

        main_layout.addLayout(top_bar)

        # Main content area with splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left side: 2D views grid
        views_widget = QWidget()
        views_layout = QGridLayout(views_widget)
        views_layout.setSpacing(5)

        # Create three orthogonal view canvases
        self.axial_canvas = SliceCanvas("Axial (Top)")
        self.coronal_canvas = SliceCanvas("Coronal (Front)")
        self.sagittal_canvas = SliceCanvas("Sagittal (Side)")

        # Connect click callbacks for click-to-navigate
        self.axial_canvas.set_click_callback(self._on_canvas_click)
        self.coronal_canvas.set_click_callback(self._on_canvas_click)
        self.sagittal_canvas.set_click_callback(self._on_canvas_click)

        views_layout.addWidget(self.axial_canvas, 0, 0)
        views_layout.addWidget(self.coronal_canvas, 0, 1)
        views_layout.addWidget(self.sagittal_canvas, 1, 0)

        # Info panel
        info_group = QGroupBox("Volume Info")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("No data loaded")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        views_layout.addWidget(info_group, 1, 1)

        splitter.addWidget(views_widget)

        # Right side: 3D view + controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.volume_widget = VolumeWidget()
        right_layout.addWidget(self.volume_widget, 1)

        # 3D Controls panel
        controls_group = QGroupBox("3D View Controls")
        controls_layout = QGridLayout(controls_group)

        # Modality selector
        controls_layout.addWidget(QLabel("Modality:"), 0, 0)
        self.modality_combo = QComboBox()
        self.modality_combo.setMinimumWidth(100)
        self.modality_combo.currentTextChanged.connect(self._on_modality_changed)
        controls_layout.addWidget(self.modality_combo, 0, 1, 1, 2)

        # Contrast (Window Level) - clim min
        controls_layout.addWidget(QLabel("Contrast Min:"), 1, 0)
        self.clim_min_slider = QSlider(Qt.Horizontal)
        self.clim_min_slider.setRange(0, 1000)
        self.clim_min_slider.setValue(0)
        self.clim_min_slider.valueChanged.connect(self._on_clim_label_update)
        self.clim_min_slider.sliderReleased.connect(self._on_clim_changed)
        controls_layout.addWidget(self.clim_min_slider, 1, 1)
        self.clim_min_label = QLabel("0%")
        self.clim_min_label.setMinimumWidth(40)
        controls_layout.addWidget(self.clim_min_label, 1, 2)

        # Contrast (Window Width) - clim max
        controls_layout.addWidget(QLabel("Contrast Max:"), 2, 0)
        self.clim_max_slider = QSlider(Qt.Horizontal)
        self.clim_max_slider.setRange(0, 1000)
        self.clim_max_slider.setValue(1000)
        self.clim_max_slider.valueChanged.connect(self._on_clim_label_update)
        self.clim_max_slider.sliderReleased.connect(self._on_clim_changed)
        controls_layout.addWidget(self.clim_max_slider, 2, 1)
        self.clim_max_label = QLabel("100%")
        self.clim_max_label.setMinimumWidth(40)
        controls_layout.addWidget(self.clim_max_label, 2, 2)

        # Window presets (quick contrast settings)
        controls_layout.addWidget(QLabel("Window Preset:"), 3, 0)
        self.window_preset_combo = QComboBox()
        self.window_preset_combo.addItems(list(WINDOW_PRESETS.keys()))
        self.window_preset_combo.currentTextChanged.connect(self._on_window_preset_changed)
        controls_layout.addWidget(self.window_preset_combo, 3, 1, 1, 2)

        # Colormap selector
        controls_layout.addWidget(QLabel("Colormap:"), 4, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(VolumeWidget.COLORMAPS)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        controls_layout.addWidget(self.colormap_combo, 4, 1, 1, 2)

        # Opacity preset selector
        controls_layout.addWidget(QLabel("Opacity:"), 5, 0)
        self.opacity_combo = QComboBox()
        self.opacity_combo.addItems(list(VolumeWidget.OPACITY_PRESETS.keys()))
        self.opacity_combo.setCurrentText("medium")
        self.opacity_combo.currentTextChanged.connect(self._on_opacity_preset_changed)
        controls_layout.addWidget(self.opacity_combo, 5, 1, 1, 2)

        # Shade toggle
        self.shade_checkbox = QCheckBox("Enable Shading")
        self.shade_checkbox.setChecked(False)
        self.shade_checkbox.stateChanged.connect(self._on_shade_changed)
        controls_layout.addWidget(self.shade_checkbox, 6, 0, 1, 3)

        # Seg overlay opacity (0% = hidden, increase to show)
        controls_layout.addWidget(QLabel("Seg Opacity:"), 7, 0)
        self.seg_opacity_slider = QSlider(Qt.Horizontal)
        self.seg_opacity_slider.setRange(0, 100)
        self.seg_opacity_slider.setValue(0)
        self.seg_opacity_slider.valueChanged.connect(self._on_seg_opacity_label_update)
        self.seg_opacity_slider.sliderReleased.connect(self._on_seg_opacity_changed)
        controls_layout.addWidget(self.seg_opacity_slider, 7, 1)
        self.seg_opacity_label = QLabel("0%")
        self.seg_opacity_label.setMinimumWidth(40)
        controls_layout.addWidget(self.seg_opacity_label, 7, 2)

        # Seg always visible (shine through)
        self.seg_always_visible_checkbox = QCheckBox("Seg Always Visible")
        self.seg_always_visible_checkbox.setChecked(False)
        self.seg_always_visible_checkbox.setToolTip("Make segmentation visible through the volume")
        self.seg_always_visible_checkbox.stateChanged.connect(self._on_seg_always_visible_changed)
        controls_layout.addWidget(self.seg_always_visible_checkbox, 8, 0, 1, 3)

        # Background color slider (0 = black, 100 = white)
        controls_layout.addWidget(QLabel("Background:"), 9, 0)
        self.bg_slider = QSlider(Qt.Horizontal)
        self.bg_slider.setRange(0, 100)
        self.bg_slider.setValue(0)  # Default to black
        self.bg_slider.valueChanged.connect(self._on_bg_label_update)
        self.bg_slider.sliderReleased.connect(self._on_bg_changed)
        controls_layout.addWidget(self.bg_slider, 9, 1)
        self.bg_label = QLabel("0%")
        self.bg_label.setMinimumWidth(40)
        controls_layout.addWidget(self.bg_label, 9, 2)

        # Reset View button
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self._on_reset_view)
        controls_layout.addWidget(self.reset_view_btn, 10, 0, 1, 3)

        right_layout.addWidget(controls_group)
        splitter.addWidget(right_widget)

        splitter.setSizes(DEFAULT_SPLITTER_SIZES)
        main_layout.addWidget(splitter, 1)

        # Bottom: slice sliders
        sliders_widget = QWidget()
        sliders_layout = QHBoxLayout(sliders_widget)

        # Axial slider
        axial_group = QGroupBox("Axial Slice")
        axial_layout = QVBoxLayout(axial_group)
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.valueChanged.connect(lambda v: self._on_slider_changed(2, v))
        self.axial_label = QLabel("0 / 0")
        self.axial_label.setAlignment(Qt.AlignCenter)
        axial_layout.addWidget(self.axial_slider)
        axial_layout.addWidget(self.axial_label)
        sliders_layout.addWidget(axial_group)

        # Coronal slider
        coronal_group = QGroupBox("Coronal Slice")
        coronal_layout = QVBoxLayout(coronal_group)
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.valueChanged.connect(lambda v: self._on_slider_changed(1, v))
        self.coronal_label = QLabel("0 / 0")
        self.coronal_label.setAlignment(Qt.AlignCenter)
        coronal_layout.addWidget(self.coronal_slider)
        coronal_layout.addWidget(self.coronal_label)
        sliders_layout.addWidget(coronal_group)

        # Sagittal slider
        sagittal_group = QGroupBox("Sagittal Slice")
        sagittal_layout = QVBoxLayout(sagittal_group)
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.valueChanged.connect(lambda v: self._on_slider_changed(0, v))
        self.sagittal_label = QLabel("0 / 0")
        self.sagittal_label.setAlignment(Qt.AlignCenter)
        sagittal_layout.addWidget(self.sagittal_slider)
        sagittal_layout.addWidget(self.sagittal_label)
        sliders_layout.addWidget(sagittal_group)

        main_layout.addWidget(sliders_widget)

    def _browse_patient(self) -> None:
        """Open file dialog to select patient directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Patient Directory", str(Path.home())
        )
        if dir_path:
            patient_path = Path(dir_path)
            self._scan_patient_directories(patient_path)
            self._load_patient(patient_path)

    def _scan_patient_directories(self, selected_path: Path) -> None:
        """Scan parent directory for all patient folders.

        Args:
            selected_path: The selected patient directory
        """
        parent = selected_path.parent

        # Find all directories in parent that contain .nii.gz files
        self.patient_list = sorted([
            d for d in parent.iterdir()
            if d.is_dir() and list(d.glob("*.nii.gz"))
        ])

        # Find index of selected patient
        try:
            self.current_patient_idx = self.patient_list.index(selected_path)
        except ValueError:
            self.current_patient_idx = 0

        self._update_nav_buttons()

    def _update_nav_buttons(self) -> None:
        """Enable/disable navigation buttons based on current position."""
        has_patients = len(self.patient_list) > 1
        self.prev_btn.setEnabled(has_patients and self.current_patient_idx > 0)
        self.next_btn.setEnabled(has_patients and self.current_patient_idx < len(self.patient_list) - 1)

    def _prev_patient(self) -> None:
        """Navigate to previous patient."""
        if self.current_patient_idx > 0:
            self.current_patient_idx -= 1
            self._load_patient(self.patient_list[self.current_patient_idx])
            self._update_nav_buttons()

    def _next_patient(self) -> None:
        """Navigate to next patient."""
        if self.current_patient_idx < len(self.patient_list) - 1:
            self.current_patient_idx += 1
            self._load_patient(self.patient_list[self.current_patient_idx])
            self._update_nav_buttons()

    def _load_patient(self, patient_dir: Path) -> None:
        """Load all NIfTI files from patient directory.

        Args:
            patient_dir: Path to patient directory containing .nii.gz files
        """
        self.patient_dir = patient_dir
        self.modalities.clear()

        # Find all NIfTI files
        nii_files = list(patient_dir.glob("*.nii.gz")) + list(patient_dir.glob("*.nii"))

        if not nii_files:
            self.path_label.setText(f"No NIfTI files found in: {patient_dir}")
            return

        # Load each modality
        for nii_path in nii_files:
            name = nii_path.name.replace('.nii.gz', '').replace('.nii', '')
            try:
                img = nib.load(str(nii_path))
                data = img.get_fdata().astype(np.float32)
                self.modalities[name] = data
            except (OSError, nib.filebasedimages.ImageFileError) as e:
                logger.error("Error loading %s: %s", nii_path, e)

        if not self.modalities:
            self.path_label.setText("Failed to load any NIfTI files")
            return

        # Update UI with patient name and index
        if self.patient_list:
            total = len(self.patient_list)
            idx = self.current_patient_idx + 1
            self.path_label.setText(f"{patient_dir.name} ({idx}/{total})")
        else:
            self.path_label.setText(f"{patient_dir.name}")

        # Populate modality combo
        self.modality_combo.blockSignals(True)
        self.modality_combo.clear()

        # Sort modalities with preferred order
        sorted_modalities = sorted(
            self.modalities.keys(),
            key=lambda x: MODALITY_ORDER.index(x) if x in MODALITY_ORDER else 999
        )
        self.modality_combo.addItems(sorted_modalities)
        self.modality_combo.blockSignals(False)

        # Set modalities on volume widget for fullscreen mode
        if sorted_modalities:
            self.volume_widget.set_modalities(
                sorted_modalities,
                sorted_modalities[0],
                self._on_modality_changed_from_fullscreen
            )

        # Set navigation callbacks for fullscreen mode
        self.volume_widget.set_nav_callbacks(self._prev_patient, self._next_patient)

        # Set callback to sync UI when exiting fullscreen
        self.volume_widget.on_exit_fullscreen_callback = self._sync_controls_from_volume_widget

        # Set patient label for fullscreen mode
        self.volume_widget.set_patient_label(self.path_label.text())

        # Load modality - preserve current if available, otherwise use first
        if sorted_modalities:
            # Try to keep current modality if it exists in new patient
            target_modality = sorted_modalities[0]
            if self.current_modality and self.current_modality in sorted_modalities:
                target_modality = self.current_modality
            # Set combo box to target modality
            self.modality_combo.blockSignals(True)
            self.modality_combo.setCurrentText(target_modality)
            self.modality_combo.blockSignals(False)
            self._on_modality_changed(target_modality, reset_camera=True)

    def _apply_modality_to_2d_views(self, data: np.ndarray) -> None:
        """Apply modality data to all 2D slice views.

        Args:
            data: 3D volume data to display
        """
        # Update max slices
        self.max_slices = list(data.shape)

        # Update sliders
        self.sagittal_slider.setMaximum(max(0, data.shape[0] - 1))
        self.coronal_slider.setMaximum(max(0, data.shape[1] - 1))
        self.axial_slider.setMaximum(max(0, data.shape[2] - 1))

        # Set to middle slices
        self.slice_indices = [s // 2 for s in data.shape]
        self.sagittal_slider.setValue(self.slice_indices[0])
        self.coronal_slider.setValue(self.slice_indices[1])
        self.axial_slider.setValue(self.slice_indices[2])

        # Update canvases with base data
        self.axial_canvas.set_data(data)
        self.coronal_canvas.set_data(data)
        self.sagittal_canvas.set_data(data)

        # Update overlay state and render
        self._update_overlay_state()
        self._update_all_views()

    def _on_modality_changed(self, modality: str, reset_camera: bool = False) -> None:
        """Handle modality selection change.

        Args:
            modality: Name of selected modality
            reset_camera: If True, reset 3D camera to fit volume
        """
        if not modality or modality not in self.modalities:
            return

        self.current_modality = modality
        data = self.modalities[modality]

        # Update 2D views
        self._apply_modality_to_2d_views(data)

        # Update 3D view (preserve contrast settings)
        self._update_3d_view(reset_camera=reset_camera, reset_contrast=False)

        # Update info and sync fullscreen modality name
        self._update_info(data, modality)
        self.volume_widget.current_modality_name = modality

    def _on_modality_changed_from_fullscreen(self, modality: str) -> None:
        """Handle modality change from fullscreen mode - update main UI.

        Args:
            modality: Name of selected modality
        """
        if not modality or modality not in self.modalities:
            return

        # Update main combo box without triggering signal to avoid loop
        self.modality_combo.blockSignals(True)
        self.modality_combo.setCurrentText(modality)
        self.modality_combo.blockSignals(False)

        self.current_modality = modality
        data = self.modalities[modality]

        # Update 2D views
        self._apply_modality_to_2d_views(data)

        # Update volume widget data for fullscreen rendering (not 3D view)
        is_seg = 'seg' in modality.lower()
        seg_data = self.modalities.get('seg') if not is_seg else None
        seg_overlay = seg_data if self.overlay_enabled else None

        self.volume_widget.current_data = data.copy()
        self.volume_widget.current_seg_overlay = seg_overlay.copy() if seg_overlay is not None else None
        self.volume_widget.seg_data_backup = seg_data.copy() if seg_data is not None else None
        self.volume_widget.is_segmentation = is_seg
        self.volume_widget.data_min = float(data.min())
        self.volume_widget.data_max = float(data.max())

        self._update_info(data, modality)

    def _update_overlay_state(self) -> None:
        """Update overlay data on all canvases based on current state."""
        seg_data = self.modalities.get('seg')
        show = self.overlay_enabled and seg_data is not None

        for canvas in [self.axial_canvas, self.coronal_canvas, self.sagittal_canvas]:
            canvas.set_overlay(seg_data if show else None, show=show)

    def _update_3d_view(self, reset_camera: bool = False, reset_contrast: bool = False) -> None:
        """Update the 3D volume view.

        Args:
            reset_camera: If True, reset camera to fit volume. Default preserves view.
            reset_contrast: If True, reset contrast sliders to full range.
        """
        if not self.current_modality or self.current_modality not in self.modalities:
            return

        data = self.modalities[self.current_modality]
        is_seg = 'seg' in self.current_modality.lower()

        # Always pass seg data for backup (so fullscreen can use it independently)
        # But only show overlay if enabled
        seg_data = self.modalities.get('seg') if not is_seg else None
        seg_overlay = seg_data if self.overlay_enabled else None

        if reset_contrast:
            # Reset to full data range
            self.volume_widget.set_volume(data, is_segmentation=is_seg, seg_overlay=seg_overlay,
                                          seg_data_for_backup=seg_data,
                                          reset_camera=reset_camera, reset_clim=True)
            self._update_contrast_sliders()
        else:
            # Preserve current contrast percentages
            min_pct = self.clim_min_slider.value() / 1000.0
            max_pct = self.clim_max_slider.value() / 1000.0

            # Set volume without resetting clim (we'll set it manually)
            self.volume_widget.set_volume(data, is_segmentation=is_seg, seg_overlay=seg_overlay,
                                          seg_data_for_backup=seg_data,
                                          reset_camera=reset_camera, reset_clim=False)

            # Recalculate clim from preserved percentages and new data range
            data_range = self.volume_widget.data_max - self.volume_widget.data_min
            if data_range > 0:
                self.volume_widget.clim_min = self.volume_widget.data_min + min_pct * data_range
                self.volume_widget.clim_max = self.volume_widget.data_min + max_pct * data_range
            self.volume_widget._render()

    def _update_contrast_sliders(self) -> None:
        """Update contrast slider ranges based on current volume data."""
        data_min = self.volume_widget.data_min
        data_max = self.volume_widget.data_max

        # Block signals while updating
        self.clim_min_slider.blockSignals(True)
        self.clim_max_slider.blockSignals(True)

        # Map data range to slider range (0-1000)
        self.clim_min_slider.setValue(0)
        self.clim_max_slider.setValue(1000)

        self.clim_min_label.setText("0%")
        self.clim_max_label.setText("100%")

        self.clim_min_slider.blockSignals(False)
        self.clim_max_slider.blockSignals(False)

    def _on_clim_label_update(self) -> None:
        """Update contrast labels while dragging (no 3D update)."""
        data_min = self.volume_widget.data_min
        data_max = self.volume_widget.data_max
        data_range = data_max - data_min

        min_val = data_min + (self.clim_min_slider.value() / 1000.0) * data_range
        max_val = data_min + (self.clim_max_slider.value() / 1000.0) * data_range

        min_pct = self.clim_min_slider.value() / 10
        max_pct = self.clim_max_slider.value() / 10
        self.clim_min_label.setText(f"{min_pct:.0f}%")
        self.clim_max_label.setText(f"{max_pct:.0f}%")

    def _on_clim_changed(self) -> None:
        """Handle contrast slider release - update 3D."""
        data_min = self.volume_widget.data_min
        data_max = self.volume_widget.data_max
        data_range = data_max - data_min

        min_val = data_min + (self.clim_min_slider.value() / 1000.0) * data_range
        max_val = data_min + (self.clim_max_slider.value() / 1000.0) * data_range

        # Ensure min < max
        if min_val >= max_val:
            max_val = min_val + 0.01 * data_range

        self.volume_widget.update_clim(min_val, max_val)

    def _on_window_preset_changed(self, preset: str) -> None:
        """Handle window preset selection - sets contrast range."""
        if preset not in WINDOW_PRESETS:
            return

        min_pct, max_pct = WINDOW_PRESETS[preset]

        # Set slider values (0-1000 maps to 0-100%)
        self.clim_min_slider.setValue(min_pct * 10)
        self.clim_max_slider.setValue(max_pct * 10)

        # Update labels and 3D view
        self._on_clim_label_update()
        self._on_clim_changed()

    def _on_colormap_changed(self, cmap: str) -> None:
        """Handle colormap selection change."""
        self.volume_widget.update_colormap(cmap)

    def _on_opacity_preset_changed(self, preset: str) -> None:
        """Handle opacity preset selection change."""
        self.volume_widget.update_opacity_preset(preset)

    def _on_shade_changed(self, state: int) -> None:
        """Handle shade checkbox toggle."""
        self.volume_widget.update_shade(state == Qt.Checked)

    def _on_seg_opacity_label_update(self, value: int) -> None:
        """Update seg opacity label while dragging (no 3D update)."""
        self.seg_opacity_label.setText(f"{value}%")

    def _on_seg_opacity_changed(self) -> None:
        """Handle seg overlay opacity slider release - update 3D."""
        opacity = self.seg_opacity_slider.value() / 100.0
        self.volume_widget.update_seg_opacity(opacity)

    def _on_seg_always_visible_changed(self, state: int) -> None:
        """Handle seg always visible checkbox toggle."""
        self.volume_widget.update_seg_always_visible(state == Qt.Checked)

    def _on_bg_label_update(self, value: int) -> None:
        """Update background label while dragging (no 3D update)."""
        self.bg_label.setText(f"{value}%")

    def _on_bg_changed(self) -> None:
        """Handle background color slider release - update 3D."""
        value = self.bg_slider.value() / 100.0
        self.volume_widget.update_bg_color(value)

    def _sync_controls_from_volume_widget(self) -> None:
        """Sync UI controls from VolumeWidget state (called when exiting fullscreen)."""
        vw = self.volume_widget

        # Sync colormap
        self.colormap_combo.blockSignals(True)
        self.colormap_combo.setCurrentText(vw.colormap)
        self.colormap_combo.blockSignals(False)

        # Sync opacity preset
        self.opacity_combo.blockSignals(True)
        self.opacity_combo.setCurrentText(vw.opacity_preset)
        self.opacity_combo.blockSignals(False)

        # Sync shading
        self.shade_checkbox.blockSignals(True)
        self.shade_checkbox.setChecked(vw.shade_enabled)
        self.shade_checkbox.blockSignals(False)

        # Sync contrast sliders (calculate percentage from clim values)
        if vw.data_max > vw.data_min:
            data_range = vw.data_max - vw.data_min
            min_pct = int((vw.clim_min - vw.data_min) / data_range * 1000)
            max_pct = int((vw.clim_max - vw.data_min) / data_range * 1000)
            self.clim_min_slider.blockSignals(True)
            self.clim_max_slider.blockSignals(True)
            self.clim_min_slider.setValue(min_pct)
            self.clim_max_slider.setValue(max_pct)
            self.clim_min_label.setText(f"{min_pct // 10}%")
            self.clim_max_label.setText(f"{max_pct // 10}%")
            self.clim_min_slider.blockSignals(False)
            self.clim_max_slider.blockSignals(False)

        # Sync seg opacity
        self.seg_opacity_slider.blockSignals(True)
        self.seg_opacity_slider.setValue(int(vw.seg_opacity * 100))
        self.seg_opacity_label.setText(f"{int(vw.seg_opacity * 100)}%")
        self.seg_opacity_slider.blockSignals(False)

        # Sync seg always visible
        self.seg_always_visible_checkbox.blockSignals(True)
        self.seg_always_visible_checkbox.setChecked(vw.seg_always_visible)
        self.seg_always_visible_checkbox.blockSignals(False)

        # Sync background color
        self.bg_slider.blockSignals(True)
        self.bg_slider.setValue(int(vw.bg_color * 100))
        self.bg_label.setText(f"{int(vw.bg_color * 100)}%")
        self.bg_slider.blockSignals(False)

    def _on_reset_view(self) -> None:
        """Handle reset view button click - resets everything to defaults."""
        if not self.current_modality or self.current_modality not in self.modalities:
            return

        data = self.modalities[self.current_modality]

        # Reset slice sliders to middle position
        mid_slices = [s // 2 for s in data.shape]
        self.sagittal_slider.setValue(mid_slices[0])
        self.coronal_slider.setValue(mid_slices[1])
        self.axial_slider.setValue(mid_slices[2])

        # Reset contrast sliders to full range
        self.clim_min_slider.setValue(0)
        self.clim_max_slider.setValue(1000)
        self.clim_min_label.setText("0%")
        self.clim_max_label.setText("100%")

        # Reset window preset to Full Range
        self.window_preset_combo.blockSignals(True)
        self.window_preset_combo.setCurrentText("Full Range")
        self.window_preset_combo.blockSignals(False)

        # Reset colormap to gray (block signals to prevent multiple re-renders)
        self.colormap_combo.blockSignals(True)
        self.colormap_combo.setCurrentText("gray")
        self.colormap_combo.blockSignals(False)

        # Reset opacity to medium
        self.opacity_combo.blockSignals(True)
        self.opacity_combo.setCurrentText("medium")
        self.opacity_combo.blockSignals(False)

        # Reset shade off
        self.shade_checkbox.blockSignals(True)
        self.shade_checkbox.setChecked(False)
        self.shade_checkbox.blockSignals(False)

        # Reset seg opacity to 0%
        self.seg_opacity_slider.setValue(0)
        self.seg_opacity_label.setText("0%")

        # Reset seg always visible off
        self.seg_always_visible_checkbox.blockSignals(True)
        self.seg_always_visible_checkbox.setChecked(False)
        self.seg_always_visible_checkbox.blockSignals(False)

        # Reset background to black
        self.bg_slider.setValue(0)
        self.bg_label.setText("0%")

        # Reset 2D view zoom/pan
        self.axial_canvas.reset_zoom()
        self.coronal_canvas.reset_zoom()
        self.sagittal_canvas.reset_zoom()

        # Apply all reset values to VolumeWidget
        self.volume_widget.colormap = "gray"
        self.volume_widget.opacity_preset = "medium"
        self.volume_widget.shade_enabled = False
        self.volume_widget.seg_opacity = 0.0
        self.volume_widget.seg_always_visible = False
        self.volume_widget.clim_min = self.volume_widget.data_min
        self.volume_widget.clim_max = self.volume_widget.data_max
        self.volume_widget.update_bg_color(0.0)  # Reset to black

        # Re-render 3D with reset camera
        self.volume_widget._render(reset_camera=True)

    def _on_slider_changed(self, axis: int, value: int) -> None:
        """Handle slice slider change.

        Args:
            axis: Axis index (0=sagittal, 1=coronal, 2=axial)
            value: New slice index
        """
        self.slice_indices[axis] = value

        # Update label
        if axis == 0:
            self.sagittal_label.setText(f"{value} / {self.max_slices[0] - 1}")
        elif axis == 1:
            self.coronal_label.setText(f"{value} / {self.max_slices[1] - 1}")
        else:
            self.axial_label.setText(f"{value} / {self.max_slices[2] - 1}")

        self._update_all_views()

    def _on_canvas_click(self, axis: int, x: int, y: int) -> None:
        """Handle click on a 2D canvas to navigate slices.

        Args:
            axis: Which canvas was clicked (0=sagittal, 1=coronal, 2=axial)
            x: X coordinate in image space (column)
            y: Y coordinate in image space (row)
        """
        if self.current_modality is None:
            return

        # Map click coordinates to slice updates based on which view was clicked
        # Note: imshow uses .T so x is the first data axis, y is the second
        if axis == 0:  # Sagittal view (Y-Z plane)
            # x -> coronal position, y -> axial position
            new_cor = max(0, min(x, self.max_slices[1] - 1))
            new_axi = max(0, min(y, self.max_slices[2] - 1))
            self.coronal_slider.setValue(new_cor)
            self.axial_slider.setValue(new_axi)
        elif axis == 1:  # Coronal view (X-Z plane)
            # x -> sagittal position, y -> axial position
            new_sag = max(0, min(x, self.max_slices[0] - 1))
            new_axi = max(0, min(y, self.max_slices[2] - 1))
            self.sagittal_slider.setValue(new_sag)
            self.axial_slider.setValue(new_axi)
        else:  # axis == 2: Axial view (X-Y plane)
            # x -> sagittal position, y -> coronal position
            new_sag = max(0, min(x, self.max_slices[0] - 1))
            new_cor = max(0, min(y, self.max_slices[1] - 1))
            self.sagittal_slider.setValue(new_sag)
            self.coronal_slider.setValue(new_cor)

    def _update_all_views(self) -> None:
        """Update all 2D slice views with crosshairs."""
        sag, cor, axi = self.slice_indices

        # Sagittal view (Y-Z plane): crosshairs at coronal (h) and axial (v) positions
        self.sagittal_canvas.set_crosshairs(h_pos=axi, v_pos=cor)
        self.sagittal_canvas.update_slice(sag, axis=0)

        # Coronal view (X-Z plane): crosshairs at axial (h) and sagittal (v) positions
        self.coronal_canvas.set_crosshairs(h_pos=axi, v_pos=sag)
        self.coronal_canvas.update_slice(cor, axis=1)

        # Axial view (X-Y plane): crosshairs at coronal (h) and sagittal (v) positions
        self.axial_canvas.set_crosshairs(h_pos=cor, v_pos=sag)
        self.axial_canvas.update_slice(axi, axis=2)

    def _update_info(self, data: np.ndarray, modality: str) -> None:
        """Update the info panel.

        Args:
            data: Volume data
            modality: Modality name
        """
        info_text = f"""
<b>Modality:</b> {modality}<br>
<b>Shape:</b> {data.shape}<br>
<b>Dtype:</b> {data.dtype}<br>
<b>Min:</b> {data.min():.2f}<br>
<b>Max:</b> {data.max():.2f}<br>
<b>Mean:</b> {data.mean():.2f}<br>
<b>Std:</b> {data.std():.2f}<br>
<br>
<b>Loaded modalities:</b><br>
{', '.join(self.modalities.keys())}
        """
        self.info_label.setText(info_text.strip())


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    viewer = NiftiViewer()
    viewer.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
