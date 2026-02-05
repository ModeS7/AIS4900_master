"""Constants for figure generation and visualization.

Centralized constants used by figures.py and related visualization modules.
"""

# Figure DPI settings
DEFAULT_FIGURE_DPI = 150

# Figure sizing (width per sample, height)
SINGLE_FIGURE_WIDTH_PER_SAMPLE = 2.5
SINGLE_FIGURE_HEIGHT = 7
DUAL_FIGURE_HEIGHT = 14

# Figure margins
FIGURE_LEFT_MARGIN = 0.05
FIGURE_RIGHT_MARGIN = 0.98
FIGURE_TOP_MARGIN = 0.92
FIGURE_BOTTOM_MARGIN = 0.02

# Font sizes
FIGURE_TITLE_FONTSIZE = 8
FIGURE_LABEL_FONTSIZE = 9
FIGURE_SUPTITLE_FONTSIZE = 10

# Spacing
FIGURE_HSPACE = 0.02
FIGURE_WSPACE = 0.02

# Mask overlay settings
MASK_CONTOUR_LINEWIDTH = 0.5
MASK_CONTOUR_ALPHA = 0.7
MASK_CONTOUR_COLOR = 'red'

# Colormaps
GRAYSCALE_COLORMAP = 'gray'
DIFFERENCE_HEATMAP_COLORMAP = 'hot'

# Image value range
IMAGE_VALUE_MIN = 0
IMAGE_VALUE_MAX = 1

# Default samples
DEFAULT_MAX_SAMPLES_PER_FIGURE = 8
