"""Shared constants for the YOLO background augmentation tool."""

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

# Color quantization bin size for background estimation
COLOR_QUANT_BIN_SIZE = 16
COLOR_QUANT_BINS = 4096  # 16^3

# Aspect ratio threshold for filtering border-touching contours
ASPECT_RATIO_THRESHOLD = 5.0

# Border tolerance for detecting edge-touching contours
BORDER_TOLERANCE = 3

# Minimum tries for biased placement
MIN_BIASED_PLACEMENT_TRIES = 10

# Maximum consecutive failures before giving up on a class
MAX_AUGMENTATION_FAILURES = 25

# Default grid size for heatmaps
DEFAULT_HEATMAP_GRID_SIZE = 5

# Spatial grid cell size for collision detection
SPATIAL_GRID_CELL_SIZE = 64

# Default maximum placement tries for random placement
DEFAULT_MAX_PLACEMENT_TRIES = 50

# Probability of adding a random seg blob during augmentation
SEG_FILL_PROBABILITY = 0.5
