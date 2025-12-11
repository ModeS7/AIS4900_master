# NIfTI 3D Viewer

A PyQt5 application for viewing 3D NIfTI medical images with volume rendering.

## Features

- **Multi-modality support**: Auto-detects and loads all NIfTI files (bravo, seg, t1_pre, t1_gd, flair)
- **Three orthogonal 2D views**: Axial, Coronal, and Sagittal slices with synchronized navigation
- **3D volume rendering**: Real-time volume visualization using PyVista/VTK
- **Click-to-navigate**: Click on any 2D view to update slice positions
- **Zoom/pan**: Scroll to zoom, middle-click to pan in 2D views
- **Fullscreen 3D mode**: Double-click the 3D view for immersive fullscreen with controls
- **Patient navigation**: Browse through multiple patients in a directory

## Controls

### 3D View Controls
| Control | Description |
|---------|-------------|
| Contrast Min/Max | Adjust intensity window |
| Window Preset | Quick presets (Full Range, Brain, Soft Tissue, Bone, Tumor) |
| Colormap | gray, bone, hot, viridis, magma, plasma |
| Opacity | soft, medium, strong |
| Enable Shading | Adds 3D lighting effects |
| Seg Opacity | Show/hide segmentation overlay |
| Seg Always Visible | Render segmentation on top of volume |
| Background | Adjust background from black to white |

### 2D View Controls
| Action | Effect |
|--------|--------|
| Left click | Navigate to clicked position |
| Scroll | Zoom in/out |
| Middle-click drag | Pan |
| Double-click | Reset zoom |

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| ESC | Exit fullscreen mode |

## Requirements

```
nibabel
numpy
matplotlib
PyQt5
pyvista
pyvistaqt
```

## Installation

```bash
pip install nibabel numpy matplotlib PyQt5 pyvista pyvistaqt
```

## Usage

```bash
python nifti_viewer.py
```

1. Click **Browse** to select a patient directory containing `.nii.gz` files
2. Use the modality dropdown to switch between available scans
3. Adjust slices using the bottom sliders or click on 2D views
4. Double-click the 3D view for fullscreen mode
5. Use **< Prev** / **Next >** buttons to navigate between patients

## Expected Directory Structure

```
patient_directory/
├── bravo.nii.gz      # T1-weighted structural
├── seg.nii.gz        # Segmentation mask
├── t1_pre.nii.gz     # T1 pre-contrast
├── t1_gd.nii.gz      # T1 post-gadolinium
└── flair.nii.gz      # FLAIR sequence
```

Any `.nii.gz` or `.nii` files in the directory will be loaded as modalities.
