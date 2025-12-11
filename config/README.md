# Configuration

Centralized path configuration for the project.

## Usage

```python
from config import PathConfig

config = PathConfig()
config.data_dir            # Base dataset directory
config.brainmet_train_dir  # Training data
config.brainmet_test_dir   # Test data
```

## Environment Detection

PathConfig auto-detects the environment:
- **Cluster**: `/cluster/work/modestas/MedicalDataSets`
- **Local**: `~/NTNU/MedicalDataSets`

Override with `MEDICAL_DATA_DIR` environment variable if needed.
