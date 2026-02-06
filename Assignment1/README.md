# Assignment 1 - Image Channel Alignment

## Requirements

- Python 3.7 or higher
- Required libraries:
  - `numpy`
  - `scikit-image`

## Installation

Install the required libraries using pip:

```bash
pip install numpy scikit-image
```

Or if using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy scikit-image
```

## Usage

1. Add your image file paths to the `pyramid_align_image_files` list in `main.py`:

```python
pyramid_align_image_files = [
    'coms4732_hw1_data/cathedral.jpg',
    'coms4732_hw1_data/monastery.jpg',
    # Add your image paths here
]
```

2. Run the script:

```bash
python main.py
```

The aligned images will be saved in the `output/` directory with the suffix `_pyramid_aligned.jpg`.

