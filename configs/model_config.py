"""
configs/model_config.py

Torchvision Mask R-CNN configuration for dental panoramic X-ray segmentation.

Dataset: AKUDENTAL — 333 panoramic X-rays
Classes: Binary (1 class: tooth) first, then 35 FDI classes
"""

BINARY       = False         # True = 1 class "tooth", False = 35 FDI classes
NUM_CLASSES  = 36            # background + 1 tooth (binary)
                             # set to 36 for 35 FDI classes
# torchvision handles resize internally — 800 min  size is default
IMAGE_MIN_SIZE = 800        # shorter side
IMAGE_MAX_SIZE = 1333       # longer side — torchvision default
EPOCHS          = 30
BATCH_SIZE      = 2         # increase if VRAM allows
NUM_WORKERS     = 4
LR              = 0.005     # SGD learning rate
MOMENTUM        = 0.9
WEIGHT_DECAY    = 0.0005
LR_STEP_SIZE    = 10        # decay LR every N epochs
LR_GAMMA        = 0.1       # multiply LR by this at each step

# Higher threshold, higher precision, but lower recall (might miss true ones)
CONF_THRESHOLD  = 0.3 #0.5       # minimum score to show detection
# tooth are naturally closer, so better higher number (as adjescent teeth can overlap)
NMS_THRESHOLD   = 0.5 #0.3       # IOU threshold for NMS
MAX_DETECTIONS  = 45        # max instances per image

# After resize to ~800px shorter side, teeth are ~50-150px
ANCHOR_SIZES    = ((32,), (64,), (128,), (256,), (512,))
ANCHOR_RATIOS   = ((0.5, 1.0, 2.0),) * 5

# Class names 
BINARY_CLASSES = ['__background__', 'tooth']

FDI_CLASSES = [
    '__background__',
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
    'Bridge', 'Filling-Crown', 'Implant',
]