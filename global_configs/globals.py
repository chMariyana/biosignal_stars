# CFG file
from pathlib import Path

import os

####################################################################
# DIRECTORIES
####################################################################
PROJECT_DIR = Path(os.path.abspath(os.path.curdir)).resolve().parent
OFFLINE_CONFIG_DIR = os.path.join(PROJECT_DIR, 'offline', 'configs')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PREP_OUTPUT_DATA_DIR = os.path.join(DATA_DIR, 'preprocess_output')
GLOBAL_CONFIG_FILE = os.path.join(PROJECT_DIR, 'global_configs', 'custom_config.yml')
####################################################################
# Strings
####################################################################
BASELINE_LENGTH = 'baseline_length'
TARGET_LENGTH = 'target_length'
HIGHLIGHT_LENGTH = 'highlight_length'
INTER_HIGHLIGHT_LENGTH = 'inter_highlight_length'
NUM_HIGHLIGHTS = 'num_highlights'
TARGET_DOWN = 'target_down'
TARGET_LEFT = 'target_left'
TARGET_RIGHT = 'target_right'
TARGET_UP = 'target_up'
HIGHLIGHT_TARGET_FALSE = 'highlight_target_false'
HIGHLIGHT_TARGET_TRUE = 'highlight_target_true'

ALL_TARGETS = {TARGET_DOWN: 7,
               TARGET_LEFT: 8,
               TARGET_RIGHT: 9,
               TARGET_UP: 10
               }
