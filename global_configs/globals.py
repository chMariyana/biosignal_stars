# CFG file
import sys
import os

####################################################################
# DIRECTORIES
####################################################################
PROJECT_DIR = os.path.abspath(os.path.curdir)
OFFLINE_CONFIG_DIR = os.path.join(PROJECT_DIR, 'offline/configs')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PREP_OUTPUT_DATA_DIR = os.path.join(DATA_DIR, 'preprocess_output')
GLOBAL_CONFIG_FILE = os.path.join(PROJECT_DIR, 'global_configs/')
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

ALL_TARGETS = {TARGET_DOWN: 7,
               TARGET_LEFT: 8,
               TARGET_RIGHT: 9,
               TARGET_UP: 10}