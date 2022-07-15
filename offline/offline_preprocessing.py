# %%
'''
Preprocessing of the P300 data.

Author:
    Karahan Yilmazer - 16.04.2022
    Nathan van Beelen
    Ayça Kepçe

'''

# !%matplotlib qt

import os
import glob
import mne
import pickle
import json
from utils import load_xdf_data

PROJECT_DIR = os.path.abspath(os.path.curdir)
OFFLINE_CONFIG_DIR = os.path.join(PROJECT_DIR, 'offline', 'configs')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

SIGNAL_DURATION = .650

# %%
# Load the configurations of the paradigm
with open(os.path.join(OFFLINE_CONFIG_DIR, 'config_short_red-yellow.json'), 'r') as opened_file:
    params = json.load(opened_file)

file = os.path.join(DATA_DIR, 'raw', 'trial_data.xdf')

epochs_list = []
for trial_no, trial in enumerate(sorted(glob.glob(file))):
    print("TRIAL NO ", trial_no)
    raw, event_arr, event_id = load_xdf_data(file=trial)

    # Implement the band-pass filter
    flow, fhigh = 0.1, 10
    raw_filt = raw.filter(flow, fhigh)

    # Apply Common Average Referencing.
    raw_car, _ = mne.set_eeg_reference(raw_filt, 'average', copy=True, ch_type='eeg')

    # Now we want to epoch our data to trials and do a baseline correction.
    total_trial_duration = params['baseline_length'] + params['target_length'] + \
                           params['highlight_length'] * params['num_highlights'] + \
                           params['inter_highlight_length'] * (params['num_highlights'] - 1)
    tmin, tmax = -params['baseline_length'], total_trial_duration
    epochs_list.append(mne.Epochs(raw_car,
                                  events=event_arr,
                                  event_id=event_id,
                                  tmin=tmin,
                                  tmax=tmax,
                                  baseline=(-params['baseline_length'], 0),
                                  # We apply baseline correction here !!!!!!!!!!!!!
                                  preload=True,
                                  event_repeated='drop'))

# Concatenate the epochs
epochs = mne.concatenate_epochs(epochs_list)

# Get a subselection of epochs that resembles the trials
sub_epochs_trials = epochs['trial_begin']

# Now we epoch the trials to get highlights
tmin, tmax = 0, SIGNAL_DURATION  # exp_durations['highlight_length']
epochs_highlights = mne.Epochs(raw_car,
                               events=event_arr,
                               event_id=event_id,
                               tmin=tmin,
                               tmax=tmax,
                               baseline=None,  # we don't apply baseline correction here !!!!!!!!!!!!!
                               preload=True,
                               event_repeated='drop')

# Get the two classes of epochs that involves the P300 response and not
sub_epochs = epochs_highlights['highlight_down', 'highlight_up']

# After this we should exclude bad epochs. Click on the epochs in the window to drop them. Close the window to save.
sub_epochs.plot(scalings='auto', butterfly=True)

# We can also check epochs with the PSD.
sub_epochs.plot_psd()

# %%
with open(os.path.join(DATA_DIR, 'sub_epochs', 'sub_epochs_trial.pkl'), 'wb') as opened_file:
    # Save the sub-epochs for signal processing
    pickle.dump(sub_epochs, opened_file)
