# %%
'''
Preprocessing of the P300 data.

Author:
    Karahan Yilmazer - 16.04.2022
    Nathan van Beelen
    Ayça Kepçe

'''

# !%matplotlib qt

import glob
import mne
import pickle
from utils import load_xdf_data


SIGNAL_DURATION = .650
# %%
# Load the configurations of the paradigm
with open(r'./config_durations.pkl', 'rb') as file:
    exp_durations = pickle.load(file)

# file = r"r'C:\Users\AYCA\PycharmProjects\biosignal_stars\data\\raw\\**\\*.xdf"
file = r'C:\Users\AYCA\PycharmProjects\biosignal_stars\data\raw\*.xdf'

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
    total_trial_duration = exp_durations['baseline_length'] + exp_durations['target_length'] + \
                           exp_durations['highlight_length'] * exp_durations['num_highlights'] + \
                           exp_durations['inter_highlight_length'] * (exp_durations['num_highlights'] - 1)
    tmin, tmax = -exp_durations['baseline_length'], total_trial_duration
    epochs_list.append(mne.Epochs(raw_car,
                                  events=event_arr,
                                  event_id=event_id,
                                  tmin=tmin,
                                  tmax=tmax,
                                  baseline=(-exp_durations['baseline_length'], 0),
                                  # We apply baseline correction here !!!!!!!!!!!!!
                                  preload=True,
                                  event_repeated='drop'))

# Concatenate the epochs
epochs = mne.concatenate_epochs(epochs_list)

# Get a subselection of epochs that resembles the trials
sub_epochs_trials = epochs['trial_begin']

# Now we epoch the trials to get highlights
tmin, tmax = 0, SIGNAL_DURATION # exp_durations['highlight_length']
epochs_highlights = mne.Epochs(raw_car,
                               events=event_arr,
                               event_id=event_id,
                               tmin=tmin,
                               tmax=tmax,
                               baseline=None,  # we don't apply baseline correction here !!!!!!!!!!!!!
                               preload=True,
                               event_repeated='drop')

# Get the two classes of epochs that involves the P300 response and not
sub_epochs = epochs['highlight_down', 'highlight_up']

# After this we should exclude bad epochs. Click on the epochs in the window to drop them. Close the window to save.
sub_epochs.plot(scalings='auto', butterfly=True)

# We can also check epochs with the PSD.
sub_epochs.plot_psd()

# %%
with open(r'C:\Users\AYCA\PycharmProjects\biosignal_stars\data\sub_epochs\sub_epochs_trial.pkl', 'wb') as file:
    # Save the sub-epochs for signal processing
    pickle.dump(sub_epochs, file)
