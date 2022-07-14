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
import yaml
from global_configs import globals as GLOB
from offline.utils import load_xdf_data, get_highlight_trials_class_based, get_labeled_dataset

SIGNAL_DURATION = .650

# TODO: Once it is clear how the raw data will be stored, the script must be tested !
# TODO: decide how to do the baseline correction

def preprocess(global_config_dict: dict):

    # Load the configurations of the paradigm
    with open(os.path.join(GLOB.OFFLINE_CONFIG_DIR, global_config_dict['experiment_file']), 'r') as opened_file:
        params = json.load(opened_file)
    preprocess_config =  global_config_dict['preprocess']

    file = os.path.join(GLOB.DATA_DIR, 'raw', 'trial_data.xdf')
    if preprocess_config['raw_data_dir'] == 'default':
        trial_data_dir = ''
    else:
        trial_data_dir = preprocess_config['raw_data_dir']

    epochs_list = []
    for trial_no, trial in enumerate(sorted(os.listdir(trial_data_dir))):
        print("TRIAL NO ", trial_no)
        raw, event_arr, event_id = load_xdf_data(file=os.path.join(trial_data_dir, trial))

        # Implement the band-pass filter
        flow, fhigh = preprocess_config['min_pass_freq'], preprocess_config['max_pass_freq']
        raw_filt = raw.filter(flow, fhigh)

        # Apply Common Average Referencing.
        raw_car, _ = mne.set_eeg_reference(raw_filt, 'average', copy=True, ch_type='eeg')

        # Now we want to epoch our data to trials and do a baseline correction.
        total_trial_duration = params[GLOB.BASELINE_LENGTH] + params[GLOB.TARGET_LENGTH] + \
                               params[GLOB.HIGHLIGHT_LENGTH] * params[GLOB.NUM_HIGHLIGHTS] + \
                               params[GLOB.INTER_HIGHLIGHT_LENGTH] * (params[GLOB.NUM_HIGHLIGHTS] - 1)
        tmin, tmax = -params[GLOB.BASELINE_LENGTH], total_trial_duration
        epochs_list.append(mne.Epochs(raw_car,
                                      events=event_arr,
                                      event_id=event_id,
                                      tmin=tmin,
                                      tmax=tmax,
                                      baseline=(-params[GLOB.BASELINE_LENGTH], 0),
                                      # We apply baseline correction here !!!!!!!!!!!!!
                                      preload=True,
                                      event_repeated='drop'))
        # get epochs for highlights based on target shown
        epochs_highlights_targets, epochs_highlights_notargets = get_highlight_trials_class_based(raw_car, event_arr,
                                                                                                  event_id)
        # get data as nd array and label vector
        x_all, y_all = get_labeled_dataset(epochs_highlights_targets, epochs_highlights_notargets)
        # create name for the current output file and save
        output_file = os.path.join(GLOB.PREP_OUTPUT_DATA_DIR, f'preprocessed_trial_{trial_no}')
        with open(output_file, 'wb') as opened_file:
            pickle.dump((x_all, y_all), opened_file)

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
    with open(os.path.join(GLOB.DATA_DIR, 'sub_epochs', 'sub_epochs_trial.pkl'), 'wb') as opened_file:
        # Save the sub-epochs for signal processing
        pickle.dump(sub_epochs, opened_file)


if __name__ == '__main__':
    with open(GLOB.GLOBAL_CONFIG_FILE) as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)
    preprocess(global_config_dict=config)