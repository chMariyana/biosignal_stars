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
import numpy as np
import mne
import pickle
import json
import yaml
from pathlib import Path

from global_configs import globals as GLOB
from offline.utils import load_xdf_data, get_avg_evoked, get_highlight_trials_class_based, get_labeled_dataset

SIGNAL_DURATION = .650

# TODO: Once it is clear how the raw data will be stored, the script must be tested !
# TODO: decide how to do the baseline correction


def preprocess(global_config_dict: dict):
    # Load the configurations of the paradigm
    with open(os.path.join(GLOB.OFFLINE_CONFIG_DIR, global_config_dict['experiment_file']), 'r') as opened_file:
        params = json.load(opened_file)
    preprocess_config = global_config_dict['preprocess']

    session_dir = os.path.join(GLOB.DATA_DIR, 'raw', preprocess_config['session_dir'], 'eeg')
    # if preprocess_config['raw_data_dir'] == 'default':
    highlights_per_trial = params['num_highlights']

    epochs_list = []
    for run_no, run in enumerate(sorted(os.listdir(session_dir))):
        print("Run NO ", run_no)
        try:
            raw, event_arr, event_id = load_xdf_data(file=os.path.join(session_dir, run))
        except Exception as e:
            print(f'Could not read file {run}, continuing on next one.')
            continue

        # Implement the band-pass filter
        flow, fhigh = preprocess_config['min_pass_freq'], preprocess_config['max_pass_freq']
        raw_filt = raw.filter(flow, fhigh)

        # Now we want to epoch our data to trials and do a baseline correction.
        total_trial_duration = params[GLOB.BASELINE_LENGTH] + params[GLOB.TARGET_LENGTH] + \
                               params[GLOB.HIGHLIGHT_LENGTH] * params[GLOB.NUM_HIGHLIGHTS] + \
                               params[GLOB.INTER_HIGHLIGHT_LENGTH] * (params[GLOB.NUM_HIGHLIGHTS] )
        print(f"getting avg evokeds: ")

        print(f"{total_trial_duration,highlights_per_trial,preprocess_config['decim_factor']}")
        highlights_final_evoked, targets_final, targets_seq, highlights_labels, highlights_seq, t_samples, h_epochs, labels_epochs=\
            get_avg_evoked(raw_filt,
                           event_arr.copy(),
                           event_id.copy(),
                           total_trial_duration,
                           highlights_per_trial,
                           decim_facor=preprocess_config['decim_factor'],
                           t_min=preprocess_config['signal_duration_min'],
                           t_max=preprocess_config['signal_duration_max']
                           )
        chnum, size = highlights_final_evoked[0].get_data().shape
        x = np.concatenate([x.get_data().reshape(1, chnum, size) for x in highlights_final_evoked])
        y = np.zeros(len(highlights_final_evoked
                         ))
        y[t_samples] = 1
        # # create name for the current output file and save
        output_file = os.path.join(GLOB.PREP_OUTPUT_DATA_DIR, preprocess_config['session_dir'],
                                   f'preprocessed_{run[:-3]}pickle')
        Path(os.path.join(GLOB.PREP_OUTPUT_DATA_DIR, preprocess_config['session_dir'])).mkdir(parents=True, exist_ok=True)
        full_data = (highlights_final_evoked, targets_final, targets_seq, highlights_labels, highlights_seq, t_samples, h_epochs, labels_epochs)

        with open(output_file, 'wb') as opened_file:
            pickle.dump((x, y, full_data), opened_file)

    # Concatenate the epochs
    # epochs = mne.concatenate_epochs(epochs_list)
    #
    # # Get a subselection of epochs that resembles the trials
    # sub_epochs_trials = epochs['trial_begin']
    #
    # # Now we epoch the trials to get highlights
    # tmin, tmax = 0, SIGNAL_DURATION  # exp_durations['highlight_length']
    # epochs_highlights = mne.Epochs(raw_car,
    #                                events=event_arr,
    #                                event_id=event_id,
    #                                tmin=tmin,
    #                                tmax=tmax,
    #                                baseline=None,  # we don't apply baseline correction here !!!!!!!!!!!!!
    #                                preload=True,
    #                                event_repeated='drop')
    #
    # # Get the two classes of epochs that involves the P300 response and not
    # sub_epochs = epochs_highlights['highlight_down', 'highlight_up']
    #
    # # After this we should exclude bad epochs. Click on the epochs in the window to drop them. Close the window to save.
    # sub_epochs.plot(scalings='auto', butterfly=True)
    #
    # # We can also check epochs with the PSD.
    # sub_epochs.plot_psd()
    #
    # # %%
    # with open(os.path.join(GLOB.DATA_DIR, 'sub_epochs', 'sub_epochs_trial.pkl'), 'wb') as opened_file:
    #     # Save the sub-epochs for signal processing
    #     pickle.dump(sub_epochs, opened_file)
    #

if __name__ == '__main__':
    with open(GLOB.GLOBAL_CONFIG_FILE) as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)
    preprocess(global_config_dict=config)