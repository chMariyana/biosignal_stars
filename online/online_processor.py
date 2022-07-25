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


class OnlineProcessor:
    """
    A class to be used in online processing of the data. Given a chunk of data (one trial) and the corresponding
     event array and ids , handles preprocessing and classification.
    """
    def __init__(self, global_config_file: str = GLOB.GLOBAL_CONFIG_FILE):
        """

        :param global_config_file:
        """
        self.global_config = yaml.load(global_config_file, Loader=yaml.FullLoader)
        self.preprocess_config = self.global_config['preprocess']
        self.online_config = self.global_config['online']
        self.X_all = None
        self.highlights_labels_all = None
        self.predictions = []

        with open(os.path.join(GLOB.OFFLINE_CONFIG_DIR, self.global_config['experiment_file']), 'r') as opened_file:
            self.params = json.load(opened_file)

        # Now we want to epoch our data to trials and do a baseline correction.
        self.total_trial_duration = self.params[GLOB.BASELINE_LENGTH] + self.params[GLOB.TARGET_LENGTH] + \
                                    self.params[GLOB.HIGHLIGHT_LENGTH] * self.params[GLOB.NUM_HIGHLIGHTS] + \
                                    self.params[GLOB.INTER_HIGHLIGHT_LENGTH] * (self.params[GLOB.NUM_HIGHLIGHTS])

        model_path = os.path.join(GLOB.MODEL_DATA_DIR, self.online_config['model_file'])
        with open(model_path, mode='rb') as opened_file:
            self.clf = pickle.load(opened_file)

    def preprocess(self, raw, event_arr: np.ndarray, event_id: dict):
        """

        :param raw: input data
        :param event_arr: ndarray of events timestamps and event ids
        :param event_id: dict name of events are the keys and corresponding int ids as values
        :return: (evoked data, highlights sequence)
        """

        highlights_per_trial = self.params['num_highlights']

        # Implement the band-pass filter
        flow, fhigh = self.preprocess_config['min_pass_freq'], self.preprocess_config['max_pass_freq']
        raw_filt = raw.filter(flow, fhigh)

        # if preprocess_config['downsample_frequency']:
        #     raw_filt = raw_filt.resample(sfreq=preprocess_config['downsample_frequency'])

        # # Apply Common Average Referencing.
        # if preprocess_config['apply_car']:
        # raw_filt, _ = mne.set_eeg_reference(raw_filt, 'average', copy=True, ch_type='eeg')

        print(f"getting avg evokeds: ")

        print(f"{self.total_trial_duration, highlights_per_trial, self.preprocess_config['decim_factor']}")
        highlights_final_evoked, targets_final, targets_seq, highlights_labels, highlights_seq, t_samples, \
        h_epochs, labels_epochs = get_avg_evoked(raw_filt,
                                                 event_arr.copy(),
                                                 event_id.copy(),
                                                 self.total_trial_duration,
                                                 highlights_per_trial,
                                                 decim_facor=self.preprocess_config['decim_factor'],
                                                 t_min=self.preprocess_config['signal_duration_min'],
                                                 t_max=self.preprocess_config['signal_duration_min'],
                                                 )

        chnum, size = highlights_final_evoked[0].get_data().shape
        x = np.concatenate([x.get_data().reshape(1, chnum, size) for x in highlights_final_evoked])
        y = np.zeros(len(highlights_final_evoked
                         ))
        y[t_samples] = 1
        full_data = (
            highlights_final_evoked, targets_final, targets_seq, highlights_labels, highlights_seq, t_samples, h_epochs,
            labels_epochs)
        #
        # with open(output_file, 'wb') as opened_file:
        #     pickle.dump((x, y, full_data), opened_file)

        return x, highlights_labels

    def classify_single_trial(self, trial_data: np.ndarray, highlights_labels: np.ndarray,):
        """

        :param trial_data: ndarray of size (num arrow types, num channels)
        :param highlights_labels: ndarray of arrow types, size (num arrow types, )
        :return: int label, predicted direction
        """

        inds_selected_ch = self.online_config['ind_selected_channels']
        # reshape data and extract features
        X = trial_data[:, inds_selected_ch, :].reshape(trial_data.shape[0], -1)
        # do some more stuff if needed
        trial_predictions = self.clf.predict(X)
        pred_label = highlights_labels[np.argmax(trial_predictions)]
        self.predictions.append(pred_label)
        self.X_all = np.concatenate([self.X_all, X])

        return pred_label

    def get_prediction(self, raw_trial, event_arr: np.ndarray, event_id: dict):
        """

        :param raw_trial: input data
        :param event_arr: ndarray of events timestamps and event ids
        :param event_id: dict name of events are the keys and corresponding int ids as values
        :return: predicted label (arrow direction)
        """
        prep_data, highlights_labels =  self.preprocess(raw_trial, event_arr, event_id)

        return self.classify_single_trial(prep_data, highlights_labels)



