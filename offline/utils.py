
import numpy as np

import mne
import pyxdf
import re
import mne.decoding.csp
import pickle
from datetime import datetime, timezone

from sklearn.preprocessing import LabelEncoder
from mne.time_frequency import psd_welch
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector


def load_xdf_data(file, with_stim=False):

    # Read the XDF file
    streams, header = pyxdf.load_xdf(file)

    # Initialize lists for the streams
    marker_streams = []
    data_stream = []

    for stream in streams:

        # Get the relevant values from the stream
        stream_name = stream['info']['name'][0].lower()
        stream_type = stream['info']['type'][0].lower()

        # Assign streams to their corresponding lists
        if 'marker' in stream_type:
            marker_streams.append(stream)
            print(f'{stream_name} --> marker stream')

        elif stream_type in ('eeg', 'data'):
            data_stream.append(stream)
            print(f'{stream_name} --> data stream')

        else:
            print(f'{stream_name} --> not sure what to do with this')

    # Check whether there is only one data stream found
    if len(data_stream) == 1:
        data_stream = data_stream[0]
    elif len(data_stream) == 0:
        raise Exception('No data stream found!')
    else:
        raise Exception('Multiple data streams found!')

    # Get the sampling frequency
    sfreq = float(data_stream['info']['nominal_srate'][0])

    # Get the data --> only the first 8 rows are for the electrodes
    data = data_stream["time_series"].T[:14]
    # Set the channel names
    ch_names =['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    #['Fz', 'T3', 'Cz', 'T4', 'Pz', 'Oz', 'C3', 'C4']  # Our config

    # Define the channel types
    ch_types = ['eeg'] * len(ch_names)

    # Get the time stamps of the EEG recording
    raw_time = data_stream["time_stamps"]

    # Check if there are available marker streams
    if marker_streams:
        for stream in marker_streams:

            # Get the cues
            cues = stream['time_series']

            # If there are no cues
            if not cues:
                # Skip the stream
                print(f"Skipping {stream['info']['name'][0]}\n")
                continue

            # Create a new list consisting of only cue strings
            # since cues is normally a list of lists containing one string
            tmp_list = []
            for cue in cues:
                # Discard the local clock
                # e.g.: 'cue_rest-1649695139.3753629-3966.48780' --> 'cue_rest'
                if '-' in cue[0]:
                    tmp_str = cue[0].split('-')[0]

                # Discard the number of occurence at the end of cues
                # e.g.: ['block_begin_1'] --> 'block_begin'
                tmp_str = re.split(r'_\d+$', tmp_str)[0]

                tmp_list.append(tmp_str)
            # Overwrite cues with the newly created list
            cues = tmp_list

            # Use the timestamps from the messages as they are more reliable.
            cue_times = [float(i[0].split("-")[1]) for i in stream['time_series']]

            # The marker stream and the data stream use different timestamps.
            # The marker stream will always use Unix time, while the Unicorn
            # stream is based on Windows timing which is the amount of time
            # since boot (plus an offset sometimes). Fortunately, the offset
            # between the Unix time and the Unicorn time is recorded. We use
            # this to convert all raw timestamps to Unix timestamps.
            # NOTE: WE ASSUME WE ONLY HAVE ONE MARKER STREAM

            # Get time offset from the start of the experiment.
            # We only use the first offset because the drift is reasonably small.
            time_offset = float(stream['footer']['info']['clock_offsets'][0]['offset'][0]['value'][0])
            # Sometimes the first offset calculation doesn't pick up sensible values,
            # check if that's the case and use the next offset instead.
            if time_offset == 0:
                time_offset = float(stream['footer']['info']['clock_offsets'][0]['offset'][1]['value'][0])

            # Convert all raw timestamps to the Unix format.
            raw_time = raw_time - time_offset
            # Get the smallest time stamp of both the data and marker stream
            offset = min(cue_times[0], raw_time[0])
            # Convert the corrected time stamps into indices, which are are basically the index
            # of the EEG sample that the cue took place.
            cue_indices = (np.atleast_1d(cue_times) - offset) * sfreq
            cue_indices = cue_indices.astype(int)

            # Initialize the label encoder
            le = LabelEncoder()
            # Encode the cues using the label encoder
            cues_encoded = le.fit_transform(cues)

            if with_stim:
                # Initalize the STIM channel
                stim = np.zeros(data[0].shape)
                stim[cue_indices] = cues_encoded + 1

                # Append the STIM channel to the EEG data array
                data = np.concatenate((data, stim.reshape(1, -1)))
                # Add stim to the channel types
                ch_types.append('stim')
                # Add STIM as a channel
                ch_names.append('STIM')

    info = mne.create_info(ch_names, sfreq, ch_types)

    # Add the system name to the info
    info['description'] = 'Unicorn Hybrid Black'

    raw = mne.io.RawArray(data, info)

    # Set the measurement date
    tmp_dt = datetime.strptime(header['info']['datetime'][0],
                               "%Y-%m-%dT%H:%M:%S%z")
    tmp_dt = tmp_dt.astimezone(timezone.utc)
    raw.set_meas_date(tmp_dt)

    # Create a montage out of the 10-20 system
    montage = mne.channels.make_standard_montage('standard_1020')

    # Apply the montage
    raw.set_montage(montage)

    # A line to supress an error message
    raw._filenames = [file]

    """# Convert time stamps of the Raw object to indices
    raw_indices = raw.time_as_index(times=raw.times)

    # Raise and error if the cue index is larger than the maximum
    # index determined by the EEG recording
    if cue_indices.max() > raw_indices.max():
        raise Exception(
            'Cue index is larger than the largest sample index of the Raw object!'
        )"""

    # Initialize the event array
    event_arr = np.zeros((len(cue_indices), 3), dtype=int)

    # Store the event information in an array of shape (len(cues), 3)
    event_arr[:, 0] = cue_indices
    event_arr[:, 2] = cues_encoded

    # Create a class-encoding correspondence dictionary for the Epochs object
    event_id = dict(zip(list(le.classes_), range(len(le.classes_))))

    return raw, event_arr, event_id


def apply_filter_bank(signal, frequency_bands, sfreq=250):
    # Function to apply filter bank to the signal
    filter_bank = []
    for fmin, fmax in frequency_bands.values():
        filter_bank.append(mne.filter.filter_data(signal,
                                                  sfreq,
                                                  fmin,
                                                  fmax,
                                                  pad='symmetric'))
    return filter_bank


def chunk_data(epochs, time_length, baseline_margin):
    # Chunk the data into given time length

    chunked_data = []
    for i in range(int(2 / time_length)):
        t_min = np.round(i * (time_length) - baseline_margin, 1)
        t_max = np.round((i + 1) * (time_length), 1)
        epoch_i = epochs.copy().crop(tmin=t_min, tmax=t_max, include_tmax=True).shift_time(tshift=-baseline_margin,
                                                                                           relative=False)
        chunked_data.append(epoch_i)
    return chunked_data


def process_features(numpy_epochs, epochs, labels, csp=None):
    """
    Funtion that extracts the variance, band powers, and CSP features. """

    # X is the feature matrix with shape of (n_trials, n_features)
    print(f'Epochs shape: {numpy_epochs.shape}.')
    X = numpy_epochs[:, :, :].var(axis=2)

    # specific frequency bands
    FREQ_BANDS = {"theta": [4, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    # FFT length and zero-padding is defined
    power_of_2 = int(np.log2(numpy_epochs.shape[2]))
    n_per_seg = 2 ** power_of_2
    n_fft = 128
    psds, freqs = psd_welch(epochs, n_fft=n_fft, n_per_seg=n_per_seg, picks='eeg', fmin=4, fmax=30.)

    # Normalize the PSDs
    # Output is of shape (samples, channels, frequencies).
    psds /= np.sum(psds, axis=-1, keepdims=True)

    # A list for all frequencies, each containing an array of shape (samples, channels)
    powers = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        powers.append(psds_band.reshape(len(psds), -1))
    powers = np.array(powers)
    powers = powers.reshape(-1, powers.shape[1])

    X = np.concatenate((X, powers.T), axis=1)  # We add the power to the feature matrix (see tutorial 4).

    # Filter bank CSP
    filter_bank = apply_filter_bank(numpy_epochs, FREQ_BANDS)
    # We are running the trainig data, thus we calculate csp
    if csp == None:
        for i in np.arange(len(filter_bank)):
            try:
                csp = mne.decoding.CSP()
                features = csp.fit_transform(filter_bank[i], y=labels)
                np.concatenate((X, features), axis=1)  # We add the features extracted by CSP to the feature matrix.
            except:
                csp = mne.decoding.CSP(reg='pca', rank='full')
                features = csp.fit_transform(filter_bank[i], y=labels)
                np.concatenate((X, features), axis=1)  # We add the features extracted by CSP to the feature matrix.

    # We are running the test data, so we transform using the train CSP.
    else:
        features = csp.transform(numpy_epochs)

    return X, csp


def epoching(sub_epochs):
    """
    Function to create the dataset in the form of numpy arrays and epochs.  """

    # Order and chunk the epochs for the given length of the epochs
    print(f"Sub_epochs shape is {sub_epochs.get_data().shape}. No of events, no of channels, no of data points.")

    highlight_down_epochs = sub_epochs['highlight_down']
    highlight_up_epochs = sub_epochs['highlight_up']
    epochs_list = [highlight_down_epochs, highlight_up_epochs]

    # Concatenate the epochs, this step automatically applies baseline correction
    dataset_epochs = mne.concatenate_epochs(epochs_list)
    print(f"Concat_epochs shape is {dataset_epochs.get_data().shape}. No of events, no of channels, no of data points.")

    # Now we extract the data to a Numpy array of the format: (n_epochs, n_channels, n_times)
    highlight_down_epochs_ndarray = dataset_epochs.get_data(item='highlight_down')
    highlight_up_epochs_ndarray = dataset_epochs.get_data(item='highlight_up')
    dataset_ndarray = np.concatenate((highlight_down_epochs_ndarray, highlight_up_epochs_ndarray), axis=0)

    eeg_labels = np.array(highlight_down_epochs_ndarray.shape[0] * [0] + highlight_up_epochs_ndarray.shape[0] * [1])

    return dataset_epochs, dataset_ndarray, eeg_labels


def k_fold_LDA(concat_epochs, concat_epochs_ndarray, eeg_labels, n_features_to_select=20):
    """
    Function that gets concatenated epochs and numpy array version of epochs, and eeg labels, and
    applies k-fold LDA algorithm while also doing a sequantial feature reduction at every k-fold.
    It returns to all the trained LDAs and the scores."""

    scores = []
    k_ldas = []
    #Determining indices of the 5 different training and testing datasets
    kf = KFold(n_splits=5, shuffle=True, random_state = 1234)
    for train_index, test_index in kf.split(concat_epochs_ndarray):
        # Data (ndarrays and MNE epochs) are split into training and testing datasets
        X_train, X_test = concat_epochs_ndarray[train_index], concat_epochs_ndarray[test_index]
        y_train, y_test = eeg_labels[train_index], eeg_labels[test_index]
        X_train_epochs, X_test_epochs = concat_epochs[train_index], concat_epochs[test_index]

        # Extract the features
        X_train, csp = process_features(X_train, X_train_epochs, y_train)
        X_test, _ = process_features(X_test, X_test_epochs, y_test, csp)

        # Initialize and train the LDA, calculate best features, test the data
        lda = LinearDiscriminantAnalysis()
        selected_feature_idx = SequentialFeatureSelector(lda, n_features_to_select=n_features_to_select, direction="backward").fit(X_train, y_train).support_
        fitted_lda = lda.fit(X_train[:,selected_feature_idx], y_train)
        k_ldas.append([fitted_lda, selected_feature_idx,  X_train, y_train, X_test, y_test ])
        score = fitted_lda.score(X_test[:,selected_feature_idx], y_test)
        scores.append(score)

    print(f'Max score: {np.max(scores)}, std: {np.std(scores)}.')

    return k_ldas, scores, selected_feature_idx

