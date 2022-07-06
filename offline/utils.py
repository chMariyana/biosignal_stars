from multiprocessing.sharedctypes import RawArray
import os
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import pyxdf
import re
import mne.decoding.csp
import pickle
from datetime import datetime, timezone

from sklearn.preprocessing import LabelEncoder


def load_xdf_data(file, device='uhb', with_stim=False):

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
    data = data_stream["time_series"].T[:8]
    # Set the channel names
    ch_names = ['Fz', 'T3', 'Cz', 'T4', 'Pz', 'Oz', 'C3', 'C4']  # Our config

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
    if device == 'uhb':
        info['description'] = 'Unicorn Hybrid Black'
    elif device == 'ac':
        info['description'] = 'actiCHamp'

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


def create_config():
    """
    Cofiguration file that creates a pkl file to hold the durations
    used in the paradigm.
    """

    exp_durations = {
            'highlight_length': 0.25,
            'target_length': 1,
            'baseline_length': 3,
            'delay_baseline_arrows': 1,
            'inter_highlight_length': 0.2,
            'inter_block_length': 3,
            'inter_trial_length': 1,
            'num_highlights': 10
        }

    with open(r'config_durations.pkl', 'wb') as file:
        pickle.dump(exp_durations, file)
    return 0