import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec, GridSpec
from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import uniform_filter1d
import mne
import mne.decoding.csp
import pickle
import time
from tqdm import tqdm
import pause
import datetime
import re
from offline.utils import get_avg_evoked_online

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_stream

X_all = None

def create_raw(data_stream, marker_stream, data_timestamp, marker_timestamp):
    # Get the sampling frequency
    sfreq = 250

    # Get the data --> only the first 8 rows are for the electrodes
    data = data_stream.T
    # Set the channel names
    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    # Define the channel types
    ch_types = ['eeg'] * len(ch_names)

    # Get the time stamps of the EEG recording
    raw_time = data_timestamp

    # Check if there are available marker streams
    cues = marker_stream

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
    cue_times = [float(i[0].split("-")[1]) for i in marker_stream]

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

    info = mne.create_info(ch_names, sfreq, ch_types)

    # Add the system name to the info
    info['description'] = 'Emotiv-EPOC'

    raw = mne.io.RawArray(data, info)

    # Create a montage out of the 10-20 system
    montage = mne.channels.make_standard_montage('standard_1020')

    # Apply the montage
    raw.set_montage(montage)

    # A line to supress an error message
    #raw._filenames = [file]

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


# Load the configurations of the paradigm
with open(os.path.join(os.path.curdir, '..', 'offline', 'configs', 'config_long_red-yellow.json'), 'r') as file:
    params = json.load(file)
SIGNAL_DURATION = 0.6
apply_CAR = 0

# %%
# Manages the animated plots.
# This was provided by Karahan, so I did not add any further comments.
class BlitManager:
    # https://matplotlib.org/stable/tutorials/advanced/blitting.html#sphx-glr-tutorials-advanced-blitting-py
    def __init__(self, fig, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = fig.canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = self.canvas.mpl_connect('draw_event', self.on_draw)
        self.cid = self.canvas.mpl_connect('close_event', self.on_close)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def on_close(self, event):
        global figure_closed
        figure_closed = True

    def add_artist(self, arts):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if type(arts) != list:
            arts = [arts]
        for art in arts:
            if art.figure != self.canvas.figure:
                raise RuntimeError
            art.set_animated(True)
            self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()


def new_subplot(fig, bm, x_min=0, x_max=5000, y_min=-0.00005, y_max=0.00005, title=None, color='navy', lw=2):
    # Create a new subplot
    tmp_ax = fig.add_subplot()
    tmp_ax.set_title(title)

    # Add artists to the BlitManager to be updated later
    # Real-time data
    bm.add_artist(tmp_ax.plot([], [], color=color, lw=lw))
    # Detection threshold
    bm.add_artist(tmp_ax.axhline(0, lw=2, ls='dashed', color='gray'))
    # Detected peaks
    bm.add_artist(tmp_ax.plot([], [], color='tomato', marker='x', lw=2))

    # Baseline period
    tmp_ax.axvspan(buffer_size_stream + base_begin,
                   buffer_size_stream + base_end,
                   alpha=0.1,
                   color='green')
    # Detection window
    tmp_ax.axvspan(buffer_size_stream + activity_begin,
                   buffer_size_stream + activity_end,
                   alpha=0.1,
                   color='red')
    # x-axis
    tmp_ax.axhline(0, color='grey')

    tmp_ax.set_xlim([x_min, x_max])
    tmp_ax.set_ylim([y_min, y_max])

    # Get all the axes from the figure
    axes = fig.axes
    # Get the number of axes
    n_axes = len(axes)
    # Iterate over the axes
    for i, ax in enumerate(axes):
        # Define the position of all the subplots using the number of total subplots
        ax.set_subplotspec(SubplotSpec(GridSpec(n_axes, 1), i))

        # Set the labels
        ax.set_ylabel('Amp.')
        if i == n_axes - 1:
            ax.set_xlabel('Sample')
        else:
            ax.set_xlabel('')

    # Set a tight layout for the titles to be clearly readable
    fig.tight_layout()

    return fig


# %%

def receive_eeg_samples(inlet,
                        samples_buffer,
                        timestamps_buffer,
                        buffer_size=5000,
                        chunk_size=100):
    """
    Receives new EEG samples and timestamps from the LSL input and stores them in the buffer variables
    :param samples_buffer: list of samples(each itself a list of values)
    :param timestamps_buffer: list of time-stamps
    :return: updated samples_buffer and timestamps_buffer with the most recent 150 samples
    """

    # Pull a chunk of maximum chunk_size samples
    chunk, timestamps = inlet.pull_chunk(max_samples=chunk_size)

    # If there are no new samples
    if chunk == []:
        # Return the unchanged buffer
        return samples_buffer, timestamps_buffer, 0
    # If there are new samples
    else:

        # Get the number of newly fetched samples
        n_new_samples = len(chunk)

        # Convert the chunk into a np.array and transpose it
        #samples = [np.ndarray(sample).T for sample in chunk]

        # Extend the buffer with the data
        samples_buffer.extend(chunk)
        # Extend the buffer with time stamps
        timestamps_buffer.extend(timestamps)

        # Get the last buffer_size samples and time stamps from the buffer
        data_from_buffer = samples_buffer[-buffer_size:]
        timestamps_from_buffer = timestamps_buffer[-buffer_size:]

        return data_from_buffer, timestamps_from_buffer, n_new_samples



# Returns the current time in milliseconds.
def current_milli_time():
    return round(time.time() * 1000)

def preprocess(raw, event_arr, event_id):
    raw_filt = raw.filter(flow, fhigh)

    # Now we crop the last trial.
    #start_trial = event_id['trial_begin']
    #end_trial = event_id['pause']
    #end_trial_ind = [index for index, value in enumerate(event_arr) if value[2] == end_trial]
    #last_trial_end = end_trial_ind[-1]

    # This ordering is important in case a new trial already started but didn't end yet.
    #start_trial_ind = [index for index, value in enumerate(event_arr[:last_trial_end+1]) if value[2] == start_trial]
    #last_trial_start = start_trial_ind[-1]

    #raw_cropped = raw_filt.crop(tmin=event_arr[last_trial_start][0], tmax=event_arr[last_trial_end][0])
    #event_arr = event_arr[last_trial_start:last_trial_end+1]

    print('Getting avg evokeds: ')

    num_highlights = params['num_highlights']
    print(f'{total_trial_duration, num_highlights, decim_factor}')

    highlights_final_evoked, highlights_labels = get_avg_evoked_online(raw_filt,
                                                                       event_arr.copy(),
                                                                       event_id.copy(),
                                                                       total_trial_duration,
                                                                       num_highlights,
                                                                       decim_facor=decim_factor,
                                                                       t_min=.1,
                                                                       t_max=.6)

    chnum, size = highlights_final_evoked[0].get_data().shape
    x = np.concatenate([x.get_data().reshape(1, chnum, size) for x in highlights_final_evoked[-4:]])
    #
    # with open(output_file, 'wb') as opened_file:
    #     pickle.dump((x, y, full_data), opened_file)

    return x, highlights_labels[-4:]

def classify_single_trial(trial_data: np.ndarray, highlights_labels: np.ndarray, inds_selected_ch, X_all=None):
    """
    :param trial_data: ndarray of size (num arrow types, num channels, num_samples)
    :param highlights_labels: ndarray of arrow types, size (num arrow types, )
    :return: int label, predicted direction
    """

    # reshape data and extract features
    X = trial_data[:, inds_selected_ch, :].reshape(trial_data.shape[0], -1)
    print(X)
    # do some more stuff if needed
    trial_predictions = clf.predict(X)
    pred_label = highlights_labels[np.argmax(trial_predictions)]
    predictions.append(pred_label)
    X_all = np.concatenate([X_all, X])

    return pred_label

def get_prediction(raw_trial, event_arr: np.ndarray, event_id: dict, inds_selected_ch):
    """
    :param raw_trial: input data
    :param event_arr: ndarray of events timestamps and event ids
    :param event_id: dict name of events are the keys and corresponding int ids as values
    :return: predicted label (arrow direction)
    """
    prep_data, highlights_labels = preprocess(raw_trial, event_arr, event_id)

    return classify_single_trial(prep_data, highlights_labels, inds_selected_ch)


if __name__ == '__main__':
    labelsss = [0, 1, 2, 3]
    print("Looking for an EEG stream")
    streams = resolve_stream()
    print("Found one")

    # Make a simple menu to choose the correct stream.
    # In case multiple unicorns are set up.
    for idx, stream in enumerate(streams):
        print(f'Press [{idx}] for stream: "{stream.name()}".')
    stream_idx = input('Select a data stream...')
    marker_idx = input('Select a marker stream...')

    stream_inlet = StreamInlet(streams[int(stream_idx)])
    marker_inlet = StreamInlet(streams[int(marker_idx)])
    print("Connected to the inlet")

    # Signal processing parameters
    X_all = None
    highlights_labels_all = None
    predictions = []

    with open("test_model.pickle", mode='rb') as opened_file:
        clf = pickle.load(opened_file)

    # Set the channel names
    ch_names =['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    # Sampling rate of the Unicorn Hybrid Black
    sfreq = 250  # Hz
    # Define the channel types
    ch_types = ['eeg'] * len(ch_names)
    # create mne info
    info = mne.create_info(ch_names, sfreq, ch_types)
    # Define the band-pass cutoff frequencies
    flow, fhigh = 2, 10
    decim_factor = 4


    # Supress MNE info messages and only show warnings
    mne.set_log_level(verbose='WARNING')

    # Create the info structure needed by MNE
    info = mne.create_info(ch_names, sfreq, 'eeg')
    # Add the system name to the info
    info['description'] = 'Unicorn Hybrid Black'

    # Create the outlet information
    info_classifier = StreamInfo(name='classifier_output_markers',
                                 type='Markers',
                                 channel_count=1,
                                 nominal_srate=0,
                                 channel_format='int8',
                                 source_id='classifier_output_markers')

    # Create the outlet to send data.
    stream_classifier = StreamOutlet(info_classifier)

    # Majority vote number
    # (number of votes needed to decide the final classification)
    majority_vote_number = 5

    # Initialise the buffers
    samples_buffer_stream = []
    timestamps_buffer_stream = []
    samples_buffer_marker = []
    timestamps_buffer_marker = []
    labels_buffer = []

    # Set the buffer sizes.
    total_trial_duration = params['baseline_length'] + params['target_length'] + \
                           params['highlight_length'] * params['num_highlights'] + \
                           params['inter_highlight_length'] * (params['num_highlights'] - 1)

    buffer_size_stream = int(total_trial_duration*500)
    buffer_size_marker = 100
    chunk_size = 50
    labels_buffer_size = 10

    # Channel index.
    idx = 0

    ch = ch_names[idx]

    # Define the length of the moving average filter.
    N = 25

    # Length of the baseline for the classifier.
    # a signal that was previously classified as non-rest.
    # len_base = 400
    len_base = 125

    # Length of the window for the classifier.
    len_win = SIGNAL_DURATION

    # Length of the space between the baseline and the activity window.
    # len_space = 100
    len_space = 0

    # Define the delay of the windows
    delay = 20

    base_begin = -len_win - delay - len_base - len_space
    base_end = -len_win - delay - len_space

    activity_begin = -len_win - delay
    activity_end = -delay

    # Set the threshold for the confidence interval.
    # What is meant with the confidence interval here?
    # It's for the eye blinks.
    conf_val = 12

    # Initialise the values for the refractory period.
    refractory_period = False
    refract_counter = 0
    class_counter = 0
    len_refractory = 15

    # Variables for the calibration.
    pbar = tqdm(desc='Calibration', total=buffer_size_stream)
    pbar_closed = False
    old_val = 0

    # Plotting

    # Set the axis limits
    y_min = -0.00005
    y_max = 0.00005
    # y_min = 0
    # y_max = 3
    x_min = 0
    x_max = buffer_size_stream

    # Define the subplot titles
    # title1 = f'Band-Pass Filtered Channel {ch}.'
    title1 = f'Classifier output'
    # title2 = f'Moving Average Smoothened Channel {ch} (N={N}).'
    title2 = f'Preprocessed signal of Fz'
    title3 = f'Band-Pass Filtered Channel {ch}.'
    title4 = f'Moving Average of Channel {ch} (N={N}).'

    # Create an empty figure.
    fig = plt.figure()
    bm = BlitManager(fig=fig)
    new_subplot(fig, bm, x_min=0, x_max=9, y_min=0, y_max=3, title=title1)
    new_subplot(fig, bm, y_min=-40, y_max=40, title=title2)

    # Define what will happen when the figure is closed
    # fig.canvas.mpl_connect('close_event', on_close)
    # Variable to keep track of the closing of the matplotlib figure
    figure_closed = False

    plt.show(block=False)
    plt.pause(0.1)

    # Plots the classifier output over time.
    labels_buffer = [0] * labels_buffer_size

    last_trial_end_time = 0
    trial_end_time = 0

    # Classification
    # Enter the endless loop
    while True:
        # If the matplotlib figure is closed.
        if figure_closed:
            # Break out of the loop.
            break

        # Get the current time.
        current_time = current_milli_time()
        #time.sleep(2400)
        # Get the EEG samples. time stamps,
        # and the index that tells from which point on the new samples have
        # been appended to the buffer.
        samples_buffer_stream, timestamps_buffer_stream, n_new_samples_stream = receive_eeg_samples(
            stream_inlet,
            samples_buffer_stream,
            timestamps_buffer_stream,
            buffer_size=buffer_size_stream,
            chunk_size=chunk_size
        )

        samples_buffer_marker, timestamps_buffer_marker, n_new_samples_marker = receive_eeg_samples(
            marker_inlet,
            samples_buffer_marker,
            timestamps_buffer_marker,
            buffer_size=buffer_size_marker,
            chunk_size=chunk_size
        )

        cue_times = [float(i[0].split("-")[1]) for i in samples_buffer_marker]
        cues = [label[0].split('-')[0] for label in samples_buffer_marker]

        if 'pause' in cues:
            trial_end_time = cue_times[cues.index('pause')]
            if type(trial_end_time) == list:
                trial_end_time = trial_end_time[-1]

        # Processing
        # Check if the calibration is over:
        if pbar_closed and trial_end_time != last_trial_end_time:

            last_trial_end_time = trial_end_time

            # The markers buffer has older data than eeg buffer, thus we need to align the two samples
            oldest_stream_data = timestamps_buffer_stream[0]
            correct_markers = [i for i in timestamps_buffer_marker if i > oldest_stream_data]
            timestamps_buffer_marker = np.pad(correct_markers, (0, len(timestamps_buffer_marker) - len(correct_markers)))
            # Create MNE epoch
            #raw = mne.io.RawArray(np.array(samples_buffer_stream).T, info)
            raw, event_arr, event_id = create_raw(np.array(samples_buffer_stream),
                                                  np.array(samples_buffer_marker),
                                                  np.array(timestamps_buffer_stream),
                                                  np.array(timestamps_buffer_marker))

            inds_selected_ch = [6, 7, 10, 5, 8, 11]
            """x, highlights_labels = get_prediction(raw, event_arr, event_id, inds_selected_ch)

            # Dummy code
            labelsss = np.roll(labelsss, 1)
            new_classification = labelsss[1]
            print(new_classification)
            labels_buffer.append(new_classification)
            labels_buffer = labels_buffer[-labels_buffer_size:]

            stream_classifier.push_sample(new_classification)

            # Plotting
            # If there are new samples in the buffer.
            # if n_new_samples > 0:
            # Update the artists (lines) in the subplots
            x_range = range(len(samples_buffer_stream))
            x_labels_range = range(len(labels_buffer))
            bm._artists[0].set_data(x_labels_range, labels_buffer)
            # bm._artists[1].set_data(x_range, thr)
            #bm._artists[3].set_data(x_range, car_data[idx])
            # bm._artists[4].set_data(x_range, thr)
            bm.update()"""

            # We want the classification to happen at 10Hz.
            unpause_time = datetime.datetime.fromtimestamp((current_time + 100) / 1000.0)
            pause.until(unpause_time)

        # Progress bar
        else:
            # Get the current number of samples.
            len_buffer = len(timestamps_buffer_stream)
            # Calculate the progress update value
            update_val = len_buffer - old_val
            # Store the current number of samples for the next iteration
            old_val = len_buffer
            # Update the progress bar
            pbar.update(update_val)
            # If the progress bar is full
            if len_buffer == buffer_size_stream:
                # Close the progress bar
                pbar.close()
                # Set the flag to get out of the loop.
                pbar_closed = True

