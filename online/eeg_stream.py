"""Streams the recorded EEG data to a pylsl stream.
Can be used to simulate an online setting."""

import time
import pyxdf

from pylsl import StreamInfo, StreamOutlet

# Get the EEG stream
# Read the XDF file
FILE_NAME = "/home/nathan/Documents/CurrentStudy/sub-Nathan/ses-long-gb/eeg/sub-Nathan_ses-long-gb_task-Default_run-003_eeg.xdf"
streams, header = pyxdf.load_xdf(FILE_NAME)

# Initialize lists for the streams
data_stream = []

for stream in streams:

    # Get the relevant values from the stream
    stream_name = stream['info']['name'][0].lower()
    stream_type = stream['info']['type'][0].lower()

    # Assign streams to their lists
    if stream_type in ('eeg', 'data'):
        data_stream.append(stream)
        print(f'{stream_name} --> data stream')
    else:
        print(f'{stream_name} --> not sure what to do with this')

print()

# Check whether there is only data stream found
if len(data_stream) == 1:
    data_stream = data_stream[0]
elif len(data_stream) == 0:
    raise Exception('No data stream found!')
else:
    raise Exception('Multiple data streams found!')

# Get the sampling frequency
sfreq = 128

# uhb: Unicorn Hybrid Black (g.tec)
# Get the data --> only the first 8 rows are for the electrodes
data = data_stream["time_series"].T

# Get the time stamps of the EEG recording
raw_time = data_stream["time_stamps"]

# first create a new stream info (here we set the name to un-2019.07.02,
# the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
# last value would be the serial number of the device or some other more or
# less locally unique identifier for the stream as far as available (you
# could also omit it but interrupted connections wouldn't auto-recover).
info = StreamInfo('un-2019.07.02', 'EEG', 14, 128, 'float32', 'myuid34234')

# next make an outlet
outlet = StreamOutlet(info)

print("now sending data...")
for sample in data.T:
    # Push the samples from the recording at the same
    # sampling frequency as the real device.
    outlet.push_sample(sample)
    time.sleep(1/sfreq)

