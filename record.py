import argparse
import yaml
from datetime import datetime
import math
import pyaudio
import struct
import time
import os
import wave
import alive_progress
import subprocess
from unittest.mock import patch

from classify import classify, prepare_model

# Loading configuration values from config.yml file
with open("config.yml") as f:
    config = yaml.safe_load(f)

# Constants used in the program
NORMALIZATION_FACTOR = 1.0 / 32768.0
SAMPLE_RATE = config["SAMPLE_RATE"]
AUDIO_CHANNELS = config["AUDIO_CHANNELS"]
BUFFER_SIZE = config["BUFFER_SIZE"]
MAX_SILENCE_LENGTH = config["MAX_SILENCE_LENGTH"]
SILENCE_THRESHOLD = config["SILENCE_THRESHOLD"]
OUTPUT_DIRECTORY = config["OUTPUT_DIRECTORY"]

# print all the config
print("SAMPLE_RATE: ", SAMPLE_RATE)
print("AUDIO_CHANNELS: ", AUDIO_CHANNELS)
print("BUFFER_SIZE: ", BUFFER_SIZE)
print("MAX_SILENCE_LENGTH: ", MAX_SILENCE_LENGTH)
print("SILENCE_THRESHOLD: ", SILENCE_THRESHOLD)
print("OUTPUT_DIRECTORY: ", OUTPUT_DIRECTORY)

# Another constant
FORMAT = pyaudio.paInt16


class Recorder:
    @staticmethod
    def rms(frame):
        """Calculates and returns the root mean square of a frame."""
        count = len(frame) // 2
        shorts = struct.unpack("%dh" % (count), frame)

        sum_squares = sum((sample * NORMALIZATION_FACTOR) ** 2 for sample in shorts)
        return math.sqrt(sum_squares / count) * 1000

    def __init__(self, input_device_id=None) -> None:
        """Initialize the Recorder class by choosing an audio device and setting up an audio stream."""

        # Set up PyAudio
        self.p = pyaudio.PyAudio()
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount")
        print(
            self.p.get_device_info_by_host_api_device_index(0, 0).get(
                "maxInputChannels"
            )
        )

        # Loop through all devices, print the info of devices with input capability
        for i in range(0, numdevices):
            if (
                self.p.get_device_info_by_host_api_device_index(0, i).get(
                    "maxInputChannels"
                )
                > 0
            ):
                print(
                    "Input Device id ",
                    i,
                    " - ",
                    self.p.get_device_info_by_host_api_device_index(0, i).get("name"),
                )

        # Ask the user to pick a device if there is more than one, otherwise, pick the only device available
        device_index = (
            input_device_id
            if input_device_id != None
            else (int(input("Select device id: ")) if numdevices > 1 else numdevices)
        )
        self.record_armed = False
        self.stream = self.p.open(
            format=FORMAT,
            channels=AUDIO_CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            output=True,
            frames_per_buffer=BUFFER_SIZE,
        )

        # Prepare the Classifier
        model, class_names = prepare_model()
        self.model = model
        self.class_names = class_names

    def run_recording(self, bar):
        print("Noise detected, recording beginning")
        rec = []
        current = time.time()
        end = time.time() + MAX_SILENCE_LENGTH

        while current <= end:
            data = self.stream.read(BUFFER_SIZE, exception_on_overflow=False)
            rms_val = self.update_progress(bar, data, "Recording")

            if rms_val >= SILENCE_THRESHOLD:
                end = time.time() + MAX_SILENCE_LENGTH

            current = time.time()
            rec.append(data)

        self.write(b"".join(rec))

    def write(self, recording):
        # Create OUTPUT_DIRECTORY if it doesn't exist
        if not os.path.exists(OUTPUT_DIRECTORY):
            os.makedirs(OUTPUT_DIRECTORY)

        filename = os.path.join(
            OUTPUT_DIRECTORY,
            "autorecord-{}.wav".format(
                datetime.strftime(datetime.now(), "%d-%m-%Y %H:%M:%S")
            ),
        )

        wf = wave.open(filename, "wb")
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(recording)
        wf.close()
        print("Written to file: {}".format(filename))

        # Run the classifier on the file
        inferred_class, _ = classify(self.model, filename, self.class_names)

        print("Inferred class: ", inferred_class)
        # Move the file to a sub-folder named after the inferred class
        new_filename = os.path.join(
            OUTPUT_DIRECTORY, inferred_class, os.path.basename(filename)
        )
        os.makedirs(os.path.dirname(new_filename), exist_ok=True)
        os.rename(filename, new_filename)
        print("Moved to: {}".format(new_filename))

        self.copy_to_gdrive()
        print("Returning to listening")

    def copy_to_gdrive(self):
        result = subprocess.run(
            ["rclone", "move", OUTPUT_DIRECTORY, "gdrive:/"],
            capture_output=True,
            text=True,
        )
        print("Copied. ", result.stdout)

    def listen(self):
        """Continuously listens to the audio stream and starts recording when sound level gets above a certain threshold."""
        print("Listening beginning")
        with alive_progress.alive_bar(
            24,
            manual=True,
            theme="smooth",
            monitor=True,
            stats=False,
            elapsed=False,
            title="Listening",
        ) as bar:
            while True:
                input = self.stream.read(BUFFER_SIZE, exception_on_overflow=False)
                rms_val = self.update_progress(bar, input, "Listening")
                if rms_val > SILENCE_THRESHOLD:
                    self.run_recording(bar)

    def update_progress(self, bar, input, title):
        rms_val = self.rms(input)
        bar(rms_val / 24.0)
        bar.title(title)
        return rms_val

    def listen_buffer(self, bar):
        rec_buffer = []
        current = time.time()
        end = time.time() + MAX_SILENCE_LENGTH

        while current <= end:
            data = self.stream.read(BUFFER_SIZE, exception_on_overflow=False)

            rms_val = self.rms(data)
            bar(rms_val / 24.0)
            if rms_val >= SILENCE_THRESHOLD:
                end = time.time() + MAX_SILENCE_LENGTH


def test_recorder_write():
    rec = Recorder()
    with patch.object(rec, "write", return_value=None) as mock_method:
        rec.write(b"")
        mock_method.assert_called_once()


def test_recorder_run_recording():
    rec = Recorder()
    rec.run


if __name__ == "__main__":
    # See if --device-id was passed as an argument using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device-id",
        help="The device id of the audio device to use. If not specified, the default device will be used.",
    )
    args = parser.parse_args()

    # If --device-id was passed, use that device, otherwise, use the default device
    device_id = int(args.device_id) if args.device_id else None

    a = Recorder(input_device_id=device_id)
    a.listen()
