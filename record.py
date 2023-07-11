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


FORMAT = pyaudio.paInt16


class Recorder:
    @staticmethod
    def rms(frame):
        count = len(frame) / 2
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * NORMALIZATION_FACTOR
            sum_squares += n * n

        rms = math.pow(sum_squares / count, 0.5)
        return rms * 1000

    def __init__(self) -> None:
        self.p = pyaudio.PyAudio()
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount")
        for i in range(0, numdevices):
            if (
                self.p.get_device_info_by_host_api_device_index(0, i).get(
                    "maxInputChannels"
                )
            ) > 0:
                print(
                    "Input Device id ",
                    i,
                    " - ",
                    self.p.get_device_info_by_host_api_device_index(0, i).get("name"),
                )
        device_index = (
            int(input("Select device id: ")) if numdevices > 1 else numdevices
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

    def record(self, bar):
        print("Noise detected, recording beginning")
        rec = []
        current = time.time()
        end = time.time() + MAX_SILENCE_LENGTH

        while current <= end:
            data = self.stream.read(BUFFER_SIZE, exception_on_overflow=False)
            rms_val = self.rms(data)
            bar(rms_val / 24.0)
            bar.title("Recording")

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
        result = subprocess.run(
            ["rclone", "move", "/home/pranjal/autorecord/output", "gdrive:/"],
            capture_output=True,
            text=True,
        )
        print("Copied. ", result.stdout)
        print("Returning to listening")

    def listen(self):
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
                rms_val = self.rms(input)
                bar(rms_val / 24.0)
                bar.title("Listening")
                if rms_val > SILENCE_THRESHOLD:
                    self.record(bar)

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


if __name__ == "__main__":
    a = Recorder()
    a.listen()
