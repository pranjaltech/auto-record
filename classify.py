# Take input directory as a user input, and classify all the music files in the directory
# using the trained model.
# Output the classification result to a csv file.

import os
import sys
import csv
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from alive_progress import alive_bar
from scipy import signal
from scipy.io import wavfile


def load_file_list(input_dir):
    input_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                input_files.append(os.path.join(root, file))
    return input_files


# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row["display_name"])

    return class_names


def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(
            round(float(len(waveform)) / original_sample_rate * desired_sample_rate)
        )
        waveform = signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


def classify(model, file, class_names):
    # Read the file
    sample_rate, wav_data = wavfile.read(file, "rb")
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    # Show some basic info about the file
    duration = len(wav_data) / sample_rate
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total duration: {duration:.2f}s")

    # Convert wav_data to mono format.
    if len(wav_data.shape) == 2:
        wav_data = np.mean(wav_data, axis=1).astype(np.int16)

    # Normalise the waveform
    waveform = wav_data / tf.int16.max

    # Run the model, check the output.
    scores, embeddings, spectrogram = model(waveform)

    mean_scores = np.mean(scores, axis=0)
    top_n = 4
    top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
    top_classes = [class_names[index] for index in top_class_indices]

    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    inferred_class = class_names[scores_np.mean(axis=0).argmax()]
    print(f"The main sound is: {inferred_class}")
    print(f"Other classes are... {top_classes}")
    print("#" * 36)

    return inferred_class, top_classes


def prepare_model():
    # Load the trained model.
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    # Create a csv file to store the classification result.
    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)

    return model, class_names


def run_classification(input_files, output_file):
    # Configure a progress bar
    progress_bar = alive_bar(len(input_files), title="Classifying...")

    model, class_names = prepare_model()

    # Save the classification result, that can be saved to a CSV later.
    # The first column is the file name, and the second column is the classification result, third column is top classes.
    classification_result = []

    with alive_bar(len(input_files), title="Classifying...") as progress_bar:
        for _file in input_files:
            try:
                # Save the classification result
                inferred_class, top_classes = classify(model, _file, class_names)
                classification_result.append([_file, inferred_class, top_classes])
            except:
                print(f"Error processing file: {_file}")
            progress_bar()

    # Save the classification result to a csv file.
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(classification_result)

    print(f"Classification result saved to {output_file}.")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--input_dir",
        type=str,
        default="data/test",
        help="Directory of the music files to be classified.",
    )
    argparse.add_argument(
        "--output_file", type=str, default="data/test.csv", help="Output file name."
    )

    args = argparse.parse_args()
    input_dir = args.input_dir
    output_file = args.output_file

    input_files = load_file_list(input_dir)
    run_classification(input_files, output_file)
