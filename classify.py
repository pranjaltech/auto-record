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
import zipfile


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
    """Resample waveform and ensure fixed length."""
    desired_length = 15600  # YAMNet expected length

    # If sample rate is not the desired one, resample
    if original_sample_rate != desired_sample_rate:
        resample_length = int(
            round(float(len(waveform)) / original_sample_rate * desired_sample_rate)
        )
        waveform = signal.resample(waveform, resample_length)

    length = len(waveform)

    # If longer, pick center part
    if length > desired_length:
        offset = (length - desired_length) // 2
        waveform = waveform[offset : offset + desired_length]
    # If shorter, pad with zeros
    elif length < desired_length:
        offset = (desired_length - length) // 2
        padding = np.zeros(desired_length - length)
        waveform = np.concatenate((padding, waveform, padding))

    return desired_sample_rate, waveform


def ensure_sample_rate_old(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(
            round(float(len(waveform)) / original_sample_rate * desired_sample_rate)
        )
        waveform = signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


def prepare_model(use_tflite=False):
    if use_tflite:
        # Check if the model exists in the current directory
        # if not, download it
        if not os.path.exists("yamnet.tflite"):
            print("Downloading model...")
            os.system(
                "wget https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite -O yamnet.tflite"
            )
            print("Download complete.")

        model = tf.lite.Interpreter(model_path="yamnet.tflite")
        model.allocate_tensors()

        # Load the class map (convert class index to class name)
        class_map_path = zipfile.ZipFile("yamnet.tflite").open("yamnet_label_list.txt")
        class_names = [l.decode("utf-8").strip() for l in class_map_path.readlines()]
    else:
        # Load the trained model.
        model = hub.load("https://tfhub.dev/google/yamnet/1")

        # Load the class map (convert class index to class name)
        class_map_path = model.class_map_path().numpy()
        class_names = class_names_from_csv(class_map_path)

    return model, class_names


def classify(model, file, class_names, use_tflite=False):
    # Read the file
    sample_rate, wav_data = wavfile.read(file, "rb")
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    # Convert wav_data to mono format.
    if len(wav_data.shape) == 2:
        wav_data = np.mean(wav_data, axis=1).astype(np.int16)

    # Normalise the waveform
    waveform = wav_data / tf.int16.max

    if use_tflite:
        # Convert the waveform to a 32-bit float
        waveform = waveform.astype(np.float32)

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        print(
            f'Waveform shape: {waveform.shape}, Expected shape: {input_details[0]["shape"]}'
        )

        model.set_tensor(input_details[0]["index"], waveform)
        model.invoke()
        scores = model.get_tensor(output_details[0]["index"])
        scores = tf.convert_to_tensor(scores)
    else:
        # Run the model, check the output.
        scores, _, _ = model(waveform)

    mean_scores = np.mean(scores, axis=0)
    top_n = 4
    top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
    top_classes = [class_names[index] for index in top_class_indices]

    scores_np = scores.numpy()
    inferred_class = class_names[scores_np.mean(axis=0).argmax()]

    return inferred_class, top_classes


def run_classification(input_files, output_file, tflite=False):
    model, class_names = prepare_model(tflite)

    classification_result = []

    # Progress bar
    with alive_bar(len(input_files), title="Classifying...") as progress_bar:
        for _file in input_files:
            try:
                inferred_class, top_classes = classify(
                    model, _file, class_names, tflite
                )
                classification_result.append([_file, inferred_class, top_classes])
            except:
                print(f"Error processing file: {_file}")
            progress_bar()

    # Save results to CSV
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
    argparse.add_argument(
        "--tflite",
        action="store_true",
        help="Use Tensorflow lite model for classification.",
    )

    args = argparse.parse_args()
    tflite = args.tflite
    input_dir = args.input_dir
    output_file = args.output_file

    input_files = load_file_list(input_dir)
    run_classification(input_files, output_file, tflite)
