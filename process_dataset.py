from classify import prepare_model, classify
import argparse
import alive_progress
import os


def run_classification(input_dir):
    model, class_names = prepare_model(use_tflite=False)
    input_files = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.endswith(".wav")
    ]

    with alive_progress.alive_bar(len(input_files)) as bar:
        for file in input_files:
            try:
                inferred_class, _ = classify(model, file, class_names, use_tflite=False)

                # Move the file to a subdir based on the classification
                subdir = os.path.join(input_dir, inferred_class)
                # Create the subdir if it doesn't exisclt
                if not os.path.exists(subdir):
                    os.makedirs(subdir)

                # Move the file
                os.rename(file, os.path.join(subdir, os.path.basename(file)))
            except:
                print(f"Error processing file: {file}")

            bar()


if __name__ == "__main__":
    # Take input_dir as an argument with argparse
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--dir",
        type=str,
        default="data/test",
        help="Directory of the music files to be classified.",
    )

    args = argparse.parse_args()
    input_dir = args.dir
    run_classification(input_dir)
