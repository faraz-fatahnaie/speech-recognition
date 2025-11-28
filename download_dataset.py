import tensorflow_datasets as tfds
import os
import numpy as np


def main():
    print("LibriSpeech Dataset Download")
    print("=" * 50)

    # Configuration
    config = {
        'data_dir': './tensorflow_datasets',
        'download_size': 'clean'  # Options: 'clean', 'other', 'all'
    }

    # Create directories
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('utils', exist_ok=True)

    print("Downloading LibriSpeech dataset...")
    print("This may take a while depending on your internet connection...")

    try:
        # Download dataset
        datasets = tfds.load(
            'librispeech',
            split=['train', 'validation', 'test'],
            data_dir=config['data_dir'],
            download=True,
            shuffle_files=False
        )

        train_ds, val_ds, test_ds = datasets

        print("✓ Download completed successfully!")
        print(f"Training samples: {len(list(train_ds))}")
        print(f"Validation samples: {len(list(val_ds))}")
        print(f"Test samples: {len(list(test_ds))}")

        # Show sample
        for example in train_ds.take(1):
            audio = example['audio'].numpy()
            text = example['text'].numpy().decode('utf-8')
            print(f"\nSample au"
                  f"dio shape: {audio.shape}")
            print(f"Sample text: '{text}'")
            print(f"Audio duration: {len(audio) / 16000:.2f} seconds")

    except Exception as e:
        print(f"Error during download: {e}")
        print("Please check your internet connection and try again.")

    print(f"\nDataset location: {os.path.abspath(config['data_dir'])}")
    print("You can now proceed to data preprocessing!")


if __name__ == "__main__":
    main()
