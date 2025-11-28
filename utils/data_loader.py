import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class LibriSpeechDataLoader:
    def __init__(self, config):
        self.config = config
        self.sampling_rate = config.get('sampling_rate', 16000)
        self.max_audio_length = config.get('max_audio_length', 16000 * 10)  # 10 seconds
        self.char_to_num = None
        self.num_to_char = None

    def load_dataset(self, splits=['train', 'validation', 'test']):
        """Load LibriSpeech dataset using TensorFlow Datasets"""
        print("Loading LibriSpeech dataset...")

        datasets = {}
        all_texts = []

        for split in splits:
            try:
                ds = tfds.load('librispeech',
                               split=split,
                               shuffle_files=True,
                               data_dir=self.config.get('data_dir', None))
                datasets[split] = ds

                # Collect text for character mapping
                for example in ds.take(1000):
                    text = example['text'].numpy().decode('utf-8').lower()
                    all_texts.append(text)

                print(f"Loaded {split} split: {len(list(ds))} examples")

            except Exception as e:
                print(f"Warning: Could not load {split} split: {e}")
                datasets[split] = None

        # Create character mappings
        self._create_char_mappings(all_texts)

        return datasets

    def _create_char_mappings(self, texts):
        """Create character to numerical mapping"""
        chars = set()
        for text in texts:
            chars.update(text.lower())

        # Add common characters if missing
        default_chars = set("abcdefghijklmnopqrstuvwxyz '!?.,-")
        chars = chars.union(default_chars)
        chars = sorted(list(chars))

        # Create mappings
        self.char_to_num = {char: idx for idx, char in enumerate(chars)}
        self.num_to_char = {idx: char for idx, char in enumerate(chars)}

        print(f"Created character mapping with {len(self.char_to_num)} characters:")
        print(f"Characters: {''.join(chars)}")

    def preprocess_audio(self, audio):
        """Preprocess audio signal"""
        if tf.is_tensor(audio):
            audio = audio.numpy()

        audio = audio.flatten()
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # Remove silence
        if len(audio) > 1000:
            audio, _ = librosa.effects.trim(audio, top_db=25)

        # Ensure minimum length
        if len(audio) < 1600:
            padding = 1600 - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        elif len(audio) > self.max_audio_length:
            audio = audio[:self.max_audio_length]

        return audio

    def text_to_numbers(self, text):
        """Convert text to numerical sequence"""
        text = text.lower().strip()
        return [self.char_to_num.get(char, self.char_to_num[' ']) for char in text]

    def numbers_to_text(self, numbers):
        """Convert numerical sequence back to text"""
        return ''.join([self.num_to_char.get(num, '') for num in numbers if num != -1 and num != 0])

    def get_dataset_info(self, datasets):
        """Print dataset information"""
        print("\n" + "=" * 50)
        print("LibriSpeech Dataset Information")
        print("=" * 50)

        for split_name, dataset in datasets.items():
            if dataset is not None:
                try:
                    count = len(list(dataset))
                    print(f"{split_name.capitalize():12}: {count:>6} examples")
                except:
                    print(f"{split_name.capitalize():12}: Unable to count")

        print(f"\nCharacter vocabulary size: {len(self.char_to_num)}")

        # Show sample statistics
        if datasets.get('train'):
            self._show_sample_statistics(datasets['train'])

    def _show_sample_statistics(self, dataset):
        """Show statistics for sample dataset"""
        audio_lengths = []
        text_lengths = []

        for example in dataset.take(100):
            audio = example['audio'].numpy()
            text = example['text'].numpy().decode('utf-8')

            audio_lengths.append(len(audio) / self.sampling_rate)
            text_lengths.append(len(text))

        print(f"\nSample Statistics (100 examples):")
        print(f"Audio duration - Mean: {np.mean(audio_lengths):.2f}s, "
              f"Min: {np.min(audio_lengths):.2f}s, Max: {np.max(audio_lengths):.2f}s")
        print(f"Text length - Mean: {np.mean(text_lengths):.1f} chars, "
              f"Min: {np.min(text_lengths)}, Max: {np.max(text_lengths)}")