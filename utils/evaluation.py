import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import librosa


class ModelEvaluator:
    def __init__(self, char_to_num, num_to_char):
        self.char_to_num = char_to_num
        self.num_to_char = num_to_char

    def evaluate_model(self, model, test_dataset):
        """Evaluate model on test dataset"""
        print("Evaluating model...")

        total_loss = 0
        total_samples = 0

        for batch in test_dataset:
            features, labels = batch
            batch_size = features.shape[0]

            # Create dummy inputs for CTC
            input_length = np.ones((batch_size, 1)) * features.shape[1]
            label_length = np.ones((batch_size, 1)) * labels.shape[1]
            dummy_labels = np.zeros_like(labels)

            loss = model.test_on_batch(
                [features, labels, input_length, label_length],
                dummy_labels
            )

            total_loss += loss * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        print(f"Test Loss: {avg_loss:.4f}")

        return avg_loss

    def predict_and_decode(self, model, features, num_to_char):
        """Make predictions and decode to text"""
        # Add batch dimension if needed
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)

        # Get predictions
        predictions = model.predict(features)

        # Decode predictions
        decoded_texts = self.decode_predictions(predictions, num_to_char)

        return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts

    def decode_predictions(self, predictions, num_to_char):
        """Decode model predictions to text"""
        input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
        results = keras.backend.ctc_decode(predictions,
                                           input_length=input_len,
                                           greedy=True)[0][0]

        texts = []
        for result in results:
            text = ""
            for idx in result:
                if idx != -1:
                    text += num_to_char.get(idx.numpy(), '')
            texts.append(text)

        return texts

    def calculate_wer(self, reference, hypothesis):
        """Calculate Word Error Rate"""
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        # Create distance matrix
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

        for i in range(len(ref_words) + 1):
            d[i, 0] = i
        for j in range(len(hyp_words) + 1):
            d[0, j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    substitution = d[i - 1, j - 1] + 1
                    insertion = d[i, j - 1] + 1
                    deletion = d[i - 1, j] + 1
                    d[i, j] = min(substitution, insertion, deletion)

        wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
        return wer

    def calculate_cer(self, reference, hypothesis):
        """Calculate Character Error Rate"""
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)

        # Create distance matrix
        d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))

        for i in range(len(ref_chars) + 1):
            d[i, 0] = i
        for j in range(len(hyp_chars) + 1):
            d[0, j] = j

        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i - 1] == hyp_chars[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    substitution = d[i - 1, j - 1] + 1
                    insertion = d[i, j - 1] + 1
                    deletion = d[i - 1, j] + 1
                    d[i, j] = min(substitution, insertion, deletion)

        cer = d[len(ref_chars), len(hyp_chars)] / len(ref_chars)
        return cer

    def visualize_predictions(self, model, test_dataset, num_samples=5):
        """Visualize model predictions"""
        plt.figure(figsize=(15, 3 * num_samples))

        sample_count = 0
        for features, labels in test_dataset:
            for i in range(features.shape[0]):
                if sample_count >= num_samples:
                    break

                # Get prediction
                prediction = self.predict_and_decode(
                    model, features[i:i + 1], self.num_to_char
                )

                # Get actual text
                actual_numbers = labels[i].numpy()
                actual_text = self.numbers_to_text(actual_numbers)

                # Plot
                plt.subplot(num_samples, 1, sample_count + 1)
                plt.imshow(features[i].numpy().T, aspect='auto', origin='lower', cmap='viridis')
                plt.title(f"Actual: '{actual_text}'\nPredicted: '{prediction}'")
                plt.xlabel('Time Steps')
                plt.ylabel('Features')

                sample_count += 1

            if sample_count >= num_samples:
                break

        plt.tight_layout()
        plt.show()

    def numbers_to_text(self, numbers):
        """Convert numerical sequence to text"""
        return ''.join([self.num_to_char.get(num, '') for num in numbers if num != 0])