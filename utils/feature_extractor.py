import numpy as np
import librosa
import pywt
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt


class FeatureExtractor:
    def __init__(self, sampling_rate=16000, n_mfcc=13, n_mels=80, n_fft=1024, hop_length=256):
        self.sampling_rate = sampling_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_mfcc(self, audio):
        """Extract MFCC features"""
        audio = self._preprocess_audio(audio)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sampling_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Add derivatives
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Stack features
        mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

        return mfcc_features.T  # Time steps first

    def extract_mel_spectrogram(self, audio):
        """Extract Mel Spectrogram features"""
        audio = self._preprocess_audio(audio)

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db.T  # Time steps first

    def extract_wavelet(self, audio, wavelet='db4', max_level=5):
        """Extract Wavelet Transform features"""
        audio = self._preprocess_audio(audio)

        # Pad audio to suitable length for wavelet transform
        target_length = 2 ** max_level * (len(audio) // (2 ** max_level) + 1)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

        # Perform Discrete Wavelet Transform
        coeffs = pywt.wavedec(audio, wavelet, level=max_level)

        # Extract features from wavelet coefficients
        wavelet_features = []
        for i, coeff in enumerate(coeffs):
            # Statistical features from each level
            if len(coeff) > 0:
                wavelet_features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.median(coeff),
                    np.max(np.abs(coeff)),
                    np.mean(np.abs(coeff))
                ])

        # Create time-frequency representation using CWT
        widths = np.arange(1, 65)
        cwt_matrix = signal.cwt(audio, signal.ricker, widths)

        # Reduce dimensionality of CWT
        cwt_features = self._reduce_cwt_dimensionality(cwt_matrix)

        # Combine statistical and CWT features
        combined_features = np.concatenate([wavelet_features, cwt_features.flatten()])

        # Reshape to 2D for compatibility with models
        time_steps = min(100, len(combined_features) // 13)
        if time_steps > 0:
            combined_features = combined_features[:time_steps * 13]
            combined_features = combined_features.reshape(time_steps, 13)
        else:
            combined_features = combined_features.reshape(1, -1)

        return combined_features

    def extract_wavelet_time_frequency(self, audio, wavelet='db4', level=4):
        """Extract Wavelet features as time-frequency representation"""
        audio = self._preprocess_audio(audio)

        # Perform DWT and create time-frequency matrix
        coeffs = pywt.wavedec(audio, wavelet, level=level)

        # Create time-frequency representation
        tf_representation = []
        max_len = max(len(coeff) for coeff in coeffs)

        for coeff in coeffs:
            # Interpolate to common length
            if len(coeff) < max_len:
                coeff_interp = np.interp(
                    np.linspace(0, len(coeff) - 1, max_len),
                    np.arange(len(coeff)),
                    coeff
                )
            else:
                coeff_interp = coeff[:max_len]

            tf_representation.append(coeff_interp)

        tf_matrix = np.array(tf_representation)
        return tf_matrix.T  # Time steps first

    def extract_combined_features(self, audio):
        """Extract all three feature types and combine"""
        mfcc = self.extract_mfcc(audio)
        mel_spec = self.extract_mel_spectrogram(audio)
        wavelet = self.extract_wavelet_time_frequency(audio)

        # Align time steps
        min_time_steps = min(mfcc.shape[0], mel_spec.shape[0], wavelet.shape[0])

        mfcc_aligned = mfcc[:min_time_steps, :]
        mel_spec_aligned = mel_spec[:min_time_steps, :]
        wavelet_aligned = wavelet[:min_time_steps, :]

        # Combine features
        combined = np.concatenate([mfcc_aligned, mel_spec_aligned, wavelet_aligned], axis=1)

        return combined

    def _preprocess_audio(self, audio):
        """Preprocess audio for feature extraction"""
        if tf.is_tensor(audio):
            audio = audio.numpy()

        audio = audio.flatten()
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        return audio

    def _reduce_cwt_dimensionality(self, cwt_matrix, target_dim=13):
        """Reduce dimensionality of CWT matrix"""
        # Use PCA or simple downsampling
        if cwt_matrix.shape[1] > target_dim:
            # Simple downsampling
            step = cwt_matrix.shape[1] // target_dim
            return cwt_matrix[:, ::step][:, :target_dim]
        else:
            # Padding if needed
            return np.pad(cwt_matrix, ((0, 0), (0, target_dim - cwt_matrix.shape[1])),
                          mode='constant')

    def visualize_features(self, audio, title="Feature Visualization"):
        """Visualize all three feature types"""
        features = {
            'MFCC': self.extract_mfcc(audio),
            'Mel Spectrogram': self.extract_mel_spectrogram(audio),
            'Wavelet Transform': self.extract_wavelet_time_frequency(audio)
        }

        plt.figure(figsize=(15, 12))

        for i, (name, feature) in enumerate(features.items()):
            plt.subplot(3, 1, i + 1)

            if name == 'Wavelet Transform':
                im = plt.imshow(feature.T, aspect='auto', origin='lower', cmap='viridis')
            else:
                im = plt.imshow(feature.T, aspect='auto', origin='lower', cmap='viridis')

            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f'{name} - Shape: {feature.shape}')
            plt.xlabel('Time Steps')
            plt.ylabel('Features' if name != 'Wavelet Transform' else 'Scale')

        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()

        return features