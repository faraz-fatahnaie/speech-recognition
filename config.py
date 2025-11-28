# Configuration for the speech recognition project

# Dataset configuration
DATASET_CONFIG = {
    'data_dir': './tensorflow_datasets',
    'sampling_rate': 16000,
    'max_audio_length': 16000 * 10,  # 10 seconds
    'batch_size': 16,
    'shuffle_buffer_size': 1000
}

# Feature extraction configuration
FEATURE_CONFIG = {
    'n_mfcc': 13,
    'n_mels': 80,
    'n_fft': 1024,
    'hop_length': 256,
    'feature_types': ['mfcc', 'mel_spectrogram', 'wavelet']
}

# Model configuration
MODEL_CONFIG = {
    'cnn': {
        'filters': 64,
        'dropout_rate': 0.3,
        'learning_rate': 0.001
    },
    'lstm': {
        'lstm_units': 128,
        'dropout_rate': 0.3,
        'learning_rate': 0.001
    },
    'transformer': {
        'd_model': 128,
        'num_heads': 8,
        'ff_dim': 512,
        'learning_rate': 0.001
    }
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5
}