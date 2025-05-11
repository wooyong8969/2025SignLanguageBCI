import numpy as np

class EEGAugmenter:
    def __init__(self, noise_level=0.01, max_shift=10, seed=None):
        self.noise_level = noise_level
        self.max_shift = max_shift
        if seed is not None:
            np.random.seed(seed)

    def _add_noise(self, eeg):
        noise = np.random.normal(0, self.noise_level, eeg.shape)
        return eeg + noise

    def _time_shift(self, eeg):
        shift = np.random.randint(-self.max_shift, self.max_shift)
        return np.roll(eeg, shift=shift, axis=-1)

    def augment(self, eeg_data, labels, num_augments=10):
        augmented_data = []
        augmented_labels = []

        for x, y in zip(eeg_data, labels):
            augmented_data.append(x)  # 원본
            augmented_labels.append(y)

            for _ in range(num_augments - 1):  # N-1개 증강본
                choice = np.random.choice(['noise', 'shift', 'noise+shift'])
                if choice == 'noise':
                    x_aug = self._add_noise(x)
                elif choice == 'shift':
                    x_aug = self._time_shift(x)
                else:
                    x_aug = self._time_shift(self._add_noise(x))
                
                augmented_data.append(x_aug)
                augmented_labels.append(y)

        augmented_data = np.stack(augmented_data)
        return augmented_data, augmented_labels

