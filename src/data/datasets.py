"""
PyTorch dataset classes for wind angle data.
"""
import numpy as np
from torch.utils.data import Dataset, Sampler


class WindAngleDataset(Dataset):
    """
    PyTorch Dataset for wind angle data.

    Attributes:
        data: Feature tensor
        labels: Label tensor (wind angles)
    """

    def __init__(self, data, labels):
        """
        Initialize dataset.

        Args:
            data: Feature tensor
            labels: Label tensor
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get item at index."""
        return self.data[idx], self.labels[idx]


class BalancedWindAngleSampler(Sampler):
    """
    Sampler that ensures balanced sampling across wind angles.

    Attributes:
        dataset: WindAngleDataset instance
        wind_angles: List of wind angles to balance
        indices: Precomputed balanced indices
    """

    def __init__(self, dataset, wind_angles):
        """
        Initialize sampler.

        Args:
            dataset: WindAngleDataset instance
            wind_angles: List of wind angles
        """
        self.dataset = dataset
        self.wind_angles = wind_angles
        self.indices = self._create_indices()

    def _create_indices(self):
        """Create balanced indices across wind angles."""
        angle_indices = {angle: np.where(self.dataset.labels == angle)[0]
                        for angle in self.wind_angles}
        balanced_indices = []

        while True:
            batch = []
            for angle in self.wind_angles:
                batch.extend(np.random.choice(angle_indices[angle], size=1))
            balanced_indices.extend(batch)
            if len(balanced_indices) > len(self.dataset):
                break

        return balanced_indices

    def __iter__(self):
        """Return iterator over indices."""
        return iter(self.indices[:len(self.dataset)])

    def __len__(self):
        """Return length of dataset."""
        return len(self.dataset)
