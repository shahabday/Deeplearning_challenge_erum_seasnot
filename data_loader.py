import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class XarraySegmentationDataset(Dataset):
    """
    PyTorch Dataset for segmentation from an xarray.DataArray.

    Expected dims:
        (sample, channel, x, y)

    Convention:
        - all channels except `label_channel` are model inputs
        - `label_channel` is the segmentation mask target
        - normalization is applied only to the input channels
    """

    def __init__(
        self,
        da,
        label_channel="labels",
        normalize=True,
        epsilon=1e-8,
        return_sample_id=False,
        inflate=False,
    ):
        self.da = da
        self.label_channel = label_channel
        self.normalize = normalize
        self.epsilon = epsilon
        self.return_sample_id = return_sample_id
        if inflate:
            self.inflate()

        required_dims = {"sample", "channel", "x", "y"}
        missing_dims = required_dims - set(self.da.dims)
        if missing_dims:
            raise ValueError(f"Missing required dimensions: {missing_dims}")

        if "channel" not in self.da.coords:
            raise ValueError("DataArray must have 'channel' coordinates.")

        self.channel_names = list(self.da.coords["channel"].values)

        if self.label_channel not in self.channel_names:
            raise ValueError(
                f"Label /dachannel '{self.label_channel}' not found in channels: {self.channel_names}"
            )

        self.input_channels = [
            ch for ch in self.channel_names if ch != self.label_channel
        ]

        if len(self.input_channels) == 0:
            raise ValueError("No input channels found after excluding label channel.")

    def __len__(self):
        return self.da.sizes["sample"]

    def _normalize_per_channel(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize each input channel independently.

        Parameters
        ----------
        x : np.ndarray
            Shape (C, H, W)

        Returns
        -------
        np.ndarray
            Shape (C, H, W), normalized channel-wise
        """
        mean = x.mean(axis=(1, 2), keepdims=True)  # (C, 1, 1)
        std = x.std(axis=(1, 2), keepdims=True)  # (C, 1, 1)
        return (x - mean) / (std + self.epsilon)

    def _augment_array(self, arr: np.ndarray):
        """
        Generate 8 variants of a single sample array with shape (C, H, W):
        4 rotations, and a horizontally flipped version of each rotation.
        """
        variants = []
        for k in range(4):
            rotated = np.rot90(arr, k=k, axes=(-2, -1))
            variants.append(rotated)
            variants.append(np.flip(rotated, axis=-1))
        return variants

    def inflate(self):
        """
        Augment the dataset by appending rotated and flipped copies of each sample
        along the sample dimension.
        """
        augmented_samples = []
        augmented_sample_ids = []

        sample_ids = self.da.coords["sample"].values

        for i, sample_id in enumerate(sample_ids):
            sample = self.da.isel(sample=i).values  # shape: (C, H, W)
            variants = self._augment_array(sample)

            for j, variant in enumerate(variants):
                augmented_samples.append(variant)
                augmented_sample_ids.append(f"{sample_id}_aug{j}")

        augmented_data = np.stack(augmented_samples, axis=0)  # (N*8, C, H, W)

        self.da = xr.DataArray(
            augmented_data,
            dims=("sample", "channel", "x", "y"),
            coords={
                "sample": augmented_sample_ids,
                "channel": self.da.coords["channel"].values,
                "x": self.da.coords["x"].values,
                "y": self.da.coords["y"].values,
            },
            name=self.da.name,
            attrs=self.da.attrs,
        )

    def __getitem__(self, idx):
        sample = self.da.isel(sample=idx)

        # Inputs: all channels except labels
        x = sample.sel(channel=self.input_channels).values.astype(
            np.float32
        )  # (C, H, W)

        # Target: labels only
        y = sample.sel(channel=self.label_channel).values  # (H, W)

        if self.normalize:
            x = self._normalize_per_channel(x)

        # Conventional segmentation target handling:
        # convert labels to integer class indices
        y = np.asarray(y, dtype=np.int64)

        x = torch.from_numpy(x)  # float32
        y = torch.from_numpy(y)  # int64

        if self.return_sample_id:
            sample_id = sample.coords["sample"].item()
            return x, y, sample_id

        return x, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = xr.open_dataset("savetheoceans.nc")
    da = data["data"]
    dataset = XarraySegmentationDataset(da, inflate=True)

    print(len(dataset))

    x, y = dataset[0]
    print(x.shape)  # expected: [11, 240, 240]
    print(y.shape)  # expected: [240, 240]
    print(x.dtype)  # float32
    print(y.dtype)  # long
    plt.imshow(x[0])
    plt.show()
