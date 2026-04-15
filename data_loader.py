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
    ):
        self.da = da
        self.label_channel = label_channel
        self.normalize = normalize
        self.epsilon = epsilon
        self.return_sample_id = return_sample_id

        required_dims = {"sample", "channel", "x", "y"}
        missing_dims = required_dims - set(self.da.dims)
        if missing_dims:
            raise ValueError(f"Missing required dimensions: {missing_dims}")

        if "channel" not in self.da.coords:
            raise ValueError("DataArray must have 'channel' coordinates.")

        self.channel_names = list(self.da.coords["channel"].values)

        if self.label_channel not in self.channel_names:
            raise ValueError(
                f"Label channel '{self.label_channel}' not found in channels: {self.channel_names}"
            )

        self.input_channels = [
            ch for ch in self.channel_names if ch != self.label_channel
        ]

        if len(self.input_channels) == 0:
            raise ValueError("No input channels found after excluding label channel.")

    def __len__(self):
        return self.da.sizes["sample"]

    def _normalize_per_channel(self, x: torch.tensor) -> torch.tensor:
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

    def __getitem__(self, idx):
        sample = self.da.isel(sample=idx)
        x = sample.sel(channel=self.input_channels).values  # (C, H, W)
        y = sample.sel(channel=self.label_channel).values  # (H, W)
        if self.normalize:
            x = self._normalize_per_channel(x)

        x = torch.tensor(x, dtype=torch.float32)  # float32
        y = torch.tensor(y, dtype=torch.int64)  # int64

        if self.return_sample_id:
            sample_id = sample.coords["sample"].item()
            return x, y, sample_id


if __name__ == "__main__":
    data = xr.open_dataset("savetheoceans.nc")
    da = data["data"]
    dataset = XarraySegmentationDataset(da)

    print(len(dataset))

    x, y = dataset[0]
    print(x.shape)  # expected: [11, 240, 240]
    print(y.shape)  # expected: [240, 240]
    print(x.dtype)  # float32
    print(y.dtype)  # long
