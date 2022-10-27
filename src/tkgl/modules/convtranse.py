import torch


class ConvTransE(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.Dropout(dropout),
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size * out_channels, hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
        )

    def forward(self, *tuples: torch.Tensor):
        """
        Arguments:
            inputs: (num_triplets, in_channels, hidden_size)
            emb: (num_classes, hidden_size)
        Return:
            output: (num_triplets, hidden_size)
        """
        # (num_triplets, out_channels, embed_size)
        feat_maps = self.conv(torch.stack(tuples, dim=1))
        feat_hid = feat_maps.view(feat_maps.size(0), -1)
        return self.mlp(feat_hid)
