"""HAR models: LSTM, Transformer, CfC. Input (B, 128, 9) -> logits (B, 6).

Models return logits only (no loss). Param target: ~80k-100k each.
"""
import math
import torch
import torch.nn as nn

DROPOUT = 0.1


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class LSTMModel(nn.Module):
    def __init__(self, input_dim=9, num_classes=6, hidden=144, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(self.drop(h[-1]))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_dim=9, num_classes=6, d_model=64, nhead=4,
                 num_layers=2, dim_ff=192, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        z = self.encoder(self.pos(self.proj(x)))
        return self.fc(self.drop(z.mean(dim=1)))


class CfCModel(nn.Module):
    def __init__(self, input_dim=9, num_classes=6, units=64, dropout=DROPOUT):
        super().__init__()
        from ncps.torch import CfC  # lazy import so LSTM/Transformer work without ncps
        self.cfc = CfC(input_dim, units, batch_first=True,
                       return_sequences=False, backbone_dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(units, num_classes)

    def forward(self, x):
        out, _ = self.cfc(x)
        return self.fc(self.drop(out))


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return LSTMModel(**kwargs)
    if name == "transformer":
        return TransformerModel(**kwargs)
    if name == "cfc":
        return CfCModel(**kwargs)
    raise ValueError(f"Unknown model '{name}'. Use lstm | transformer | cfc.")


if __name__ == "__main__":
    x = torch.randn(32, 128, 9)
    for n in ["lstm", "transformer", "cfc"]:
        m = build_model(n)
        y = m(x)
        assert y.shape == (32, 6), y.shape
        print(f"{n:12s} params={count_params(m):>8,d} out={tuple(y.shape)}")