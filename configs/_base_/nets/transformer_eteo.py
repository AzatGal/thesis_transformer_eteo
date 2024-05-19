act = dict(
    type="transformerETEOstacked",
    num_layers=3,
    d_model=156,
    nhead=3,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=False,
    time_steps=10
)

cri = dict(
    type="transformerETEOstacked",
    num_layers=3,
    d_model=156,
    nhead=3,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=False,
    time_steps=10
)