act = dict(
    type="ETEOStacked",
    num_layers=3,
    d_model=156,
    nhead=3,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=False,
    time_steps=10
)

cri = dict(
    type="ETEOStacked",
    num_layers=3,
    d_model=156,
    nhead=3,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=False,
    time_steps=10
)


"""
act = dict(
    type="ETEOStacked",
    dims=[128,128],
    time_steps=10,
    action_dim=2,
    state_dim =10,
    explore_rate=0.25
)

cri = dict(
    type="ETEOStacked",
    dims=[128,128],
    time_steps=10,
    action_dim=2,
    state_dim =10,
    explore_rate=0.25
)
"""
