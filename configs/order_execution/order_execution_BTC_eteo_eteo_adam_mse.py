task_name = "order_execution"
dataset_name = "BTC"
net_name = "eteo"
agent_name = "eteo"
optimizer_name = "adam"
loss_name = "mse"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/eteo_trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
    f"../_base_/transition/transition.py"
]

batch_size = 64

data = dict(
    type="OrderExecutionDataset",
    data_path="data/order_execution/BTC",
    train_path="data/order_execution/BTC/train.csv",
    valid_path="data/order_execution/BTC/valid.csv",
    test_path="data/order_execution/BTC/test.csv",
    test_dynamic_path="data/order_execution/BTC/Market_Dynamics_Model/OE_BTC/test_labeled_slice_and_merge_model_3dynamics_minlength12_quantile_labeling.csv",
    tech_indicator_list=[
        "midpoint",
        "spread",
        "buys",
        "sells",
        "bids_distance_0",
        "bids_distance_1",
        "bids_distance_2",
        "bids_distance_3",
        "bids_distance_4",
        "bids_distance_5",
        "bids_distance_6",
        "bids_distance_7",
        "bids_distance_8",
        "bids_distance_9",
        "bids_distance_10",
        "bids_distance_11",
        "bids_distance_12",
        "bids_distance_13",
        "bids_distance_14",
        "bids_notional_0",
        "bids_notional_1",
        "bids_notional_2",
        "bids_notional_3",
        "bids_notional_4",
        "bids_notional_5",
        "bids_notional_6",
        "bids_notional_7",
        "bids_notional_8",
        "bids_notional_9",
        "bids_notional_10",
        "bids_notional_11",
        "bids_notional_12",
        "bids_notional_13",
        "bids_notional_14",
        "bids_cancel_notional_0",
        "bids_cancel_notional_1",
        "bids_cancel_notional_2",
        "bids_cancel_notional_3",
        "bids_cancel_notional_4",
        "bids_cancel_notional_5",
        "bids_cancel_notional_6",
        "bids_cancel_notional_7",
        "bids_cancel_notional_8",
        "bids_cancel_notional_9",
        "bids_cancel_notional_10",
        "bids_cancel_notional_11",
        "bids_cancel_notional_12",
        "bids_cancel_notional_13",
        "bids_cancel_notional_14",
        "bids_limit_notional_0",
        "bids_limit_notional_1",
        "bids_limit_notional_2",
        "bids_limit_notional_3",
        "bids_limit_notional_4",
        "bids_limit_notional_5",
        "bids_limit_notional_6",
        "bids_limit_notional_7",
        "bids_limit_notional_8",
        "bids_limit_notional_9",
        "bids_limit_notional_10",
        "bids_limit_notional_11",
        "bids_limit_notional_12",
        "bids_limit_notional_13",
        "bids_limit_notional_14",
        "bids_market_notional_0",
        "bids_market_notional_1",
        "bids_market_notional_2",
        "bids_market_notional_3",
        "bids_market_notional_4",
        "bids_market_notional_5",
        "bids_market_notional_6",
        "bids_market_notional_7",
        "bids_market_notional_8",
        "bids_market_notional_9",
        "bids_market_notional_10",
        "bids_market_notional_11",
        "bids_market_notional_12",
        "bids_market_notional_13",
        "bids_market_notional_14",
        "asks_distance_0",
        "asks_distance_1",
        "asks_distance_2",
        "asks_distance_3",
        "asks_distance_4",
        "asks_distance_5",
        "asks_distance_6",
        "asks_distance_7",
        "asks_distance_8",
        "asks_distance_9",
        "asks_distance_10",
        "asks_distance_11",
        "asks_distance_12",
        "asks_distance_13",
        "asks_distance_14",
        "asks_notional_0",
        "asks_notional_1",
        "asks_notional_2",
        "asks_notional_3",
        "asks_notional_4",
        "asks_notional_5",
        "asks_notional_6",
        "asks_notional_7",
        "asks_notional_8",
        "asks_notional_9",
        "asks_notional_10",
        "asks_notional_11",
        "asks_notional_12",
        "asks_notional_13",
        "asks_notional_14",
        "asks_cancel_notional_0",
        "asks_cancel_notional_1",
        "asks_cancel_notional_2",
        "asks_cancel_notional_3",
        "asks_cancel_notional_4",
        "asks_cancel_notional_5",
        "asks_cancel_notional_6",
        "asks_cancel_notional_7",
        "asks_cancel_notional_8",
        "asks_cancel_notional_9",
        "asks_cancel_notional_10",
        "asks_cancel_notional_11",
        "asks_cancel_notional_12",
        "asks_cancel_notional_13",
        "asks_cancel_notional_14",
        "asks_limit_notional_0",
        "asks_limit_notional_1",
        "asks_limit_notional_2",
        "asks_limit_notional_3",
        "asks_limit_notional_4",
        "asks_limit_notional_5",
        "asks_limit_notional_6",
        "asks_limit_notional_7",
        "asks_limit_notional_8",
        "asks_limit_notional_9",
        "asks_limit_notional_10",
        "asks_limit_notional_11",
        "asks_limit_notional_12",
        "asks_limit_notional_13",
        "asks_limit_notional_14",
        "asks_market_notional_0",
        "asks_market_notional_1",
        "asks_market_notional_2",
        "asks_market_notional_3",
        "asks_market_notional_4",
        "asks_market_notional_5",
        "asks_market_notional_6",
        "asks_market_notional_7",
        "asks_market_notional_8",
        "asks_market_notional_9",
        "asks_market_notional_10",
        "asks_market_notional_11",
        "asks_market_notional_12",
        "asks_market_notional_13",
        "asks_market_notional_14",
    ],
    length_keeping=30,
    state_length=10,
    target_order=1,
    initial_amount=100000,
    test_dynamic=0
)

environment = dict(
    type="OrderExecutionETEOEnvironment",
)

transition = dict(
    type = "Transition"
)

agent = dict(
    type="OrderExecutionETEO",
    gamma=0.9,
    climp=0.2,
    reward_scale=1,
    repeat_times=1,
)

trainer = dict(
    type="OrderExecutionETEOTrainer",
    epochs=3,
    work_dir=work_dir,
    seeds_list=(12345,),
    batch_size=batch_size,
    horizon_len=2,
    buffer_size=2 * batch_size,
    num_threads=8,
    if_remove=False,
    if_discrete=True,
    if_off_policy=False,
    if_keep_save=True,
    if_over_write=False,
    if_save_buffer=False,
)

loss = dict(type='MSELoss')

optimizer = dict(type='Adam', lr=0.001)

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
