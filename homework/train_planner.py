"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-4,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    # logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    loss_fn = torch.nn.L1Loss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    global_step = 0
    
    planner_metrics = PlannerMetric()
    
    for epoch in range(num_epoch):
        model.train()
        
        for batch in train_data:
            waypoints, waypoints_mask, track_left, track_right = batch["waypoints"].to(device), batch["waypoints_mask"].to(device), batch["track_left"].to(device), batch["track_right"].to(device)
            
            pred = model(track_left, track_right)

            masked_pred = pred[waypoints_mask]
            masked_target = waypoints[waypoints_mask]

            loss_val = loss_fn(masked_pred, masked_target)

            # loss_val = loss_fn(pred, waypoints)
            
            optim.zero_grad()
            loss_val.backward()
            optim.step()
            
            global_step += 1
            
        with torch.inference_mode():
            model.eval()
            
            for batch in val_data:
                 waypoints, waypoints_mask, track_left, track_right = batch["waypoints"].to(device), batch["waypoints_mask"].to(device), batch["track_left"].to(device), batch["track_right"].to(device)
                 
                 pred = model(track_left, track_right)
                 
                 planner_metrics.add(pred, waypoints, waypoints_mask)
    
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"planner_metrics={planner_metrics.compute()}"
            )
            
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="mlp_planner")
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # pass all arguments to train
    train(**vars(parser.parse_args()))