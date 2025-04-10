import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from src.utils.io import save_results
import time


def test(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    eval_interval: int,
    save_interval: int,
    exp_name: str,
    device: str,
) -> pd.DataFrame:

    results = pd.DataFrame(columns=["Batch", "Loss", "Accuracy", "Time"])

    model.eval()
    print("TESTING BEGUN")
    print(exp_name)
    start_time = time.time()

    total_loss = 0
    total_accuracy = 0

    with torch.inference_mode():
        for batch_num, batch in enumerate(dataloader, start=1):
            x, y_target = batch

            x = x.to(device)
            y_target = y_target.to(device)

            logits: torch.Tensor = model(x)

            y_pred = logits[:, -1, :]

            loss = loss_fn(
                y_pred,
                y_target,
            )

            batch_size = x.size(0)
            total_batches = len(dataloader)

            accuracy = (
                y_target == torch.argmax(F.softmax(y_pred, dim=-1), dim=-1)
            ).sum().item() / batch_size

            total_loss += loss.item()
            total_accuracy += accuracy

            is_final_batch = batch_num == total_batches

            if batch_num % eval_interval == 0 or is_final_batch:

                avg_loss = total_loss / batch_num
                avg_accuracy = total_accuracy / batch_num
                elapsed_time = time.time() - start_time

                print(
                    f"Batch: {batch_num} | "
                    f"Loss: {avg_loss} | "
                    f"Accuracy: {avg_accuracy} | "
                    f"Time: {elapsed_time:.3f}"
                )

                results.loc[len(results)] = [
                    batch_num,
                    avg_loss,
                    avg_accuracy,
                    elapsed_time,
                ]

            if batch_num % save_interval == 0 or is_final_batch:
                save_results(results, exp_name)

    return results
