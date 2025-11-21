import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt

def train(
        model,
        config,
        train_loader,
        optimizer,
        lr_scheduler,
        best_val_loss = 1e8,
        start_epoch = 0,
        valid_loader=None,
        valid_epoch_interval=5,
        foldername="",
        seed = 0,
):
    # Control random seed in the current script.
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p0 = int(0.25 * config["epochs"])
    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    p3 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p0, p1, p2, p3], gamma=0.1
    )
    history = {'train_loss': [], 'val_rmse': []}
    best_valid_loss = best_val_loss
    train_loss = []
    for epoch_no in range(start_epoch, config["epochs"]):
        avg_loss = 0
        train_batches = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                # The forward method returns loss.
                loss = model(train_batch)
                train_loss.append(loss.detach().cpu().numpy())
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                train_batches = batch_no
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
            history['train_loss'].append(avg_loss / train_batches)
        if (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for val_batch in valid_loader:
                    val_loss = model(val_batch)
                    avg_val_loss += val_loss.item()
                    val_batches += 1
            avg_val_loss /= val_batches
            history['val_rmse'].append(avg_val_loss)
            print(f"Epoch {epoch_no}: Validation Loss = {avg_val_loss:.4f}")
            if avg_val_loss < best_valid_loss:
                best_valid_loss = avg_val_loss
                checkpoint_path = os.path.join(foldername, f'checkpoint_epoch_{epoch_no + 1}.ckpt')
                torch.save({
                    'epoch': epoch_no + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_valid_loss': best_valid_loss,
                }, checkpoint_path)
                print(f"New best model saved with validation loss: {best_valid_loss:.4f}")
            elif (epoch_no+1)%5 == 0:
                checkpoint_path = os.path.join(foldername, f'checkpoint_epoch_{epoch_no + 1}.ckpt')
                torch.save({
                    'epoch': epoch_no + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_valid_loss': best_valid_loss,
                }, checkpoint_path)
        
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    
    # Plot validation loss with correct epoch intervals
    val_epochs = range(valid_epoch_interval - 1, config['epochs'], valid_epoch_interval)
    plt.plot(val_epochs, history['val_loss'], label='Validation Loss', marker='o')
    
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(foldername + '/loss_curve.png')
    plt.close()

def evaluate(model, test_loader, nsample=1000, scaler=1, mean_scaler=0, foldername=""):
    torch.manual_seed(0)
    np.random.seed(0)

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)
                c_target = c_target.permute(0, 2, 1)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                                      ((samples_median.values - c_target) * eval_points) ** 2
                              ) * (scaler ** 2)
                mae_current = (
                                  torch.abs((samples_median.values - c_target) * eval_points)
                              ) * scaler
                mse_total += torch.sum(mse_current, dim=0)
                mae_total += torch.sum(mae_current, dim=0)
                evalpoints_total += torch.sum(eval_points, dim=0)
                it.set_postfix(
                    ordered_dict={
                        "rmse_total": torch.mean(
                            torch.sqrt(torch.div(mse_total, evalpoints_total))
                        ).item(),
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

    print("RMSE:", torch.mean(torch.sqrt(torch.div(mse_total, evalpoints_total))).item(), )


def genera(model, genera_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", max_arr=None, attention_mech='', gene_names = None):
    if max_arr is None:
        max_arr = np.array([])
    torch.manual_seed(0)
    np.random.seed(0)

    imputed_samples = []
    observed_data_all = []
    cond_mask_all = []
    observed_mask_all = []

    with torch.no_grad():
        model.eval()

        with tqdm(genera_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, gen_batch in enumerate(it, start=1):
                output = model.genera(gen_batch, nsample)
                samples, observed_data, target_mask, observed_mask, _ = output
                observed_data_all.append(observed_data.squeeze())
                cond_mask_all.append(target_mask.squeeze())
                observed_mask_all.append(observed_mask.squeeze())
                samples = samples.permute(0, 1, 3, 2)
                samples_median = samples.mean(dim=1)
                imputed_samples.append(samples_median.squeeze())

                # if batch_no % 100 == 0:
                #     imputed_samples_ndarray = torch.cat(imputed_samples, dim=0).cpu().numpy()
                #     observed_data_all_ndarray = torch.cat(observed_data_all, dim=0).cpu().numpy()
                #     cond_mask_all_ndarray = torch.cat(cond_mask_all, dim=0).cpu().numpy()
                #     indices = np.where(cond_mask_all_ndarray == 1)
                #     observed_data_all_ndarray[indices] = imputed_samples_ndarray[indices]
                #     observed_data_all_ndarray = np.abs(
                #         (observed_data_all_ndarray * (max_arr + 1) - 1))
                #     if gene_names:
                #         pd.DataFrame(observed_data_all_ndarray).to_csv(f'./imputed_{batch_no}.csv', header=gene_names, index=False)
                #     else:
                #         pd.DataFrame(observed_data_all_ndarray).to_csv(f'./imputed_{batch_no}.csv', header=False, index=False)


        # very end
        imputed_samples_ndarray = torch.cat(imputed_samples, dim=0).cpu().numpy()
        observed_data_all_ndarray = torch.cat(observed_data_all, dim=0).cpu().numpy()
        cond_mask_all_ndarray = torch.cat(cond_mask_all, dim=0).cpu().numpy()
        indices = np.where(cond_mask_all_ndarray == 1)
        observed_data_all_ndarray[indices] = imputed_samples_ndarray[indices]
        observed_data_all_ndarray = np.abs(
            (observed_data_all_ndarray * (max_arr + 1) - 1))
        # if gene_names:
        #     pd.DataFrame(observed_data_all_ndarray).to_csv(f'./imputed_total.csv', header=gene_names, index=False)
        # else:
        #     pd.DataFrame(observed_data_all_ndarray).to_csv(f'./imputed_total.csv', header=False, index=False)
        return observed_data_all_ndarray
