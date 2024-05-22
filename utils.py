import torch
import torchvision
from dataset import CVPDataset
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F

def save_checkpoint(state, epoch, folder="model_state"):
    filename = f"my_checkpoint_{epoch}.pth.tar"
    filepath = os.path.join(folder,filename)
    print("--> Saving Checkpoint to {}".format(filepath))
    try:
        torch.save(state, filepath)
    except RuntimeError as e:
        print("Could not write file {}".format(filename))
        filename = f"my_checkpoint_{epoch}.txt"
        filepath = os.path.join(folder,filename)
        out = "Error in saving model: \n{}".format(e)
        with open(filepath, 'wb') as f:
            f.write(out)
        

def load_checkpoint(checkpoint, model):
    print("--> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    print("Loading Complete")
    
def get_loaders(
        train_dir,
        train_gt_dir,
        val_dir,
        val_gt_dir,
        batch_size,
        num_workers=4,
        pin_memory=True
):
    train_ds = CVPDataset(
        image_dir=train_dir,
        gt_dir=train_gt_dir
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = CVPDataset(
        image_dir=val_dir,
        gt_dir=val_gt_dir
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    return train_loader, val_loader

def check_accuracy(loader, model, loss_fn, device="cuda"):
    model.eval()
    mse = 0
    with torch.no_grad():
        for filename, x,y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x.float())
            mse += loss_fn(preds, y)

    print("MSE : {}".format(mse/len(loader)))
    model.train()

def save_prediction_as_imgs(
        loader,model, epoch, folder="saved_images",device="cuda"
):
    model.eval()
    for filename, x,y in loader:
        x = x.to(device)
        with torch.no_grad():
            preds = model(x.float())
        for idx, pred in enumerate(preds):
            filename1 = "pred_"+filename[idx]
            folder_path = os.path.join(folder,str(epoch))
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            dest_path=os.path.join(folder_path,filename1)
            try:
                torchvision.utils.save_image(
                    pred,dest_path       
                )
            except:
                print(f"Could not save image {filename1}")
                pass
        # torchvision.utils.save_image(
        #     y, f"{folder}y_{idx}.png"        
        # )
    model.train()





