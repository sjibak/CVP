import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
from tqdm import tqdm
# from model import UNetPixelwiseRegression
from unet import UNET
from dataset import CVPDataset
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_prediction_as_imgs
)


# def train_model(model, dataloader, criterion, optimizer, num_epochs):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     for epoch in range(num_epochs):
#         model.train()  # Set the model to training mode
#         total_loss = 0.0

#         with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
#             for inputs, targets in dataloader:
#                 inputs, targets = inputs.to(device), targets.to(device)

#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 loss.backward()
#                 optimizer.step()

#                 total_loss += loss.item()
#                 pbar.set_postfix(loss=f'{loss.item():.4f}')
#                 pbar.update()

#         average_loss = total_loss / len(dataloader)
#         print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}')


# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/train_img_dir/"
TRAIN_GT_DIR = "data/train_gt_dir/"
VAL_IMG_DIR = "data/val_img_dir/"
VAL_GT_DIR = "data/val_gt_dir"
MODEL_STATE_DIR = "model_state/"
PRED_IMG_DIR = "saved_images/" 


def train_func(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    # model = model.float()
    # model.to(device=DEVICE)
    for batch_idx, (filename, data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        # print(DEVICE)
        targets = targets.to(device=DEVICE)   #.unsqueeze(1)

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data.float())
            loss = loss_fn(predictions, targets)

        #backward
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        loop.set_postfix(loss=loss.item())


def main():
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_GT_DIR,
        VAL_IMG_DIR,
        VAL_GT_DIR,
        BATCH_SIZE
    )
    train_losses = []
    valid_losses = []

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_func(train_loader, model, optimizer, loss_fn, scaler)

        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint, epoch, folder=MODEL_STATE_DIR)

        #check accuracy
        check_accuracy(val_loader, model, loss_fn, device=DEVICE)

        save_prediction_as_imgs(
            val_loader,model,epoch,folder=PRED_IMG_DIR,device=DEVICE
        )
if __name__ == "__main__":
    main()


