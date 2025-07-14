import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F


def calculate_dice_coefficient(ground_truth, predicted):
    gt_np = ground_truth.cpu().numpy()
    pred_np = predicted.detach().cpu().numpy()
    intersection = np.logical_and(gt_np, pred_np)
    dice_coefficient = (2. * np.sum(intersection)) / (np.sum(gt_np) + np.sum(pred_np))

    # intersection = np.logical_and(ground_truth, predicted)
    # dice_coefficient = (2 * np.sum(intersection)) / (np.sum(ground_truth) + np.sum(predicted))
    return dice_coefficient

def calculate_dice_coefficients(ground_truths, predictions):
    num_samples = len(ground_truths)
    dice_coefficients = np.zeros(num_samples)
    for i in range(num_samples):
        dice_coefficients[i] = calculate_dice_coefficient(ground_truths[i], predictions[i])
    return dice_coefficients

# Membuat kelas BinaryDiceLoss, yang merupakan turunan dari nn.Module (modul dasar dalam PyTorch untuk model dan loss function).
class BinaryDiceLoss(nn.Module):
    # Parameter smoothing untuk mencegah pembagian dengan nol dan menstabilkan perhitungan.
    # Pangkat yang digunakan untuk mengukur kontribusi dari prediksi dan target. Biasanya p=2 mirip dengan L2-norm.
    # Metode penggabungan nilai loss dari tiap sampel dalam batch. Opsi yang umum yaitu 'mean', 'sum', atau 'none'.
    def __init__(self, smooth:float=1, p:int=2, reduction:str='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    # Mendefinisikan metode forward, yang merupakan inti dari operasi komputasi dalam modul PyTorch.
    # Parameter predict adalah tensor prediksi dari model. (biasanya setelah sigmoid, bentuk [B, C, H, W] → [B, H, W] jika binary)
    # Parameter target adalah tensor target (label) yang sebenarnya.
    # Fungsi ini diharapkan mengembalikan tensor loss.
    def forward(self, predict:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        # Pastikan batch size (B) dari predict dan target sama.
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        # Mengubah bentuk tensor dari [B, C, H, W] atau [B, H, W] → [B, N], di mana N adalah jumlah piksel per sampel.
        # Mengubah bentuk tensor sehingga tiap sampel (batch) dijadikan satu dimensi (flatten),
        # dengan dimensi pertama mewakili batch size dan dimensi kedua menyatukan semua elemen lainnya.
        # contiguous() memastikan memori tensor tersusun linear. (berurutan dalam memori)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Untuk mengukur seberapa besar overlap antara prediksi dan label target.
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        # Untuk menghitung ukuran total dari prediksi dan ground truth mask dalam satu batch sample
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
            
# Mendefinisikan class DiceLoss sebagai subclass dari nn.Module.
# weight: tensor opsional untuk memberi bobot pada tiap kelas dalam perhitungan loss.
# ignore_index: kelas yang akan diabaikan saat menghitung loss.
# **kwargs: parameter tambahan untuk dikirim ke BinaryDiceLoss.
class DiceLoss(nn.Module):
    def __init__(self, weight:torch.Tensor=None, ignore_index:int=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index


    def forward(self, predict:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        if predict.shape != target.shape:
            num_classes = predict.shape[1]
            target = torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = torch.softmax(predict, dim=1)

        # Loop untuk setiap kelas (i adalah indeks kelas).
        for i in range(target.shape[1]):
            # Mengecek apakah indeks kelas sekarang tidak diabaikan.
            if i != self.ignore_index:
                # Menghitung dice loss untuk kelas ke-i, dengan mengambil channel i dari prediksi dan target.
                dice_loss = dice(predict[:, i], target[:, i])

                # Jika self.weight tidak None, artinya ingin memberi bobot khusus pada kelas.
                # Dice loss akan dikalikan dengan bobot kelas tertentu.
                # Mengecek bahwa jumlah bobot sesuai jumlah kelas (target.shape[1]).
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    
                    dice_loss *= self.weight[i] # Jika valid, maka dice loss untuk kelas ke-i dikalikan bobotnya.
                total_loss += dice_loss # Menambahkan dice loss kelas ke-i ke total loss.

        return total_loss/target.shape[1] # Mengembalikan average dice loss dari seluruh kelas (atau seluruh kelas selain yang di-ignore).

class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.5, ce_weight=0.5, ignore_index=None):
        super().__init__()
        if ignore_index is not None:
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.ce = nn.CrossEntropyLoss(weight=weight)

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)

        # Dice loss expects one-hot targets and softmax inputs
        num_classes = inputs.shape[1]
        smooth = 1e-5

        inputs_soft = F.softmax(inputs, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        intersection = torch.sum(inputs_soft * targets_onehot, dim=(2, 3))
        union = torch.sum(inputs_soft + targets_onehot, dim=(2, 3))
        dice_score = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score.mean()

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


def train_engine(dataloader, model, loss_fn, optim, scaler=torch.cuda.amp.GradScaler()):
    model.train()
    total_loss = 0.0
    
    loop = tqdm(dataloader)
    for data, targets in loop:
        # Move data to GPU
        data = data.to('cuda')
        targets = targets.to('cuda')
        
        # Convert targets to long integers (required for CrossEntropyLoss)
        targets = targets.long()
        
        # Verify target values are valid
        assert targets.min() >= 0, "Target values contain negative numbers"
        assert targets.max() < model.out_channels, f"Target values exceed number of classes ({model.out_channels})"

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type='cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass and optimization
        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        
        total_loss += loss.item()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

def val_engine(dataloader, model, loss_fn):
    model.eval()
    loss_one_step = 0
    loop = tqdm(dataloader)

    for data, targets in loop:
        data = data.to('cuda')
        targets = targets.float().to(device="cuda")
        # Convert targets to long integers (required for CrossEntropyLoss)
        targets = targets.long()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
        loss_one_step += loss.item()

        loop.set_postfix(loss=loss.item())

    return loss_one_step / len(dataloader)

# def train_epoch(model, dataloader, criterion, optimizer, device):
#     #please do training step in this function
#     model.train()
#     loss_one_step = 0
#     loop = tqdm(dataloader)
#     for data, targets in loop:
#         data = data.to('cuda')
#         targets = targets.float().to(device="cuda")
#         # forward
#         with torch.cuda.amp.autocast():
#             predictions = model(data)
#             loss = loss_fn(predictions, targets)
    
#         optim.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optim)
#         scaler.update()
#         loss_one_step += loss.item()
    
#         # update tqdm loop
#         loop.set_postfix(loss=loss.item())
    
#     return loss_one_step / len(dataloader)

# def evaluate(model, dataloader, criterion, device):
#     #please do evaluation step that calculate evaluation loss and evaluation metrics dice_score_coefficient
#     model.eval()
#     loss_one_step = 0
#     loop = tqdm(dataloader)

#     for data, targets in loop:
#         data = data.to('cuda')
#         targets = targets.float().to(device="cuda")
#         with torch.no_grad():
#             with torch.cuda.amp.autocast():
#                 predictions = model(data)
#                 loss = loss_fn(predictions, targets)
#         loss_one_step += loss.item()

#         loop.set_postfix(loss=loss.item())

#     return loss_one_step / len(dataloader)

def train(train_dataloaders, val_dataloaders, model, loss_fn, optim, num_epochs, log_freq=10, save_best_model=False, 
          best_model_name='best_model.pth', last_model_name='last_model.pth', save_path='./'):
    """
    Train the model for a given number of epochs.
    :param train_dataloaders: A dictionary of dataloaders for training and validation.
    :param val_dataloaders: A dictionary of dataloaders for validation.
    :param model: The model to train.
    :param loss_fn: The loss function to use.
    :param optim: The optimizer to use.
    :param num_epochs: The number of epochs to train for.
    :param log_freq: The frequency with which to log training metrics.
    :return: The trained model.
    """
    best_model = None
    best_val_loss = float('inf')

    logs = []
    # best_model_name = os.path.join('drive/MyDrive/UNet/ckpt_save', best_model_name)
    # last_model_name = os.path.join('drive/MyDrive/UNet/ckpt_save', last_model_name)
    
    best_model_name = os.path.join(save_path, best_model_name)
    last_model_name = os.path.join(save_path, last_model_name)

    for epoch in range(num_epochs):
        train_loss = train_engine(train_dataloaders, model, loss_fn, optim)
        val_loss = val_engine(val_dataloaders, model, loss_fn)

        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_best_model:
                best_model = model
                torch.save(best_model.state_dict(), best_model_name)
                torch.save(model.state_dict(), last_model_name)
                is_best = True

        if epoch % log_freq == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            if is_best:
                print(f'✅ Best model saved! (Val Loss: {format(val_loss)})')
            print()
            logs.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })


    return model, logs