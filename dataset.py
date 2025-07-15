
import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

data_transforms = A.Compose([
    A.Resize(256,256),
    A.Normalize(),
    ToTensorV2()
])

from torch.utils.data import Dataset

class CityScapes(Dataset):
    def __init__(self, img_folder, mask_folder, files_list):
        super().__init__()
        self.files_list = files_list
        self.img_folder = img_folder
        self.mask_folder = mask_folder
    #magic method
    def __len__(self): #ada berapa banyak sih datapoint kita yang akan kita train
        return len(self.files_list)

    def __getitem__(self, idx):
        # get the filename
        file_name = self.files_list[idx]

        # get the image and mask file
        file_img = os.path.join(self.img_folder, file_name)
        file_mask = os.path.join(self.mask_folder, file_name)

        image = Image.open(file_img).convert('RGB')
        image = np.array(image)

        mask = Image.open(file_mask).convert('L')
        mask = np.array(mask)

        output = data_transforms(image=image, mask=mask)
        image = output['image']
        masks = output['mask']

        return image, masks
