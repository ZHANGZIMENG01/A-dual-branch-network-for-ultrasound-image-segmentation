from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

class DBU_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(DBU_datasets, self)
        if train:
            images_list = os.listdir(path_Data + 'train/images/')
            enhanceimage_list = os.listdir(path_Data + 'train/enhanceimages/')
            masks_list = os.listdir(path_Data + 'train/masks/')
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/' + images_list[i]
                en_path = path_Data + 'train/enhanceimages/' + enhanceimage_list[i]
                mask_path = path_Data + 'train/masks/' + masks_list[i]
                self.data.append([img_path, en_path, mask_path])
            self.transformer = config.train_transformer

        else:
            images_list = os.listdir(path_Data + 'val/images/')
            enhanceimage_list = os.listdir(path_Data + 'val/enhanceimages/')
            masks_list = os.listdir(path_Data + 'val/masks/')
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                en_path = path_Data + 'val/enhanceimages/' + enhanceimage_list[i]
                mask_path = path_Data + 'val/masks/' + masks_list[i]
                self.data.append([img_path, en_path, mask_path])
            self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path, en_path, mask_path = self.data[indx]
        # 检查图像、增强图像和掩码是否都存在
        if not (os.path.exists(img_path) and os.path.exists(en_path) and os.path.exists(mask_path)):
            raise FileNotFoundError("Missing image, enhance image, or masks file.")

        img = np.array(Image.open(img_path).convert('RGB'))
        en = np.array(Image.open(en_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(mask_path).convert('L')), axis=2) / 255

        img, en, msk = self.transformer((img, en, msk))
        return img, en, msk

    def __len__(self):
        return len(self.data)

