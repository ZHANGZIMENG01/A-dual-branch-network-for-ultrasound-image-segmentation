import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split

class DBU_datasets(Dataset):
    def __init__(self, path_Data, config, fold_index=0, num_folds=5, train=True):
        super(DBU_datasets, self).__init__()
        self.num_folds = num_folds
        self.fold_index = fold_index
        self.train = train
        self.path_Data = path_Data
        self.config = config

        images_list = os.listdir(path_Data + 'val/images/')
        enhanceimage_list = os.listdir(path_Data + 'val/enhanceimages/')
        masks_list = os.listdir(path_Data + 'val/masks/')

        data = []
        for i in range(len(images_list)):
            img_path = path_Data + 'val/images/' + images_list[i]
            en_path = path_Data + 'val/enhanceimages/' + enhanceimage_list[i]
            mask_path = path_Data + 'val/masks/' + masks_list[i]
            data.append([img_path, en_path, mask_path])

        folds = np.array_split(data, num_folds)

        if self.train:
            self.data = []
            for i, fold in enumerate(folds):
                if i != self.fold_index:
                    self.data.extend(fold)
            self.transformer = config.train_transformer
        else:
            self.data = folds[self.fold_index]
            self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path, en_path, mask_path = self.data[indx]

        if not (os.path.exists(img_path) and os.path.exists(en_path) and os.path.exists(mask_path)):
            raise FileNotFoundError("Missing image, enhance image, or masks file.")

        img = np.array(Image.open(img_path).convert('RGB'))
        en = np.array(Image.open(en_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(mask_path).convert('L')), axis=2) / 255

        img, en, msk = self.transformer((img, en, msk))
        return img, en, msk

    def __len__(self):
        return len(self.data)

