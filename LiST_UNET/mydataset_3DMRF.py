import os
import h5py
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class MRFdataset(Dataset):
    def __init__(self):
        super().__init__()
        self.root_dir = None
        self.Dp = None
        self.per = None

        self.img_list = []
        self.label_list = []
        self.mask_list = []
        self.person = []

    def getpath(self, arg, Dp, person):
        self.root_dir = arg.root_dir
        self.Dp = Dp
        self.per = person

        person_dirs = os.listdir(self.root_dir + "_48_500")
        person_dirs.sort(key=lambda x: int(x.split("_")[0]))

        img_list = []
        label_list = []
        mask_list = []
        person_list = []

        for name in person_dirs:
            img_path = os.path.join(
                self.root_dir + "_48_500",
                name,
                "48g_500tr",
                "Rec_dl_input.mat",
            )
            label_path = os.path.join(
                self.root_dir,
                name,
                "48gro_1000",
                "T1T2.mat",
            )
            mask_path = os.path.join(
                self.root_dir,
                name,
                "48gro_1000",
                "T1w_synthseg.nii",
            )

            img_list.append(img_path)
            label_list.append(label_path)
            mask_list.append(mask_path)
            person_list.append(name)

        self.img_list = img_list
        self.label_list = label_list
        self.mask_list = mask_list
        self.person = person_list

        print(self.person)
        return self.person

    def _load_input_image(self, index):
        rec_data = h5py.File(self.img_list[index], "r")["Rec"]
        rec_data = rec_data.transpose(0, 2, 3, 1)
        img_abs = np.abs(rec_data["real"] + 1j * rec_data["imag"]).astype("float32")
        return torch.from_numpy(img_abs)

    def _load_mask(self, index):
        mask = nib.load(self.mask_list[index]).get_fdata()
        mask[mask != 0] = 1

        hole_mask = np.empty((220, 220, 220), dtype=np.float32)
        for i in range(220):
            hole = mask[:, :, i].astype(int)
            hole_mask[:, :, 219 - i] = np.flipud(hole)

        hole_mask = np.rot90(hole_mask, k=1, axes=(1, 0))
        hole_mask = hole_mask[4:196, 4:196, 20:148].astype("float32")

        return torch.from_numpy(hole_mask)

    def _normalize_with_mask(self, img, hole_mask):
        repeat_mask = hole_mask.unsqueeze(0).repeat(5, 1, 1, 1)
        img = img * repeat_mask

        size = img.size()

        max_value = torch.max(torch.max(torch.max(img, 1).values, 1).values, 1).values
        min_value = torch.min(torch.min(torch.min(img, 1).values, 1).values, 1).values

        repeat_max = max_value.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
            1, size[1], size[2], size[3]
        )
        repeat_min = min_value.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
            1, size[1], size[2], size[3]
        )

        normalized = (img - repeat_min) / (repeat_max - repeat_min)
        return normalized

    def _load_t1_label(self, index):
        t1 = h5py.File(self.label_list[index], "r")["T1_find_all"][20:148, 4:196, 4:196]
        t1 = t1.transpose(1, 2, 0).astype("float32")
        t1 = torch.from_numpy(t1) / 4000
        return t1

    def _load_t2_label(self, index):
        t2 = h5py.File(self.label_list[index], "r")["T2_find_all"][20:148, 4:196, 4:196]
        t2 = t2.transpose(1, 2, 0).astype("float32")
        t2 = torch.from_numpy(t2) / 300
        return t2

    def __getitem__(self, item):
        img = self._load_input_image(item)
        hole_mask = self._load_mask(item)
        norvalue = self._normalize_with_mask(img, hole_mask)

        t1 = self._load_t1_label(item)
        t2 = self._load_t2_label(item)

        return norvalue, t1, hole_mask, self.person[item]

    def __len__(self):
        return len(self.img_list)

   
