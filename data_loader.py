from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import sys
import pickle
import numpy as np
import lmdb
import torch

def default_loader(path):
    """Load an image, raising FileNotFoundError if it cannot be read."""
    try:
        im = Image.open(path)
        # Handle palette images with transparency properly to avoid UserWarning
        if im.mode == 'P' and 'transparency' in im.info:
            im = im.convert('RGBA')
        return im.convert('RGB')
    except Exception as e:
        raise FileNotFoundError(f"Could not load image at {path}") from e


def _build_img_path(img_entry, img_root):
    """Construct the filesystem path for a given image metadata entry.

    Recipe1M+ images use a 4-level directory structure keyed on the first four
    characters of the image stem, e.g.:
        id: 3e23a9b850.jpg
        path: <img_root>/3/e/2/3/3e23a9b850.jpg
    """
    stem = img_entry['id']
    return os.path.join(img_root, stem[0], stem[1], stem[2], stem[3], stem)


def _load_first_valid_image(candidates, img_root, loader):
    """Try each candidate image in order; return (img, path) for the first
    one that loads successfully, or (None, None) if all fail."""
    for cand in candidates:
        cand_path = _build_img_path(cand, img_root)
        try:
            img = loader(cand_path)
            return img, cand_path
        except FileNotFoundError:
            continue
    return None, None


class ImagerLoader(data.Dataset):
    def __init__(self, img_path, transform=None, target_transform=None,
                 loader=default_loader, square=False, data_path=None, partition=None, sem_reg=None):

        if data_path is None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition

        self.env = lmdb.open(os.path.join(data_path, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with open(os.path.join(data_path, partition + '_keys.pkl'), 'rb') as f:
            self.ids = pickle.load(f)

        self.square = square
        self.imgPath = img_path
        self.mismtch = 0.8
        self.maxInst = 20

        # Maximum number of resample attempts during training before giving up.
        # This prevents infinite loops if the dataset has widespread missing images.
        self._max_resample = 10

        if sem_reg is not None:
            self.semantic_reg = sem_reg
        else:
            self.semantic_reg = False

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def _get_recipe(self, index):
        """Fetch a serialised sample from LMDB by index."""
        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index].encode('latin1'))
        return pickle.loads(serialized_sample, encoding='latin1')

    def __getitem__(self, index):
        is_train = (self.partition == 'train')

        # ------------------------------------------------------------------
        # Training: resample up to _max_resample times if no image is found.
        # Val/test: raise immediately — no missing images should reach here
        #           after correct LMDB preprocessing.
        # ------------------------------------------------------------------
        for attempt in range(self._max_resample if is_train else 1):

            recipId = self.ids[index]
            sample  = self._get_recipe(index)

            # Decide match/mismatch
            if is_train:
                match = np.random.uniform() > self.mismtch
            else:
                match = True  # val/test are always matched pairs

            target = 1 if match else -1

            imgs = sample['imgs']

            # ---- Pick candidate image list --------------------------------
            if target == 1:
                candidates = imgs[:min(5, len(imgs))]
                if is_train:
                    idxs = np.random.permutation(len(candidates))
                    candidates = [candidates[i] for i in idxs]
            else:
                # Randomly pick a non-matching recipe for the mismatch case
                all_idx = range(len(self.ids))
                rndindex = np.random.choice(all_idx)
                while rndindex == index:
                    rndindex = np.random.choice(all_idx)

                rndsample = self._get_recipe(rndindex)
                rndimgs   = rndsample['imgs']
                candidates = rndimgs[:min(5, len(rndimgs))]
                if is_train:
                    idxs = np.random.permutation(len(candidates))
                    candidates = [candidates[i] for i in idxs]

            # ---- Load image -----------------------------------------------
            img, path = _load_first_valid_image(candidates, self.imgPath, self.loader)

            if img is not None:
                break  # success — exit the resample loop

            # img is None — handle depending on partition
            if not is_train:
                # Val/test must never reach here if LMDB was built correctly.
                raise RuntimeError(
                    f"[{self.partition}] No valid image found for recipe '{recipId}'. "
                    "This should not happen after correct LMDB preprocessing. "
                    "Re-run mk_dataset_subset.py with --img_path to rebuild the dataset."
                )

            # Training: try a different random recipe next iteration
            index = np.random.randint(0, len(self.ids))

        else:
            # Exhausted all resample attempts during training
            raise RuntimeError(
                f"[train] Failed to find a valid image after {self._max_resample} attempts. "
                "The training set likely has widespread missing images. "
                "Re-run mk_dataset_subset.py with --img_path to rebuild the dataset."
            )

        # ------------------------------------------------------------------
        # Build instruction tensor
        # ------------------------------------------------------------------
        instrs = sample['intrs']
        itr_ln = len(instrs)
        t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
        t_inst[:itr_ln][:] = instrs
        instrs = torch.FloatTensor(t_inst)

        # Build ingredient tensor
        ingrs   = sample['ingrs'].astype(int)
        ingrs   = torch.LongTensor(ingrs)
        igr_ln  = max(np.nonzero(sample['ingrs'])[0]) + 1

        # Apply image transforms
        if self.square:
            img = img.resize(self.square)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        rec_class = sample['classes'] - 1
        rec_id    = self.ids[index]

        if target == -1:
            img_class = rndsample['classes'] - 1
            img_id    = self.ids[rndindex]
        else:
            img_class = sample['classes'] - 1
            img_id    = self.ids[index]

        # Output
        if is_train:
            if self.semantic_reg:
                return [img, instrs, itr_ln, ingrs, igr_ln], [target, img_class, rec_class]
            else:
                return [img, instrs, itr_ln, ingrs, igr_ln], [target]
        else:
            if self.semantic_reg:
                return [img, instrs, itr_ln, ingrs, igr_ln], [target, img_class, rec_class, img_id, rec_id]
            else:
                return [img, instrs, itr_ln, ingrs, igr_ln], [target, img_id, rec_id]

    def __len__(self):
        return len(self.ids)
