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
    """Construct the filesystem path for a given image metadata entry."""
    loader_path = [img_entry['id'][i] for i in range(4)]
    loader_path = os.path.join(*loader_path)
    return os.path.join(img_root, loader_path, img_entry['id'])
       
class ImagerLoader(data.Dataset):
    def __init__(self, img_path, transform=None, target_transform=None,
                 loader=default_loader, square=False, data_path=None, partition=None, sem_reg=None):

        if data_path == None:
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

        if sem_reg is not None:
            self.semantic_reg = sem_reg
        else:
            self.semantic_reg = False

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img = None
        while img is None:
            recipId = self.ids[index]
            # we force 80 percent of them to be a mismatch
            if self.partition == 'train':
                match = np.random.uniform() > self.mismtch
            elif self.partition == 'val' or self.partition == 'test':
                match = True
            else:
                raise 'Partition name not well defined'

            target = match and 1 or -1

            with self.env.begin(write=False) as txn:
                serialized_sample = txn.get(self.ids[index].encode('latin1'))
            sample = pickle.loads(serialized_sample,encoding='latin1')
            imgs = sample['imgs']

            # image — try candidate images in order, skipping missing ones
            if target == 1:
                candidates = imgs[:min(5, len(imgs))]
                if self.partition == 'train':
                    # Shuffle so we see variety across epochs
                    idxs = np.random.permutation(len(candidates))
                    candidates = [candidates[i] for i in idxs]
            else:
                # we randomly pick one non-matching recipe
                all_idx = range(len(self.ids))
                rndindex = np.random.choice(all_idx)
                while rndindex == index:
                    rndindex = np.random.choice(all_idx)  # pick a random index

                with self.env.begin(write=False) as txn:
                    serialized_sample = txn.get(self.ids[rndindex].encode('latin1'))

                rndsample = pickle.loads(serialized_sample, encoding='latin1')
                rndimgs = rndsample['imgs']
                candidates = rndimgs[:min(5, len(rndimgs))]
                if self.partition == 'train':
                    idxs = np.random.permutation(len(candidates))
                    candidates = [candidates[i] for i in idxs]

            path = None
            for cand in candidates:
                cand_path = _build_img_path(cand, self.imgPath)
                try:
                    img = self.loader(cand_path)
                    path = cand_path
                    break
                except FileNotFoundError:
                    continue

            if img is None:
                if self.partition == 'train':
                    # Resample a new random recipe instead of using a blank placeholder during training
                    index = np.random.randint(0, len(self.ids))
                else:
                    # All candidate images were missing — use a blank placeholder as last resort
                    print(f"Warning: no valid image found for recipe {recipId}, using blank placeholder.",
                          file=sys.stderr)
                    img = Image.new('RGB', (224, 224), 'white')
                    break

        # instructions
        instrs = sample['intrs']
        itr_ln = len(instrs)
        t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
        t_inst[:itr_ln][:] = instrs
        instrs = torch.FloatTensor(t_inst)

        # ingredients
        ingrs = sample['ingrs'].astype(int)
        ingrs = torch.LongTensor(ingrs)
        igr_ln = max(np.nonzero(sample['ingrs'])[0]) + 1

        # load image
        # (image is already loaded above)

        if self.square:
            img = img.resize(self.square)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        rec_class = sample['classes'] - 1
        rec_id = self.ids[index]

        if target == -1:
            img_class = rndsample['classes'] - 1
            img_id = self.ids[rndindex]
        else:
            img_class = sample['classes'] - 1
            img_id = self.ids[index]

        # output
        if self.partition == 'train':
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
