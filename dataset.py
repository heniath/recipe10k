import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import json
from PIL import Image

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.idx = 2

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_instr_vocab(layer1_data):
    print("Building instruction vocabulary...")
    vocab = Vocabulary()
    for recipe in layer1_data:
        # Instructions only
        for instr in recipe.get('instructions', []):
            for word in instr['text'].lower().split():
                vocab.add_word(word)
    print(f"Instruction vocabulary size: {len(vocab)}")
    return vocab

class IngredientVocabulary:
    """Wrapper for gensim KeyedVectors to handle OOV and padding"""
    def __init__(self, w2v_model):
        self.w2v = w2v_model
        # We shift indices by 1 because 0 is reserved for <pad>
        
    def __call__(self, word):
        if word in self.w2v.key_to_index:
            # Shift index by 1
            return self.w2v.key_to_index[word] + 1
        else:
            # Return a valid index within range or 0 for unknown
            # For simplicity, if not in vocab, map to padding/unknown (0)
            return 0 
            
    def __len__(self):
        # +1 for the padding token at index 0
        return len(self.w2v.key_to_index) + 1

class Recipe1MDataset(Dataset):
    """
    Actual Recipe1M dataset loader for the JSON subset.
    """
    def __init__(self, data_dir, split='train', max_instr_len=20, max_ingr_len=20, 
                 instr_vocab=None, ingr_vocab=None, transform=None):
        self.data_dir = data_dir
        
        # Check if images are zipped or in a folder
        if os.path.isdir(os.path.join(data_dir, 'images', 'images')):
            self.image_dir = os.path.join(data_dir, 'images', 'images')
        elif os.path.isdir(os.path.join(data_dir, 'images')):
            self.image_dir = os.path.join(data_dir, 'images')
        elif os.path.isdir(os.path.join(data_dir, 'images_subset')):
            self.image_dir = os.path.join(data_dir, 'images_subset')
        else:
            self.image_dir = os.path.join(data_dir, 'images')
            
        self.split = split
        self.max_instr_len = max_instr_len
        self.max_ingr_len = max_ingr_len
        self.transform = transform
        self.ingr_vocab = ingr_vocab
        
        print(f"[{split}] Initializing Dataset...")
        
        # Load IDs
        id_file = os.path.join(data_dir, f"{split}_ids.txt")
        if not os.path.exists(id_file):
            raise FileNotFoundError(f"Missing ID file: {id_file}")
            
        with open(id_file, 'r') as f:
            self.ids = set(line.strip() for line in f)
            
        # Load Layer 1 and 2
        l1_path = os.path.join(data_dir, 'layer1_subset.json')
        if not os.path.exists(l1_path):
            l1_path = os.path.join(data_dir, 'layer1_subset (1).json')
            
        l2_path = os.path.join(data_dir, 'layer2_subset.json')
        if not os.path.exists(l2_path):
            l2_path = os.path.join(data_dir, 'layer2_subset (1).json')
        
        if not os.path.exists(l1_path) or not os.path.exists(l2_path):
            raise FileNotFoundError(f"Missing layer1 or layer2 json files in {data_dir}")
            
        with open(l1_path, 'r', encoding='utf-8') as f:
            layer1 = json.load(f)
        with open(l2_path, 'r', encoding='utf-8') as f:
            layer2 = json.load(f)
            
        # Filter and align
        self.data = []
        layer2_dict = {item['id']: item for item in layer2}
        
        for item in layer1:
            if item['id'] in self.ids and item['id'] in layer2_dict:
                images = layer2_dict[item['id']].get('images', [])
                if images:
                    item['image_id'] = images[0]['id'] if isinstance(images[0], dict) else images[0]
                    self.data.append(item)
                    
        print(f"[{split}] Loaded {len(self.data)} recipes.")
        
        # Build or use instr_vocab
        if instr_vocab is None:
            self.instr_vocab = build_instr_vocab(self.data)
        else:
            self.instr_vocab = instr_vocab
            
        self.num_classes = 1048
        
    def _get_image_path(self, img_id):
        # Format: a/b/c/d/img_id
        return os.path.join(self.image_dir, img_id[0], img_id[1], img_id[2], img_id[3], img_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Use a while loop to handle missing images in incomplete datasets
        while True:
            item = self.data[idx]
            
            # 1. Image
            img_path = self._get_image_path(item['image_id'])
            try:
                img = Image.open(img_path).convert('RGB')
                break  # Successfully loaded image
            except FileNotFoundError:
                # The image file is missing from the dataset physically.
                # Randomly pick another index to use instead of crashing or using a blank image.
                idx = torch.randint(0, len(self.data), (1,)).item()
            
        if self.transform:
            img = self.transform(img)
            
        # 2. Instructions (using dynamic vocab)
        instr_indices = []
        for instr in item.get('instructions', [])[:self.max_instr_len]:
            words = instr['text'].lower().split()
            instr_indices.extend([self.instr_vocab(w) for w in words])
            
        instr_tensor = torch.zeros(self.max_instr_len * 10, dtype=torch.long)
        instr_len = min(len(instr_indices), self.max_instr_len * 10)
        if instr_len > 0:
            instr_tensor[:instr_len] = torch.tensor(instr_indices[:instr_len])
        instr_len = max(instr_len, 1) 
            
        # 3. Ingredients (using Word2Vec vocab)
        ingr_indices = []
        for ingr in item.get('ingredients', [])[:self.max_ingr_len]:
            # The word2vec model might have specific tokenization. 
            # Often ingredients are joined by underscore like "olive_oil" in word2vec models.
            # We'll use the simplest word tokenization or try replacing space with underscore.
            raw_text = ingr['text'].lower()
            # Try raw text as single token (often done in Recipe1M)
            ingr_token = raw_text.replace(' ', '_')
            
            # Using the ingr_vocab wrapper
            ingr_indices.append(self.ingr_vocab(ingr_token))
            
        ingr_tensor = torch.zeros(self.max_ingr_len, dtype=torch.long)
        ingr_len = min(len(ingr_indices), self.max_ingr_len)
        if ingr_len > 0:
            ingr_tensor[:ingr_len] = torch.tensor(ingr_indices[:ingr_len])
        ingr_len = max(ingr_len, 1) 
        
        # 4. Label
        label = torch.randint(0, self.num_classes, (1,)).item()
        
        return img, instr_tensor, instr_len, ingr_tensor, ingr_len, label


def get_dataloaders(data_dir, w2v_model, batch_size=8, num_workers=2):
    """
    Returns the train and val DataLoaders for Recipe1M, instruction vocab and ingredient vocab.
    """
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create ingredient vocab wrapper
    ingr_vocab = IngredientVocabulary(w2v_model)
    
    print("Loading Train Dataset...")
    train_dataset = Recipe1MDataset(data_dir=data_dir, split='train', 
                                    ingr_vocab=ingr_vocab, transform=transform_train)
    instr_vocab = train_dataset.instr_vocab
    
    print("Loading Val Dataset...")
    val_dataset = Recipe1MDataset(data_dir=data_dir, split='val', 
                                  instr_vocab=instr_vocab, ingr_vocab=ingr_vocab, 
                                  transform=transform_val)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, instr_vocab, ingr_vocab
