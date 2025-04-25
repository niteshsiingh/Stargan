#coding: utf-8

import os
import time
import random
import torch
import torchaudio

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 validation=False,
                 ):
        
        _data_list = [l[:-1].split('|') for l in data_list][:-1]
        for x in _data_list:
            if "\ufeff" in x[0]:
                x[0] = x[0].split("\ufeff")[-1]
        
        self.data_list = [(path, int(label)) for path, label in _data_list]
        
        # Track all labels
        all_labels = list(set([label for _, label in self.data_list]))
        self.data_list_per_class = {
            target: [(path, label) for path, label in self.data_list if label == target] \
            for target in all_labels
        }

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
        self.mean, self.std = -4, 4
        self.validation = validation
        self.max_mel_length = 192
        
        # Track valid sample IDs (initially all are assumed valid)
        self.valid_sample_ids = list(range(len(self.data_list)))
        
        # Create a small default tensor for fallback cases
        self.fallback_tensor = torch.ones(1, 80, 192) * 0.01

    def __len__(self):
        return len(self.valid_sample_ids)

    def __getitem__(self, idx):
        # Convert to actual index in the data list
        try:
            real_idx = self.valid_sample_ids[idx % len(self.valid_sample_ids)]
            data = self.data_list[real_idx]
            
            # Try to load the main data
            try:
                mel_tensor, label = self._load_data(data)
                
                # Try to load reference samples
                try:
                    # Choose a random sample as reference
                    ref_idx = random.choice(self.valid_sample_ids)
                    ref_data = self.data_list[ref_idx]
                    ref_mel_tensor, ref_label = self._load_data(ref_data)
                    
                    # Try to get a second reference from the same class
                    try:
                        if ref_label in self.data_list_per_class and self.data_list_per_class[ref_label]:
                            ref2_data = random.choice(self.data_list_per_class[ref_label])
                            ref2_mel_tensor, _ = self._load_data(ref2_data)
                        else:
                            # If no samples in this class, use the first reference
                            ref2_mel_tensor = ref_mel_tensor
                    except Exception as e:
                        logger.warning(f"Error loading second reference: {e}")
                        ref2_mel_tensor = ref_mel_tensor
                        
                except Exception as e:
                    logger.warning(f"Error loading reference sample: {e}")
                    # Use the main sample as reference if there's a problem
                    ref_mel_tensor, ref_label = mel_tensor, label
                    ref2_mel_tensor = mel_tensor
                
                return mel_tensor, label, ref_mel_tensor, ref2_mel_tensor, ref_label
                
            except Exception as e:
                logger.warning(f"Error loading main sample {data[0]}: {e}")
                # Mark this sample as invalid
                if real_idx in self.valid_sample_ids:
                    self.valid_sample_ids.remove(real_idx)
                
                # Retry with another sample
                if len(self.valid_sample_ids) > 0:
                    return self.__getitem__(random.randint(0, len(self.valid_sample_ids)-1))
                else:
                    # If no valid samples at all, use fallback tensor
                    logger.error("No valid samples in dataset!")
                    return (self.fallback_tensor, 0, self.fallback_tensor, self.fallback_tensor, 0)
                    
        except Exception as e:
            logger.error(f"Critical error getting sample: {e}")
            # Last resort fallback
            return (self.fallback_tensor, 0, self.fallback_tensor, self.fallback_tensor, 0)
    
    def _load_data(self, path):
        try:
            wave_tensor, label = self._load_tensor(path)
            
            if wave_tensor.numel() == 0:
                raise ValueError(f"Empty wave tensor from {path[0]}")
                
            if not self.validation: # random scale for robustness
                random_scale = 0.5 + 0.5 * np.random.random()
                wave_tensor = random_scale * wave_tensor

            mel_tensor = self.to_melspec(wave_tensor)
            mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
            
            mel_length = mel_tensor.size(1)
            if mel_length > self.max_mel_length:
                random_start = np.random.randint(0, mel_length - self.max_mel_length)
                mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]
                
            # Ensure we have the expected shape
            if mel_tensor.size(1) < self.max_mel_length:
                # Pad if too short
                pad_length = self.max_mel_length - mel_tensor.size(1)
                mel_tensor = F.pad(mel_tensor, (0, pad_length))

            return mel_tensor, label
            
        except Exception as e:
            raise ValueError(f"Error in _load_data for {path[0]}: {str(e)}")

    def _preprocess(self, wave_tensor):
        if wave_tensor.numel() == 0:
            raise ValueError("Empty wave tensor provided for preprocessing")
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def _load_tensor(self, data):
        wave_path, label = data
        label = int(label)
        
        # Verify file exists and has content
        if not os.path.exists(wave_path):
            raise FileNotFoundError(f"File not found: {wave_path}")
            
        if os.path.getsize(wave_path) == 0:
            raise ValueError(f"File is empty: {wave_path}")
        
        try:
            wave, sr = sf.read(wave_path)
            
            if len(wave) == 0:
                raise ValueError(f"Empty audio file: {wave_path}")
                
            wave_tensor = torch.from_numpy(wave).float()

            if len(wave_tensor.shape) > 1 and wave_tensor.shape[1] == 2:
                # Convert stereo to mono by averaging the channels
                wave_tensor = torch.mean(wave_tensor, dim=1)
            
            # Make sure the tensor is 1D
            wave_tensor = wave_tensor.squeeze()
            
            # Final check to ensure non-empty tensor
            if wave_tensor.numel() == 0:
                raise ValueError(f"Empty tensor after processing: {wave_path}")
                
            return wave_tensor, label
            
        except Exception as e:
            raise ValueError(f"Error reading audio file {wave_path}: {str(e)}")

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        labels = torch.zeros((batch_size)).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref2_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()

        for bid, (mel, label, ref_mel, ref2_mel, ref_label) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref2_mel_size = ref2_mel.size(1)
            ref2_mels[bid, :, :ref2_mel_size] = ref2_mel
            
            labels[bid] = label
            ref_labels[bid] = ref_label

        z_trg = torch.randn(batch_size, self.latent_dim)
        z_trg2 = torch.randn(batch_size, self.latent_dim)
        
        mels, ref_mels, ref2_mels = mels.unsqueeze(1), ref_mels.unsqueeze(1), ref2_mels.unsqueeze(1)
        return mels, labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2

def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    try:
        dataset = MelDataset(path_list, validation=validation)
        collate_fn = Collater(**collate_config)
        
        # Only create dataloader if we have enough valid samples
        if len(dataset) >= batch_size:
            data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=(not validation),
                                 num_workers=num_workers,
                                 drop_last=True,
                                 collate_fn=collate_fn,
                                 pin_memory=(device != 'cpu'))
            return data_loader
        else:
            logger.error(f"Not enough valid samples: {len(dataset)} < {batch_size}")
            # Create a smaller batch size if needed
            adjusted_batch = max(1, len(dataset) // 2)
            logger.warning(f"Adjusting batch size to {adjusted_batch}")
            
            data_loader = DataLoader(dataset,
                                 batch_size=adjusted_batch,
                                 shuffle=(not validation),
                                 num_workers=num_workers,
                                 drop_last=True,
                                 collate_fn=collate_fn,
                                 pin_memory=(device != 'cpu'))
            return data_loader
            
    except Exception as e:
        logger.error(f"Error building dataloader: {e}")
        raise
