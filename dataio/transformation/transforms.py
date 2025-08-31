import torch
import torchvision.transforms as ts
from pprint import pprint
import numpy as np
import random
from torchvision.transforms import InterpolationMode

class Transformations:
    def __init__(self, name):
        self.name = name

        # Input patch and scale size
        self.scale_size = (192, 192, 1)
        self.patch_size = (128, 128, 1)
        # self.patch_size = (208, 272, 1)

        # Affine and Intensity Transformations
        self.shift_val = (0.1, 0.1)
        self.rotate_val = 15.0
        self.scale_val = (0.7, 1.3)
        self.inten_val = (1.0, 1.0)
        self.random_flip_prob = 0.0

        # Divisibility factor for testing
        self.division_factor = (16, 16, 1)

    def get_transformation(self):
        return {
            # 'test_sax': self.test_3d_sax_transform,
            'acdc_sax': self.cmr_3d_sax_transform,
        }[self.name]()

    def print(self):
        print('\n\n############# Augmentation Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')

    def initialise(self, opts):
        t_opts = getattr(opts, self.name)

        # Affine and Intensity Transformations
        if hasattr(t_opts, 'scale_size'):       self.scale_size = t_opts.scale_size
        if hasattr(t_opts, 'patch_size'):       self.patch_size = t_opts.patch_size
        if hasattr(t_opts, 'shift_val'):        self.shift_val = t_opts.shift
        if hasattr(t_opts, 'rotate_val'):       self.rotate_val = t_opts.rotate
        if hasattr(t_opts, 'scale_val'):        self.scale_val = t_opts.scale
        if hasattr(t_opts, 'inten_val'):        self.inten_val = t_opts.intensity
        if hasattr(t_opts, 'random_flip_prob'): self.random_flip_prob = t_opts.random_flip_prob
        if hasattr(t_opts, 'division_factor'):  self.division_factor = t_opts.division_factor

    def cmr_3d_sax_transform(self):
      train_transform = ts.Compose([PadNumpy(self.scale_size),
                        ChannelsFirst(),
                        TypeCast(['float', 'float']),
                        # ts.ToTensor(),
                        RandomFlip(h=True, v=True, p=self.random_flip_prob),
                        # ts.ToTensor(),
                        ToTensorND(),
                        ts.RandomAffine(degrees=self.rotate_val, translate=self.shift_val,
                                scale=self.scale_val, interpolation= InterpolationMode.BILINEAR),
                        # ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                        ts.Normalize(mean=[0.0], std=[1.0]),
                        ChannelsLast(),
                        AddChannel(axis=0),
                        ToTensorND(),
                        # ts.RandomCrop(size=self.patch_size[:2]),
                        TypeCast(['float', 'long'])
                      ])

      valid_transform = ts.Compose([PadNumpy(size=self.scale_size),
                        ChannelsFirst(),
                        TypeCast(['float', 'float']),
                        #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                        ts.Normalize(mean=[0.0], std=[1.0]),
                        ChannelsLast(),
                        AddChannel(axis=0),
                        SpecialCrop(size=self.patch_size, crop_type=0),
                        TypeCast(['float', 'long'])
                      ])
      return {'train': train_transform, 'valid': valid_transform}

    # def test_3d_sax_transform(self):
    #     test_transform = ts.Compose([ts.PadFactorNumpy(factor=self.division_factor),
    #                                  ts.ToTensor(),
    #                                  ChannelsFirst(),
    #                                  ts.TypeCast(['float']),
    #                                  #ts.NormalizeMedicPercentile(norm_flag=True),
    #                                  ts.Normalize(),
    #                                  ChannelsLast(),
    #                                  AddChannel(axis=0),
    #                                  ])

    #     return {'test': test_transform}
class ToTensorND(object):
    def __call__(self, img):
        # Converts any numpy array to a torch tensor, preserving shape
        return torch.from_numpy(np.ascontiguousarray(img))
    
class PadNumpy(object):
    def __init__(self, size):
        self.size = size  # (target_x, target_y, target_z)

    def __call__(self, img):
        pad_width = []
        for i in range(3):
            diff = self.size[i] - img.shape[i]
            if diff > 0:
                pad_before = diff // 2
                pad_after = diff - pad_before
            else:
                pad_before = pad_after = 0
            pad_width.append((pad_before, pad_after))
        return np.pad(img, pad_width, mode='constant')

class ChannelsFirst(object):
    def __call__(self, img):
        # For 3D: (H, W, D, C) -> (C, H, W, D)
        if img.ndim == 4:
            return np.transpose(img, (3, 0, 1, 2))
        # For 2D: (H, W, C) -> (C, H, W)
        elif img.ndim == 3:
            return np.transpose(img, (2, 0, 1))
        else:
            return img

class TypeCast(object):
    def __init__(self, dtypes):
        self.dtypes = dtypes  # e.g., ['float', 'long']

    def __call__(self, sample):
        # If sample is a tuple/list, cast each element
        if isinstance(sample, (tuple, list)):
            return tuple(self._cast(s, dt) for s, dt in zip(sample, self.dtypes))
        else:
            # If only one dtype specified, cast the whole sample
            return self._cast(sample, self.dtypes[0])

    def _cast(self, x, dtype):
        if dtype == 'float':
            if isinstance(x, np.ndarray):
                return x.astype(np.float32)
            elif torch.is_tensor(x):
                return x.float()
        elif dtype == 'long':
            if isinstance(x, np.ndarray):
                return x.astype(np.int64)
            elif torch.is_tensor(x):
                return x.long()
        # Add more types as needed
        return x
    
class SpecialCrop(object):
    def __init__(self, size, crop_type=0):
        self.size = size
        self.crop_type = crop_type

    def __call__(self, img):
        # Center crop for 2D or 3D images
        if self.crop_type == 0:
            shape = img.shape
            crop = self.size
            # Support for 2D (H, W) or 3D (H, W, D)
            slices = []
            for i in range(len(crop)):
                start = max((shape[i] - crop[i]) // 2, 0)
                end = start + crop[i]
                slices.append(slice(start, end))
            # For images with channels, keep all channels
            while len(slices) < len(shape):
                slices.append(slice(None))
            return img[tuple(slices)]
        else:
            # Other crop types can be added here
            return img
        
class AddChannel(object):
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, img):
        # Add a new channel axis at the specified position
        return np.expand_dims(img, axis=self.axis)
    
class ChannelsLast(object):
    def __call__(self, img):
        # For 3D: (C, H, W, D) -> (H, W, D, C)
        if img.ndim == 4:
            return np.transpose(img, (1, 2, 3, 0))
        # For 2D: (C, H, W) -> (H, W, C)
        elif img.ndim == 3:
            return np.transpose(img, (1, 2, 0))
        else:
            return img
        
class RandomFlip(object):
    def __init__(self, h=True, v=True, p=0.5):
        self.h = h
        self.v = v
        self.p = p

    def __call__(self, img):
        # img can be 2D, 3D, or 4D numpy array
        if random.random() < self.p:
            if img.shape[1] > 0 and self.h:
                img = np.flip(img, axis=1)  # horizontal flip (axis=1)
            if  img.shape[0] > 0 and self.v:
                img = np.flip(img, axis=0)  # vertical flip (axis=0)
        return img
