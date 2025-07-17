# -*- coding: utf-8 -*-
""" datasets/SportImages.py """

import os
from pathlib import Path

from datasets.Coco import Coco
from settings import DATA_PATH
from utils.tools import dict_update


__all__ = [
    'SportImages',
]


class SportImages(Coco):
    """
    SportImages Dataset
    """

    def __init__(self, transform=None, task='train', **config):

        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, config)

        self.transforms = transform
        self.action = 'train' if task == 'train' else 'val'

        # FIXME: MONEY PATH, fix me!!!
        task = 'valid' if task == 'val' else task
        # get files
        base_path = Path(os.path.join(DATA_PATH, config['dataset'], task))
        image_paths = list(base_path.iterdir())
        # if config['truncate']:
        #     image_paths = image_paths[:config['truncate']]
        names = [p.stem for p in image_paths]
        image_paths = [str(p) for p in image_paths]
        files = {'image_paths': image_paths, 'names': names}

        sequence_set = []
        self.labels = False
        if self.config['labels']:
            self.labels = True
            # from models.model_wrap import labels2Dto3D
            # self.labels2Dto3D = labels2Dto3D
            print("load labels from: ", self.config['labels']+'/'+task)
            count = 0
            for (img, name) in zip(files['image_paths'], files['names']):
                p = Path(self.config['labels'], task, '{}.npz'.format(name))
                if p.exists():
                    sample = {'image': img, 'name': name, 'points': str(p)}
                    sequence_set.append(sample)
                    count += 1
                # if count > 100:
                #     print ("only load %d image!!!", count)
                #     print ("only load one image!!!")
                #     print ("only load one image!!!")
                #     break
        else:
            for (img, name) in zip(files['image_paths'], files['names']):
                sample = {'image': img, 'name': name}
                sequence_set.append(sample)
        self.samples = sequence_set

        self.init_var()
