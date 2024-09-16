r""" Dataloader builder for few-shot semantic segmentation dataset  """
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.datasets_prompt import MAPData



# from data.coco2pascal import DatasetCOCO2PASCAL


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'map': MAPData
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize
        
        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])


    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, args, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0
        if args.backbone == 'golden_muscat':
            patch_size = 256

        patch_size = 256 if args.backbone == 'golden_muscat' else 512
        if args.img_size_for_models == 256:
            patch_size = 256
        elif args.img_size_for_models == 1024:
            patch_size = 1024

        #overlap = '64' if args.backbone == 'golden_muscat' else '15'
        overlap = '32' if args.backbone == 'golden_muscat' else '15'
        if args.overlap is not None:
            overlap = args.overlap


        dataset = cls.datasets[benchmark](cls.datapath, 
                                          fold=fold, 
                                          transform=cls.transform, 
                                          split=split, 
                                          shot=shot, 
                                          use_original_imgsize=cls.use_original_imgsize,
                                          backbone=args.backbone,
                                          map_size=patch_size,
                                          overlap=overlap,
                                          args=args,
                                          )
        
        # if split == 'trn':
        #     if args.parallel:
        #         # sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        #         # sampler = DistributedSampler(dataset, shuffle=shuffle)
        #         sampler = None
        #     else:
        #         sampler = None
        #     shuffle = False
        # else:
        #     if args.parallel:
        #         # sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        #         #sampler = DistributedSampler(dataset, shuffle=shuffle)
        #         sampler = None
        #     else:
        #         sampler = None
        #     pin_memory = True
        # dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, pin_memory=True, num_workers=nworker, sampler=sampler)

        # return dataloader
        if args.parallel:

            # Assume rank and world_size are provided in args
            rank = args.local_rank
            # world_size = torch.cuda.device_count() 
            world_size = args.world_size
            
            # # number of workers per GPU
            # nworker = nworker // world_size

            # Split dataset indices based on rank
            total_size = len(dataset)
            indices = list(range(total_size))
            split_size = total_size // world_size

            # Assign indices for the current rank
            start_index = rank * split_size
            if rank == world_size - 1:
                end_index = total_size  # Ensure the last rank gets the remaining data
            else:
                end_index = start_index + split_size

            # Subset the dataset
            dataset = torch.utils.data.Subset(dataset, indices[start_index:end_index])


        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, pin_memory=True, num_workers=nworker)

        return dataloader