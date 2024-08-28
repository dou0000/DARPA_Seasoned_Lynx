## Seasoned Lynx -- Class-agnotic Map Segmentation

currently in the development for the class-agnostic map segmentation based on the proposed concept from the VRP-SAM model
<p align="middle">
    <img src="assets/vrp_sam.jpg" height="360">
</p>

## Requirements

- Python 3.10
- PyTorch 1.12
- cuda 11.6

Conda environment settings:
```bash
conda create -n vrpsam python=3.10
conda activate vrpsam

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

Segment-Anything-Model setting:
```bash
cd ./segment-anything
pip install -v -e .
cd ..
```



#### Based Repo
this repo is based on VRPSAM