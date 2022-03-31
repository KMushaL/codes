# Complementary Factorization towards Outfit Compatibility Modeling
This repo is the official implementation of our [paper]() by PyTorch. 

## Framework


## Usage

### Python Environment by Conda

1. create a conda env: `conda create -n ocm-cf python=3.8`
2. install packages:
- [PyTorch](https://pytorch.org/get-started/previous-versions/). e.g. 
    ```shell
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
    ``` 
- [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). e.g. 
  ```shell
  CUDA=cu101
  TORCH=1.6.0
  pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
  pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
  pip install torch-geometric
  ```
- conda install scikit-learn  tensorboard  
  
### Dataset
1. Download [Polyovre](https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view?usp=sharing) dataset, and unzip it into `data` directory. e.g. `unzip -d data/ polyvore_outfits.zip`.
2. The `data` directory tree like this:
```shell
data/
└── polyvore_outfits
    ├── disjoint
    ├── images
    ├── maryland_polyvore_hardneg
    ├── nondisjoint
    └── retrieval
```

### Inference
We released the pre-trained [model](https://drive.google.com/drive/folders/10tG-AhU3nblgfk7RegWsRtllcmx95uVa). Please download and move files into `checkpoints`, likely 
```shell
checkpoints/
├── disjoint_best.pt
└── nondisjoint_best.pt
```
You can run script to inference, as follow,
```shell
cd inference
bash inference_all_tasks.sh ${gpu_id}
```

### Train
For the `Polyvore Outfits` dataset, run `python main.py --polyvore-split nondisjoint`.

For the `Polyvore Outfits-D` dataset, run `python main.py --polyvore-split disjoint`.

## Reference
If you find this code or our paper useful in your research, please consider citing:
```latex

```

 