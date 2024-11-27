<div align="center">
<h1>Proxy and Cross-Stripes Integration Transformer for Remote Sensing Image Dehazing</h3>


[[Paper]](https://ieeexplore.ieee.org/document/10677537)
[[Hazy-DIOR Dataset]](https://huggingface.co/datasets/SmileShaun/Hazy-DIOR/tree/main)
[[Hazy-LoveDA Dataset]](https://huggingface.co/datasets/SmileShaun/Hazy-LoveDA/tree/main)
</div>

This repo contains our proposed RS image dehazing model PCSformer and the two proposed benchmarks, namely Hazy-DIOR and Hazy-LoveDA.

## Installation

```
git clone https://github.com/SmileShaun/PCSformer.git
cd PCSformer
pip install -r requirements.txt
cd loss/robust_loss_pytorch
pip install -e .[dev]
```

## Train
Please first set 'train_data_dir' and 'val_data_dir' in ```config.py``` (This is the path to your own dataset), then ```python train.py```.

## Proposed Benchmarks

### Download Link

* Hazy-DIOR: [https://huggingface.co/datasets/SmileShaun/Hazy-DIOR](https://huggingface.co/datasets/SmileShaun/Hazy-DIOR/tree/main)
* Hazy-LoveDA: [https://huggingface.co/datasets/SmileShaun/Hazy-LoveDA](https://huggingface.co/datasets/SmileShaun/Hazy-LoveDA/tree/main)

### Dataset folder structure

```
    Hazy-DIOR/Hazy-LoveDA
    ├── train
    │   ├── haze
    │   │   │── 00001.png
    │   │   │── 00002.png
    │   │   ├── ...
    │   ├── gt
    │   │   ├── 00001.png
    │   │   ├── 00002.png
    │   │   ├── ...
    ├── val
    │   ├── haze
    │   │   ├── 00001.png
    │   │   ├── 00002.png
    │   │   ├── ...
    │   ├── gt
    │   │   ├── 00001.png
    │   │   ├── 00002.png
    │   │   ├── ...
    ├── test
    │   ├── haze
    │   │   ├── thin
    │   │   │   ├── 00001.png
    │   │   │   ├── 00002.png
    │   │   │   ├── ...
    │   │   ├── moderate
    │   │   │   ├── 00001.png
    │   │   │   ├── 00002.png
    │   │   │   ├── ...
    │   │   ├── thick
    │   │   │   ├── 00001.png
    │   │   │   ├── 00002.png
    │   │   │   ├── ...
    │   ├── gt
    │   │   ├── thin
    │   │   │   ├── 00001.png
    │   │   │   ├── 00002.png
    │   │   │   ├── ...
    │   │   ├── moderate
    │   │   │   ├── 00001.png
    │   │   │   ├── 00002.png
    │   │   │   ├── ...
    │   │   ├── thick
    │   │   │   ├── 00001.png
    │   │   │   ├── 00002.png
    │   │   │   ├── ...
```

## Citation

If you use this codebase or the proposed benchmarks, please kindly cite our work:
```
@article{zhang2024proxy,
  title={Proxy and Cross-Stripes Integration Transformer for Remote Sensing Image Dehazing},
  author={Zhang, Xiaozhe and Xie, Fengying and Ding, Haidong and Yan, Shaocheng and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```