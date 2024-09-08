<div align="center">
<h1>Proxy and Cross-Stripes Integration Transformer for Remote Sensing Image Dehazing</h3>
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

## Proposed Benchmarks

### Download Link

* Hazy-DIOR: [https://pan.quark.cn/s/cf275400e7ca](https://pan.quark.cn/s/cf275400e7ca) (password: GMwk)
* Hazy-LoveDA: [https://pan.quark.cn/s/18ed1fe0e2ce](https://pan.quark.cn/s/18ed1fe0e2ce) (password: jCDW)

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
