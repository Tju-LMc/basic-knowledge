# 服务器 conda 操作记录

## 基础操作备忘录

```sh
# 安装指定版本
conda install pytorch=1.4

# 列出当前环境
conda env list

# 删除环境
conda env remove -n xxx

# 使用本地包进行安装
conda install ~/anaconda3/pkgs/xxx.tar.bz2
```

## 离线情况瞎配置环境

当流量不够时, 创建环境可以基于已有同学的环境克隆出自己的环境

```sh
# 创建自己的环境 BBB, 以 xxx 为模板
conda create -n BBB --clone xxx
# ~/path 为环境的路径
conda create -n BBB --clone ~/path
```

用 path 路径方法的话, 可以在不同机器之间克隆, 只要把当前环境打包发送到另一台机器,再解压, 用路径进行克隆

### 当前已有基础环境

- p36t18 ( python3.6 + torch 1.8 + cuda 10.1 )

> 遇到克隆报错, python not found ..待解决

### 离线配置教程

本质就是安装 gpu 版 torch, 再安装驱动的 toolkit

```sh
# 创建基础环境
conda create -n p36t18 python=3.6

# 进入
conda activate p36t18

# 一些离线安装torch所必须的依赖
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# 查看nvcc提示的cuda版本
nvcc --version

# 打开网址 https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/
# 找对应版本 如搜索 pytorch-1.8 会看到对应版本的 torch, cuda 和 cudnn, 目前安装10.x版本
# 下载后上传到服务器比如 Downloads 文件夹下
conda install -c local Downloads/pytorch-1.8.0-py3.8_cuda10.1_cudnn7.6.3_0.tar.bz2

# 再下载对应版本的 cudatoolkit 上传安装
# 网址 https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/
conda install -c local Downloads/cudatoolkit-10.1.243-h6bb024c_0.tar.bz2

# 确认当前环境已有的包, 会发现离线安装的url为file路径
conda list
```

进入 python 命令行, 检查有无报错, 正常则返回 true

```python
import torch
torch.cuda.is_available()
```

## 离线安装其他包

以安装 torchtext 为例

进入[https://pypi.org/project/torchtext/#description](https://pypi.org/project/torchtext/#description) 找到包和 python 版本之间的关系

比如我当前环境 python3.8, torch 1.8, 则应该下载 torchtext version 为 0.9

使用 ` conda info torchtext` 会检索到 n 个版本的下载地址

找到需要的版本, 下载之, 传到服务器~

如果包实在找不到, 去 conda 官网 https://anaconda.org/ 上寻找对应的包下载

同样使用本地安装命令安装

`conda install xxx.tar.bz2`
