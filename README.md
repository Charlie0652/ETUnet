# ETUnet
ETU-Net: edge enhancement-guided U-Net with transformer for skin lesion segmentation
Authors: LiFang Chen，Jiawei Li，Yunmin Zou， Tao Wang
<br/>
Research paper: https://doi.org/10.1088/1361-6560/ad13d2

## Abstract
<div align="justify">
Objective. Convolutional neural network (CNN)-based deep learning algorithms have been widely
used in recent years for automatic skin lesion segmentation. However, the limited receptive fields of
convolutional architectures hinder their ability to effectively model dependencies between different
image ranges. The transformer is often employed in conjunction with CNN to extract both global and
local information from images, as it excels at capturing long-range dependencies. However, this
method cannot accurately segment skin lesions with blurred boundaries. To overcome this difficulty,
we proposed ETU-Net. Approach. ETU-Net, a novel multi-scale architecture, combines edge
enhancement, CNN, and transformer. We introduce the concept of edge detection operators into
difference convolution, resulting in the design of the edge enhanced convolution block (EC block) and
the local transformer block (LT block), which emphasize edge features. To capture the semantic
information contained in local features, we propose the multi-scale local attention block (MLA block),
which utilizes convolutions with different kernel sizes. Furthermore, to address the boundary
uncertainty caused by patch division in the transformer, we introduce a novel global transformer
block (GT block), which allows each patch to gather full-size feature information. Main results.
Extensive experimental results on three publicly available skin datasets(PH2, ISIC-2017, and ISIC-
2018) demonstrate that ETU-Net outperforms state-of-the-art hybrid methods based on CNN and
Transformer in terms of segmentation performance. Moreover, ETU-Net exhibits excellent generalization ability in practical segmentation applications on dermatoscopy images contributed by the
Wuxi No.2 People’s Hospital. Significance. We propose ETU-Net, a novel multi-scale U-Net model
guided by edge enhancement, which can address the challenges posed by complex lesion shapes and
ambiguous boundaries in skin lesion segmentation tasks.
</div>

## Citation
Please cite our paper if you find it useful: 
<pre>
@article{chen2023etu,
  title={ETU-Net: edge enhancement-guided U-Net with transformer for skin lesion segmentation},
  author={Chen, Lifang and Li, Jiawei and Zou, Yunmin and Wang, Tao},
  journal={Physics in Medicine \& Biology},
  volume={69},
  number={1},
  pages={015001},
  year={2023},
  publisher={IOP Publishing}
}
</pre>
