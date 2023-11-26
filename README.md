# mmDiff
This is the code for the paper "mmDiff: Context and Consistency Awareness for mmWave Human Pose Estimation via Multi-Conditional Diffusion"





## Training and inference
We provide the training and inference code in runner.sh



## Environment

The code is developed and tested under the following environment:

-   Python 3.8.2
-   PyTorch 1.7.1
-   CUDA 11.0

You can create the environment via:

```bash
conda env create -f environment.yml
```

## Dataset
Our datasets are based on [mmBody](https://chen3110.github.io/mmbody/index.html) and [mm-Fi](https://ntu-aiot-lab.github.io/mm-fi). We provide the pretrained mmBody dataset as [mmBody.zip](https://www.dropbox.com/scl/fo/xqs7viqn6bjlolmu0qsjj/h?rlkey=hleuxio64kp43b5yx75lsszow&dl=0). To run the code, please download all the `mmBody.zip` and extract the .npy files to the `mmBody/` folder. Meanwhile, to follow the setting of Human 3.6m dataset of human pose estimation, please download the `pose_format.zip` and extract it to the `pose_format\` folder. 


## Pretrained mmDiff model
We provide the pretrained mmDiff parameter [here](https://www.dropbox.com/scl/fo/xqs7viqn6bjlolmu0qsjj/h?rlkey=hleuxio64kp43b5yx75lsszow&dl=0). Before running the code, please download the `checkpoints.zip` and extract all the .pth files to the `checkpoints/` folder. Before running the code, please specify the checkpoint path in the runner.sh shell file as: 
```bash
--model_diff_path checkpoints/[name].pth \
```
## Models
The fundamental model design can be found in `models/mmDiff.py`.


## Running experiments
### Evaluating pre-trained models

We provide the pre-trained diffusion model [ckpt_71.pth](https://www.dropbox.com/scl/fo/xqs7viqn6bjlolmu0qsjj/h?rlkey=hleuxio64kp43b5yx75lsszow&dl=0). To evaluate it, put it into the `./checkpoint` directory and run:

```bash
CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py \
--config human36m_diffpose_uvxyz_cpn.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_cpn.pth \
--model_diff_path checkpoints/diffpose_uvxyz_cpn.pth \
--doc t_human36m_diffpose_uvxyz_cpn --exp exp --ni \
>exp/t_human36m_diffpose_uvxyz_cpn.out 2>&1 &
```

We also provide the pre-trained diffusion model (with Ground truth 2D pose as input) [here](https://www.dropbox.com/sh/jhwz3ypyxtyrlzv/AABivC5oiiMdgPePxekzu6vga?dl=0). To evaluate it, put it into the `./checkpoint` directory and run:

```bash
CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py \
--config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_gt.pth \
--model_diff_path checkpoints/diffpose_uvxyz_gt.pth \
--doc t_human36m_diffpose_uvxyz_gt --exp exp --ni \
>exp/t_human36m_diffpose_uvxyz_gt.out 2>&1 &
```

### Training new models

-   To train a model from scratch (CPN 2D pose as input), run:

```bash
CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py --train \
--config human36m_diffpose_uvxyz_cpn.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_cpn.pth \
--doc human36m_diffpose_uvxyz_cpn --exp exp --ni \
>exp/human36m_diffpose_uvxyz_cpn.out 2>&1 &
```

-   To train a model from scratch (Ground truth 2D pose as input), run:

```bash
CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py --train \
--config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_gt.pth \
--doc human36m_diffpose_uvxyz_gt --exp exp --ni \
>exp/human36m_diffpose_uvxyz_gt.out 2>&1 &
```

## Video-based experiments
The code and pretrained model will be released at the end of May.

### Bibtex

If you find our work useful in your research, please consider citing:

    @InProceedings{gong2023diffpose,
        author    = {Gong, Jia and Foo, Lin Geng and Fan, Zhipeng and Ke, Qiuhong and Rahmani, Hossein and Liu, Jun},
        title     = {DiffPose: Toward More Reliable 3D Pose Estimation},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2023},
    }

## Acknowledgement

Part of our code is borrowed from [DDIM](https://github.com/ermongroup/ddim), [VideoPose3D](https://github.com/facebookresearch/VideoPose3D), [Graformer](https://github.com/Graformer/GraFormer), [MixSTE](https://github.com/JinluZhang1126/MixSTE) and [PoseFormer](https://github.com/zczcwh/PoseFormer). We thank the authors for releasing the codes.
