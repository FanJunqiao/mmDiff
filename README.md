# mmDiff
This is the code for the paper "mmDiff: Context and Consistency Awareness for mmWave Human Pose Estimation via Multi-Conditional Diffusion"



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
We provide the pretrained mmBody dataset as [mmBody.zip](https://www.dropbox.com/scl/fo/xqs7viqn6bjlolmu0qsjj/h?rlkey=hleuxio64kp43b5yx75lsszow&dl=0). To run the code, please download all the `mmBody.zip` and extract the .npy files to the `mmBody/` folder. Meanwhile, to follow the setting of the Human 3.6m dataset of human pose estimation, please download the `pose_format.zip` and extract it to the `pose_format\` folder. 


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
### inference with pretrain ###
CUDA_VISIBLE_DEVICES=0 python main_mmDiff.py \
--config mmDiff_config.yml --batch_size 512 \
--model_diff_path checkpoints/ckpt_71.pth \ # add your pretrained model
--doc test --exp exp --ni \
```

### Training models from scratch
To train a model from scratch, run

```bash
### train ###
CUDA_VISIBLE_DEVICES=0 python main_mmDiff.py --train \
--config mmDiff_config.yml --batch_size 512 \
--doc test --exp exp --ni \
```
