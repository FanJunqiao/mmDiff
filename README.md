# mmDiff
This is the code for the paper "mmDiff: Diffusion model is a good pose estimator for RF-Vision". It implements a diffusion framework for radar PC-based human pose estimation.

## Latest update (from 30/10/2024)
-   We fix bugs in original codes, please update
-   We provide a tutorial for running our code, which helps you uses diffsion model for mmWave radar-based projects.
-   We provide pretrained weights in url.txt
-   Some bugs still exists in dataset implementation, e.g. the sequence id is not considered when concatenate adjacent radar frames and pose frames. This lead to slight performance drops compared to original paper. (To be implemented)


## Environment

The code is developed and tested under the following environment:

-   Python 3.8.2
-   PyTorch 1.8.1
-   CUDA 11.6

You can create the environment via:

```bash
conda env create -f environment.yml
```
For manual setup, we build our environment following [P4transformer](https://github.com/hehefan/P4Transformer).

## Dataset
Please follow [mmBody](https://github.com/Chen3110/mmBody) and [mmFi](https://ntu-aiot-lab.github.io/mm-fi) for dataset setup.


## Pretrained mmDiff model
We provide the pretrained mmDiff parameter in url.txt. Please download and extract to `pretrained/`.
When running tutorial.ipynb, the parameter can be automatically downloaded in lines, and you do not need to download by hand.




## Running experiments
### Models
The fundamental model design can be found in `pysensing/mmwave/PC/model/hpe/mmDiff.py`.
### Trained from scratch and evaluate
```bash
python hpe.py
```

### phase 1 training
```python
mmDiffRunner.phase1_train(train_dataset, test_dataset, is_train=True, is_save=True) # is_train = True means training phase 1 from scratch, is_save = True means saves the pretrained features and poses..
```
### phase 2 training
```python
mmDiffRunner.phase2_train(train_loader = None, is_train = True) # is_train = True means training phase 2 frome scratch.
```
### testing
```python
mmDiffRunner.test() # evaluating.
```
