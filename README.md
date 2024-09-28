# mmDiff
This is the code for the paper "mmDiff: Context and Consistency Awareness for mmWave Human Pose Estimation via Multi-Conditional Diffusion"



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
We provide the pretrained mmDiff parameter [here]. Please download and extract to `pretrained/`,

## Models
The fundamental model design can be found in `pysensing/mmwave/PC/model/hpe/mmDiff.py`.


## Running experiments
### Evaluating pre-trained models

```bash
python hpe.py
```

