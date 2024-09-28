import torch
import torch.nn as nn

from pysensing.mmwave.PC.dataset.hpe import load_hpe_dataset
from pysensing.mmwave.PC.model.hpe.mmDiff.load_mmDiff import load_mmDiff


if __name__ == "__main__":
    dataset = "mmBody" # ["mmFi", "mmBody"]
    if dataset == "mmFi":
        train_dataset, test_dataset = load_hpe_dataset("MetaFi", '/home/junqiao/projects/data/MMFi_Dataset/', config=None)
        mmDiffRunner = load_mmDiff("MetaFi")
        mmDiffRunner.phase1_train(train_dataset, test_dataset, is_train=True, is_save=True)
        mmDiffRunner.phase2_train(train_loader = None, is_train = True)
        mmDiffRunner.test()
    elif dataset == "mmBody":
        # train_dataset, test_dataset = load_hpe_dataset("MetaFi", '/home/junqiao/projects/data/MMFi_Dataset/', config=None)
        train_dataset, test_dataset = load_hpe_dataset("mmBody", '/home/junqiao/projects/data/mmpose/', config=None)
        mmDiffRunner = load_mmDiff("mmBody")
        mmDiffRunner.phase1_train(train_dataset, test_dataset, is_train=False, is_save=False)
        mmDiffRunner.phase2_train(train_loader = None, is_train = False)
        mmDiffRunner.test()



    






