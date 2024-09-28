import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from pysensing.mmwave.PC.dataset.hpe import load_hpe_dataset
from pysensing.mmwave.PC.model.hpe import load_hpe_model, load_hpe_pretrain
from pysensing.mmwave.PC.inference import load_pretrain
from pysensing.mmwave.PC.model.hpe.mmDiff.load_mmDiff import load_mmDiff


def hpe_train(model, train_loader, num_epochs, optimizer, criterion, device):
    r"""
    This function provide human pose estimation (hpe) training.

    Args:
        model (torch.nn.Module): Pytorch model.

        train_loader (torch.utils.data.DataLoader): Pytorch data_loader.

        num_epochs (int): Training epochs.

        optimizer (torch.optim.Optimizer): Optimizer, e.g. torch.optim.Adam().

        criterion (torch.nn.LossFunction): Criterion or loss function for training.

        device (torch.device): torch.device("cuda" if torch.cuda.is_available() else "cpu").


    """
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_mpjpe = 0

        with tqdm(total=len(train_loader), desc=f'Train round{epoch}/{num_epochs}', unit='batch') as pbar:
            for data in train_loader:
                inputs, labels = data
                inputs = inputs.type(torch.FloatTensor).to(device)
                labels = labels.to(device)
                labels = labels.type(torch.FloatTensor)

                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs.to(device)
                outputs = outputs.type(torch.FloatTensor)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_mpjpe += (outputs - labels).square().mean().item() / labels.size(0)
                pbar.update(1)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        epoch_loss = epoch_loss / len(train_loader.dataset)
        epoch_mpjpe = epoch_mpjpe / len(train_loader)
        print('Epoch:{}, MPJPE:{:.4f},Loss:{:.9f}'.format(epoch + 1, float(epoch_mpjpe), float(epoch_loss)))
        # save model weights
        savepath = f"train_{epoch}.pth"
        print(f'Save model at {savepath}...')
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
    return

def hpe_test(model, test_loader, criterion, device):
    r"""
    This function provide human pose estimation (hpe) inference.

    Args:
        model (torch.nn.Module): Pytorch model.

        test_loader (torch.utils.data.DataLoader): Pytorch data_loader.

        criterion (torch.nn.LossFunction): Loss function or criterion function, e.g. nn.CrossEntropyLoss().

        device (torch.device): torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Return:
        Output criterion metrics based on model and data from test_loader.

    """
    model.to(device)
    model.eval()
    test_mpjpe = 0
    test_loss = 0
    num = 0
    for data in tqdm(test_loader, total=len(test_loader), desc='Test round', unit='batch', leave=False):
        inputs, labels = data


        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor).to(device)


        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor).to(device)
        

        loss = criterion(outputs, labels)
        mpjpe = (outputs - labels).square().sum(-1).sqrt().mean(-1)
        test_mpjpe += mpjpe.sum().item()*1000
        test_loss += loss.item() * inputs.size(0)
        num += labels.shape[0]
    test_mpjpe = test_mpjpe / num
    test_loss = test_loss / num
    print("validation mpjpe:{:.4f}, loss:{:.5f}".format(float(test_mpjpe), float(test_loss)))
    return

# def hpe_predict(dataset, model_name, pretrain=False):

#     '''
#     load the weights of model from the pretrain files
#     '''
#     if dataset == "mmBody":
#         dataset_root = '../projects/data/mmpose'
#         train_loader, test_loader = load_hpe_dataset(dataset, dataset_root)
#     elif dataset == "MetaFi":
#         dataset_root = '../projects/data/MMFi_Dataset'
#         with open('config.yaml', 'r') as fd:
#             config = yaml.load(fd, Loader=yaml.FullLoader)
#         train_loader, test_loader = load_hpe_dataset(dataset, dataset_root, config=config)
        
   
#     pretrain_root = None

#     model = load_hpe_model(dataset, model_name)
#     criterion = nn.MSELoss()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     if pretrain == True:
#         model = load_hpe_pretrain(model, pretrain_root, dataset, model_name)
#     model.to(device)


#     hpe_test(
#         model=model,
#         test_loader=test_loader,
#         criterion=criterion,
#         device=device
#     )

if __name__ == "__main__":
    # dataset_root = '/home/junqiao/projects/data/MMFi_Dataset/'
    # # xian zai shi yong de shi Radar_Fused
    # with open('config.yaml', 'r') as fd:
    #     config = yaml.load(fd, Loader=yaml.FullLoader)
    # train_dataset, test_dataset = load_hpe_dataset("mmBody", '/home/junqiao/projects/data/mmpose/', config=config)
    # train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=16)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16)
    # model = load_hpe_model("mmBody", "P4Transformer")

    # criterion = nn.MSELoss()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # model = load_pretrain(model, "mmBody", "P4Transformer")
    # hpe_test(model, test_loader, criterion=criterion, device=device)
    # # hpe_train(model, train_loader, num_epochs=5, optimizer=optimizer, criterion=criterion, device=device)

    train_dataset, test_dataset = load_hpe_dataset("mmBody", '/home/junqiao/projects/data/mmpose/', config=None)

    mmDiffRunner = load_mmDiff("mmBody")
    mmDiffRunner.phase1_train(train_dataset, test_dataset, is_train=False, is_save=False)
    mmDiffRunner.phase2_train(train_loader = None, is_train = False)
    mmDiffRunner.test()






