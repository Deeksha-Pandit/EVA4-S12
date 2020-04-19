from torch.utils.data import DataLoader
import torch
def load(train_dataset,test_dataset,batch_size):
    SEED = 1

	# CUDA?
    cuda = torch.cuda.is_available()

	# For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return trainloader,testloader