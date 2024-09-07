import torch

from p_baseline_constants import CONFIG
from simple_cnn import SkinCancerDetectionModel

# Initialize the model
model = torch.load("models/simple_cnn_fold_0_AUROC_0.1903_Loss_0.3165_epoch_42.model", map_location=CONFIG['device'])

# Load the model weights from the .model file
torch.save(model.state_dict(), 'models/simple_cnn_fold_0_AUROC_0.1903_Loss_0.3165_epoch_42.bin')

# Move the model to the configured device
model.cuda()

# Set the model to evaluation mode
model.eval()

print("done")