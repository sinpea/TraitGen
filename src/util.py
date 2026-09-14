import torch
def load_model(ModelClass,path,device):
    model = ModelClass()

    # 2. Load the raw weights file
    checkpoint = torch.load(path, map_location=device)

    # 3. Load the weights into the base model
    model.load_state_dict(checkpoint['model_state_dict'])
    return model