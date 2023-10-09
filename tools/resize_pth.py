import torch
pretrained_weights=torch.load('yolos_ti_raw.pth')
num_classes=1
pretrained_weights["model"]["class_embed.layers.2.weight"].resize_(num_classes+1,192)
pretrained_weights["model"]["class_embed.layers.2.bias"].resize_(num_classes+1)
# del pretrained_weights["model"]["class_embed.layers.2.weight"]
# del pretrained_weights["model"]["class_embed.layers.2.bias"]
pretrained_weights["epoch"] = 0
torch.save(pretrained_weights,"yolos_ti_%d.pth"%num_classes)