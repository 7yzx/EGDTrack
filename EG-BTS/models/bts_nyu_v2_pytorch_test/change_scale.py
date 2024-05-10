import torch
state_dict = torch.load("model-36000-best_rms_0.43576")#xxx.pth或者xxx.pt就是你想改掉的权重文件
torch.save(state_dict, "model_rms_grad08bts16_4357", _use_new_zipfile_serialization=False)

