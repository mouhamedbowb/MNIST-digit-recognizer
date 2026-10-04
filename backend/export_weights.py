"""Export mnist_cnn.pt -> ../web/public/model.json (read by the browser)."""
import json, torch
from model import MnistNet

net = MnistNet()
net.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))
out = {}
for name in ["conv1", "conv2", "fc1", "out"]:
    m = getattr(net, name)
    out[name] = {"w": [round(v, 6) for v in m.weight.detach().flatten().tolist()],
                 "b": [round(v, 6) for v in m.bias.detach().tolist()]}
json.dump(out, open("../web/public/model.json", "w"))
print("wrote ../web/public/model.json")
