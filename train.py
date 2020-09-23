from mydataset import Mydataset
from model import *
import torch
from torch.utils.data import DataLoader
import os
import cfg
import cfg1
path = r"./model_param1/model.t"

def loss_fn(output, target, alpha):
    conf_loss_fn = torch.nn.BCEWithLogitsLoss()     #置信度
    center_loss_fn = torch.nn.BCEWithLogitsLoss()   #中心点
    wh_loss_fn = torch.nn.MSELoss()              #宽高
    cls_loss_fn = torch.nn.CrossEntropyLoss()     #类别
    output = output.permute(0, 2, 3, 1)
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

    target = target.to(DEVICE)
    mask_obj = target[..., 0] > 0.1
    output_obj = output[mask_obj]
    target_obj = target[mask_obj]
    loss_obj_conf = conf_loss_fn(output_obj[:, 0], target_obj[:, 0])
    loss_obj_center = center_loss_fn(output_obj[:, 1:3], target_obj[:, 1:3])
    loss_obj_wh = wh_loss_fn(output_obj[:, 3:5], target_obj[:, 3:5])
    loss_obj_cls = cls_loss_fn(output_obj[:, 5:], target_obj[:, 5].long())
    loss_obj = loss_obj_conf + loss_obj_center + loss_obj_wh + loss_obj_cls
    mask_noobj = target[..., 0] <= 0.1
    output_noobj = output[mask_noobj]
    target_noobj = target[mask_noobj]
    loss_noobj = conf_loss_fn(output_noobj[:, 0], target_noobj[:, 0])
    loss = alpha * loss_obj + (1 - alpha) * loss_noobj
    return loss


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Mydataset()
    train_loader = DataLoader(data, batch_size=3, shuffle=True)

    net =MainNet(len(cfg.name))
    net.to(DEVICE)
    if os.path.exists(path):  # 如果文件存在，接着继续训练
        pretrained_dict = torch.load(path)
        #for name, param in pretrained_dict.items():
            # if name == 'detetion_13.1.weight':
            #     param.data = torch.randn(3*(5+len(cfg.name)),1024,1,1)
            #     model[name] = param
            # if name == 'detetion_13.1.bias':
            #     param.data = torch.randn(3*(5+len(cfg.name)))
            #     model[name] = param
            # if name == 'detetion_26.1.weight':
            #     param.data = torch.randn(3*(5+len(cfg.name)),512,1,1)
            #     model[name] = param
            # if name == 'detetion_26.1.bias':
            #     param.data = torch.randn(3*(5+len(cfg.name)))
            #     model[name] = param
            # if name =='detetion_52.1.weight':
            #     param.data = torch.randn(3*(5+len(cfg.name)),256,1,1)
            #     model[name] = param
            # if name == 'detetion_52.1.bias':
            #     param.data = torch.randn(3*(5+len(cfg.name)))
            #     model[name] = param
            #print('{},size:{}'.format(name, param.data.size()))

        net_dict = net.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}

        net_dict.update(pretrained_dict)

        net.load_state_dict(net_dict)
    net.train()

    opt = torch.optim.Adam(net.parameters())
    for epoch in range(1000):
        for i,(target_13, target_26, target_52, img_data) in enumerate(train_loader):
            img_data = img_data.to(DEVICE)
            output_13, output_26, output_52 = net(img_data)
            loss_13 = loss_fn(output_13, target_13, 0.9)
            loss_26 = loss_fn(output_26, target_26, 0.9)
            loss_52 = loss_fn(output_52, target_52, 0.9)

            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()
            print("epoch:{0} ".format(epoch),loss.item())
        if epoch % 10 == 0:
            torch.save(net.state_dict(), f"./model_param1/{epoch}.t")