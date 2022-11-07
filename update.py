import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import copy


# train image net
class LocalImgUpdate(object):
    def __init__(self, opt, F_buffer, Y, B, train_img, train_label):
        self.opt = opt
        self.F_buffer = F_buffer
        self.Y = Y
        self.B = B
        self.train_image = train_img
        self.train_label = train_label

    def update_weights_het(self, global_protos, model):
        # set mode to train model
        num_train = self.train_image.shape[0]
        batch_size = self.opt.batch_size

        lr = self.opt.lr
        optimizer_img = torch.optim.Adam(model.parameters(), lr=lr)

        # proto similarity loss
        l2_x = 0.
        # proto loss
        loss_proto = 0.
        agg_protos_label = {}

        # train image net
        index = np.random.permutation(num_train)
        for idx in tqdm(range(num_train // batch_size + 1)):
            # use the last batch dataset
            remaining = num_train - idx * batch_size
            if remaining <= 0:
                break
            ind = index[idx * batch_size: (idx + 1) * batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = self.train_label[ind, :]
            labels = [np.argmax(item) for item in sample_L.cpu()]
            image = self.train_image[ind].type(torch.float)
            if self.opt.use_gpu:
                image = image.cuda()
                sample_L = sample_L.cuda()

            cur_f, protos = model(image)  # cur_f: (batch_size, bit)
            self.F_buffer[ind, :] = cur_f.data

            # calculate loss
            l1_x = torch.sum(torch.pow(self.opt.bit * sample_L - torch.matmul(cur_f, self.Y.t()), 2))
            quantization_x = torch.sum(torch.pow(self.B[ind, :] - cur_f, 2))
            loss_x = self.opt.alpha * l1_x + self.opt.gamma * quantization_x
            # proto loss
            loss_mse = nn.MSELoss()
            # loss_mse = nn.KLDivLoss()
            if len(global_protos) == 0:
                l2_x = 0 * loss_x
                loss_proto = 0 * loss_x
            else:
                # proto similarity loss
                global_protos_list = [[0.0] * self.opt.bit] * self.opt.num_class
                for key, value in global_protos.items():
                    global_protos_list[key] = value[0].cpu().numpy().tolist()
                P = torch.tensor(global_protos_list)
                if self.opt.use_gpu:
                   P = P.cuda()
                l2_x = torch.sum(torch.pow(self.opt.bit * sample_L - torch.matmul(cur_f, P.t()), 2))
                # proto loss
                proto_new = copy.deepcopy(protos.data)
                for i, label in enumerate(labels):
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                loss_proto = loss_mse(proto_new, protos)
            loss_x += self.opt.beta * l2_x + self.opt.eta * loss_proto
            loss_x /= (batch_size * num_train)
            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()

            # global proto
            for i, label in enumerate(labels):
                if label.item() in agg_protos_label:
                    agg_protos_label[label.item()].append(protos[i, :])
                else:
                    agg_protos_label[label.item()] = [protos[i, :]]

        return model.state_dict(), self.F_buffer, agg_protos_label, loss_proto


# train text net
class LocalTxtUpdate(object):
    def __init__(self, opt, G_buffer, Y, B, train_txt, train_label):
        self.opt = opt
        self.G_buffer = G_buffer
        self.Y = Y
        self.B = B
        self.train_text = train_txt
        self.train_label = train_label

    def update_weights_het(self, global_protos, model):
        # set mode to train model
        num_train = self.train_text.shape[0]
        batch_size = self.opt.batch_size

        lr = self.opt.lr
        optimizer_txt = torch.optim.Adam(model.parameters(), lr=lr)

        # proto similarity loss
        l2_y = 0.
        # proto loss
        loss_proto = 0.
        agg_protos_label = {}

        # train text net
        index = np.random.permutation(num_train)
        for idx in tqdm(range(num_train // batch_size + 1)):
            # use the last batch dataset
            remaining = num_train - idx * batch_size
            if remaining <= 0:
                break
            ind = index[idx * batch_size: (idx + 1) * batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = self.train_label[ind, :]
            labels = [np.argmax(item) for item in sample_L.cpu()]
            text = self.train_text[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            if self.opt.use_gpu:
                text = text.cuda()
                sample_L = sample_L.cuda()

            cur_g, protos = model(text)  # cur_g: (batch_size, bit)
            self.G_buffer[ind, :] = cur_g.data

            # calculate loss
            l1_y = torch.sum(torch.pow(self.opt.bit * sample_L - torch.matmul(cur_g, self.Y.t()), 2))
            quantization_y = torch.sum(torch.pow(self.B[ind, :] - cur_g, 2))
            loss_y = self.opt.alpha * l1_y + self.opt.gamma * quantization_y
            # proto loss
            loss_mse = nn.MSELoss()
            # loss_mse = nn.KLDivLoss()
            if len(global_protos) == 0:
                l2_y = 0 * loss_y
                loss_proto = 0 * loss_y
            else:
                # proto similarity loss
                global_protos_list = [[0.0] * self.opt.bit] * self.opt.num_class
                for key, value in global_protos.items():
                    global_protos_list[key] = value[0].cpu().numpy().tolist()
                P = torch.tensor(global_protos_list)
                if self.opt.use_gpu:
                    P = P.cuda()
                l2_y = torch.sum(torch.pow(self.opt.bit * sample_L - torch.matmul(cur_g, P.t()), 2))
                # proto loss
                proto_new = copy.deepcopy(protos.data)
                for i, label in enumerate(labels):
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                loss_proto = loss_mse(proto_new, protos)
            loss_y += self.opt.beta * l2_y + self.opt.eta * loss_proto
            loss_y /= (num_train * batch_size)
            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()

            # global proto
            for i, label in enumerate(labels):
                if label.item() in agg_protos_label:
                    agg_protos_label[label.item()].append(protos[i, :])
                else:
                    agg_protos_label[label.item()] = [protos[i, :]]

        return model.state_dict(), self.G_buffer, agg_protos_label, loss_proto
