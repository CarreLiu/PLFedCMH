import copy
import gc
from config import opt
from data_handler import *
import numpy as np
import torch
from tqdm import tqdm
from models import ImgModule, TxtModule
from utils import get_dataset, calc_map_k, split_data, agg_func, proto_aggregation
from update import LocalImgUpdate, LocalTxtUpdate
from models.hypernetwork_module import ImageHyperNetwork, TextHyperNetwork
from collections import OrderedDict
import os
import datetime
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def FedProto_taskheter(opt, X, Y, L, user_groups, img_model, txt_model, S, local_F_buffer_dict,
                       local_G_buffer_dict, local_Y_dict, local_B_dict, img_client_model_params_list,
                       txt_client_model_params_list, img_all_params_name, txt_all_params_name):
    train_L = torch.from_numpy(L['train'])
    train_x = torch.from_numpy(X['train'])
    train_y = torch.from_numpy(Y['train'])

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    if opt.use_gpu:
        train_L = train_L.cuda()

    result = {
        'loss': [],
        'proto_loss': [],
        'mapi2t': [[] for idx in range((opt.num_users // 3))],
        'mapt2i': [[] for idx in range((opt.num_users // 3))]
    }

    max_mapi2t = [0. for idx in range((opt.num_users // 3))]
    max_mapt2i = [0. for idx in range((opt.num_users // 3))]

    # HyperNetwork
    img_hypernet = ImageHyperNetwork(
        modal='img',
        embedding_dim=opt.embedding_dim_img,
        client_num=opt.num_users,
        hidden_dim=opt.hidden_dim_img,
        backbone=img_model,
        gpu=opt.use_gpu,
    )
    txt_hypernet = TextHyperNetwork(
        modal='txt',
        embedding_dim=opt.embedding_dim_txt,
        client_num=opt.num_users,
        hidden_dim=opt.hidden_dim_txt,
        backbone=txt_model,
        gpu=opt.use_gpu,
    )

    # fedproto begin
    global_img_protos, global_txt_protos = [], []
    idxs_users = np.arange(opt.num_users)

    for round in range(opt.max_epoch):
        local_losses, local_proto_losses = [], []
        local_img_protos, local_txt_protos = {}, {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        # proto_img_loss, proto_txt_loss = 0, 0
        for idx in idxs_users:
            # layer-wise
            img_layer_params_dict = dict(
                zip(img_all_params_name, list(zip(*img_client_model_params_list))))
            txt_layer_params_dict = dict(
                zip(txt_all_params_name, list(zip(*txt_client_model_params_list))))
            img_alpha = img_hypernet(idx)
            txt_alpha = txt_hypernet(idx)
            img_aggregated_parameters, txt_aggregated_parameters = {}, {}
            for name in img_all_params_name:
                a = img_alpha[name.split(".")[-2]]
                img_aggregated_parameters[name] = torch.sum(
                    a
                    / a.sum()
                    * torch.stack(img_layer_params_dict[name], dim=-1).cuda(),
                    dim=-1,
                )
            img_client_model_params_list[idx] = list(img_aggregated_parameters.values())
            for name in txt_all_params_name:
                a = txt_alpha[name.split(".")[-2]]
                txt_aggregated_parameters[name] = torch.sum(
                    a
                    / a.sum()
                    * torch.stack(txt_layer_params_dict[name], dim=-1).cuda(),
                    dim=-1,
                )
            txt_client_model_params_list[idx] = list(txt_aggregated_parameters.values())

            # theta * alpha, put it into client update
            img_model_net = copy.deepcopy(img_model)
            img_model_net.load_state_dict(img_aggregated_parameters, strict=True)
            txt_model_net = copy.deepcopy(txt_model)
            txt_model_net.load_state_dict(txt_aggregated_parameters, strict=True)

            img_protos, txt_protos = {}, {}
            idxs = user_groups[idx]
            # pick up the certain client's training image, text and label dataset
            local_train_x = train_x[idxs]
            local_train_y = train_y[idxs]
            local_train_L = train_L[idxs]

            epoch_loss = {'total': [], 'proto': []}
            epoch_weights = {'image': img_model_net.state_dict(),
                             'text': txt_model_net.state_dict()}
            # the weights before train
            img_frz_model_params = OrderedDict({
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in epoch_weights['image'].items()})
            txt_frz_model_params = OrderedDict({
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in epoch_weights['text'].items()})

            for iter in range(opt.train_ep):
                # update local weights every epoch
                img_model_net.load_state_dict(epoch_weights['image'])
                txt_model_net.load_state_dict(epoch_weights['text'])

                local_img_model = LocalImgUpdate(opt=opt, F_buffer=local_F_buffer_dict[idx], Y=local_Y_dict[idx],
                                                 B=local_B_dict[idx], train_img=local_train_x,
                                                 train_label=local_train_L)
                img_w, local_F_buffer, img_protos, loss_img_proto_loss = local_img_model.update_weights_het(
                    global_img_protos, model=img_model_net)
                local_F_buffer_dict[idx] = local_F_buffer

                local_txt_model = LocalTxtUpdate(opt=opt, G_buffer=local_G_buffer_dict[idx], Y=local_Y_dict[idx],
                                                 B=local_B_dict[idx], train_txt=local_train_y,
                                                 train_label=local_train_L)
                txt_w, local_G_buffer, txt_protos, loss_txt_proto_loss = local_txt_model.update_weights_het(
                    global_txt_protos, model=txt_model_net)
                local_G_buffer_dict[idx] = local_G_buffer
                # update B
                local_B = torch.sign(opt.gamma * (local_F_buffer + local_G_buffer))
                local_B_dict[idx] = local_B

                # update Y
                local_Q = opt.bit * (torch.matmul(local_F_buffer.t(), local_train_L.to(torch.float32))
                                     + torch.matmul(local_G_buffer.t(), local_train_L.to(torch.float32))
                                     + opt.s_param * torch.matmul(local_Y_dict[idx].t(), S.to(torch.float32)))

                for i in range(3):
                    local_Y_buffer = local_Y_dict[idx]
                    for k in range(opt.bit):
                        local_sel_ind = np.setdiff1d([ii for ii in range(opt.bit)], k)
                        local_yk = local_Y_dict[idx][:, k]
                        local_Y_ = local_Y_dict[idx][:, local_sel_ind]
                        local_Fk = local_F_buffer[:, k]
                        local_F_ = local_F_buffer[:, local_sel_ind]
                        local_Gk = local_G_buffer[:, k]
                        local_G_ = local_G_buffer[:, local_sel_ind]

                        local_y = torch.sign(
                            local_Q[k, :] - (torch.matmul(local_Y_, torch.matmul(local_F_.t(), local_Fk))
                                             + torch.matmul(local_Y_, torch.matmul(local_G_.t(), local_Gk))))
                        local_Y_dict[idx][:, k] = local_y
                    if np.linalg.norm(
                            local_Y_dict[idx].cpu().numpy() - local_Y_buffer.cpu().numpy()) < 1e-6 * np.linalg.norm(
                        local_Y_buffer.cpu().numpy()):
                        break

                loss, loss_proto = calc_loss(local_B, local_F_buffer, local_G_buffer, local_Y_dict[idx], local_train_L,
                                             loss_img_proto_loss, loss_txt_proto_loss, opt)
                print(
                    'Local loss...global round: %3d, client: %3d, local epoch: %3d, loss: %3.3f, proto loss: %3.3f, comment: update B'
                    % (round + 1, idx + 1, iter + 1, loss.item(), loss_proto.item()))
                epoch_loss['total'].append(loss.item())
                epoch_loss['proto'].append(loss_proto.item())
                epoch_weights['image'] = img_w
                epoch_weights['text'] = txt_w

            agg_img_protos = agg_func(img_protos)
            agg_txt_protos = agg_func(txt_protos)
            # prepare for the valid use
            epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
            epoch_loss['proto'] = sum(epoch_loss['proto']) / len(epoch_loss['proto'])
            local_losses.append(copy.deepcopy(epoch_loss['total']))
            local_proto_losses.append(copy.deepcopy(epoch_loss['proto']))
            local_img_protos[idx] = agg_img_protos
            local_txt_protos[idx] = agg_txt_protos
            # calculate delta for local update
            img_diff = OrderedDict({
                k: p1 - p0
                for (k, p1), p0 in zip(
                    epoch_weights['image'].items(),
                    img_frz_model_params.values(), )})
            txt_diff = OrderedDict({
                k: p1 - p0
                for (k, p1), p0 in zip(
                    epoch_weights['text'].items(),
                    txt_frz_model_params.values(), )})
            # update hypernetwork
            img_hypernet = update_hypernetwork(img_hypernet, img_client_model_params_list, idx, img_diff, opt)
            txt_hypernet = update_hypernetwork(txt_hypernet, txt_client_model_params_list, idx, txt_diff, opt)
            # update client model
            img_updated_params = []
            for param, diff in zip(
                    img_client_model_params_list[idx], img_diff.values()
            ):
                img_updated_params.append((param + diff).detach())
            img_client_model_params_list[idx] = img_updated_params
            txt_updated_params = []
            for param, diff in zip(
                    txt_client_model_params_list[idx], txt_diff.values()
            ):
                txt_updated_params.append((param + diff).detach())
            txt_client_model_params_list[idx] = txt_updated_params

            del img_model_net, txt_model_net
            gc.collect()

        # calculate avg clients' losses as global
        loss_avg = sum(local_losses) / len(local_losses)
        loss_proto_avg = sum(local_proto_losses) / len(local_proto_losses)
        print('Global loss...global round: %3d, loss: %3.3f, proto loss: %3.3f' % (round + 1, loss_avg, loss_proto_avg))
        result['loss'].append(float(loss_avg))
        result['proto_loss'].append(float(loss_proto_avg))

        global_img_protos = proto_aggregation(local_img_protos)
        global_txt_protos = proto_aggregation(local_txt_protos)

        # valid
        if opt.valid:
            for idx in range(opt.num_users // 3):
                img_parameters = OrderedDict(
                    {
                        k: p
                        for k, p in zip(
                        img_all_params_name,
                        img_client_model_params_list[idx]
                        )
                    }
                )
                txt_parameters = OrderedDict(
                    {
                        k: p
                        for k, p in zip(
                        txt_all_params_name,
                        txt_client_model_params_list[idx]
                        )
                    }
                )
                img_model.load_state_dict(img_parameters)
                txt_model.load_state_dict(txt_parameters)
                mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
                                       query_L, retrieval_L)
                print('...round: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (round + 1, mapi2t, mapt2i))
                result['mapi2t'][idx].append(mapi2t)
                result['mapt2i'][idx].append(mapt2i)
                if mapt2i >= max_mapt2i[idx] and mapi2t >= max_mapi2t[idx]:
                    max_mapi2t[idx] = mapi2t
                    max_mapt2i[idx] = mapt2i
                    datasetName = opt.data_path.split('/')[2]
                    iid_unequal = 'iid' if opt.iid else 'noniid_unequal' if opt.unequal else 'noniid_equal'
                    if not os.path.exists('checkpoints/' + iid_unequal + '/layerwise/'):
                        os.makedirs('checkpoints/' + iid_unequal + '/layerwise/')
                    img_model.save('checkpoints/' + iid_unequal + '/layerwise/', img_model.module_name + '_'
                                   + datasetName + '_' + 'layerwise' + '_' + 'S' + str(opt.s_param) + '_'
                                   + str(opt.max_epoch) + '_' + str(opt.train_ep) + '_' + str(opt.embedding_dim_img) + '_' + str(opt.embedding_dim_txt) + '_' + iid_unequal + '_' + str(opt.bit) + '.pth')
                    txt_model.save('checkpoints/' + iid_unequal + '/layerwise/', txt_model.module_name + '_'
                                   + datasetName + '_' + 'layerwise' + '_' + 'S' + str(opt.s_param) + '_'
                                   + str(opt.max_epoch) + '_' + str(opt.train_ep) + '_' + str(opt.embedding_dim_img) + '_' + str(opt.embedding_dim_txt) + '_' + iid_unequal + '_' + str(opt.bit) + '.pth')

    print('...training procedure finish')
    if opt.valid:
        max_mapi2t_avg = sum(max_mapi2t) / len(max_mapi2t)
        max_mapt2i_avg = sum(max_mapt2i) / len(max_mapt2i)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t_avg, max_mapt2i_avg))
        result['max_mapi2t_list'] = max_mapi2t
        result['max_mapt2i_list'] = max_mapt2i
        result['max_mapi2t'] = max_mapi2t_avg
        result['max_mapt2i'] = max_mapt2i_avg
    else:
        mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
                               query_L, retrieval_L)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
        result['max_mapi2t'] = mapi2t
        result['max_mapt2i'] = mapt2i

    write_result(result)
    img_hypernet.clean_models()
    txt_hypernet.clean_models()


def update_hypernetwork(hypernet, client_model_params_list, client_id, diff, opt):
    # calculate gradients
    hn_grads = torch.autograd.grad(
        outputs=list(client_model_params_list[client_id]),
        inputs=hypernet.mlp_parameters() + hypernet.fc_layer_parameters() + hypernet.emd_parameters(),
        grad_outputs=list(
            map(lambda tup: tup[1], diff.items())
        ),
        allow_unused=True,
    )
    mlp_grads = hn_grads[: len(hypernet.mlp_parameters())]
    fc_grads = hn_grads[
               len(hypernet.mlp_parameters()): len(
                   hypernet.mlp_parameters() + hypernet.fc_layer_parameters()
               )
               ]
    emd_grads = hn_grads[
                len(hypernet.mlp_parameters() + hypernet.fc_layer_parameters()):
                ]

    for param, grad in zip(hypernet.fc_layer_parameters(), fc_grads):
        if grad is not None:
            param.data -= opt.hn_lr * grad

    for param, grad in zip(hypernet.mlp_parameters(), mlp_grads):
        param.data -= opt.hn_lr * grad

    for param, grad in zip(hypernet.emd_parameters(), emd_grads):
        param.data -= opt.hn_lr * grad

    hypernet.save_hn()

    return hypernet


def train(**kwargs):
    start_time = datetime.datetime.now()
    opt.parse(kwargs)

    n_list = np.random.randint(max(2, opt.ways - opt.stdev), min(opt.num_class, opt.ways + opt.stdev + 1),
                               opt.num_users)
    k_list = np.random.randint(opt.shots - opt.stdev + 1, opt.shots + opt.stdev - 1, opt.num_users)

    pretrain_model = load_pretrain_model(opt.pretrain_model_path)

    X, Y, L, user_groups = get_dataset(opt, n_list, k_list)
    y_dim = Y['train'].shape[1]
    print('...loading and splitting data finish')

    # layer-wise
    img_client_model_params_list = []
    txt_client_model_params_list = []

    # build models for clients and save weights in local
    img_model = ImgModule(opt.bit, pretrain_model)
    txt_model = TxtModule(y_dim, opt.bit)
    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
    img_all_params_name = [name for name in img_model.state_dict().keys()]
    txt_all_params_name = [name for name in txt_model.state_dict().keys()]

    for client in range(opt.num_users):
        img_client_model_params_list.append(list(img_model.state_dict().values()))
        txt_client_model_params_list.append(list(txt_model.state_dict().values()))

    # init F_buffer, G_buffer, Y_buffer(that is Y, to differ from txt dataset Y), B for clients
    local_F_buffer_dict, local_G_buffer_dict, local_Y_dict, local_B_dict = {}, {}, {}, {}
    for i in range(opt.num_users):
        num_train = len(user_groups[i])
        # num_class = len(classes_list[i])
        F_buffer = torch.randn(num_train, opt.bit)
        G_buffer = torch.randn(num_train, opt.bit)
        Y_buffer = torch.sign(torch.randn(opt.num_class, opt.bit))
        if opt.use_gpu:
            F_buffer = F_buffer.cuda()
            G_buffer = G_buffer.cuda()
            Y_buffer = Y_buffer.cuda()
        B = torch.sign(F_buffer + G_buffer)
        local_F_buffer_dict[i] = F_buffer
        local_G_buffer_dict[i] = G_buffer
        local_Y_dict[i] = Y_buffer
        local_B_dict[i] = B

    S = torch.tensor(np.eye(opt.num_class)).cuda()

    FedProto_taskheter(opt, X, Y, L, user_groups, img_model, txt_model, S,
                       local_F_buffer_dict, local_G_buffer_dict, local_Y_dict,
                       local_B_dict, img_client_model_params_list, txt_client_model_params_list,
                       img_all_params_name, txt_all_params_name)
    end_time = datetime.datetime.now()
    process_time = end_time - start_time
    print('The program runs ', process_time.seconds, 'seconds')


def valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L):
    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    return mapi2t, mapt2i


def test(**kwargs):
    opt.parse(kwargs)

    images, tags, labels = load_data(opt.dataset, opt.num_class, opt.data_path)
    y_dim = tags.shape[1]

    X, Y, L = split_data(opt, images, tags, labels)
    print('...loading and splitting data finish')

    img_model = ImgModule(opt.bit)
    txt_model = TxtModule(y_dim, opt.bit)

    if opt.load_img_path:
        img_model.load(opt.load_img_path)

    if opt.load_txt_path:
        txt_model.load(opt.load_txt_path)

    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    if opt.use_gpu:
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    print('...test MAP: MAP(i->t): %3.3f, MAP(t->i): %3.3f' % (mapi2t, mapt2i))


def calc_loss(B, F, G, Y, train_L, loss_img_proto, loss_txt_proto, opt):
    term1 = torch.sum(
        torch.pow(opt.bit * train_L - torch.matmul(F, Y.t()), 2) + torch.pow(opt.bit * train_L - torch.matmul(G, Y.t()),
                                                                             2))
    term2 = opt.gamma * torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    loss = term1 + term2

    loss_proto = loss_img_proto + loss_txt_proto
    loss += opt.eta * loss_proto

    return loss, loss_proto


def generate_image_code(img_model, X, bit):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        if len(ind) == 0:
            break
        image = X[ind].type(torch.float)
        if opt.use_gpu:
            image = image.cuda()
        cur_f, _ = img_model(image)
        B[ind, :] = cur_f.data
    B = torch.sign(B)
    return B


def generate_text_code(txt_model, Y, bit):
    batch_size = opt.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        if len(ind) == 0:
            break
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
        if opt.use_gpu:
            text = text.cuda()
        cur_g, _ = txt_model(text)
        B[ind, :] = cur_g.data
    B = torch.sign(B)
    return B


def write_result(result):
    import os
    datasetName = opt.data_path.split('/')[2]
    iid_unequal = 'iid' if opt.iid else 'noniid_unequal' if opt.unequal else 'noniid_equal'
    if not os.path.exists(opt.result_dir + '/' + iid_unequal + '/layerwise/'):
        os.makedirs(opt.result_dir + '/' + iid_unequal + '/layerwise/')
    with open(os.path.join(opt.result_dir + '/' + iid_unequal + '/layerwise', 'result_' + datasetName + '_' + 'layerwise' + '_'
                                           + 'S' + str(opt.s_param) + '_' + str(opt.max_epoch) + '_'
                                           + str(opt.train_ep) + '_' + str(opt.embedding_dim_img) + '_' + str(opt.embedding_dim_txt) + '_' + iid_unequal + '_' + str(opt.bit) + '_new' + '.txt'), 'w') as f:
        for k, v in result.items():
            f.write(k + ' ' + str(v) + '\n')


def help():
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --lr=0.01
            python {0} help
    avaiable args:'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print('\t\t{0}: {1}'.format(k, v))


if __name__ == '__main__':
    train()
