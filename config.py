import warnings


class DefaultConfig(object):
    load_img_path = None  # load model path
    load_txt_path = None

    dataset = 'FashionVC'
    # dataset = 'Ssense'
    # data parameters
    if dataset == 'FashionVC':
        data_path = './data/FashionVC/'
        pretrain_model_path = './data/imagenet-vgg-f.mat'
        training_size = 16862
        query_size = 3000
        database_size = 16862
        batch_size = 128
        # local arguments
        num_class = 27
        num_users = 10
        ways = 15
        shots = 20
        train_shots_max = 20
        stdev = 2
        train_ep = 5  # the number of local episodes
    elif dataset == 'Ssense':
        data_path = './data/Ssense/'
        pretrain_model_path = './data/imagenet-vgg-f.mat'
        training_size = 13696
        query_size = 2000
        database_size = 13696
        batch_size = 128
        # local arguments
        num_class = 28
        num_users = 10
        ways = 15
        shots = 20
        train_shots_max = 20
        stdev = 2
        train_ep = 5  # the number of local episodes

    # hyper-parameters
    # layer-wise
    hn_lr = 0.001
    embedding_dim_img = 32
    embedding_dim_txt = 32
    hidden_dim_img = 32
    hidden_dim_txt = 32
    max_epoch = 50
    alpha = 0.5
    beta = 0.5
    s_param = 0
    gamma = 10
    eta = 100000
    bit = 32  # final binary code length
    lr = 0.0001

    iid = False
    unequal = False
    use_gpu = True
    valid = True

    print_freq = 2  # print info every N epoch

    result_dir = 'result'
    weight_dir = 'clients_weight/'

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()
