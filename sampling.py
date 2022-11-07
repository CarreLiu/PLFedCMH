import numpy as np
import random


def iid(opt, train_label):
    """
    Sample I.I.D. client data from dataset
    """
    num_items = int(len(train_label) / opt.num_users)
    dict_users, all_idxs = {}, [i for i in range(len(train_label))]
    for i in range(opt.num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users


def noniid(opt, train_label, n_list, k_list):
    """
    Sample non-I.I.D client data from dataset
    """
    num_shards, num_imgs = 0, 0
    if opt.dataset == 'FashionVC':
        # 16,862 training imgs
        num_shards, num_imgs = 27, 624
    elif opt.dataset == 'Ssense':
        # 13,696 training imgs
        num_shards, num_imgs = 28, 489
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards * num_imgs)
    labels = [np.argmax(item) for item in train_label]
    labels = np.array(labels)[:num_shards * num_imgs]
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    # the num for each class
    class_num = [label_begin[item+1] - label_begin[item] for item in range(opt.num_class-1)]
    class_num.append(num_shards*num_imgs-label_begin[opt.num_class-1])
    class_sort = np.array(class_num)[np.array(class_num).argsort()]
    # each class for k_len
    class_divided = np.floor(np.array(class_num) / opt.train_shots_max)
    idxs_class_divided = np.vstack((np.zeros(opt.num_class).astype(int), class_divided.astype(int)))

    classes_list = []
    for i in range(opt.num_users):
        class_candidate = []
        for index, each_class in enumerate(idxs_class_divided[1]):
            if each_class >= 1:
                class_candidate.append(index)
        n = n_list[i]
        k = k_list[i]
        k_len = opt.train_shots_max
        classes = random.sample(class_candidate, n)
        classes = np.sort(classes)

        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = idxs_class_divided[0][each_class] * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

        # classes里面出现了的类的数量在idxs_class_divided[1]中减1
        for each_class in classes:
            idxs_class_divided[0][each_class] = idxs_class_divided[0][each_class] + 1
            idxs_class_divided[1][each_class] = idxs_class_divided[1][each_class] - 1

    return dict_users, classes_list


def noniid_unequal(opt, train_label):
    """
    Sample non-I.I.D client data from dataset s.t clients
    have unequal amount of data
    """
    num_shards, num_imgs = 0, 0
    if opt.dataset == 'FashionVC':
        num_shards, num_imgs = 843, 20
    elif opt.dataset == 'Ssense':
        num_shards, num_imgs = 228, 60
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(opt.num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = [np.argmax(item) for item in train_label]
    labels = np.array(labels)[:num_shards * num_imgs]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=opt.num_users)
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    # caused by np.around
    if sum(random_shard_size) > num_shards:

        for i in range(opt.num_users):
            # First assign each client 1 shard to ensure every client has at least one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(opt.num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    else:

        for i in range(opt.num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        # caused by np.around
        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate((dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users
