import h5py
import scipy.io as sio


def load_data(dataset, num_class, path):

    image_path = path + "image.mat"
    label_path = path + "label.mat"
    tag_path = path + "tag.mat"

    images, tags, labels = [], [], []
    if dataset == 'FashionVC':
        images = sio.loadmat(image_path)['Image'].transpose(0, 3, 1, 2)  # FashionVC:19862,3,224,224
        tags = sio.loadmat(tag_path)['Tag']  # FashionVC:19862,2685
        labels = sio.loadmat(label_path)['Label']
        labels = labels[:, labels.shape[1] - num_class:labels.shape[1]]  # FashionVC:19862,27
    elif dataset == 'Ssense':
        images = h5py.File(image_path)['Image'][:].transpose(3, 0, 1, 2)  # Ssense:15696,3,224,224
        tags = sio.loadmat(tag_path)['Tag']  # Ssense:15696,4945
        labels = sio.loadmat(label_path)['Label']
        labels = labels[:, labels.shape[1] - num_class:labels.shape[1]]  # Ssense:15696,28

    return images, tags, labels


def load_pretrain_model(path):
    return sio.loadmat(path)


if __name__ == '__main__':
    a = {'s': [12, 33, 44],
         's': 0.111}
    import os
    with open('result.txt', 'w') as f:
        for k, v in a.items():
            f.write(k, v)