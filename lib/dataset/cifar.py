import torch
import torchvision


category_name = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}


class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, subtype, root_dir, divide):
        assert subtype in ['cifar10', 'cifar100']
        assert divide in ['train', 'test']

        if subtype == 'cifar10':
            cifar = torchvision.datasets.CIFAR10(
                root=root_dir, train=(divide == 'train'), download=False,
            )
        elif subtype == 'cifar100':
            cifar = torchvision.datasets.CIFAR100(
                root=root_dir, train=(divide == 'train'), download=False
            )

        images = torch.tensor(cifar.data).float()   # (50000, 32, 32, 3)
        images = images.permute((0, 3, 1, 2))       # (50000, 3, 32, 32)
        labels = torch.tensor(cifar.targets).int()  # (50000, )

        # all to PIL.Image
        t = torchvision.transforms.ToPILImage()
        images = [t(image) for image in images]

        num_samples = len(images)
        num_classes = len(labels.unique())

        self.images = images
        self.category_name = category_name
        self.labels = labels
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return image, label

    def __repr__(self):
        return f'CifarDataset(num_samples={self.num_samples}, ' \
               f'num_classes={self.num_classes})'
