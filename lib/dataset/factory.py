from .voc import Voc2007Dataset, Voc2012Dataset
from .coco import Coco2014Dataset, Coco2017Dataset
from .nuswide import NusWideDataset
from .cifar import CifarDataset
from .cub200 import Cub200Dataset
from .rand import RandDataset
from .customize import customize
from os.path import join, expanduser
# import torchvision


# TODO: rewrite it as a register
def supported_multilabel_datasets():
    return ('voc2007', 'voc2012', 'coco2014', 'coco2017',
            'nuswide', 'cub200', 'rand')


def supported_multiclass_datasets():
    return ('cifar10', 'cifar100')


def supported_datasets():
    return supported_multilabel_datasets() + supported_multiclass_datasets()


def supported_divides():
    return ('train', 'test')


def build_dataset(
    name, divide,
    base=expanduser('~/dataset'),
    **kwargs,
):
    assert name in supported_datasets()
    assert divide in supported_divides()

    def _customize(dataset_class):
        return customize(dataset_class, **kwargs)

    if name in ['voc2007']:
        divide = {'train': 'trainval', 'test': 'test'}[divide]
        dataset = _customize(Voc2007Dataset)(
            root_dir=join(base, name),
            divide=divide
        )

    elif name in ['voc2012']:
        divide = {'train': 'train', 'test': 'val'}[divide]
        dataset = _customize(Voc2012Dataset)(
            root_dir=join(base, name),
            divide=divide
        )

    elif name in ['coco2014']:
        divide = {'train': 'train', 'test': 'val'}[divide]
        dataset = _customize(Coco2014Dataset)(
            root_dir=join(base, name),
            divide=divide
        )

    elif name in ['coco2017']:
        divide = {'train': 'train', 'test': 'val'}[divide]
        dataset = _customize(Coco2017Dataset)(
            root_dir=join(base, name),
            divide=divide
        )

    elif name in ['nuswide']:
        dataset = _customize(NusWideDataset)(
            root_dir=join(base, name), divide=divide
        )

    elif name in ['cifar10', 'cifar100']:
        dataset = _customize(CifarDataset)(
            subtype=name, root_dir=join(base, name), divide=divide
        )

    elif name in ['cub200']:
        dataset = _customize(Cub200Dataset)(
            root_dir=join(base, name), divide=divide
        )

    elif name == 'rand':
        dataset = _customize(RandDataset)(
            num_samples=100,
            num_classes=20,
            image_size=(300, 500)
        )

    return dataset
