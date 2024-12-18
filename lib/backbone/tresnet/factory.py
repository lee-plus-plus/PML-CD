from .tresnet_v2 import TResnetL_V2
from .tresnet import TResnetM, TResnetL, TResnetXL

try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401


def create_model(name, pretrained=False):
    """Create a model
    """
    base_url = 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet'

    if name == 'tresnet_m':
        model = TResnetM()
        if pretrained:
            state_dict = load_state_dict_from_url(
                url=f'{base_url}/tresnet_m.pth', progress=True, map_location='cpu',
                check_hash=True)['model']
            model.load_state_dict(state_dict, strict=True)

    elif name == 'tresnet_l':
        model = TResnetL()
        if pretrained:
            state_dict = load_state_dict_from_url(
                url=f'{base_url}/tresnet_l.pth', progress=True, map_location='cpu',
                check_hash=True)['model']
            model.load_state_dict(state_dict, strict=True)

    elif name == 'tresnet_xl':
        model = TResnetXL()
        if pretrained:
            state_dict = load_state_dict_from_url(
                url=f'{base_url}/tresnet_xl.pth', progress=True, map_location='cpu',
                check_hash=True)['model']
            model.load_state_dict(state_dict, strict=True)

    elif name == 'tresnet_l_v2':
        model = TResnetL_V2()
        if pretrained:
            state_dict = load_state_dict_from_url(
                url=f'{base_url}/stanford_cars_tresnet-l-v2_96_27.pth',
                progress=True, map_location='cpu', check_hash=True)['model']
            model.load_state_dict(state_dict, strict=True)

    else:
        raise ValueError(f'unsupported model name ‘{name}’')

    return model
