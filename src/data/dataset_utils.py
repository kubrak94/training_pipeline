import src.train.transforms as transforms


def prepare_transforms(transform_params):
    transforms_list = parse_transforms(transform_params)
    
    transform = transforms.Compose(transforms_list)

    return transform


def parse_transforms(transform_params):
    transforms_list = []
    for transform_info in transform_params:
        transform_name = transform_info.pop('name')

        transform_class = transforms.__dict__[transform_name]
        
        if transform_name in ['Compose', 'OneOf', 'OneOrOther']:
            sub_transforms = parse_transforms(transform_info['augs'])
            transform = transform_class(sub_transforms)
        else:
            if transform_info is not None:
                transform = transform_class(**transform_info)
            else:
                transform = transform_class()

        transforms_list.append(transform)
        
    return transforms_list
