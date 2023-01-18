import logging


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_model(model_name, num_classes, enable_dar_bn=False, dropout_rate=0.3):
    net = None

    if model_name == 'WideResNet-28-10-torchdistill':
        from models.wide_resnet_torchdistill import WideBasicBlock, WideResNet
        net = WideResNet(
            depth=28,
            k=10,
            dropout_p=dropout_rate,
            block=WideBasicBlock,
            num_classes=num_classes,
            enable_dar_bn=enable_dar_bn,
        )
    elif model_name == 'WideResNet-28-10-xternalz':
        from networks import WideResNet
        net = WideResNet(
            depth=28,
            widen_factor=10,
            dropRate=dropout_rate,
            num_classes=num_classes,
        )
    elif model_name == 'ResNet-32-m2m':
        from models.m2m_models import resnet32
        net = resnet32(num_classes=num_classes)
    elif model_name == 'ResNet-32-akamaster':
        from models.akamaster_resnet32 import resnet32
        net = resnet32(num_classes=num_classes, enable_dar_bn=enable_dar_bn)
    else:
        logging.error(f"{model_name} is not a supported model name.")
        assert ValueError(f"{model_name} is not a supported model name.")

    logging.info(f"Successfully loaded model {model_name}.")
    logging.info(f"Model parameter count: {_count_parameters(net)}")

    return net
