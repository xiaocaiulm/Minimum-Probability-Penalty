import torch
import torchvision.models as models


def get_model(opt, num_classes):
    if opt.model_name == 'vgg':
        net = models.vgg16_bn(pretrained=True)
        in_features = net.classifier[0].in_features
        net._fc = torch.nn.Linear(
            in_features=in_features, out_features=num_classes)
    elif opt.model_name == 'resnet':
        net = models.resnet50(pretrained=True)
        in_features = net.fc.in_features
        net._fc = torch.nn.Linear(
            in_features=in_features, out_features=num_classes)
    elif opt.model_name == 'resnext':
        net = models.resnext101_32x8d(pretrained=True)
        in_features = net.fc.in_features
        net._fc = torch.nn.Linear(
            in_features=in_features, out_features=num_classes)
    elif opt.model_name == 'googlenet':
        net = models.googlenet(pretrained=True)
        in_features = net.fc.in_features
        net._fc = torch.nn.Linear(
            in_features=in_features, out_features=num_classes)
    elif opt.model_name == 'densenet':
        net = models.densenet161(pretrained=True)
        in_features = net.classifier.in_features
        net._fc = torch.nn.Linear(
            in_features=in_features, out_features=num_classes)
    elif opt.model_name == 'seresnext':
        import pretrainedmodels
        net = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
        in_features = net.last_linear.in_features
        net.last_linear = torch.nn.Linear(
            in_features=in_features, out_features=num_classes)
    elif opt.model_name == 'inception':
        net = models.inception_v3(pretrained=True)
        in_features = net.classifier.in_features
        net.classifier = torch.nn.Linear(
            in_features=in_features, out_features=num_classes)
    else:
        raise NotImplementedError

    return net
