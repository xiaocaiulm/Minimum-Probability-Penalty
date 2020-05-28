from utils import get_dataset_config
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.custom_dataset import CustomDataset


def data_loader(opt):
    data_config = get_dataset_config(opt.dataset)
    rate = 0.875

    transform_train = transforms.Compose([
        transforms.Resize((int(opt.input_size // rate), int(opt.input_size // rate))),
        transforms.RandomCrop((opt.input_size, opt.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = CustomDataset(
        data_config['train'], data_config['train_root'], transform=transform_train)
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)

    transform_test = transforms.Compose([
        transforms.Resize((opt.image_size, opt.image_size)),
        transforms.CenterCrop(opt.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dataset = CustomDataset(
        data_config['val'], data_config['val_root'], transform=transform_test)
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)

    print('Dataset Name:{dataset_name}, Train:[{train_num}], Val:[{val_num}]'.format(
        dataset_name=opt.dataset,
        train_num=len(train_dataset),
        val_num=len(val_dataset)))
    print('Batch Size:[{0}], Total:::Train Batches:[{1}],Val Batches:[{2}]'.format(
        opt.batch_size, len(train_loader), len(val_loader)
    ))

    return train_dataset, train_loader, val_dataset, val_loader
