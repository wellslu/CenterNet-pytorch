import mlconfig
import torchvision.transforms as transforms
from torch.utils import data
from .datagen import CenternetDataset


@mlconfig.register
class VOCDataLoader(data.DataLoader):

    def __init__(self, root: str, list_file: str, train: bool, batch_size: int, scale: int, num_classes: int, **kwargs):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((scale,scale)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        with open(list_file) as f:
            train_lines = f.readlines()
            
        dataset = CenternetDataset(transform, train_lines, scale, num_classes, train)

        super(VOCDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=train, **kwargs)
