from torchvision import transforms
from augmentation import five_crops, HorizontalFlip, make_transforms

normalize_05 = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.25, 0.25, 0.25]
)

normalize_torch = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def preprocess(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
def preprocess_with_augmentation(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(30, expand=True),
        transforms.RandomResizedCrop(size=image_size,scale=(0.7,1.5),ratio=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_with_augmentation_affine(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.7,1.5), shear=5),
        transforms.RandomCrop(size=image_size,pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_hflip(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        HorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

