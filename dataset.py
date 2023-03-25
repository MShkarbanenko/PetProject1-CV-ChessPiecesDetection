from config import *


def transform_image(image):
    """Applies a sequence of transformations to image."""

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image)


class CPDDataset(Dataset):
    """Chess pieces detection dataset."""

    def __init__(self, root_dir, classes_dict, transform=transform_image):
        """
        Args:
            root_dir (string): Directory with all the images and their annotations.
            transform (callable, optional): Transformations applied to images.
        """

        self.images_dir = os.path.join(root_dir, 'images')
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        self.transform = transform
        self.images_names = sorted(os.listdir(self.images_dir), key=lambda i: i[:-4])
        self.annotations_names = sorted(os.listdir(self.annotations_dir), key=lambda i: i[:-4])
        self.classes_dict = classes_dict

    def __len__(self):
        return len([entry for entry in os.listdir(self.annotations_dir)])

    def __getitem__(self, idx):
        annotation_path = os.path.join(self.annotations_dir, self.annotations_names[idx])
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        xmin_iter, ymin_iter = root.iter('xmin'), root.iter('ymin')
        xmax_iter, ymax_iter = root.iter('xmax'), root.iter('ymax')
        name_iter = root.iter('name')
        image_bboxes, image_labels = [], []
        for (xmin, ymin, xmax, ymax, name) in zip(xmin_iter, ymin_iter, xmax_iter, ymax_iter, name_iter):
            image_bboxes.append([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)])
            image_labels.append(self.classes_dict[name.text])
        image_path = os.path.join(self.images_dir, self.images_names[idx])
        image = io.imread(image_path)
        image_annotations = {'boxes': torch.Tensor(image_bboxes), 'labels': torch.Tensor(image_labels).to(int)}
        if self.transform:
            image = self.transform(image)
        return image, image_annotations
