import os
import torch
import pandas as pd
from skimage import io, transform
import torch.nn.functional as F
from torchvision import utils, transforms
from extras.image_manip import ManipDetectionModel
from torch.utils.data import DataLoader, Dataset
from extras.util import visDet
from extras.boxes import nms

COCO_DIR = "train2014"
SYNTHETIC_DIR = "coco_synthetic"
TEST_FILE = "test_filter.txt"


class ImageManipDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, txt_file, transform=None, test_mode=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_info_frame = pd.read_csv(txt_file, delimiter=" ", header=None)
        if test_mode:
            self.file_info_frame = pd.read_csv(
                txt_file, delimiter=" ", header=None).head(2048)
        self.transform = transform

    def __len__(self):
        return len(self.file_info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = get_image(self.file_info_frame.iloc[idx, 0])
        bbox = self.file_info_frame.iloc[idx, 1:5].values
        is_authentic = 1 if self.file_info_frame.iloc[idx,
                                                      5] == "authentic" else 0
        sample = {'image': image, 'bbox': bbox.reshape(1, -1)}

        if self.transform:
            sample = self.transform(sample)

        sample["authentic"] = is_authentic

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bboxs = sample['image'], sample['bbox']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        bboxs = (bboxs * [new_w / w, new_h / h,
                 new_w / w, new_h / h]).astype(float)

        return {'image': img, 'bbox': bboxs}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'bbox': torch.from_numpy(bbox)}


def get_gt_boxes():
    """
    Generate 192 boxes where each box is represented by :
    [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

    Each anchor position should generate 3 boxes according to the scales and ratios given.

    Return this result as a numpy array of size [192,4]
    """
    stride = 16  # The stride of the final feature map is 16 (the model compresses the image from 128 x 128 to 8 x 8)
    map_sz = 128  # this is the length of height/width of the image

    scales = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ratios = torch.tensor([[1, 1], [0.7, 1.4], [1.4, 0.7], [0.8, 1.2], [
                          1.2, 0.8], [0.6, 1.8], [1.8, 0.6]]).view(1, 14)

    half_stride = int(stride / 2)
    num_grids = int((map_sz / stride) ** 2)
    boxes_size = (ratios.T * scales).T.reshape(-1, 2)
    num_boxes = boxes_size.shape[0] * num_grids
    gt_boxes = torch.zeros((num_boxes, 4))

    for i in range(num_boxes):
        grid_index = i // (scales.shape[0] * ratios.shape[1] // 2)
        box_index = i % (scales.shape[0] * ratios.shape[1] // 2)
        center_x = int(grid_index % (map_sz / stride) * stride + half_stride)
        center_y = int(grid_index // (map_sz / stride) * stride + half_stride)
        top_left_x = center_x - (boxes_size[box_index, 0] / 2)
        top_left_y = center_y - (boxes_size[box_index, 1] / 2)
        bottom_right_x = center_x + (boxes_size[box_index, 0] / 2)
        bottom_right_y = center_y + (boxes_size[box_index, 1] / 2)
        gt_boxes[i, :] = torch.tensor([top_left_x, top_left_y,
                                       bottom_right_x, bottom_right_y])

    return gt_boxes


def visPred(model, sample):
    # visualize your model predictions on the sample image.
    model.eval()
    sample = sample.float()
    out_pred, out_box = model(sample)
    gt_boxes = get_gt_boxes()
    probs = F.softmax(out_pred, dim=1)
    preds = torch.argmax(probs, axis=1)
    boxes = []
    select_preds = []
    for i in range(sample.shape[0]):
        curr_pred = preds[i, :]
        curr_probs = probs[i, :]

        gt_width = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_height = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_center_x = gt_boxes[:, 0] + 0.5 * gt_width
        gt_center_y = gt_boxes[:, 1] + 0.5 * gt_height

        ex_width = gt_width / torch.exp(out_box[i, :, 2])
        ex_height = gt_height / torch.exp(out_box[i, :, 3])
        ex_center_x = gt_center_x - out_box[i, :, 0] * ex_width
        ex_center_y = gt_center_y - out_box[i, :, 1] * ex_height

        left_top_x = (ex_center_x - ex_width / 2).unsqueeze(1)
        left_top_y = (ex_center_y - ex_height / 2).unsqueeze(1)
        right_bottom_x = (ex_center_x + ex_height / 2).unsqueeze(1)
        right_bottom_y = (ex_center_y + ex_height / 2).unsqueeze(1)

        max_probs, _ = torch.max(curr_probs, axis=0)

        selected_boxes = (curr_pred != 0) & (max_probs > 0.7)

        ex_boxes = torch.cat(
            (left_top_x, left_top_y, right_bottom_x, right_bottom_y), dim=1)
        ex_boxes = ex_boxes[selected_boxes]
        curr_probs = curr_probs[:, selected_boxes]
        curr_pred = curr_pred[selected_boxes]
        max_probs = max_probs[selected_boxes]

#         print(curr_probs.shape)
#         print(torch.argmax(curr_probs, axis=0).shape)

        select_result = nms(ex_boxes, max_probs, 0.3)
        ex_boxes = ex_boxes[select_result, :]
        curr_pred = curr_pred[select_result]

        boxes.append(ex_boxes.detach())
        # convert to 1 for auth 0 for tampered
        select_preds.append(curr_pred.detach() - 1)

    visDet(sample, boxes, select_preds)


def get_image(filename):
    imdir = SYNTHETIC_DIR if filename[:2] == "Tp" else COCO_DIR
    return io.imread(os.path.join(imdir, filename))


coco_transform = transforms.Compose([
    Rescale((128, 128)),
    ToTensor()
])


if __name__ == "__main__":

    print("Select Base Model")
    print("(a) ResNet 18")
    print("(b) ResNet 34")
    print("(c) ResNet 50")
    choice = input()

    base = '18'

    if choice in 'bB':
        base = '34'
    elif choice in 'cC':
        base = '50'

    print("Select Learning Rate")
    print("(a) 1")
    print("(b) 0.1")
    print("(c) 0.01")
    choice = input()

    lr = '0.1'
    if choice in 'aA':
        lr = '1'
    elif choice in 'cC':
        lr = '0.01'

    print("Use Pretrained Model")
    print("(a) Yes")
    print("(b) No")
    choice = input()

    pretrained = choice in 'aY'

    model_filename = "model_resnet" + base + "_lr" + lr
    if pretrained:
        model_filename += '_pretrained'

    model_filename = 'models/model.pth'

    transformed_test = ImageManipDataset(
        txt_file=TEST_FILE, transform=coco_transform, test_mode=False)
    # test_loader = DataLoader(transformed_test, batch_size=4,
    #                          shuffle=False, num_workers=4)

    # data_dict = iter(test_loader).next()
    data_dict = transformed_test[:4]
    visDet(data_dict['image'][:4], data_dict['bbox'][:4],
           data_dict["authentic"][:4].reshape(-1, 1))

    model = ManipDetectionModel(base=int(base))
    model.load_state_dict(torch.load(model_filename, map_location='cpu'))
    visPred(model, data_dict['image'].to(dtype=torch.float))
