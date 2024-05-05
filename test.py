import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from model import SolarPannelsDetectionModel
from utils.transforms import get_transform

def test():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image = read_image("data/solar_pannels/2021-11-06 11_33_45-Google Earth.jpg")
    eval_transform = get_transform(train=False)

    model = SolarPannelsDetectionModel()
    model.load_state_dict(torch.load("checkpoints/saved_model.ckpt"))
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]


    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    # masks = (pred["masks"] > 0.7).squeeze(1)
    # output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.show()

if __name__ == "__main__":
    test()