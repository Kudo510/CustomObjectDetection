import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
def SolarPannelsDetectionModel(num_classes=2):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 2  # 1 class (person) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model