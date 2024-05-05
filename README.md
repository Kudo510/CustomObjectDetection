# CustomObjectDetection
Object Detection on Custom Dataset (Solar Pannels Dataset)

## Setup 
```
conda env create environment.yml
conda activate obj_detection
```
## Training
```python train.py```

## Testing
```python test.py```

## Dataset
Input as 
- Images
- bboxes are in format (Xmin, Ymin, Xmax, Ymax) and unnormalized e.g  (420, 171) - (535, 486) not in YOLO format

### What is done
Using zip for for loop
Using if __name__ == "__main__"
Building Pytorch project
### What is left
Using pytorch lighting for training and testing (rewrite the train loop as well as eval loop) 

Using wandb



