# CustomObjectDetection
Object Detection on Custom Dataset (Solar Pannels Dataset)

create environment.yml for conda virtual environment
conda env create environment.yml

using zip for for loop
use pytorch lighting for training (rewrite the train loop as well as eval loop) 

build pytorch project
build pytorch lighting project

Dataset
images
bboxes : (Xmin, Ymin) - (Xmax, Ymax) - the bboxes are not normalized in range [0,1] e.g  (420, 171) - (535, 486)
