# DS-UA 301 Final Project

This project achieves object manipulation detection through R-CNN. It is simple, but most importantly it is implemented from scratch.

The input image would be reshaped to 128x128 and serves as the input to a ResNet encoder. The feature map after the third set of layers in the ResNet encoder (8x8) would be extracted, an 3x3 convolutional layer on top of it. Then, two sibling 1x
1 convolutional layers for classification and regression respectively would be applied to generate the final output.

The cross-entropy loss is applied on the classification output, while smooth-l1 loss is applied on the regression output.

![Fast R-CNN](/imgs/faster_rcnn.png)

For more details, refer to the presentation slide attached in this repository.

## About this repository

The main file of this repository is `DS-UA 301 Final Project.ipynb`. By running this jupyter notebook cell by cell, one could instantiate the dataset, model, loss function and everything needed to train the model. Then, a ray tune task will be started to find the best set of hyperparameters for the model.

The datasets that will be used are not included in the repository so you need to download them. The datasets needed are:

- [COCO2014 Dataset](http://images.cocodataset.org/zips/train2014.zip)
- [COCO Synthetic Dataset](https://drive.google.com/open?id=1vIAFsftjmHg2J5lJgO92C1Xmyw539p_B) provided by `@pengzhou1108` on GitHub

After download these two datasets, unzip and put them under the root directory of the repository.

`test_filter.txt` and `train_filter.txt` are two annotation files indicating the filenames of images for the training/testing set. For each image file, the corresponding bounding box and the label indicating whether the object in the bounding box is original or tampered.

Before running the Jupyter notebook, change the value of `COCO_DIR`, `SYNTHETIC_DIR`, `MODEL_DIR`, `TRAIN_FILE`, and `TEST_FILE` in the second cell to make sure they point to the correct datasets/annotation files. Change these values in `demo.py` and `remove.py` as well. Specifically, to make sure raytune could run properly, it is recommended to use absolute path for `COCO_DIR`, `SYNTHETIC_DIR`, and `MODEL_DIR` in the Jupyter notebook.

If you cannot run `raytune`, you could instead run another cell which manually implements the grid search. If you use this cell, the performance metrics could be found in an output file called `performance.json`.

Also, if you don't want to run the Jupyter notebook cell by cell, you could instead run the equivalent `DS-UA 301 Final Project.py`. You could even use the command `nohup python3 -u "DS-UA 301 Final Project.py" &` to run it in the background. You could always monitor the training progress in `nohup.out`.

After finish running the Jupyter notebook or the equivalent python script, there will be a list of `.pth` model files under the `models` folder. Since hyperband scheduler is used, the models that are stopped early may perform badly. Therefore, make sure to check the result files or output of raytune to remember the best set of hyperparameter.

Then, run `demo.py` in the terminal. Follow the prompt and enter the desired parameters. This script first shows the ground truth bounding box of some sample image, and then visualizes the model prediction. You could get a sense of how this model performs based on the comparison between the ground truth and the model output. Note that red box means there is authentic object in the box, blue box means that there is tampered object in the box.

There are also multiple helper files: run `remove.py` if you want to delete unused images in COCO2014 dataset. The scripts in `extras` folder contains useful helper functions, as well as the definition of the R-CNN model and ResNet encoder that are used in the jupyter notebook. Check it out if you are interested in the detailed implementation.

## Results

First of all, during the training phase, both the loss from regression head and classification head decreases steadily, which indicates that our model successfully captures the relationship between image features and bounding boxes. Below is a typical loss curve for the models in this repository.

![Loss Curve](/imgs/loss_curve.png)

### Charts

### Prediction Visualization

The following is the ground truth bounding box of some sample images

![Ground Truth](/imgs/ground_truth.png)

Then this is the prediction of our model (`base=resnet34 init_lr=0.1 pretrained_base=True`)

![Prediction](/imgs/prediction.png)

We could observe that our model get the correct prediction, and the bounding boxes it produces basically covers the correct part of the image. However, the prediction is still quite messy, which is partly due to the fact that the IoU threshold in the non-maximum-suppression stage may need to be adjusted. If we assume that there will only be one tampered/authentic image in the image, then we could, instead of using non-maximum-suppression, simply choose the bounding box with maximum prediction probability for authentic image and tampered image to get one bounding box.
