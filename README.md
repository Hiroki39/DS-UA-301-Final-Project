# DS-UA 301 Final Project

This project achieves object manipulation detection through R-CNN. It is simple, but most importantly it is implemented from scratch.

The input image would be reshaped to 128x128 and serves as the input to a ResNet encoder. The feature map after the third set of layers in the ResNet encoder (8x8) would be extracted, an 3x3 convolutional layer on top of it. Then, two sibling 1x
1 convolutional layers for classification and regression respectively would be applied to generate the final output.

The cross-entropy loss is applied on the classification output, while smooth-l1 loss is applied on the regression output.

![Fast R-CNN](/imgs/faster_rcnn.png)

For more details, refer to the presentation slide attached in this repository.

## About this repository

The main file of this repository is `DS-UA 301 Final Project.ipynb`. By running this Jupyter notebook cell by cell, one could instantiate the dataset, model, loss function and everything needed to train the model. Then, a ray tune task will be started to find the best set of hyperparameters for the model.

The datasets that will be used are not included in the repository so you need to download them. The datasets needed are:

- [COCO2014 Dataset](http://images.cocodataset.org/zips/train2014.zip)
- [COCO Synthetic Dataset](https://drive.google.com/open?id=1vIAFsftjmHg2J5lJgO92C1Xmyw539p_B) provided by `@pengzhou1108` on GitHub

After download these two datasets, unzip and put them under the root directory of the repository.

`test_filter.txt` and `train_filter.txt` are two annotation files indicating the filenames of images for the training/testing set. For each image file, the corresponding bounding box and the label indicating whether the object in the bounding box is original or tampered.

Before running the Jupyter notebook, change the value of `COCO_DIR`, `SYNTHETIC_DIR`, `MODEL_DIR`, `TRAIN_FILE`, and `TEST_FILE` in the second cell to make sure they point to the correct datasets/annotation files. Change these values in `demo.py` and `remove.py` as well. Specifically, to make sure raytune could run properly, it is recommended to use absolute path for `COCO_DIR`, `SYNTHETIC_DIR`, and `MODEL_DIR` in the Jupyter notebook.

If you cannot run `raytune`, you could instead run another cell which manually implements the grid search. If you use this cell, the performance metrics could be found in an output file called `performance.json`.

Also, if you don't want to run the Jupyter notebook cell by cell, you could instead run the equivalent `DS-UA 301 Final Project.py`. You could even use the command `nohup python3 -u "DS-UA 301 Final Project.py" &` to run it in the background. You could always monitor the training progress in `nohup.out`.

After finish running the Jupyter notebook or the equivalent python script, there will be a list of `.pth` model files under the `models` folder. Since hyperband scheduler is used, the models that are stopped early may perform badly.

Next, run `visualize.ipynb` to get the charts and graphs in the **Results** section. You could also run the equivalent `visualize.py` file instead. After running the Jupyter notebook or its equivalent python script, you could find the loss curves for all models under the `imgs` folder.

Then, run `demo.py` in the terminal. Follow the prompt and enter the desired parameters. This script first shows the ground truth bounding box of some sample image, and then visualizes the model prediction. You could get a sense of how this model performs based on the comparison between the ground truth and the model output. Note that red box means there is authentic object in the box, blue box means that there is tampered object in the box.

There are also multiple helper files: run `remove.py` if you want to delete unused images in COCO2014 dataset. The scripts in `extras` folder contains useful helper functions, as well as the definition of the R-CNN model and ResNet encoder that are used in the jupyter notebook. Check it out if you are interested in the detailed implementation.

## Results

We train our models with different sets of hyperparameters. Each model will be trained for 40 epochs, and the learning rate will decay with `gamma=0.1` every 15 epochs. The search space for hyperparameters is:

- Base ResNet: ResNet18, ResNet34, ResNet50
- Initial Learning RateL: 1.0, 0.1, 0.01
- Pretrained: True, False

For almost all models, during the training phase, both the loss from regression head and classification head decreases steadily, which indicates that our model successfully captures the relationship between image features and bounding boxes. Nevertheless, when observing the testing loss curves, we notice that there are several models with certain hyperparameter combinations experiencing overfitting. Therefore, earlystopping may need to be further enforced to produce better experiences.

Also the regression head and the classification head may have different optimal learning rate: instead of simply adding them up like `loss = clf_loss + reg_loss`, we may achieve better result by searching for the optimal ratio between these two losses. (i.e. `loss = w * clf_loss + reg_loss`, search for the best `w`)

### Graphs and Charts

Top 5 models in terms of classification loss:

| Base Model | Initial Learning Rate | Pretrained | Classification Loss (Test) |
| :--------: | :-------------------: | :--------: | :------------------------: |
|  ResNet50  |          1.0          |    True    |           0.3060           |
|  ResNet18  |          1.0          |   False    |           0.3072           |
|  ResNet34  |          1.0          |    True    |           0.3166           |
|  ResNet50  |         0.01          |   False    |           0.3170           |
|  ResNet18  |         0.01          |    True    |           0.3196           |

Top 5 models in terms of regression loss:

| Base Model | Initial Learning Rate | Pretrained | Regression Loss (Test) |
| :--------: | :-------------------: | :--------: | :--------------------: |
|  ResNet34  |          0.1          |    True    |       0.0001040        |
|  ResNet34  |         0.01          |   False    |       0.0001401        |
|  ResNet50  |          0.1          |   False    |       0.0001140        |
|  ResNet50  |         0.01          |    True    |       0.0001828        |
|  ResNet50  |          0.1          |    True    |       0.0002212        |

Loss Curves for model with `base=resnet50 init_lr=1.0 pretrained_base=True`

![Loss 1](/imgs/50_1.0_True_loss.png)

Loss Curves for model with `base=resnet34 init_lr=0.1 pretrained_base=True` (This one is probably experiencing overfitting, but still gives the best result)

![Loss 2](/imgs/34_0.1_True_loss.png)

### Prediction Visualization

The following is the ground truth bounding box of some sample images

![Ground Truth](/imgs/ground_truth.png)

Then this is the prediction of our model (`base=resnet34 init_lr=0.1 pretrained_base=True`)

![Prediction](/imgs/prediction.png)

We could observe that our model get the correct prediction, and the bounding boxes it produces basically covers the correct part of the image. However, the prediction is still quite messy, which is partly due to the fact that the IoU threshold in the non-maximum-suppression stage may need to be adjusted. If we assume that there will only be one tampered/authentic image in the image, then we could, instead of using non-maximum-suppression, simply choose the bounding box with maximum prediction probability for authentic image and tampered image to get one bounding box.

### Conclusion

Our model accomplishes two objectives, object detection and manipulation detection, at once. The implementation is quite straightforward as it uses the basic R-CNN. Nevertheless, we implement this model from scratch, which definitely greatly enhances our understanding to every R-CNN concepts.

We also come up with many innovative approaches when trying to make this R-CNN model fit our objectives: We come up with a customized dataloader with customized transformations to transform bounding box along with the image. Also, instead of reading the image files during loading, it read and transforms all image files at the initialization stage, which significantly reduces the data loading time, making each training/testing iteration 5 times faster. We substitute the original binary classification head with multi-class classification head to achieve object detection and manipulation detection at once. Besides, we downsampled the anchors with negative labels to address the imbalanced data problem.

Of course, our model still has many limitations: regardless of whether the object in the input is tampered or not, there is only one ground truth bounding box in each image while in reality an image could include multiple authentic and/or tampered objects, so we may need to look for better training dataset or make it by ourselves. Also,the predictions are not accurate enough visually, suggesting that we may need to use larger dataset, more complex module (e.g. ResNet101), and implement the SRM filter layer mentioned in [Peng Zhou et al's paper](https://arxiv.org/abs/1805.04953) to enhance the performance.

### References

- Peng Zhou's original paper [Learning Rich Features for Image Manipulation Detection](https://arxiv.org/abs/1805.04953)
- [Peng Zhou's Github repository](https://github.com/pengzhou1108/RGB-N), which provides the link to COCO synthetic dataset
- [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) on writing custom datasets and dataloaders
- [Object Detection Jupyter Notebook File](https://colab.research.google.com/drive/14T75p9pjmQTNHoiVQseHPQ8Yyzg5qxgN?usp=sharing) from NYU Computer Vision Course
