# Solar segmentation

Finding solar panels using USGS satellite imagery.

## 1. Introduction

This repository leverages the [distributed solar photovoltaic array location and extent dataset for remote sensing object identification](https://www.nature.com/articles/sdata2016106)
to train a segmentation model which identifies the locations of solar panels from satellite imagery.

Training happens in two steps:

1. Using an Imagenet-pretrained ResNet34 model, a classifier is trained to identify whether or not solar panels are present
in a `[224, 224]` image.
2. The classifier base is then used as the downsampling base for a U-Net, which segments the images to isolate solar panels. 

## 2. Results

The classifier was trained on 80% of the data, with 10% being used for validation and 10% being used as a holdout test set.
On this test set, with a threshold of `0.5` differentiating positive and negative examples, the model achieved a **precision
of 98.8%**, and a **recall of 97.7%**. This is competitive with [DeepSolar](http://web.stanford.edu/group/deepsolar/home) 
(precision of 93.1% - 93.7%, and recall of 88.5% - 90.5%) despite being trained on a smaller, publically available dataset.

<img src="diagrams/test_auc_roc.png" alt="AUC ROC results" height="400px"/>

The segmentation model achieved a [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
of 0.89:

<img src="diagrams/segmentation_predictions.png" alt="segmentation results" height="500px"/>

## 3. Pipeline

The main entrypoint into the pipeline is [`run.py`](solarnet/run.py). Note that each component reads files from the 
previous step, and saves all files that later steps will need, into the [`data`](data) folder.

In order to run this pipeline, follow the instructions in the [data readme](data/README.md) to download the data.

[Python Fire](https://github.com/google/python-fire) is used to generate command line interfaces.

#### 3.1. Make masks

This step goes through all the polygons defined in `metadata/polygonVertices_PixelCoordinates.csv`, and constructs masks
for each image, where `0` indicates background and `1` indicates the presence of a solar panel.

```bash
python run.py make_masks
```
This step takes quite a bit of time to run. Using an Macbook Pro M1 took the following times for each city (min:secs):

- Fresno: 4:30
- Modesto: 0:10
- Oxnard: 0:31
- Stockton: 0:48

#### 3.2. Split images

This step breaks the `[5000, 5000]` images into `[224, 224]` images. To do this, [`polygonDataExceptVertices.csv`](data/metadata/polygonDataExceptVertices.csv)
is used to identify the centres of solar panels. This ensures the model will see whole solar panels during the segmentation step.

Negative examples are taken by randomly sampling the image, and ensuring no solar panels are present in the randomly sampled example.

```bash
python run.py split_images
```

This yields the following images (examples with panels above, and without below):

<img src="diagrams/positive_splits.png" alt="examples with panels" height="200px"/>

<img src="diagrams/negative_splits.png" alt="examples without panels" height="200px"/>

#### 3.3. Train classifier

This step trains and saves the classifier. In addition, the test set results are stored for future analysis.

```bash
python run.py train_classifier

# On MacOS, training on Metal Performance Shaders (MPS) is possible
python run.py train_classifier --device mps
```

#### 3.4. Train segmentation model

This step trains and saved the segmentation model. In addition, the test set results are stored for future analysis.
By default, this step expects the classifier to have been run, and will try to use it as a pretrained base.

```bash
python run.py train_segmenter

# On MacOS, training on Metal Performance Shaders (MPS) is possible
python run.py train_segmenter --device mps
```

Both models can be trained consecutively, with the classifier automatically being used as the base of the segmentation
model, by running
```bash
python run.py train_both
```

#### 3.5. Use segmentation model

Place your image in a new folder and create a folder in-between (e.g. `unclassified`):

```
data/example_images
└── unclassified
    ├── image1.png
    └── image2.jpg
```

Note that the images will be resized to the network's input shape (e.g. `224` pixels) and
cropped in case the images are not square-shaped (i.e. W != H).

Provide both input and output folders and run:

```bash
python run.py segment_new_data data/example_images data/example_outputs
```

The result should look like this:

```
data/example_outputs
├── image1.bmp
└── image2.bmp
```

## 4. Setup

Pyenv and Poetry are used to setup the repository:

```
# we assume that pyenv is set-up and `which python` points to the right python version

# install the dependencies
poetry install
```

This pipeline can be tested by running `pytest`.

[Docker](https://www.docker.com/) can also be used to run this code. To do this, first build the docker image:

```bash
docker build -t solar .
```

Then, use it to run a container, mounting the data folder to the container:

```bash
docker run -it \
--mount type=bind,source=<PATH_TO_DATA>,target=/solar/data \
solar /bin/bash
```
