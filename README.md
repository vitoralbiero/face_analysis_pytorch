# Face Analysis PyTorch

- This repository contains code to train race, gender, and age models separately.
- The race and gender models use weighted cross-entropy loss.
- The age model use [ordinal regression](https://ieeexplore.ieee.org/document/7780901) loss with a small modification to sigmoid activation instead of softmax.
- Along with the attribute predictors, it also contains code to train face recognition models (ArcFace and CosFace).

## Trained models
[Model-Zoo](https://github.com/vitoralbiero/face_analysis_pytorch/wiki/Model-Zoo)

## Training Dasets
Training/testing datasets should be a list of image paths and class number.
Examples are inside the datasets folder, the attribute training/testing files consists of: [image_path race_class gender_class age_class] for attributes, and [image_path person_class] for recognition.

If you want to retrain on your own dataset, aligned the images first and create a similar list.

## Training
### Attributes
To train the attribute predictors, you will need to pass the path to images main folder, along with the image list, or an image list that contains the absolute path to the images.

```
python3 train.py --train_source /path_to_train_dataset_main_folder/ --train_list ./datasets/age_train.txt --val_source ../path_to_val_dataset_main_folder/ --val_list ./datasets/age_val.tx -a age --prefix age --multi_gpu
```
An alternate faster way to train is to convert the datasets to LMDB format. For this end, use the [imagelist2lmdb.py](https://github.com/vitoralbiero/face_analysis_pytorch/blob/master/utils/imagelist2lmdb.py) or [folder2lmdb.py](https://github.com/vitoralbiero/face_analysis_pytorch/blob/master/utils/folder2lmdb.py) to convert a dataset to LMDB. Then, train using the command below.
```
python3 train.py --train_source ./train_dataset.lmdb --val_source ./val_dataset.lmdb/ --val_list ./datasets/age_val.tx -a age --prefix age --multi_gpu
```
### Recognition
To train for recognition, the [LFW, CFP-FP and AgeDB-30](https://github.com/deepinsight/insightface) should be converted using [utils/prepare_test_sets.py](https://github.com/vitoralbiero/face_analysis_pytorch/blob/master/utils/prepare_test_sets.py).

```
python3 train.py --train_source ./ms1m_v2.lmdb --val_source ./path_to_val_datasets/ --val_list ['lfw', 'cpf_fp', 'agedb_30'] -a recognition --prefix arcface --multi_gpu --head arcface
```

If you train using [ArcFace](https://arxiv.org/abs/1801.07698) or [CosFace](https://arxiv.org/abs/1801.09414), please cite the apppropriate papers.

## Attribute Predict
To predict, you will need to pass the trained models (race, gender and/or age) to the predict file, along with path to the images and image list. The predictor assumes that images are already aligned, since I am still trying to add MTCNN to the dataloader as it crashes, since it is done in parallel.

```
python3 predict.py -s /path_to_images_main_folder/ -i ../ext_vol2/training_datasets/ms1m_v2/ms1m_v2_images.txt -d /path_to_save_predictions_file/ -rm ./path_to_race_model -gm ./path_to_gender_model -am ./path_to_age_model
```
- Gender: Males = 1, Females = 0
- Race: Caucasian = 0, African-American = 1, Asian = 2, Indian = 3, Other (e.g., hispanic, latino, middle eastern) = 4
- Age: 0 to 100

## Recognition
### Feature Extraction
Run feature extractor for recognition models. Path to the main folder is optional if the image list does not contain the absolute path.
```
python3 feature_extraction.py -s ./path_to_main_folder -i image_list.txt -d ./path_to_save_features/ -m ./model_to_be_loaded
```

### Feature Matching
Feature match between a probe and a gallery, if matching probe to probe, leave gallery file empty.
```
python3 feature_match.py -p ./probe_file.txt -g ./optional_gallery_file.txt -o ./output_path -d dataset_name -gr prefix_of_outputs
```

## Credit
Some implementations in this repository were heavily inspired by:
* [TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
