# Face Analysis PyTorch

- This repository contains code to train race, gender, and age models separately.
- The race and gender models use weighted cross-entropy loss.
- The age model use [ordinal regression](https://ieeexplore.ieee.org/document/7780901) loss with a small modification to sigmoid activation instead of softmax.
- Along with the attribute predictors, it also contains code to train face recognition models (ArcFace and CosFace).

## Trained models
[Model-Zoo](https://github.com/vitoralbiero/face_analysis_pytorch/wiki/Model-Zoo).

## Training Dasets
Training/testing datasets should be a list of image paths and class number.
Examples are inside the datasets folder, the attribute training/testing files consists of: [image_path race_class gender_class age_class] for attributes, and [image_path person_class] for recognition.

If you want to retrain on your own dataset, aligned the images first and create a similar list.

## Training
To train the attribute predictors, you will need to pass the path to images main folder, along with the image list.

```
python3 main.py --train_source /path_to_train_dataset_main_folder/ --train_list ./datasets/age_train.txt --val_source ../path_to_val_dataset_main_folder/ --val_list ./datasets/age_val.tx -a age --prefix age --multi_gpu
```

To train for recognition, the [LFW, CFP-FP and AgeDB-30](https://github.com/deepinsight/insightface) should be converted using [utils/prepare_test_sets.py](https://github.com/vitoralbiero/face_analysis_pytorch/blob/master/utils/prepare_test_sets.py).

If you train using [ArcFace](https://arxiv.org/abs/1801.07698) or [CosFace](https://arxiv.org/abs/1801.09414), please cite the apppropriate papers.

## Attribute Predict
To predict, you will need to pass the trained models (race, gender and/or age) to the predict file, along with path to the images and image list. The predictor assumes that images are already aligned, since I am still trying to add MTCNN to the dataloader as it crashes, since it is done in parallel.

```
python3 predict.py -s /path_to_images_main_folder/ -i ../ext_vol2/training_datasets/ms1m_v2/ms1m_v2_images.txt -d /path_to_save_predictions_file/ -rm ./workspace/race_run_02/final/model_2020-04-03-02-41_accuracy\:0.9705_step\:27051.pth -gm ./workspace/gender_run_01/final/model_2020-04-03-20-33_accuracy\:0.9723_step\:34505.pth -am ./workspace/age_run_01/final/model_2020-04-03-18-31_accuracy\:4.6575_step\:35532.pth
```

## Recognition
### Feature Extraction
TO-DO

### Feature Matching
TO-DO

## Credit
Some implementations in this repository were heavily inspired by:
* [TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
