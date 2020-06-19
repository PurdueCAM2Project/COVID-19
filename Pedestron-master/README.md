[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pedestrian-detection-the-elephant-in-the-room/pedestrian-detection-on-citypersons)](https://paperswithcode.com/sota/pedestrian-detection-on-citypersons?p=pedestrian-detection-the-elephant-in-the-room)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pedestrian-detection-the-elephant-in-the-room/pedestrian-detection-on-caltech)](https://paperswithcode.com/sota/pedestrian-detection-on-caltech?p=pedestrian-detection-the-elephant-in-the-room)


# Pedestron
<img title="Amsterdam" src="gifs/1.gif" width="400" /> <img title="Amsterdam" src="gifs/2.gif" width="400"/>
<img title="Amsterdam" src="gifs/3.gif" width="400"/> <img title="Amsterdam" src="gifs/4.gif" width="400"/>


[Pedestron](https://128.84.21.199/pdf/2003.08799.pdf) is a [MMetection](https://github.com/open-mmlab/mmdetection) based repository that focuses on the advancement of research on pedestrian detection. We provide a list of detectors, both general purpose and pedestrian specific to train and test. Moreover, we provide pre-trained models and benchmarking of several detectors on different pedestrian detection datasets. Additionally, we provide processed annotations and scripts to process the annotation of different pedestrian detection benchmarks. If you use Pedestron, please cite us (see at the end) and other respective sources.

# :fire: **Updates** :fire:
* [NEW] **Ported testing of Mask-Guided Attention Network for Occluded Pedestrian Detection (MGAN). (ICCV'19), into Pedestron along with pretrained model**
* [NEW] **Pedestron now supports general person and pedestrian detection in crowded scenarios. Pre-trained model along with evaluation script on CrowdHuman benchmark is added**
* [NEW] **Pre-trained model for WIDER pedestrian datasets has been added to the Pedestron, for general person detection**
* :fire: **Configuration along with a pre-trained model for RetinaNet with Gudied Anchoring added. Its fast and accurate**
* :fire: **Configuration along with a pre-trained model for Faster R-CNN with HRNet for EuroCity Persons has been adeed**
* :fire: **[Google Colab](https://github.com/hasanirtiza/Pedestron/blob/master/colab/PedestronColab.ipynb) step-by-step instruction on how to setup Pedestron and run demo by [Gokulan Vikash](https://github.com/gokulanv)** 




### YouTube demo
* [Caltech](https://www.youtube.com/watch?v=cemN7JbgxWE&feature=youtu.be) and [EuroCity Person](https://www.youtube.com/watch?v=ZFObRPa_yMM). Pre-Trained model available.

### Installation
We refer to the installation and list of dependencies to [installation](https://github.com/hasanirtiza/Pedestron/blob/master/INSTALL.md) file.
Clone this repo and follow [installation](https://github.com/hasanirtiza/Pedestron/blob/master/INSTALL.md). Alternatively, [Google Colab](https://github.com/hasanirtiza/Pedestron/blob/master/colab/PedestronColab.ipynb) step by step instruction can be followed for installation 

### List of detectors
Currently we provide configurations for the following detectors, with different backbones
* Cascade Mask-R-CNN
* Faster R-CNN
* RetinaNet
* RetinaNet with Guided Anchoring
* Hybrid Task Cascade (HTC)  
* MGAN  


### Following datasets are currently supported 
* [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)
* [CityPersons](https://bitbucket.org/shanshanzhang/citypersons/src/default/)
* [EuroCity Persons](https://eurocity-dataset.tudelft.nl/)
* [CrowdHuman](https://www.crowdhuman.org/)
* [WiderPedestrian Challenge](https://competitions.codalab.org/competitions/20132)


### Datasets Preparation
* We refer to [Datasets preparation file](Datasets-PreProcessing.md) for detailed instructions


# Benchmarking 

### Benchmarking of pre-trained models on pedestrian detection datasets (autonomous driving)
|    Detector                | Dataset   | Backbone| Reasonable  | Heavy    | 
|--------------------|:--------:|:--------:|:--------:|:--------:|
| [Cascade Mask R-CNN](https://drive.google.com/open?id=1B487ljaU9FxTSFaLoirOSqadZ-39QEH8) | CityPersons        | HRNet | 7.5        |   28.0      | 
| [Cascade Mask R-CNN](https://drive.google.com/open?id=1ysWlzN92EInIjDD_QwIq3iQEZJDo8wtT) | CityPersons        | MobileNet | 10.2        |   37.3      | 
| [Faster R-CNN](https://drive.google.com/open?id=1aanqAEFBc_KGU8oCFCji-wqmLmqTd749) | CityPersons        | HRNet | 10.2        |   36.2      |
| [RetinaNet](https://drive.google.com/open?id=1MGxZitqLzQtd2EF8cVGYNzSKt73s9RYY) | CityPersons        | ResNeXt | 14.6        |   39.5      |
| [RetinaNet with Guided Anchoring](https://drive.google.com/open?id=1eBxkjhl12ELlv7VFyH9jS2O4I3SO_t8V) | CityPersons        | ResNeXt | 11.7        |   41.5      |
| [Hybrid Task Cascade (HTC)](https://drive.google.com/open?id=1qPEJ1r48Ggl2TdE1ohcDoprZomC2j3SX) | CityPersons        | ResNeXt | 9.5       |   35.8      | 
| [MGAN](https://drive.google.com/open?id=1c191nSSRUGd0LfkjYXKcyJEZCtjUeWr-) | CityPersons        | VGG | 11.2       |   52.5      | 
| [Cascade Mask R-CNN](https://drive.google.com/open?id=1HkoUPlONSF04AKsPkde4gmDLMBf_vrnv) | Caltech        | HRNet |   1.7      |    25.7     | 
| [Cascade Mask R-CNN](https://drive.google.com/open?id=1GzB3O1JxPN5EusJSyl7rl9h0sQAVKf15) | EuroCity Persons | HRNet |    4.4     |  21.3       |
| [Faster R-CNN](https://drive.google.com/open?id=19xBNw_wJPGNFIYsylcxPGIuoDygxHa2D) | EuroCity Persons        | HRNet | 6.1        |   27.0      |
 
 
### Benchmarking of pre-trained models on general human/person detection datasets
 
 |    Detector                | Dataset   | Backbone| AP  |
 |--------------------|:--------:|:--------:|:--------:|
 |    [Cascade Mask R-CNN](https://drive.google.com/open?id=1MqI1-Bbn0vl5Ft1RnhD70YWl7JHRyVMx) | CrowdHuman   | HRNet| 84.1  |

### Pre-Trained models
Cascade Mask R-CNN
1) [CityPersons](https://drive.google.com/open?id=1B487ljaU9FxTSFaLoirOSqadZ-39QEH8)
2) [Caltech](https://drive.google.com/open?id=1HkoUPlONSF04AKsPkde4gmDLMBf_vrnv)
3) [EuroCity Persons](https://drive.google.com/open?id=1GzB3O1JxPN5EusJSyl7rl9h0sQAVKf15)
4) [CrowdHuman 1](https://drive.google.com/open?id=1rXopG04Dv-HKge3ZyqNYAHdCi9gmLuNe)
5) [CrowdHuman 2](https://drive.google.com/open?id=1MqI1-Bbn0vl5Ft1RnhD70YWl7JHRyVMx) (higher AP)
6) [WIDER Pedestrian](https://drive.google.com/open?id=1Z1LTASbr-VTIfbk4Fu-d2VCBpKPqADH8)

Faster R-CNN
1) [CityPersons](https://drive.google.com/open?id=1aanqAEFBc_KGU8oCFCji-wqmLmqTd749)
2) [EuroCity Persons](https://drive.google.com/open?id=19xBNw_wJPGNFIYsylcxPGIuoDygxHa2D)

RetinaNet
1) [CityPersons](https://drive.google.com/open?id=1MGxZitqLzQtd2EF8cVGYNzSKt73s9RYY)

RetinaNet with Guided Anchoring
1) [CityPerson](https://drive.google.com/open?id=1eBxkjhl12ELlv7VFyH9jS2O4I3SO_t8V)


Hybrid Task Cascade (HTC)
1) [CityPersons](https://drive.google.com/open?id=1qPEJ1r48Ggl2TdE1ohcDoprZomC2j3SX)

MGAN
1) [CityPersons](https://drive.google.com/open?id=1c191nSSRUGd0LfkjYXKcyJEZCtjUeWr-)

# Getting Started

### Running a demo using pre-trained model on few images
1) Pre-trained model can be evaluated on sample images in the following way

```shell 
python tools/demo.py config checkpoint input_dir output_dir
```
Download one of our provided pre-trained model and place it in  models_pretrained folder. Demo can be run using the following command

```shell 
python tools/demo.py configs/elephant/cityperson/cascade_hrnet.py ./models_pretrained/epoch_5.pth.stu demo/ result_demo/ 
```

See [Google Colab demo](https://github.com/hasanirtiza/Pedestron/blob/master/colab/PedestronColab.ipynb). 



### Training

- [x] single GPU training
- [x] multiple GPU training

Train with single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

Train with multiple GPUs
```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

For instance training on CityPersons using single GPU 

```shell
python tools/train.py configs/elephant/cityperson/cascade_hrnet.py
```

Training on CityPersons using multiple(7 in this case) GPUs 
```shell
./tools/dist_train.sh configs/elephant/cityperson/cascade_hrnet.py 7  
```

### Testing

- [x] single GPU testing
- [x] multiple GPU testing

Test can be run using the following command.
 
```shell 
python ./tools/TEST_SCRIPT_TO_RUN.py PATH_TO_CONFIG_FILE ./models_pretrained/epoch_ start end\
 --out Output_filename --mean_teacher 
``` 

For example for CityPersons inference can be done the following way

1) Download the pretrained [CityPersons](https://drive.google.com/open?id=1B487ljaU9FxTSFaLoirOSqadZ-39QEH8) model and place it in the folder "models_pretrained/".
2) Run the following command:

```shell 
python ./tools/test_city_person.py configs/elephant/cityperson/cascade_hrnet.py ./models_pretrained/epoch_ 5 6\
 --out result_citypersons.json --mean_teacher 
```
Alternatively, for EuroCity Persons

```shell
python ./tools/test_euroCity.py configs/elephant/eurocity/cascade_hrnet.py ./models_pretrained/epoch_ 147 148 --mean_teacher
```

or without mean_teacher flag for MGAN

```shell 
python ./tools/test_city_person.py configs/elephant/cityperson/mgan_vgg.py ./models_pretrained/epoch_ 1 2\
 --out result_citypersons.json  
```

Testing with multiple GPUs on CrowdHuman

```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

```shell
./tools/dist_test.sh configs/elephant/crowdhuman/cascade_hrnet.py ./models_pretrained/epoch_19.pth.stu 8 --out CrowdHuman12.pkl --eval bbox
```

* Similarly change respective paths for EuroCity Persons
* For Caltech refer to [Datasets preparation file](Datasets-PreProcessing.md)

### Please cite the following work
[ArXiv version](https://arxiv.org/pdf/2003.08799.pdf)
```
@article{hasan2020pedestrian,
  title={Pedestrian Detection: The Elephant In The Room},
  author={Hasan, Irtiza and Liao, Shengcai and Li, Jinpeng and Akram, Saad Ullah and Shao, Ling},
  journal={arXiv preprint arXiv:2003.08799},
  year={2020}
}
```
