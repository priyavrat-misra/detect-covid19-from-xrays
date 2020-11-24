# Diagnosing COVID-19 and Viral Pneumonia from Chest X-Rays

![cover](https://raw.githubusercontent.com/priyavrat-misra/detect-covid19-from-xrays/master/images/test_results.png "sample test set results")

This project uses the [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) for training.
It consists of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. There are `219` COVID-19 positive images, `1341` normal images and `1345` viral pneumonia images.

<br>

## Steps:
> * [Exploring the dataset](https://github.com/priyavrat-misra/detect-covid19-from-xrays/blob/master/data_exploration.ipynb "data_exploration.ipynb")
> * [Dealing with the class imbalance problem](https://github.com/priyavrat-misra/detect-covid19-from-xrays/blob/master/custom_dataset.py "custom_dataset.py")
> * [Fine-tuning the ResNet-18 model](https://github.com/priyavrat-misra/detect-covid19-from-xrays/blob/master/finetune_resnet18.ipynb "finetune_resnet18.ipynb")
> * [Evaluating the model's results](https://github.com/priyavrat-misra/detect-covid19-from-xrays/blob/master/results.ipynb "results.ipynb")


## Results:
> || Train Accuracy | Test Accuracy |
> | :- | -: | -: |
> | fine-tuning (ResNet-18) | 99.32% | 96.67% |