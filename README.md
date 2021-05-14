# Zero-shot plant health determination


## Theoretical idea
The goal of this project was to identify a plant leaf from a type of plant (crop species) unseen during training as healthy or unhealthy (not to classify the specific disease). Since plant leaves often look very different, a siamese model was proposed. The model consists of an embedding network (Resnet-18), and a classification model. An exemplar of a healthy leaf of the plant species being classified was always fed into the embedding network to generate a "healthy embedding". The plant leaf image we desired a classification for (same species as the healthy exemplar) was also fed to the embedder to generate an "unknown embedding". These embeddings were concatenated and fed to the classification model (MLP) to generate the final classification. Since the network was tested on species it had not seen during training, the healthy embedding was key for the classification model to achieve maximum performance since it gave a frame of reference for the model to compare the unknown image against. 

In order to improve performance, a triplet loss embedding was used as a pre-training step for the embedding network. The embedding network was pre-trained on only crop species it would see during the later classification training to maintain the testing split as completely unseen.

This project was done as part of 16-889 (Robotics & AI for Agriculture) at CMU in the Spring of 2021.

## Code and data explanation
The following files contain the main experiment code (built using pytorch-lightning):
```
triplet_loss_model.ipynb
classification_model.ipynb
healthy_image_branch_ablation.ipynb
fully_supervised_model.ipynb
```

Most of the code should be relatively straightforward to use in other places (and I hope it is useful, if a bit messy currently). One thing I was never able to resolve was the threshold to use in actually classifying plants: ideally this would consistently be 0.5, but often the model would learn a space where a value closer to 0.1 or even 1e-3 would achieve excellent performance. I'm not sure why this is happening. The threshold was adjusted on the validation split, so in general the model was built using only data available for training and never touched the test split until the evaluation phase, but for more real world usage this issue needs to be resolved. Perhaps it is related to dataset imbalance (there are far more unhealthy images than healthy images). I attempted to correct this by weighing samples in the loss function, but maybe these weights need to be dramatically increased for healthy samples.

A triplet loss model must be trained with the correct split first for most experiments, then a classification model (the proposed model for this project) can be trained with the triplet loss model as a pretrained component. The same is true of the healthy image branch ablation model, which is the same as proposed but removes the branch of the system which takes a healthy example of the plant as input. The fully supervised model is different from the healthy image branch ablation model only in the split it is trained on: all types of crop species. 

The datamodules used throughout support specifying which crops are to be trained and tested on, allowing easy variation of the splits. 

The PlantVillage dataset has been filtered of plants which either do not have healthy examples of the leaves or do not have  unhealthy examples of the leaves. This ends up leaving the following plants: Apple, Cherry, Corn, Grape, Peach, Bell Pepper, Potato, Strawberry, Tomato. The splits used in this project are specified in the dataset folder under `splits`. I do not upload the full dataset used to save space, but it should be recoverable from numerous locations online, including [here](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color).
