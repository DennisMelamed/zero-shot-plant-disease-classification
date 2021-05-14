The following files contain the main experiment code (built using pytorch-lightning):
```
triplet_loss_model.ipynb
classification_model.ipynb
healthy_image_branch_ablation.ipynb
fully_supervised_model.ipynb
```

A triplet loss model must be trained with the correct split first for most experiments, then a classification model (the proposed model for this project) can be trained with the triplet loss model as a pretrained component. The same is true of the healthy image branch ablation model, which is the same as proposed but removes the branch of the system which takes a healthy example of the plant as input. The fully supervised model is different from the healthy image branch ablation model only in the split it is trained on: all types of crop species. 

The datamodules used throughout support specifying which crops are to be trained and tested on, allowing easy variation of the splits. 

The PlantVillage dataset has been filtered of plants which either do not have healthy examples of the leaves or do not have  unhealthy examples of the leaves. This ends up leaving the following plants: Apple, Cherry, Corn, Grape, Peach, Bell Pepper, Potato, Strawberry, Tomato. The splits used in this project are specified in the dataset folder under `splits`. I do not upload the full dataset used to save space, but it should be recoverable from numerous locations online, including [here](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color).
