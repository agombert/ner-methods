# Named Entity Recognition: easy implementation on spaCy, BERT fine tuning & BERT Distillation

In this repo, I constructed a quick overview/tutorial to use Named Entity Recognition (NER) algorithms on an e-commerce dataset. We go over a few methods motivated by an article on [BERT distillation with spaCy](https://towardsdatascience.com/distilling-bert-models-with-spacy-277c7edc426c) and we are going through the three steps of the process:

1. Create a [NER](https://github.com/explosion/spaCy/blob/master/examples/training/train_ner.py) algorithm with spaCy
2. [Fine tune BERT](https://github.com/google-research/bert#fine-tuning-with-bert) to create a more efficient algorithm
3. [Distill](https://arxiv.org/pdf/1503.02531.pdf) the BERT algorithm to get an efficient model but lighter than the BERT

The distillation phase is a bit different from the [texcat algorithm](https://github.com/agombert/textcat), especially in the data augmentation step. 

The dataset I used for this experiment was created from scratch. I cannot provide the datasets (I give only a tiny sample to highlights the results) I used as I scrapped some data on [eBay](www.ebay.com). I scrapped it with [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#) and managed to create a dataset around car adds on eBay. From that I decided to try to focus on the car make, the model / submodel of the car, and some characteristics of the car when available. 

I trained my model on this dataset nevertheless, you can use this code to perform you own NER with any dataset in `data/` where the `text_train.npy` and `text_test.npy` are arrays of text and `labels_train.py` and `labels_test.npy` are arrays of label (with as many label as you want). To perform this go directly to the [*spaCy textcat implementation*](https://github.com/agombert/ner-methods/blob/master/README.md#spacy-ner-implementation).

## Main results

Our [train dataset](https://github.com/agombert/textcat/data/Text_train.npy) has a length 4000. It's balanced between the four classes. The [test dataset](https://github.com/agombert/textcat/data/Text_test.npy) has a length 4000. Thus we will be able to see the difference with the transfert learning method (BERT fine tuning).

For each algorithm I trained for 15 epochs with batch size 32 and a dropout at 0.5. 

I computed the macro recall / macro precision for each model.

|      Model     | Recall | Precision | Model Size|
|:--------------:|:------:|:---------:|:---------:|
| *spaCy Imp*    | % |   %  |   b   |
|*BERT FineTuned*| % |   %  |   Gb   |
|*Distilled BERT*| % |   %  |   B   | 

I also computed the recall/precision for each label. 

|      Model     |   Car Make  |  Car Model  |
|:--------------:|:-----------:|:-----------:|
|*spaCy Imp*     |-%/-%|-%/-%|
|*BERT FineTuned*|-%/-%|-%/-%|
|*Distilled BERT*|-%/-%|-%/-%|

## Data

Here I provide an example of the add we want to extract the info from. I use [doccano](https://github.com/doccano/doccano) to highlights the example. It's an open source annotator that you can use to create/revise annotations on text corpus for text classification or Name Entity Recognition.

![alt text](https://raw.githubusercontent.com/agombert/ner-methods/master/img/example.png)

## spaCy NER implementation

<!--When you have your dataset with the texts and labels you can use `run_textcat.py` to make the classification and save your model.-->

<!--Save your train test in `data/` with a name followed by the mention `_train` with a `.npy` format (with numpy.save) and your test set with the mention `_test` with the same extention `.npy`.-->


<!--```bash
python3 run_textcat.py --is predict False
                       --name_model name_of_model
                       --cats cat1 cat2 cat3 ...
                       --X name of features data (without _train)
                       --y name of labels data (without _train)
                       --bs_m           The minimum batch size for training
                       --bs_M           The maximum batch size for training
                       --step           The step to go from min batchsize to max batchsize
                       --epoch          Number of epoch for the training
                       --drop           Drop out to apply the model
```
-->
<!--Your model will be save in `models/` with one file `name_of_model_nlp` which is the spaCy model associated, `name_of_model_textcat` which is the spacy Textcomponent component and `name_of_models_scores` which are the scores on the evaluation dataset (automatically created) during training.-->

<!--Then you can use the same function to get metrics on the test dataset:-->

<!--```bash
python3 run_textcat.py --is predict True
                       --name_model name_of_model
                       --X name of features data (without _test)
                       --y name of labels data (without _test)
```-->

<!--And you'll get the evaluations in logs. -->

## BERT NER implementation (with google Colab)

<!--For this part I used google colab, as it's really cool to get free GPU access and perform BERT fine tuning on small datasets. I used the code from this [colab](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb#scrollTo=dsBo6RCtQmwx) which uses BERT to review movies. I aranged a bit the code to adapt it to my problem. -->

<!--The code is [here](https://colab.research.google.com/drive/1MShG1gDV5TfvVEYDTBgr5LCzVNVmTf03#scrollTo=NwW9OH0CBJx9&uniqifier=1), and adapting to your problem you can use it for any text classification by fine tuning BERT. -->

<!--You just follow each cell, but previously you'll need some storage ([GCS](https://console.cloud.google.com/storage) for instance) to store the results and your datasets. -->

## BERT distillation implementation

### [Data Augmentation](https://arxiv.org/abs/1503.02531)

<!--For the augmented data we follow two out of three methods from [Distilling Task-Specific Knowledge from BERT intoSimple Neural Networks](https://arxiv.org/pdf/1903.12136.pdf). We mask some tokens and also change the order of randomly chosen n-grams in the sentence. At the end we go from 4k to 200k data for our augmented set.-->

<!--You can use a [google colab notebook](https://colab.research.google.com/drive/128apJ8WAMVyXxocCY9CRapsr1Qs8Mu0w#scrollTo=zXkPH_rUatS6) once again or you can do it in local and stock your augmented data in the bucket you previously created. -->

<!--You can also perform this code locally to get 50 times more data. Then we will use the previously trained model to compute the prevision for the augmented data and also the probabilities associated at each text. -->

### Training on augmented data

<!--When you have your augmented data and also the BERT predictions on those data (labels and probabilities) we are going to train a new model from spaCy on those data.-->

<!--First we use only the labels predicted by the BERT fineted model. So we use exactly the same method as in the section on spaCy textcat. -->

<!--We outperformed first model by 13% on hatred speech datasets and 4.5% on the wine datasets ! We still are below the BERT from a significagive margin, but we should add the loig probs to the loss function to try to improve the model. And the size of the model is a bit higher from the first textcat spaCy classifier we had. -->

<!--Thus by using the encoder on each sentence, without finetuning or new training to focus on our data we can try to classify the text. -->

<!--It looks really good as we outperform all the models except the BERT and we are not so far as we are only 4.5% below for the hatred speech and X% for the wine classification.-->
