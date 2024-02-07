# Ad Creative Recognition With Computer Vision

### Table Of Content:
* [Visual Rhetoric In Ad Creative](#first-bullet)
* [Using CNNs](#CNN)
* [Using CLIP](#using-vitvision-tranformers-based-clip-model)
* [Explaining Visual Features of Ad]


## Results

| Model Name | Train Accuracy  | Validation Accuracy |
|----------|----------|----------|
|  ResNet152   | 79.28   | 79.96   |
| CLIP(Original Weights)   | --   | 90.2   |
| CLIP(Finetuned)   | 98   | 94.6   |


<a class="anchor" id="first-bullet"></a>

Ads are persuasive because they convey a certain message that appeals to the viewer.

These are just a few examples of how ads use different types of **visual rhetoric** to convey their message, namely:

- common-sense reasoning,
- symbolism, and
- recognition of non-photorealistic objects.

![Image with Complex Rhetoric Image](paper1.png "Few Images with Complex Rhetoric Image")
![Image with Complex Rhetoric Image](paper2.png "Few Images with Complex Rhetoric Image")
![Image with Complex Rhetoric Image](complex_img.png "Few Images with Complex Rhetoric Image")


<a class="anchor" id="CNN"></a>

## We need to develop a method to decode symbolism in ads to better understand the visual rhetoric in an Ad Creative

## CNNs to decode the Visual Rhetoric in an Ad Creative

### Checkout the below notebook for CNN based solution

[CNNs performance on Ad Creative Image Understanding](ImageUnderstanding.ipynb)

| Model Name | Train Accuracy  | Validation Accuracy: |
|----------|----------|----------|
|  ResNet152   | 79.28   | 79.96   |

## Visual Question Answering
![Image with Complex Rhetoric Image](paper3.png "Few Images with Complex Rhetoric Image")

## Using ViT(Vision Tranformers) based CLIP model
![image](https://github.com/anonymous-atom/AdGod/assets/74659873/47d48bdb-2275-4f57-acb2-62498a719cc2)
![image](https://github.com/anonymous-atom/AdGod/assets/74659873/c1770833-6e9a-4735-a460-9c42495b7d91)

### Using Pretrained
| Model Name | Train Accuracy  | Validation Accuracy |
|----------|----------|----------|
| CLIP(Original Weights)   | --   | 90.2   |

#### FPR(False Positive Rate) = 0.515

As you can see above even though the accuracy of CLIP is really high but also have high FPR, that's is because of CLIP being trained on very large general data.

## The FPR can be significatly reduced to ~(0.1 to 0.04) which can be achieved using full training of the model CLIP model and increasing the dataset size. 
### CLIP can't be easily trained on small system and takes large time and compute


### Finetuned CLIP
| Model Name | Train Accuracy  | Validation Accuracy |
|----------|----------|----------|
| CLIP(Finetuned)   | 98   | 94.6   |


#### FPR(False Positive Rate) = 0.41

# Explaining Visual Features of Ad Creative
To explain the visual rhetoric of the image we merge our model CLIP with an LLM using a projection matrix.
To do so I used already available model named LLaVA.

### Also to finetune LLaVA for Ad Creative Visual explanatin I created custom ** Visual Instruction tuning ** dataset and further finetune the model.

## Example explanation of an Ad Creative by LLaVA finetuned on custom data for ad specific purposes.
![Image with Complex Rhetoric Image](LLaVA_adv.png "LLaVA on Ad Creative")
