# Ad Creative Recognition With Computer Vision

### Table Of Content:
* [Visual Rhetoric In Ad Creative](#first-bullet)
* [Using CNNs](#CNN)


## Results

| Model Name | Train Accuracy  | Validation Accuracy: |
|----------|----------|----------|
|  ResNet152   | 79.28   | 79.96   |
| CLIP(Original Weights)   | --   | Data 6   |
| CLIP(Finetuned)   | Data 5   | Data 6   |


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