# Ad Creative Recognition With Computer Vision

## Table of Contents

- [Visual Rhetoric In Ad Creative](#visual-rhetoric-in-ad-creative)
- [Methodology](#methodology)
    - [Using CNNs](#using-cnns)
    - [Using CLIP](#using-clip)
- [Results](#results)
- [Explaining Visual Features of Ad](#explaining-visual-features-of-ad-creative)
- [Citation](#citation)

## Results

| Model Name         | Train Accuracy | Validation Accuracy |
|--------------------|----------------|---------------------|
| 🖼️ ResNet152          | 79.28          | 79.96               |
| 🧠 CLIP(Original Weights) | --             | 90.2                |
| ✨ CLIP(Finetuned)    | 98             | 94.6                |



## Visual Rhetoric In Ad Creative

Advertisements are not merely commercial messages but sophisticated visual compositions crafted to convey compelling narratives. They leverage various visual techniques including:

- Common-sense reasoning,
- Symbolism, and
- Recognition of non-photorealistic objects.

![Example of Visual Rhetoric](paper1.png)
![Example of Visual Rhetoric](paper2.png)
![Example of Visual Rhetoric](complex_img.png)

## Methodology

### Using CNNs

To unravel the secrets hidden within advertisements, we employ Convolutional Neural Networks (CNNs) to decode their visual rhetoric.

#### CNNs to Decode the Visual Rhetoric in an Ad Creative

For detailed insights, refer to our [notebook](ImageUnderstanding.ipynb) showcasing CNN-based solutions.

| Model Name | Train Accuracy | Validation Accuracy |
|------------|----------------|---------------------|
| ResNet152  | 79.28          | 79.96               |

### Using CLIP

We also leverage the Vision Transformers based CLIP model to further enhance our understanding of ad creatives.

#### Using ViT(Vision Transformers) based CLIP model

Explore our [CLIP Finetuning](Adv_CLIP/Adv_CLIP_Custom.ipynb) for detailed analysis.

| Model Name         | Train Accuracy | Validation Accuracy |
|--------------------|----------------|---------------------|
| CLIP(Original Weights) | --             | 90.2                |
| CLIP(Finetuned)    | 98             | 94.6                |



## Explaining Visual Features of Ad Creative

To provide a deeper understanding of ad creatives, we merge our CLIP model with a Language-Model using projection matrix. To do the same easily we can use LLaVA model which works the same way.

## Example Explanation of an Ad Creative by LLaVA

![Example Explanation by LLaVA](LLaVA_adv.png "LLaVA on Ad Creative")

## Citation

If you find our work helpful, please consider citing:

