<h2 style="border-bottom: 1px solid lightgray;">Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability</h2>

<!-- Badges and Links Section -->
<div style="display: flex; align-items: center; justify-content: center;">

</div>

[![arxiv](https://img.shields.io/badge/arXiv-2407.14949-red)](https://arxiv.org/abs/2505.03641)
[![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&up_color=brightgreen&up_message=online&url=https://eaterminator.github.io/BAM/)](https://eaterminator.github.io/BAM/)


</div>
This is the github code and data repository for the <em>ICML 2025</em> paper <em>Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability</em>. For convenience, model checkpoints and the variMNIST dataset can be found in <strong>Releases</strong>.

### Get started
You can set up a conda environment with all dependencies by running the following commands:

```
conda env create -f environment.yml
conda activate HumanPercVar
```

### Dataset
- Images: `varMNIST/dataset_PNG`
- Responses: `varMNIST/df_DigitRecog.csv`


### Demo 1: Generating images to elicit human perceptual variability (Figure 2 in manuscript)
- `Generating_images_human_perceptual_variability.ipynb` 

### Demo 2: Generating images to manipulate Individual choice (Figure 5 in manuscript)
- `customized_image_generation.ipynb`
