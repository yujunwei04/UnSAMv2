<p align="center">

  <h1 align="center">UnSAMv2: Self-Supervised Learning Enables Segment Anything at Any Granularity</h1>
  <p align="center">
    <a href="https://yujunwei04.github.io/"><strong>Junwei Yu</strong></a>,
    <a href="https://people.eecs.berkeley.edu/~trevor/"><strong>Trevor Darrell</strong></a>,
    <a href="https://people.eecs.berkeley.edu/~xdwang/"><strong>XuDong Wang<sup>*</sup></strong></a>
  </p>
  <p align="center">
    <strong>UC Berkeley</strong><br />
    <small style="font-size:0.75em;"><em>* Corresponding author</em></small>
  </p>
</p>

<h3 align="center">
  <a href="https://yujunwei04.github.io/UnSAMv2-Project-Page/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/111"><strong>ArXiv (coming soon)</strong></a>
  |
  <a href="https://x.com/111"><strong>Demo 😊 (TODO)</strong></a>
</h3>

<div align="center">
  <img src="./assets/teaser.png" alt="" style="width: 100%; margin: 10px 0;">
  <!-- <img src="./assets/edit_demo.jpg" alt="" style="width: 100%; margin: 10px 0;"> -->
</div>

## News 🎉
- 11/02/2025: We released UnSAMv2.

## Installation ⚙️

## Method Overview 💡
UnSAMv2 has two stages. (1) We generate pseudo mask-granularity pairs with granularity-aware divide-and-conquer. (2) We utilize these unsupervised data to finetune SAM-2 with granularity module.

### 1. Granularity-Aware Divide-and-Conquer ✌️

### 2. Segment Anything at Any Granularity 🔥

### UnSAMv2: Inference Demo for Interative Image Segmentation

### UnSAMv2: Inference Demo for Whole Image Segmentation

### UnSAMv2: Inference Demo for Video Segmentation

## Model Zoo 🥳

| Method | NoC<sub>80</sub> ↓ | NoC<sub>90</sub> ↓ | 1-IoU ↑ | AR<sub>1000</sub> ↑ |
| --- | --- | --- | --- | --- |
| SimpleClick | 3.32 | 4.87 | 60.2 | - |
| UnSAM | - | - | - | 39.2 |
| GraCo* | 2.35 | 3.42 | 74.4 | - |
| SAM-2 | 2.44 | 3.63 | 69.0 | 49.6 |
| UnSAMv2* | 2.28 | 3.40 | 79.3 | 64.8 |
| UnSAMv2+* | 2.07 | 3.10 | 81.7 | 68.1 |

<sub>* search for optimal granularity across [0.1, 1.0] with step 0.1.</sub>

## Evaluation 😎

## License 📋

## Acknowledgements 🙏
This codebase is built on UnSAM, SAM-2, CutLER, DINOv3, HQ-SAM, and GraCo. We sincely appreciate the authors for open-sourcing their code.

## Contact ☎️
If you have any general questions, feel free to email us at yujunwei04@berkeley.edu and xdwang@eecs.berkeley.edu. If you have code questions, we encourage you to open an issue in this repo as your question may help others.

## Citation ✨
