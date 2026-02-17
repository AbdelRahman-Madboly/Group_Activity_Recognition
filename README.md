# Group Activity Recognition

Reimplementation of "Hierarchical Deep Temporal Models for Group Activity Recognition"
(Ibrahim et al., CVPR 2016) using PyTorch and ResNet50.

The original paper used AlexNet trained in Caffe and reported 81.9% accuracy on the
volleyball dataset. This repo uses ResNet50 as the backbone and works through the
paper's ablation baselines from simplest to most complex.

## Results so far

| Baseline | Description | Test Accuracy |
|----------|-------------|---------------|
| B1 | Frame-level image classification | 77.6% |
| B3 | Person-level feature pooling | in progress |
| B8 | Full hierarchical LSTM model | planned |

Paper (AlexNet): 81.9%
Our target (ResNet50): 85%+


## Dataset

The volleyball dataset contains 55 YouTube videos with 4,830 annotated frames.
Each frame has bounding boxes and action labels for every player, plus a team-level
group activity label.

- 8 group activity classes: r_set, r_spike, r-pass, r_winpoint, l_set, l_spike, l-pass, l_winpoint
- 9 person action classes: blocking, digging, falling, jumping, moving, setting, spiking, standing, waiting

The dataset is not included in this repo. On Kaggle it lives at:
`/kaggle/input/datasets/ahmedmohamed365/volleyball/`

Update the paths at the top of `config/config.py` if you are running locally.


## Project structure

```
Group_Activity_Recognition/
    config/
        config.py           all hyperparameters, label maps, and dataset paths
    data/
        annotation_parser.py    parses annotations.txt and tracking files
        datasets.py             PyTorch Dataset classes for B1, B3, and B5
        transforms.py           image transforms for training and evaluation
    models/
        baselines/
            b1_image_classifier.py   ResNet50 frame classifier
            b3_person_pooling.py     person action classifier + group activity MLP
    training/
        trainer.py      shared train/eval loop used by all baselines
        train_b1.py     B1 training script
        train_b3.py     B3 training script (3-stage pipeline)
    evaluation/
        metrics.py      confusion matrix, classification report, result summaries
    utils/
        visualization.py    dataset exploration and frame visualization
    tests/
        test_data.py    sanity checks for the data pipeline
        test_models.py  sanity checks for model forward passes
    notebooks/
        kaggle_training.ipynb   runs training on Kaggle by cloning this repo
    requirements.txt
```


## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/AbdelRahman-Madboly/Group_Activity_Recognition.git
cd Group_Activity_Recognition
pip install -r requirements.txt
```

If you are working locally, update `config/config.py` to point `VIDEOS_ROOT` and
`ANNOT_ROOT` at wherever you have the dataset stored.


## Running locally in VSCode

The training scripts can be run directly from the terminal:

```bash
# Run data and model sanity checks first
python tests/test_data.py
python tests/test_models.py

# Train B1
python training/train_b1.py

# Train B3 (takes longer, three stages)
python training/train_b3.py
```

Outputs (checkpoints and result plots) go to `outputs/` by default.
That folder is in .gitignore so it will not be pushed to GitHub.


## Training on Kaggle from VSCode

The workflow is:
1. Write and edit code locally in VSCode
2. Push changes to GitHub with `git push`
3. Open `notebooks/kaggle_training.ipynb` on Kaggle
4. The notebook clones the repo and calls the training functions
5. Kaggle provides the GPU and the dataset

This way all your code lives in version control and Kaggle is just used
for its GPU, not as a code editor.

To update Kaggle with your latest changes:
```bash
# On your laptop
git add .
git commit -m "your message"
git push

# Then in the Kaggle notebook, re-run the git pull cell
```


## Connecting VSCode to the repo

```bash
cd C:\Dan_WS
git clone https://github.com/AbdelRahman-Madboly/Group_Activity_Recognition.git
cd Group_Activity_Recognition
code .
```

Then in VSCode, open the integrated terminal and work normally.
The repo is already connected to GitHub so `git push` and `git pull` work as expected.


## Baselines

### B1 - Frame image classifier

Simplest baseline. ResNet50 takes the middle frame of each clip and predicts
the group activity. No temporal information, no person-level reasoning.
Achieves around 77-78%, which is already 10 points above the paper's AlexNet
baseline for the same setup.

### B3 - Person-level feature pooling

Three stages:
- Stage A: train ResNet50 on individual player crops to recognize person actions
- Stage B: use the trained backbone to extract features per player, max-pool across players
- Stage C: train a small MLP on the pooled features to predict group activity

This teaches the model to reason about individual players before aggregating.

### B8 - Full hierarchical model (planned)

The full model from the paper. An LSTM runs per person across the temporal
window, then player representations are pooled into two group representations
(left and right team), then a second LSTM runs at the group level.
Target accuracy: 85%+.


## Paper reference

Ibrahim, M. S., Muralidharan, S., Deng, Z., Vahdat, A., & Mori, G. (2016).
A hierarchical deep temporal model for group activity recognition.
In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR).
