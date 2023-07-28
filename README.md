# Difficulty-Aware-Simulator-for-Open-Set-Recognition
[[Arxiv](https://arxiv.org/abs/2207.10024)] | [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850360.pdf)] |
[[Video](https://www.youtube.com/watch?v=0_q9wxDIYbE&ab_channel=WonJunMoon)]

Official PyTorch Repository of "Difficulty-Aware Simulator for Open Set Recognition" (ECCV 2022 Paper)

## 1. Requirements
- python 3.6+
- torch 1.2+
- torchvision 0.4+
- CUDA 10.1+
- scikit-learn 0.22+

### Split information & Dataset

Split information for all datasets can be found in `split.py`.

`splits_F1` and `splits_AUROC` are split information for each benchmark with F1 score and AUROC.

When you run the code, datasets except tiny-ImageNet will be automatically downloaded.

[[Tiny_ImageNet Download Link](https://drive.google.com/file/d/1Z0cHrTVZxjOCBol5jrQASYucWXVB0I4i/view?usp=share_link)]

## 2. Training & Evaluation
```train
python osr.py --dataset 'cifar10'
```
To run the code, execute `osr.py`.
Then, the results will be saved under the "logs" directory.

```
sh osr.sh
```
For simplicity, we provide the training scripts for running all datasets.
You can execute the shell file by the above command. 


++ Since [Open-Set Recognition: a Good Closed-Set Classifier is All You Need?] elaborated additional techniques with searched hyperparameters can boost OSR performances, we here simply conduct and compare the performances of DIAS and ARPL+cs.

| Tiny-ImageNet | 800 epochs training |
| ------------- |:-----:|
| ARPL+cs       | 71.9  |
| DIAS (Ours)   | 75.6  |

##  Cite DIAS (Difficulty-Aware Simulator for Open Set Recognition)

If you find this repository useful, please use the following entry for citation.
```
@inproceedings{moon2022difficulty,
  title={Difficulty-Aware Simulator for Open Set Recognition},
  author={Moon, WonJun and Park, Junho and Seong, Hyun Seok and Cho, Cheol-Ho and Heo, Jae-Pil},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
```

## Contributors and Contact

If there are any questions, feel free to contact with the authors: WonJun Moon (wjun0830@gmail.com), JunHo Park (pjh3974@gmail.com), Hyun Seok Seong (gustjrdl95@gmail.com), Cheol-Ho Cho (hoonchcho@gmail.com).

## Acknowledgement

This repository is built based on [ARPL](https://github.com/iCGY96/ARPL) repository.
Thanks for the great work.


