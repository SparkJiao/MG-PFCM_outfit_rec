# GP-BPR: Personalized Compatibility Modeling for Clothing Matching

Code for paper [GP-BPR: Personalized Compatibility Modeling for Clothing Matching](https://dl.acm.org/doi/abs/10.1145/3343031.3350956).

## Dependencies

This project currently requires the stable version of [Pytorch](pytorch.org) 

- torch 1.0.0

or

- torch 1.0.1.post2

you need to run this program using GPU

## Data Preparation

### /data

- train(valid/test).scv

format: UserID|TopID|PositiveBottomID|NegativeBottomID
 
### /feat

- smallnwjc2vec

- textfeatures

- visualfeatures

Can be download from [there](https://drive.google.com/file/d/1ILz1P4BiyQ0rTwOJD-vqs2J4cF77alUM/view).

### Meta data

format: user/outfit/item

Can be download from [there](https://drive.google.com/open?id=1sTfUoNPid9zG_MgV--lWZTBP1XZpmcK8).

## Running command

CUDA_VISIBLE_DEVICE=0 python main.py

## Citations

```
@inproceedings{song2019gp,
  title={GP-BPR: Personalized Compatibility Modeling for Clothing Matching},
  author={Song, Xuemeng and Han, Xianjing and Li, Yunkai and Chen, Jingyuan and Xu, Xin-Shun and Nie, Liqiang},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={320--328},
  year={2019}
}
```
