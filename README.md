# Sequential Image Generation using GANs

Pytorch implementation for sequential image generation using GANs based on our AAAI 2019 paper:

Turkoglu, MO, et al. " A Layer-Based Sequential Framework for Scene Generation with GANs. " 
In AAAI. 2019.

# [[Paper]](https://arxiv.org/abs/1902.00671)    [[Poster]](https://drive.google.com/open?id=1MJhVce9a5jWI6GnW45k4gNFGe-Jie0-z)  [[Blog]](https://medium.com/@moturkoglu/sequential-image-generation-with-gans-acee31a6ca55)




<img src="https://raw.githubusercontent.com/0zgur0/Seq_Scene_Gen/master/imgs/intro.png" width="600" height="360">

## Getting Started

Background model can be trained with 
```bash
python train_bg_model.py
```

Foreground model can be trained with 
```bash
python train_fg_model.py
```

Baseline model can be trained with 
```bash
python train_baseline_model.py
```

## Citation
```bash
@inproceedings{turkoglu2019layer,
  title={A Layer-Based Sequential Framework for Scene Generation with GANs},
  author={Turkoglu, Mehmet Ozgur and Spreeuwers, Luuk and Thong, William and Kicanaoglu, Berkay},
  booktitle={Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19)},
  year={2019}
}
```
