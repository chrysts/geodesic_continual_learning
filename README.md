
This is a public repository for:<br/>
On learning the geodesic path for incremental learning<br/>
CVPR 2021
Paper: [https://arxiv.org/abs/2104.08572](https://arxiv.org/abs/2104.08572)


## Dataset

The ImageNet-Subset can be downloaded at:
https://drive.google.com/drive/folders/1cuqI8yuqc8u1lN_N5CbFBxEdF_VSUNbG

## Run

Run on ImageNet-subset with 50 classes as an initialization and 10 tasks:
```
python3 class_incremental_imagenet.py --dataset imagenet --datadir  {your imagenetsubset dir} --num_classes 100 --nb_cl_fg 50 --nb_cl 5 --nb_protos 20 --rs_ratio 0.0 --imprint_weights --less_forget --resume --lamda 10 --adapt_lamda
```




incremental_train_and_eval_LF.py is the file containing the distillation loss with our method.


## Citation

```` 
@inproceedings{Christian2020ModGrad,
author = {Simon, Christian and Koniusz, Piotr and  Harandi, Mehrtash},
title = {On Learning the Geodesic Path for Incremental Learning},
booktitle = {IEEE Computer Vision and Pattern Recognition},
year = {2021}
}
````


Thanks to LUCIR codebase https://github.com/hshustc/CVPR19_Incremental_Learning
