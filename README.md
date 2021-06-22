
This is a public repository for:<br/>
On learning the geodesic path for incremental learning<br/>
CVPR 2021
Paper: [https://arxiv.org/abs/2104.08572](https://arxiv.org/abs/2104.08572)



Run on ImageNet-subset with 50 classes as an initialization and 10 tasks:
```
python3 class_incremental_cosine_imagenet.py --dataset imagenet --datadir /flush5/sim314/cs1/imagenet/seed_1993_subset_100_imagenet/data/ --num_classes 100 --nb_cl_fg 50 --nb_cl 5 --nb_protos 20 --rs_ratio 0.0 --imprint_weights --less_forget --resume --lamda 10 --adapt_lamda
```


incremental_train_and_eval_LF.py is the file containing the distillation loss with our method.



Thanks to LUCIR codebase https://github.com/hshustc/CVPR19_Incremental_Learning
