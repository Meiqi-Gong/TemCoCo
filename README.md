# TemCoCo
Code of [TemCoCo: Temporally Consistent Multi-modal Video Fusion with Visual-Semantic Collaboration](https://www.sciencedirect.com/science/article/pii/S1077314222000352)

Tips
---------
#### To train:<br>
* Fllow run_command.txt to run CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=7542 train_dist.py --opt options/fusion.yml --launcher pytorch
  or CUDA_VISIBLE_DEVICES=0 python train_dist_iso.py --opt options/fusion.yml

#### To test with the pre-trained model:<br>
* Run test_folder.py.

If this work is helpful to you, please cite it as:
```
@article{xu2022cufd,
  title={CUFD: An encoder--decoder network for visible and infrared image fusion based on common and unique feature decomposition},
  author={Xu, Han and Gong, Meiqi and Tian, Xin and Huang, Jun and Ma, Jiayi},
  journal={Computer Vision and Image Understanding},
  pages={103407},
  year={2022},
  publisher={Elsevier}
}
```
If you have any question, please email to me (meiqigong@whu.edu.cn).
