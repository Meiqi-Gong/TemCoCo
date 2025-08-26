# TemCoCo
Code of [TemCoCo: Temporally Consistent Multi-modal Video Fusion with Visual-Semantic Collaboration](https://www.sciencedirect.com/science/article/pii/S1077314222000352)

Tips
---------
#### To train:<br>
* Fllow run_command.txt to run with multiple GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=7542 train_dist.py --opt options/fusion.yml --launcher pytorch
```
  or run with single GPU
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --opt options/fusion.yml
```

#### To test with the pre-trained model:<br>
* Run
```bash
python test_folder.py
```

If this work is helpful to you, please cite it as:
```
@article{gong2025temcoco,
  title={TemCoCo: Temporally Consistent Multi-modal Video Fusion with Visual-Semantic Collaboration},
  author={Gong, Meiqi and Zhang, Hao and Yi, Xunpeng and Tang, Linfeng and Ma, Jiayi},
  journal={arXiv preprint arXiv:2508.17817},
  year={2025},
  note={Accepted by ICCV 2025}
}
```
If you have any question, please email to me (meiqigong@whu.edu.cn).
