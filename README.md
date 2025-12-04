<h1 align="center">TemCoCo: Temporally Consistent Multi-modal Video Fusion with Visual-Semantic Collaboration</h1>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/ICCV2025/html/Gong_TemCoCo_Temporally_Consistent_Multi-modal_Video_Fusion_with_Visual-Semantic_Collaboration_ICCV_2025_paper.html">Paper</a> |
  <a src="examples/1207_1739.mp4">Video results</a>
</p>

---------
## To train:<br>
* Fllow run_command.txt to run with multiple GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=7542 train_dist.py --opt options/fusion.yml --launcher pytorch
```
  or run with single GPU
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --opt options/fusion.yml
```

## To test with the pre-trained model:<br>
* Run
```bash
python test_folder.py --opt options/fusion.yml
```

## Regarding the DCN runtime environment:
Enter the dcn/src folder and run the command ``` pip install -e .  ```

### If this work is helpful to you, please cite it as:
```
@inproceedings{gong2025temcoco,
  title={Temcoco: Temporally consistent multi-modal video fusion with visual-semantic collaboration},
  author={Gong, Meiqi and Zhang, Hao and Yi, Xunpeng and Tang, Linfeng and Ma, Jiayi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14326--14335},
  year={2025}
}
```
If you have any question, please email to me (meiqigong@whu.edu.cn).

---

### Acknowledgement
We sincerely thank the authors of [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) for their great contribution.  
Our flowD metric is computed based on their implementation, and it is compatible with any other optical flow estimation algorithm.
