# 边缘引导的单目深度估计网络EG-BTS

## 更新记录 
5.10 上传主要代码（可能不完整）

---

主要参考BTS：
From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation   
[arXiv](https://arxiv.org/abs/1907.10326)  
[Supplementary material](https://arxiv.org/src/1907.10326v4/anc/bts_sm.pdf)  
[code](https://github.com/cleinc/bts)  
我的论文主要使用pytorch版本


## 目录

- [部署](#部署)
- [测试](#test)
- [评估](#evaluate)
- [训练](#train)
### 部署
#### NYU V2 Depth dataset
[google drive](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view)  
这个数据集是修复后的，50k训练集，645测试集（4.1G大小）

#### 环境
1. 普通GPU python3.6 pytorch1.2 cuda10  
2. 3080 python3.7 pytorch1.10 cuda11.3  
3. A100GPU 用pytorch1.6版本  
都可以
#### 训练模型
最优模型：grad rms 0.3869 [baidu链接](https://pan.baidu.com/s/1nYdpmtm5C66HgTPIvtsXfw?pwd=ufdi) 提取码: ufdi  
其他模型：链接: https://pan.baidu.com/s/1nBlKGLP1SKq82iuVdFRHgg?pwd=6bwb 提取码: 6bwb
### test
**离线测试**  
[bts_test_debug.py](bts_test_debug.py)
生成结果图  
``` python
python bts_test_debug.py --checkpoint_path './models/bts_nyu_v2_pytorch_test/model_rms_gradorin4392' --save_name 'grad08'
```

### evaluate
[eval_nyu_with_inpaint.py](eval_nyu_with_inpaint.py)

### train
[bts_main_grad.py](bts_main_grad.py)  
可以直接运行，但是需要修改自己的数据集位置，log_directory等。