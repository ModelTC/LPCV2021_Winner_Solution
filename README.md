# LPCV Winner Solution of Spring Team

## Background
Challenge link: https://lpcv.ai/2021LPCVC/fpga-track

Leaderboard: https://lpcv.ai/scoreboard/FPGA21

## Environment
Xilinx Board: Ultra-96 V2

System: https://github.com/Avnet/Ultra96-PYNQ/releases v2.6

```
git clone --recursive --shallow-submodules https://github.com/Xilinx/DPU-PYNQ
cd DPU-PYNQ/upgrade
make
pip3 install pynq-dpu
```

Note: replace the image.ub with [the LPCV provided one](https://drive.google.com/file/d/1UuZBG5GrhwGFUJWx_eiZYPtxy-tR-q_V/view).


## Build
```
cd nms
python3 build.py install
```

## Evaluation
Evaluate in the jupyternote book: evaluation.ipynb.


## Contributors
[Jiahao Hu](https://github.com/Joker-co), [Pu Li](https://github.com/SpursLipu), [Yongqiang Yao](https://github.com/yqyao), [Ruihao Gong](https://xhplus.github.io/)


## Reference Links
The detection model is trained by [EOD](https://github.com/ModelTC/EOD).
The Quantization can be fulfilled with [MQBench](https://github.com/ModelTC/mqbench).