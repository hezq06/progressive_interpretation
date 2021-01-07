Code for reproducing results of paper:
"An Information-theoretic Progressive Framework for Interpretation"
Authors: Zhengqi He @ NCA Lab, CBS, RIKEN, Japan
Taro Toyoizumi @ NCA Lab, CBS, RIKEN & UTokyo, Japan

Dependencies:
python (3.0)
jupyter notebook
numpy
matplotlib
pytorch (1.2.0)
pycocotools

Dataset:
We use standard CLEVR dataset in this project:
https://cs.stanford.edu/people/jcjohns/clevr/
Download CLEVR v1.0 (18 GB)
Extra data like calculated object mask and pretrained model is available at:
https://riken-share.box.com/s/rc95iet4af680o3pll2sqywc6kevbduk
Put folders "dataset" and "pretrained_models" inside the progressive_interpretation main folder

Run the code:
To reproduce results in the paper, check out the jupyter notebooks:
CLEVRTask1SuperviseLearning.ipynb
CLEVRTask2MultipleChoice.ipynb

