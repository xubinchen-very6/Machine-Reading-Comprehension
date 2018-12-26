## Introduction
During my internship at GammaLab, I participated in the SQUAD competition with team and won the first place in the world.  [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)  
This project is an exploration of Chinese machine reading comprehension and also a summary of my internship work.

![squad](pic/squad.png)



## Dependencies
Tensorflow 1.7  
tqdm  
jieba  
synonym  

## Dataset
Dureader   [http://ai.baidu.com/broad/subordinate?dataset=dureader](http://ai.baidu.com/broad/subordinate?dataset=dureader)  
Sogou   [http://task.www.sogou.com/cips-sogou_qa/](http://task.www.sogou.com/cips-sogou_qa/)  
CMRC 2017  [https://hfl-rc.github.io/cmrc2017/](https://hfl-rc.github.io/cmrc2017/)  

## Mission Details
`Paragraph`:
胰腺移植：指因胰腺功能衰竭，已经实施了在全身麻醉下进行的胰腺的异体器官移植手术。__单纯胰岛移植、部分胰腺组织或细胞的移植__ 不在本保障范围之内。  
`Question`: 哪些移植不在胰腺移植的保障范围呢？  
`Answer`: 单纯胰岛移植、部分胰腺组织或细胞的移植

## Performance
通用MRC|Dev EM|Dev F1|Test EM|Test F1|step
----|----|----|----|----|---
QANet96|49.25|76.82|41.93|73.68|88000
QANet256|48.57|76.46|42.04|73.80|100000
QANet256+cw2vec*|47.42|77.05|47.18|75.42|57000
QANet256+stroke-aware*|47.79|77.07|47.62|76.44|59000
QANet256+wordpiece-aware*|-|-|-|-|-
BERT-base|46.58|75.94|46.34|76.12|100000

>cw2vec: Learning Chinese Word Embeddings with Stroke n-gram Information  
stroke-aware: Original method   
wordpiece-aware: Original method

## Analysis
![](/pic/length.png)
![](/pic/ansl.png)
![](/pic/f11.png)
![](/pic/f12.png)
![](/pic/score.png)

## TODO
- [x] Update README.MD
- [] Update QANetBaseline
- [] Update cw2vecs
- [] Update stroke-aware language model
- [] Update wordpiece-aware language model
- [] Update Question Augument:FrameNet+Semi-CRF-RNN
