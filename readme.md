# bertMetric
使用huggingface/timm实现的高质量多模态相似Embedding抽取方案
* 使用Roberta预训练模型进行文本embedding抽取
* 使用efficientnet模型进行图像embedding抽取
* 使用图文Two stream实现多模态embedding抽取
* arcface实现metric learning优化embedding质量
* pyspark拉取数据

# References
https://github.com/deepinsight/insightface
https://github.com/auroua/InsightFace_TF
https://github.com/MuggleWang/CosFace_pytorch
https://github.com/rwightman/pytorch-image-models
https://github.com/huggingface/transformers

# pretrained model
nlp pretrained model 默认使用roberta-wwm-ext-large 
cv pretrained model 默认使用efficientnet_b4
可以按需求替换