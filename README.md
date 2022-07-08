<p align="center">
    <img src="https://raw.githubusercontent.com/atomicai/clai/master/docs/clai-logo.png"/>
</p>

> This repo is designed to provide simple yet flexible flow to classify textual dataset into fixed number of classes using Deep Learning techniques.

On the one hand you can model different architectures to enocde your text in a Euclidean space preserving semantic sense. On the other side you can vary training pipeline. Assuming you already have the best in its kind architecture, switching from standard "log loss" to, let's say "focal loss", might give you significant boost on several metrics and overall model quality. 

The real world challenges usually impose you on the several constraints. The most important is, of course, latency. Engineering and deploying AI models in an already built heavy loaded pipeline makes . The model size is secondlimiting the choi We provide several models out of the box that works for both english and russian languages varying in sizes and 

The text classification is one of the "in-demand" subtask in almost any production ready systems. Query classification in highly complex chatbot intent flow or simple positive/negative review classification fall in that category. This repo falls in "building blocks" design rather than end-2-end pipeline. While you can pick any of the already pre-built and pretrained models, the most efficient way to achieve the best score on your data is to finetune already prebuilt models on your own data 🎩.

