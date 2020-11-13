# Asynchronous Advantage Actor-Critic(A3C)
論文  
[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783v1.pdf)  
参考にしたコード  
[pytorch_a3c/rarilurelo](https://github.com/rarilurelo/pytorch_a3c)  
[pytorch-A3C/MorvanZhou](https://github.com/MorvanZhou/pytorch-A3C)

### Requirment  
python 3.7  
pytorch 1.6.0  
gym

### Usage
#### Built environment
> docker/Dockerfile をお使いください. 
>> - remark  
>> docker run -it -d -p [host側のport]:9000 -v [host側の作業ディレクトリ(このprojectのあるところ)]:/home/develop --name [image名] [container名] bash  

#### Training
> python3 train.py --test  
>> もし学習後のtestを動画保存したい場合  
>> python3 train.py --test --monitor  

>> もしdockerを使わず実行する場合はtestの様子を描画できます  
>> python3 train.py --test --render  
