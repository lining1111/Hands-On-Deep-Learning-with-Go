


# Hands-On Deep Learning with Go 

<a href="https://www.packtpub.com/in/big-data-and-business-intelligence/hands-deep-learning-go?utm_source=github&utm_medium=repository&utm_campaign="><img src="https://www.packtpub.com/media/catalog/product/cache/e4d64343b1bc593f1c5348fe05efa4a6/9/7/9781789340990-original.jpeg" alt="Hands-On Deep Learning with Go " height="256px" align="right"></a>

本书的代码仓库在 [Hands-On Deep Learning with Go](https://www.packtpub.com/in/big-data-and-business-intelligence/hands-deep-learning-go?utm_source=github&utm_medium=repository&utm_campaign=), Packt出版社发布

**一种使用Gorgonia构建神经网络模型的实用方法.**

## 这本书是关于什么的?
Go生态系统包含一些非常强大的深度学习工具.本书向您展示如何使用这些工具来训练和部署可扩展的深度学习模型.
您将探索许多现代神经网络架构，例如CNN、RNN等.最后,您将能够使用Go的强大功能从头开始训练自己的深度学习模型.
本书涵盖了以下令人兴奋的功能:
* 探索深度学习Go生态系统中为了深度学习的库和社区
* 掌握神经网络的历史和工作原理
* 用Go设计和实现深度神经网络
* 获得反向传播和动量等概念的坚实基础(在梯度下降方法下，动量代表参数在参数空间移动的方向和速度)
* 使用Go构建变分自动编码器和受限玻尔兹曼机
* 使用CUDA和基准测试CPU和GPU模型构建模型


如果你感到书籍适合你, 可以在 [copy](https://www.amazon.com/dp/1789340993) 平台购买!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## 导航说明
所有代码都组织到文件夹中.例如,Chapter02

代码将如下所示:
```
type nn struct {
    g *ExprGraph
    w0, w1 *Node

    pred *Node
}
```

**以下是本书所需的内容:**
本书面向对深度学习感兴趣的机器学习工程师、数据分析师、数据科学家，他们希望探索在PyTorch中实现高级算法.机器学习的一些知识是有帮助的，但不是强制性的. 要求具有Python编程的工作知识.

使用以下软件和硬件列表，您可以运行本书中存在的所有代码文件 (Chapter 1-10).
### Software and Hardware List
| Chapter | Software required | OS required |
| -------- | ------------------------------------ | ----------------------------------- |
| All | Gorgonia package for Go | Windows, Mac OS X, and Linux (Any) |
| 4,6 | Cu package for Go | Windows, Mac OS X, and Linux (Any) |
| 4,6 | CUDA (plus drivers) from NVIDIA | Windows, Mac OS X, and Linux (Any) |
| 4,6 | NVIDIA GPU that supports CUDA | Windows, Mac OS X, and Linux (Any) |
| 9 | Docker | Windows, Mac OS X, and Linux (Any) |
| 10 | AWS Account, Kubernetes/Docker/kops, Pachyderm | Windows, Mac OS X, and Linux (Any) |


我们还提供了一个PDF文件，其中包含本书中使用的屏幕截图图表的彩色图像. [Click here to download it](https://static.packt-cdn.com/downloads/9781789340990_ColorImages.pdf).

### 相关书籍
* Deep Learning By Example  [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-example?utm_source=github&utm_medium=repository&utm_campaign=) [[Amazon]](https://www.amazon.com/dp/1788399900)

* Deep Learning with PyTorch  [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-pytorch?utm_source=github&utm_medium=repository&utm_campaign=) [[Amazon]](https://www.amazon.com/dp/1788624335)


## 了解作者
**Gareth Seneque** 是一名机器学习工程师，在金融和媒体行业大规模构建和部署系统方面拥有11年的经验.他在2014年开始对深度学习感兴趣，目前正在他的组织内建立一个搜索平台,使用神经语言编程和其他机器学习技术生成内容元数据并推动推荐.他为许多开源项目做出了贡献,包括CoREBench 和 Gorgonia. 他还在现代DevOps实践方面拥有丰富的经验, 使用 AWS, Docker, 还有 Kubernetes 去有效分配机器学习工作负载的处理.

**Darrell Chua** 是一位拥有超过10年经验的高级数据科学家. 他开发了不同复杂程度的模型,从使用逻辑回归构建信用记分卡到为名片创建图像分类模型.他大部分时间都在金融科技公司工作,尝试将机器学习技术带入金融世界.他已经在 Go 中编程了几年，并且一直在研究深度学习模型. 他的成就之一是创建了众多商业智能和数据科学管道，实现顶级自动化核保系统的交付, 产生近乎即时的批准决策.



### 建议和反馈
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.


### 下载免费的PDF

 <i>如果您已经购买了本书的印刷版或Kindle版，则可以免费获得无DRM的PDF版本.<br>只需单击链接即可领取您的免费PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781789340990">https://packt.link/free-ebook/9781789340990 </a> </p>

### 章节概述

    Chapter01 简单的介绍了用GO实现深度学习。介绍了深度学习的历史和应用，同时给了GO实现机器学习的概述
    Chapter02 什么是神经网络，怎么训练一个呢？介绍怎么构建一个简单的神经网络，怎么查看一个图、以及许多的通用且实用的激活函数。
                本章同样讨论了一些梯度下降算法的选项和神经网络的优化。
    Chapter03 超越基本的神经网络。自动编码器和RBMs( 受限玻尔兹曼机,一种可通过输入数据集,学习概率分布的随机生成型神经网络).
                展示了如何构建一个简单的多层神经网络和一个自动编码器。本章同样探讨了概率图模型和寿险波尔兹曼机的设计和实现，
                用无监督的方式创建一个电影推荐引擎
    Chapter04 CUDA-GPU加速训练。查看深度学习的硬件层面，同时查看CPUs和GPUs是怎么为我们的计算需求服务的
    Chapter05 用RNN循环神经网络进行下一个单词预测。详细描述了一个基本循环神经网络，还有怎么训练它，你会得到一个对于
                包含GRU和LSTM的循环神经网络的清楚认知。(GRU 门控循环单元，为了更高的捕捉时序数据中间隔较大的依赖关系;
                                                    LSTM 长短记忆网络，是为了解决一般的RNN（循环神经网络）存在的长期依赖问题而专门设计出来的)
    Chapter06 用CNN卷积神经网络进行物体识别。展示了如何构建一个CNN，怎么调整一些超参(诸如迭代次数、批次大小)，去获取期望的结果，使它在
                不同的电脑内顺畅的运行。
    Chapter07 用DQN深度Q值网络来解决迷宫问题。介绍了强化学习和Q值学习，以及怎么构建一个DQN来解决迷宫问题
    Chapter08 用VAE变分自编码器来生成在生成模型。展示了如何构建一个VAE，查看VAE对于标准自动编码器的优势，
                本章同样展示了如何理解一个网络的可变的隐性空间维度。
    Chapter09 构建一个深度学习的生产线。查看数据线和使用Pachyderm来构建和操作它们
    Chapter10 扩展部署。查看大量的技术，在Pachyderm之下的，包括Docker和k8s，同样检查我们如何部署大量的云服务。