# Hands-On Deep Learning with Go

<a href="https://www.packtpub.com/in/big-data-and-business-intelligence/hands-deep-learning-go?utm_source=github&utm_medium=repository&utm_campaign="><img src="https://www.packtpub.com/media/catalog/product/cache/e4d64343b1bc593f1c5348fe05efa4a6/9/7/9781789340990-original.jpeg" alt="Hands-On Deep Learning with Go " height="256px" align="right"></a>

本书的代码仓库在 [Hands-On Deep Learning with Go](https://www.packtpub.com/in/big-data-and-business-intelligence/hands-deep-learning-go?utm_source=github&utm_medium=repository&utm_campaign=),
Packt出版社发布

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

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img
src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png"
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
本书面向对深度学习感兴趣的机器学习工程师、数据分析师、数据科学家，他们希望探索在PyTorch中实现高级算法.机器学习的一些知识是有帮助的，但不是强制性的.
要求具有Python编程的工作知识.

使用以下软件和硬件列表，您可以运行本书中存在的所有代码文件 (Chapter 1-10).

### Software and Hardware List

| Chapter | Software required                              | OS required                        |
|---------|------------------------------------------------|------------------------------------|
| All     | Gorgonia package for Go                        | Windows, Mac OS X, and Linux (Any) |
| 4,6     | Cu package for Go                              | Windows, Mac OS X, and Linux (Any) |
| 4,6     | CUDA (plus drivers) from NVIDIA                | Windows, Mac OS X, and Linux (Any) |
| 4,6     | NVIDIA GPU that supports CUDA                  | Windows, Mac OS X, and Linux (Any) |
| 9       | Docker                                         | Windows, Mac OS X, and Linux (Any) |
| 10      | AWS Account, Kubernetes/Docker/kops, Pachyderm | Windows, Mac OS X, and Linux (Any) |

我们还提供了一个PDF文件，其中包含本书中使用的屏幕截图图表的彩色图像. [Click here to download it](https://static.packt-cdn.com/downloads/9781789340990_ColorImages.pdf).

### 相关书籍

* Deep Learning By
  Example  [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-example?utm_source=github&utm_medium=repository&utm_campaign=) [[Amazon]](https://www.amazon.com/dp/1788399900)

* Deep Learning with
  PyTorch  [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-pytorch?utm_source=github&utm_medium=repository&utm_campaign=) [[Amazon]](https://www.amazon.com/dp/1788624335)

## 了解作者

**Gareth Seneque**
是一名机器学习工程师，在金融和媒体行业大规模构建和部署系统方面拥有11年的经验.他在2014年开始对深度学习感兴趣，目前正在他的组织内建立一个搜索平台,使用神经语言编程和其他机器学习技术生成内容元数据并推动推荐.他为许多开源项目做出了贡献,包括CoREBench
和 Gorgonia. 他还在现代DevOps实践方面拥有丰富的经验, 使用 AWS, Docker, 还有 Kubernetes 去有效分配机器学习工作负载的处理.

**Darrell Chua** 是一位拥有超过10年经验的高级数据科学家.
他开发了不同复杂程度的模型,从使用逻辑回归构建信用记分卡到为名片创建图像分类模型.他大部分时间都在金融科技公司工作,尝试将机器学习技术带入金融世界.他已经在
Go 中编程了几年，并且一直在研究深度学习模型. 他的成就之一是创建了众多商业智能和数据科学管道，实现顶级自动化核保系统的交付,
产生近乎即时的批准决策.

### 建议和反馈

[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you
have any feedback or suggestions.

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

## Chapter01

### 解释深度学习

    退一步讲，从一个简单的例子来讲深度学习。随着书籍深入，我们会对深度学习有一个认识的发展。现在，我们从一个简单的例子开始。
    如果我们有一个人像的图片，我们应该怎样向电脑展示它呢？我们怎么教电脑把这个图片和一个"person"的单词关联起来？
    首先，我们找到一种展示图片的方法，将图片的每个像素的RGB值表述出来。然后我们将这些像素值的数组(和一些可训练的参数)
    输入进一系列我们熟悉的操作，比如乘法和加法。这样会产生出一个新的变量，我们用它来和我们设定的一个标签图组做对比，得到"person"
    这个单词。我们使这个对比的过程自动化，且随着我们的进行更新这些参数。
    上面的描述涵盖了一个简单的，浅显的深度学习系统。我们将在后面的章节学习更多神经网络的细节。但是现在，为了实现一个系统深度，
    我们在大量的参数之上增加了一系列的操作。这样我们就能抓住更多的关于人像的细节。影响系统设计的生物模型是人类神经系统。
    包括神经元(我们呈现出来的)和突触(可训练的参数)。
    所以，深度学习只是1957年出现的感知机的演化。最简单最原始的二分发分类器。计算机能力戏剧性的增长和转变，是系统不工作和系统能够允许车辆自动驾驶的不同
    超越自动驾驶，有一系列的深度学习软件和相关的领域，如耕种、农作物管理、卫星图片分析。先进的计算机视觉功能机可以完成除草来减少
    杀虫剂的使用。我们有高实时的快速精准的语音搜索。这些都是社会的基础元素、从食物生产到通讯领域。

### Go中的深度学习概述

    go语言中的深度学习系统相当的有限，这门语言发布于2009年，在深度学习变革带来了许多的编程人员进去此领域。
    不幸的是，这些深度学习库，却很少的适配go语言。
    平时中，都是怎么应对深度学习的问题的呢？
    1、BLAS Basic Linear Algebra Subprograms
    2、tensor
    3、SGD Stochastic Gradient Descent
    上述的问题，就是本书要解决的
    1、Automatic or symbolic differentiation
    2、Numberical stabilization fucntions
    3、Activation functions
    4、Gradient descent optimizations
    5、CUDA support
    6、Deployment tools
    本书中的库都是从Gorgonia开始的
    第一章是通过gorgonia库内的Graph元素来理解机器学习中矩阵的加减乘除。并通过ToDot()函数，来得到.dot文件，
    此文件可以用过dot命令转换为各种可读文件，比如pdf

## Chapter02

### 什么是神经网络，我应该怎么训练一个呢？

    第二章主要介绍怎么构建一个神经网络，以及使它工作。这样教你了解构建复杂神经网络的基本元素。
    1、一个基本神经网络
    2、激活函数
    3、梯度下降和反向传播
    4、高级梯度下降算法

    1）简单神经网络、用一个基本的加乘算法，使一个4*3的整数矩阵，来初始化一个权重系数，来反映一个3*1的列向量，
    然后调整这些权重系数，直到它们可以预测，已经给定的一系列输入和输出数据的关系。
    这里说的是，用4*3的输入矩阵来表示3*1的列向量输出之间的关系。
    第二章的关键目标就是理解Gorgonia库怎么操作一个基本神经网络的。神经元的学习就是一系列、反复的调整神经元的权重系数。
    激活函数、SGD(随机梯度下降)、反向传播 将在后面的章节分别展示其细节，现在讨论的是它们在神经元的作用。

## Chapter03

    第三章主要介绍超出基本神经网络的---自动编码器和RBMs(受限玻尔兹曼机)是一种可通过输入数据集学习概率分布的随机生成神经网络。
    MNIST是一个手写字体的数据。每个字体是28*28像素的灰度，然后识别后是0-9(10个数字)
    这里采用的是输入是1*784(28*28)的列向量 输出是1*10的列向量。然后中间神经元有三层，一层是784*300的权重矩阵，一层是300*100的权重矩阵，再一层是100*10的权重矩阵。
    这样计算是
    1   vector(784) *   matrix(784*300)=    vector(300)
    2   vector(300) *   matrix(300*100)=    vector(100)
    3   vector(100) *   matrix(100*10)=     vector(10)
    大致的过程如上，其中在输入的时候为了提高速度，将784的列向量变成bs*784,10的列向量变成bs*10，bs为batchsize的意思，批处理的规模。
    程序中有三个量 epochs，意思是对这次的训练执行多少代(次);batchsize，每次处理的数据集合个数，(以784为一个输入数据集合);numExamples,mnist读出的字体文件内，字体像素集合的个数。

### 在mnist_feedforward文件夹内，

    csv文件夹存的是测试后的结果：以批次为文件名;每个文件内存的是当前批，输入经过模型后的到的输出，即1*10的向量结果，其中越接近1的就是数字0-9的index。
    images文件夹内存的是测试后的结果，文件名是以批次_当前批的元素集合索引_MNIST提供的label_模型预测的label。
    这样就是images内是实际图像素的识别结果表示、csv内存的是数学上的计算结果。相互对照。

### mnist_autoencoder 自动分类器

    nn中同时包含编码器和解码器，即784*128,128*64  和   64*128  128*784
    training文件夹内存的是训练时的过程，文件名，开头的没用_批次号_训练的次数,为了观察一个批的随着训练次数的增加带来的效果
    images文件夹内存的是测试时的结果，以输入和输出为一组，文件名为 批次号_当前批的集合索引 对比输入和输出的预测情况
    ~~~~本章中关于RBMs的例子是关于网飞电影的一个例子，但是原代码中就没有，书中的代码又是片段，只能在此时作罢(20230606)，可能以后对深度学习有了把握后，再来做这个。

## Chapter04

    第四章是MNIST的训练，用的CUDA来做加速，代码内容基本不变，但是项目是在虚拟机内做的，用不了GPU也就用不CUDA，所以暂时作罢。

## 小结

    前面4章都是讲述的深度学习的基本单元 nn的相关知识，后面就是针对具体领域的实际深度学习模型。

## Chapter05

    第五章是基于RNN的语言模型。下一个单词预测。
    难点就是自然语言处理和理解力
    本章介绍的是：
    1、一个基本的RNN是什么样的
    2、怎么训练一个RNN
    3、RNN的提高训练，包括GRU(门循环单元)/LSTM(长短期记忆单元)
    4、怎么用Gorgonia来实现一个带LSTM单元的RNN

    RNN是带时间序列反射的NN，即上一层的状态会参与下一层的运算。BPTT，损失函数用交叉熵
    代码中Activate函数和前面章节的神经元的fwd函数的作用一样，用于描述这个单元做了哪些运算。
    这章没看透(20230606)

## Chapter06

    第六章讲的是CNN(卷积神经网络)，用于物体识别，这里采用的数据集是cifar由 Alex Krizhevsky，Vinod Nair 和 Geoffrey Hinton 收集整理自8000万张微型图像数据集，
    其中CIFAR数据集又根据所涉及的分类对象数量，可分为CIFAR-10和CIFAR-100。该数据集主要用于深度学习的图像分类，目前已被广泛应用
    CIFAR-10有10个类别，每个类别1000张图，每张图是32*32个像素，每个像素为32位
    二进制版本的文件解压后，将得到 data_batch_1.bin, data_batch_2.bin, ..., data_batch_5.bin,以及 test_batch.bin
    第一个字节是第一个图像的标签，它是一个范围为0-9的数字。 接下来的3072字节是图像的像素值。 前1024个字节是红色通道值，接下来的1024是绿色的，最后的1024是蓝色的
    每个文件包含10000个这样的3073字节的“行”图像，尽管没有划分行。 因此，每个文件应该正好是30730000字节长。
    前面几章的如设计MNIST的图像手写体识别，数据规模不是很大，可以用基本神经网络识别，但是在平常生活中，经常是物体图像会非常的大，但是我们又不想弄极端大的
    隐藏层在神经元内，所以通过增加数据的维数，比如图像分为 高度、宽度、深度，三个维度(这里的深度就是像素点上的值)
    在CNN卷积神经网络中有三个重要的层：卷积层、取样层、全链接层，现在的CNN发展很快，网上有很多比如知乎的https://zhuanlan.zhihu.com/p/468806876
    这里文章说跑这个代码需要至少4G的空间，现在在虚拟机中可以先不试(20230606)。

## Chapter07

    第七章讲的是用DQN来解决迷宫问题，

## Chapter08

    第八章讲的是VAE 变分自动编码器

## 总结

    从readme可以看出基础的部分1-4章，都做了详细的笔记，也就是学的较好，而5-8章，设计具体的神经网络应用的时候，就不那么好了，甚至是到了7-8章只点出了
    具体神经网络的名字，简单之极。具体的神经网络的学习，面对书中的篇幅和英文水平的受限，学习吃力，应该从网上搜集较好的中文文章来从原理性(每种神经网络含有的
    特有神经元操作)、到具体的go代码实现，应用来理解，我相信每个具体的应用展开来，都会花费巨大的时间和精力，但也受益匪浅。这就是4-8章的学习策略。
    至于9-10章，是部署相关的，等真正的参与从事相关工作的时候，可以再进行相关的学习和训练。

## 20230606

    今天也是2023年高考开始的日子，每个人都有过美好生活的愿望，而教育提供了一个公平公正的上升渠道，回首往事，只有上学的时候，每个人都像自己，干净单纯。
    同样的，这也是一段纯粹的，为了改变人生而努力的阶段。就像一个孱弱的婴儿，通过家庭、学校，不断的变得强壮，以适应将来复杂的社会环境。
    能心无旁鹜的，安心的学习，也是件美好。加上有老师同学。
    工作后，才知道学习技能提升，尤其是提升一个极具竞争力的技能，很难。不光是别人不教，更多的是，环境的糟糕。

    说了这么多，接下来，在Hands-On 这本书学到基础后，可以通过学习《深度学习》的中文译版+《Go语言机器学习实战》两本书，来加强学习了。
    至少基础理解是没问题了，就是基础神经元，都是一系列的矩阵乘操作(输入和模型当前层的权重)，而在不同的应用模型中。神经元的内部操作有小的不同、
    神经元之间连接关系有大的不同，同时伴随一些外部不同，来理解具体的神经网络。
    到现在的阶段，虽然理解不能完全的对，但书读百遍，其意自现。同样，还有疏途同归的说法，总会进步到对的那天。

## 学习机器学习的现阶段问题

    现阶段的准备工作，在简单理解神经元的操作后，
    1、需要就具体的库如本书中的Gorgonia，来理解数学操作;
    2、同样还需要找到通俗的具体神经网络的数学表达。
    3、将两者结合起来，把5-8章的代码注释补全，完成理解。
    
## 解决方案：
### 1、对照库的官网https://gorgonia.org/来学习
        开始理解Gorgonia与线性代数一个很重要的开始就是，用Gorgonia的方式来初始化线性代数的基本元素：
        标量  NewScalar...    单个数
        向量  NewVector...    一维数组
        矩阵  NewMatrix...    二维数组
        张量  NewTensor...    多维数组
        前面的元素在完成初始化的时候，都带Gorgonia库的基本单元 ExprGraph(无环图) 还有名字，这些会生成Node，
        在最后，都可以通过 图的ToDot()函数来生成byte数组，如 
            ioutil.WriteFile("pregrad.dot", []byte(g.ToDot()), 0644)
        最后通过在shell输入 dot -T pdf xx.dot -o xx.pdf 来生成可读的pdf文件来看图的各部分连接关系
        通过树形结构来理解Gorgonia的ExprGraph 
        1、ExprGraph 包含 Node (Node 是通过NewXX来生成线性代数的 标量、向量、矩阵、张量)
        2、Node 之间的关联通过 他们的集合如struct，的一个自有函数，如fwd(nn)、Activate(lstm、gru)来表示该集合内各个Node的算术关系(即它们在该框架的线性代数运算)
            属于前向传播。 Node与Node之间的算数关系用Must Add Mul Sigmoid Tanh HadamardProd Tanh SoftMax 之类的算数关系来链接
        3、在最外层的模型 建立计算输出和已知输出的loss，即损失值 是一个Node
        4、通过Grad函数将 loss 和 learnables (Node数组,各神经的权重)，联系起来
        5、通过NewTapeMachine 创建vm 传入 BindDualValues() 这里可以填 learnables 也可以空
        6、NewXXSolver创建 solver
        7、NodesToValueGrads(learnables) 创建 model
        8、通过Let UnsafeLet 来初始化 输入和输出 vm.Reset() vm.RunAll()
        9、solver.Step(model)
        10、完成一次训练、从8、9循环每一次训练

        以上10步，是一个基本模型的形成和训练过程。(预测和训练不同的是没有Step这步)
        理解大框架后，再通过小部分的线性代数关系，来融会贯通的理解库和模型

        SoftMax()函数针对的是概率分布问题。它必须面对的问题是数值的上溢和下溢问题。
            背景：连续数学在数字计算机上的根本困难是，我们需要通过有限数量的位模式来表示无限多的实数。
                这意味着我们在计算机中表示实数时，几乎总会引入一些近似误差。在许多情况下，这仅仅是舍入误差。
                舍入误差会导致一些问题，特别是当许多操作复合时，即使是理论上可行的算法，如果在设计时没有考虑
                最小化舍入误差的累积，在实践时也可能会导致算法失效。
            带来的问题：下溢：当接近零的数被四舍五入为零时发生下溢。许多函数在其参数为零而不是一个很小的正数时才会表现出质的不同。例如，我们通常要避免被零除
                      上溢：当大量级的数被近似为∞或−∞时发生上溢。进一步的运算通常会导致这些无限值变为非数字。
            
        条件数指的是函数相对于输入的微小变化而变化的快慢程度。输入被轻微扰动而迅速改变的函数对于科学计算来说可能是有问题的，因为输入中的舍入误差可能导致输出的巨大变化

        大多数深度学习算法都涉及某种形式的优化。优化指的是改变x(权重)以最小化或最大化某个函数f(x) (损失函数)的任务。我们通常以最小化f(x)指代大多数最优化问题。

        约束优化：指x在某些集合中寻找f(x)的最大值或最小值。KKT方法是针对约束优化非常通用的解决方案。
    
### 2、机器学习的数学表达

        机器学习可以让我们解决一些人为设计和使用确定性程序很难解决的问题。从科学和哲学的角度来看，机器学习之所以受到关注，
        是因为提高我们对机器学习的认识需要提高我们自身对智能背后原理的理解。

        机器学习算法是一种能够从数据中学习的算法。学习的简洁定义：对于某类任务T和性能度量P，一个计算机程序被认为可以从经验E中学习是指，
        通过经验E改进后，它在任务T上由性能度量P衡量的性能有所提升。

        学习：是我们所谓的获取完成任务的能力。

        任务：定义为 机器学习系统应该如何处理样本。样本是指我们从某些希望机器学习系统处理的对象或事件中收集到的已经量化的特征（feature）的集合。
            通常样本将表示成一个向量，其中向量的每一个元素是一个特征。例如：一张图的特征通常是值这张图片的像素值。
            常见机器学习的任务类型：
                分类：输入属于K类中的哪一类。
                输入缺失分类：
                回归：对给定的输入预测数值。一般用于金融交易中。
                转录：将一些相对非结构化表示的数据，转录信息为离散的文本形式。例如光学字符识别、语音识别。
                机器翻译：
                结构化输出：上述的混合任务。
                异常检测：在一组事件或对象筛选，并标记不正常u或非典型的个体。信用卡欺诈检测。
                合成和采样：生产一些和训练数据相似的新样本。
                缺失值填补：
                去噪：输入损坏后的样本，得到干净的样本。
                密度估计或概率质量函数估计：
            当然还有很多其他同类型或其他类型的任务。
        
        性能度量：准确率

        经验：大致说来，无监督学习涉及观察随机向量x的好几个样本，试图显式或隐式地学习出概率分布p（x），或者是该分布一些有意思的性质；
            而监督学习包含观察随机向量x及其相关 联的值或向量y，然后从x预测y，通常是估计p（y｜x）。
            术语监督学习（supervised learning）源自这样一个视角，教员或者老师提供目标y给机器学习系统，指导其应该做什么。
            在无监督学习中，没有教员或者老师，算法必须学会在没有指导的情况下理解数据。
        传统上，人们将回归、分类或者结构化输出问题称为监督学习，将支持其他任务的密度估计称为无监督学习。

    大部分机器学习算法简单地训练于一个数据集上。数据集可以用很多不同方式来表示。在所有的情况下，数据集都是样本的集合，而样本是特征的集合。
    表示数据集的常用方法是设计矩阵（design matrix）。设计矩阵的每一行包含一个不同的样本。每一列对应不同的特征。
    通常在处理包含观测特征的设计矩阵X的数据集时，我们也会提供一个标签向量y，其中yi表示样本i的标签。
    当然，有时标签可能不止一个数。例如，如果我们想要训练语音模型转录整个句子，那么每个句子样本的标签是一个单词序列。
    
    线性回归：顾名思义，线性回归解决回归问题。换言之，我们的目标是建立一个系统，将向量作为输入，预测标量作为输出。线性回归的输出是其输入的线性函数。
            一般性能或者说损失函数为均方误差。通过减少均方误差来改进权重、偏置。
    
    机器学习的主要挑战是我们的算法必须能够在先前未观测到的新输入上表现良好，而不只是在训练集上表现良好。在先前未观测到的输入上表现良好的能力被称为泛化。
        这个就像上学，平时课堂老师将方法，同时提供课堂练习作为训练，而训练的目标是提高正确率，这叫在训练集上表现良好;但真正的目的是最后的考试，考试正确率高。
    以下是决定机器学习算法效果是否好的因素：
        （1）降低训练误差。
        （2）缩小训练误差和测试误差的差距。
        这两个因素对应机器学习的两个主要挑战：欠拟合（underfitting）和过拟合（overfitting）。
        欠拟合是指模型不能在训练集上获得足够低的误差，而过拟合是指训练误差和测试误差之间的差距太大。
    通过调整模型的容量（capacity），我们可以控制模型是否偏向于过拟合或者欠拟合。
    通俗来讲，模型的容量是指其拟合各种函数的能力。
    容量低的模型可能很难拟合训练集。
    容量高的模型可能会过拟合，因为记住了不适用于测试集的训练集性质。
    
    监督学习算法是给定一组输入x和输出y的训练集，学习如何关联输入和输出。
    在许多情况下，输出y很难自动收集，必须由人来提供“监督”，不过该术语仍然适用于训练集目标可以被自动收集的情况。
    SVM（支持向量机）是监督学习中最有影响力的方法之一。
    
    几乎所有的深度学习算法都用到了一个非常重要的算法：随机梯度下降（stochastic gradi-ent descent，SGD）。
    机器学习中反复出现的一个问题是好的泛化需要大的训练集，但大的训练集的计算代价也更大。
    
    几乎所有的深度学习算法都可以被描述为一个相当简单的配方：特定的数据集、代价函数、优化过程和模型
        
### 3.深度前馈网络

    前馈神经网络之所以被称作网络（network），是因为它们通常用许多不同函数复合在一起来表示。
    该模型与一个有向无环图相关联，而图描述了函数是如何复合在一起的。

    一种理解前馈网络的方式是从线性模型开始，并考虑如何克服它的局限性。线性模型，例如逻辑回归和线性回归，是非常吸引人的，
    因为无论是通过闭解形式还是使用凸优化，它们都能高效且可靠地拟合。线性模型也有明显的缺陷，那就是该模型的能力被局限在线性函数里，
    所以它无法理解任何两个输入变量间的相互作用。

    训练算法几乎总是基于使用梯度来使得代价函数下降的各种方法。

    任何时候，当我们想要表示一个具有n个可能取值的离散型随机变量的分布时，都可以使用softmax函数。
    softmax函数最常用作分类器的输出，来表示n个不同类上的概率分布。

    整流线性单元、激活函数

## 20230613

    Gorgonia库和《深度学习》中文译版都是很好的理论学习的东西，但是数学思维不强的，看这个很容易陷进去，搞的大脑乱糟糟的。
    像有位学者说的，理论都是大神们去搞的东西，我们还是乖乖的用现成的，安静的按大神指导好的路走就好了。
