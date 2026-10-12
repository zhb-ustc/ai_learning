# 图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills

> 作者：猛猿  
> 来源：知乎专栏（本地网页存档转换）  
> 原文：https://zhuanlan.zhihu.com/p/710165390

在[分离式推理架构1](https://zhuanlan.zhihu.com/p/706761664)中，我们以DistServe为例，**解释了“为何要使用分离式推理架构”：分离式推理架构可以解耦prefill（compute-bound）和decode（memory-bound）过程，使得不管是在硬件分配还是在并行策略上，这两者都能朝着独立的方向优化，同时改进TTFT和TPOT，而无需再像合并式推理架构那样，总是在这两者之间做trade off**。  
  
  
但是，读完这篇文章，**你可能会有这样的疑惑：如果我能采取一种方法，使得处于prefill阶段的请求和处于decode阶段的请求能组成一个batch同时计算**，而在组建这样的batch的过程中，我又充分考虑了**最大化GPU计算单元利用率、最小化IO读写次数**（简而言之，怎么能榨干一块gpu我就怎么来）。**那么这时，我是不是在不解耦的情况下，同样也能同时保全TTFT和TPOT呢？**  
  
  
那么在这篇文章中，我们就来看看遵从这个思路设计的推理架构：**Sarathi-Serve**，以及它背后的核心技术**chunked-prefills（切块式prefill**）和**stall-free schedules（无停滞式调度策略）**。虽然本文是讲Sarathi-Serve，但是为了更好理清其设计思路（它也是在借鉴了其余架构的基础上改良而来），本文也会涉及对其余架构的核心技术讲解，并依次回答两个重要问题：  

- **为什么有了selective batching还可能需要chunked-prefills**
- **为什么有了chunked-prefills还可能需要分离式架构**

**【大模型计算加速系列】**

**[猛猿：图解大模型计算加速系列：FlashAttention V1，从硬件到计算逻辑](https://zhuanlan.zhihu.com/p/669926191)**

**[猛猿：图解大模型计算加速系列：Flash Attention V2，从原理到并行计算](https://zhuanlan.zhihu.com/p/691067658)**

**[猛猿：图解Mixtral 8 * 7b推理优化原理与源码实现](https://zhuanlan.zhihu.com/p/691066049)**

**[猛猿：从啥也不会到CUDA GEMM优化](https://zhuanlan.zhihu.com/p/703256080)**

**[猛猿：图解大模型计算加速系列之：vLLM核心技术PagedAttention原理](https://zhuanlan.zhihu.com/p/691038809)**

**[猛猿：图解大模型计算加速系列：vLLM源码解析1，整体架构](https://zhuanlan.zhihu.com/p/691045737)**

**[猛猿：图解大模型计算加速系列：vLLM源码解析2，调度器策略(Scheduler)](https://zhuanlan.zhihu.com/p/692540949)**

**[猛猿：图解大模型计算加速系列：vLLM源码解析3，块管理器BlockManager（上篇）](https://zhuanlan.zhihu.com/p/700780161)**

**[猛猿：图解大模型计算加速系列：vLLM源码解析3，Prefix Caching](https://zhuanlan.zhihu.com/p/707228704)（BlockManager下篇）**

**[猛猿：图解大模型计算加速系列：分离式推理架构1，从DistServe谈起](https://zhuanlan.zhihu.com/p/706761664)**

**[猛猿：图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills](https://zhuanlan.zhihu.com/p/710165390)**

**【历史文章汇总】**

**[猛猿：【必看】历史技术文章导航](https://zhuanlan.zhihu.com/p/654910335)**

**【如有帮助，欢迎大家点赞收藏和在看～～】**

---

##  一、传统batching方式

### 1.1 整体流程

我们来看早期一个传统的batching方式的例子（例如FasterTransformer的实现，图片来自Orca论文）：  

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-930af9ffcd19c0bf471fdb03802903fc_1440w.jpg>)

在这个例子中，**我们的batch_size = 2，分别装着长度相等的x1和x2序列（长度不相等时，可以采用诸如左侧padding等方法）。**  

- 我们把（左padding过后）长度相等的序列送入模型做prefill，产出第一个token。**整个prefill的过程，被称为1次iteration（中文可以理解成一次迭代，或者1个推理阶段）。**
- 接下来我们对这两个序列做decode。可以发现1次迭代后，x2已经推理完毕，x1依然还在做推理
- **由于在传统batching方法中，整个batching中的序列是一起行动的，所以尽管x2已经做完推理了，它还是没有办法被“释放”。“释放”的含义是：x2所占据的资源（例如KV cache等）不能被释放。**
- 接下来，x1又做了两次迭代。这下x1也完成推理了。然后整个batch中的数据才可以被真正“释放”。
- **当这一个batch推理完毕后。其余请求才能继续组成新batch，做下一轮推理。**

正是由于在传统batching中，需要所有的request一起行动，**因此和传统batching配套的调度方式，又被称为request-level schedules**

### 1.2 传统batching方式的缺陷

由1.1的整体流程，我们可以直观看出传统batching方式的缺点：  

1. **以牺牲TTFT的方式保全TBT（Time Between Tokens，可以理解成和TPOT是等价的）**。由于整个batch一起行动，所以在这个batch做推理的过程中，不能接受新的请求，**导致prefill的过程停滞了(stall)**。所以尽管它一气呵成完成了现有数据的decode过程，它却增加了新请求们在队列中等待被处理的时间。
2. **以牺牲吞吐（throughput）的方式降低延迟（latency）**。由于不能接受新请求，吞吐量下降了，但是由于不间断地做decode，对decode来说延迟降低了。
3. **增加了流水线并行中的气泡**。

我们对第3点做一些更详细的说明。

在大模型推理中，当模型尺寸过大时，我们需要把它切割到多张卡上，常用的并行方式有pp和tp（这里我们不谈dp，因为确认好tp和pp后，dp维度只是做模型副本拷贝而已）。**一般来说，在做推理时，我们希望用一个较大的batch，这样一来我们可以最大化利用gpu的计算单元，二来也减少从显存读取数据到cache的次数**（比如同样是从显存中读取模型权重，如果你分成很多小batch，你就要读取多次。当你合成大batch时，你只用读取1次，大家共享就可以了）。  

- **当我们使用tp时，我们是对模型做层内切割**，这样一块卡上维护的模型权重占的显存就少了，我们就有空间组织更大的batch了。但是由于tp在前向过程中涉及到2次allreduce，所以它对不同gpu间的通讯性能要求更高。**因此一般是在单机内，或者在有更好带宽的集群的情况下，我们会倾向于使用tp。**
- **当我们使用pp时，我们是对模型做层间切割**，一块卡上维护的还是完整的层，虽然此时可能batch无法像tp那样打得比较大，**但是pp间只涉及层间activation的通讯，对带宽要求更小。所以很多商用的架构都会使用pp作为推理的并行方式。**

**那么如果使用pp做推理，有一个优化点肯定是避不开的：减小pp的bubble，也就是减少gpu的空闲时间**。  

---

**All-Reduce：多卡之间“求和再同步”的通信操作**

**All-Reduce** 是一种分布式通信操作。  
它的目标是：把多张 GPU 上各自持有的数据先做一次聚合，比如求和或求平均，然后把聚合后的完整结果再发回给每一张 GPU。

可以把它理解成一个团队对账：

1. 每个 GPU 手里都有一部分数据；
2. 大家把这些数据按位求和/求平均；
3. 最后每个 GPU 都拿到同一份“全局结果”。

在 TP 里，模型被切成多块并行计算，每张卡只算出一部分结果。要得到完整的输出，就需要把各卡的部分结果合并起来，这个合并过程通常就会用到 All-Reduce 或类似的集合通信。  
所以 TP 对卡间带宽要求高：如果 All-Reduce 慢，整个推理就会被通信拖住。

**Activation：神经网络层与层之间的中间输出**

**Activation** 通常指“激活值”，也就是神经网络中某一层计算后的输出。

一层神经网络的典型计算是：

$$
z = Wx + b
$$

这里 $z$ 是线性变换后的结果，再经过激活函数，比如 ReLU、Sigmoid、GELU 等，得到：

$$
a = \sigma(z)
$$

这个 $a$ 就是 activation。

它有两个作用：

- **作为下一层的输入**，让数据逐层向前传播；
- **引入非线性**，让模型能表达更复杂的关系。

在 PP 里，不同 GPU 负责不同层，所以 GPU 之间需要传递的就是这些 activation。比如 GPU0 算完第 1 层，把 activation 发给 GPU1，GPU1 再拿它继续算第 2 层。  
这种通信量通常比 TP 里的 All-Reduce 小，因为传递的是层间中间结果，而不是整个矩阵乘法的聚合结果。

**放到 TP / PP 里看**

- **TP 里的 All-Reduce**：多卡共同算一个层，算完后要合并结果，所以通信量大、对带宽敏感。
- **PP 里的 activation 传递**：卡与卡之间只传层间输出，通信量相对小，但容易出现流水线空闲，也就是 bubble。

所以原文说“TP 需要 2 次 all-reduce，对通信要求更高；PP 只涉及 activation 通信，对带宽要求更小”，本质上就是在比较这两种并行方式的数据流动和通信压力。

---
  
  
我们来看传统batching方式下的pp bubble情况，如下图（图片来自Orca论文）：  

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-17771fea328d6ee61d09e19642762b3c_1440w.jpg>)

其中，batch_size = 2，它装了A和B两个序列，下标表示序列正在进行第几个迭代。我们假设A和B此时都处于decode阶段。partition1～3可以理解成是3张gpu，上面维护着模型的不同层。  
  
  
**由于decode阶段是token by token的，所以A和B必须在第1次迭代产出一个token后，才能做第2次迭代。这就造成了每块gpu上的bubble（空闲时间）**。  
  
  
看见传统batching方式的这3个缺陷，此时的你一定觉得很可惜，因为：  

- **已经做完推理的请求，为什么还要占据着资源呢？把位置让给新的请求，让新请求做prefill，旧请求继续做decode，那不是更好吗？**
- **在使用pp的前提下，我在那些气泡处，塞入新请求做prefill或者decode，不就既能把那些气泡填满，又不影响当前请求做推理吗？**

所以，**这一切都指向了两个迫切需要被改进的方向：**  

- **更改request-level的限制，让新请求和旧请求能接连不断组成新的batch（Orca iteration-level schedule）**
- **让prefill和decode能在一个batch中一起做（Orca selective batching）**

##   二、Orca：Selective Batching

### 2.1 Iteration-Level Schedule 

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-b356d282bc79cbf092895d3b7a4164f8_1440w.jpg>)

再复习一下：传统推理架构的调度流程如上图（图片来自Orca论文）。调度器（Scheduler）每次从请求队列中组织一个新的batch（如图中的x1和x2），然后与执行引擎（Execution Engine）交互做推理，等engine把这个batch的数据都做完推理并且返回给用户后，调度器才会继续从请求队列中组织新的batch。由于batch中的所有请求必须一起行动，我们管这种调度策略叫**Request-Level Schedule。**  
  
  
**而现在我们的目标是：及时检测出推理完毕的请求，将其从batch中移出，好腾出位置给新的请求**。  
**那怎么实现这点呢**？还记得我们在1.1中给出的那张推理流程示意图吗？**在那张图里，我们管请求做完prefill产出第一个token的过程叫1次iteration，请求每做一次decode也被称为1次iteration**。所以，对于一个batch内的数据，如果我是按iteration维度调度的，**也就是一个batch中的所有请求每做完1次iteration，scheduler就和engine交互一次**，去检查batch中是否有做完推理的请求，以此决定是否要更新batch，这样不就能达到我们的目的吗？**我们管这样的调度策略叫Iteration-Level Schedule**，整体流程可用下图表示（图片来自anyscale blog：[https://www.anyscale.com/blog/continuous-batching-llm-inference](https://link.zhihu.com/?target=https%3A//www.anyscale.com/blog/continuous-batching-llm-inference)）  
  
  

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-512d1a7d1750939c419b384069f820dc_1440w.jpg>)

这里，我们先不要管如何使用特殊的方法让这个batch中的数据能同时做推理（我们马上在下文讲解），只着重关注调度流程。这个batch中原始有4个序列s1～s4，黄色表示prefill tokens，蓝色表示decode tokens。左图展示了这4个序列刚做完prefill的过程。在此之后序列进入decode阶段，每生成1个token，scheduler就和engine做交互，即时检查序列的完成情况。在右图中，s3最先做完推理。此时scheduler检测到了这点，就把s3从batch中移除，再从队列里塞入新请求s5组成新batch继续做推理。s6～s7的推理过程同理可推。

### 2.2 Selective Batching

了解了iteration-level schedule后，现在我们来看一个大家都非常好奇的问题：**同一个batch中，那些形态、计算方式各异的请求，要如何同时做推理？**  
  
  
举例来说：

- prefill过程是长序列并行计算的，decode过程是token by token的
- prefill过程不需要读取KV cache，decode过程需要读取KV cache
- 对于prefill，各个请求的prompt长度是不一致的
- 对于decode，不同请求的decode token的index不一样，意味着它们计算attention的mask矩阵也不一样。

诸如此类，真是令人头大。  
  
  
**而解决这些问题的一个好思路是：尽量找到这些请求计算时的共同之处，使得计算能最大化合并。对于有差异的部分再单独处理**。这样说你可能觉得比较抽象，不要紧，我们先以一个transformer decode block为例，回顾一下序列要经过哪些计算，然后我们再慢慢讲解合并batch计算的细节。

### （1）Decoder block中的各种计算类型

（下图来自sarathi论文）  

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-66f4b672058f2670913f56b104deabf9_1440w.jpg>)

- <strong>`preproj`</strong>：即序列经过$W_{Q}, W_{K}, W_{V}$矩阵产出$Q, K, V$的过程。观察table1中给出的input和weights权重，可以发现重要的两点：
  - **preproj计算时需要从显存读取模型权重。**
  - **preproj计算时和input序列长度无关（只是在hidden_size维度上做线性转换）**

- **`attn`**：利用计算出的$Q, K, V$计算attention分数的过程，可以发现：
  - **attention分数计算时不需要从显存读取模型权重，你只需要利用算好的QKV即可**
  - **atttention分数计算时依赖mask矩阵，而不同序列的mask矩阵是不同的**

- **`postproj`**：使用$W_{O}$权重矩阵，对经过attention计算后的序列做映射，它的两个特性和preproj一致。
- **`FFN1与FFN2`**：道理同preproj/postproj，不再赘述。

我们把上面的介绍稍作提炼，得到如下重要信息：

- **`preproj/postproj/FFN1/FFN2`：做这些计算时，需要从显存读取模型权重，且这些计算和input序列长度无关。**
- **`attn`：做attention分数计算时，不需要从显存读取模型权重，且不同序列的mask矩阵不同。**

### （2）Selective Batching的计算细节

- **`preproj/postproj/FFN1/FFN2`的计算和序列长度无关，这意味着你可以把一个batch中所有的tokens都展平成一行进行计算（维护好各自的位置向量就好）。而这些计算都要读取模型权重，这意味着我们可以尽量增大batch size，使得一次读取能造福更多request，以此减少IO次数。**

- **`attn`的计算受各个序列的差异性影响（例如mask矩阵、是否需要读取KV cache），所以需要将序列拆分开独立处理，也即batch维度是重要的（cuBLAS batch matrix multiplication）。而由于attn部分本身不涉及到权重读取，因此你把序列拆分开处理，也不会在这一方面上带来额外的IO开销。**

整体流程如下（图片来自Orca论文）：  

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-5ea61c325a9f0765fe105da8bab2f195_1440w.jpg>)

在图中，序列x1和x2正在decode阶段（因此需要KV cache Manager帮它们取出KV cache），序列x3和x4正在prefill阶段，它们被组成了一个batch。在非attention的部分，batch中的7个tokens被拉平成一行进行计算（忽略了batch维度），等实际计算attention时，再split开。计算完毕后再拉平。

## 三、Sarathi-Serve：chunked-prefills

我们来小结一下目前为止的内容：

- 我们以分离式架构为引子，讨论了解耦prefill和decode过程带来的好处：能独立优化TTFT和TPOT/TBT，同时提升吞吐和降低延迟。
- 基于此，我们又产生了疑问：如果不采用解耦的方式，只是修改传统的batching里非prefill即decode的方法，在最大化榨干一块gpu的前提下，让prefill和decode能同时放在一个batch里做推理，是不是也能达到一样的效果？
- 为了解答这个问题，我们先回顾了以FasterTransformer为代表的早期batching方法：在推理的每个时刻，batch中的序列总是一起做prefill，或一起做decode。
- 接下来，我们介绍了Orca是如何能让各种请求（prefill+decode，长度不同的prefill，index不同的decode等）混合在一个batch里做同时做推理的。

关于混合batch对性能带来的提升，大家可以去看Orca论文中的实验部分（以FasterTransformer等更早期的推理架构为baseline），**这里就不赘述了。我们来看一个更有趣的问题：为什么混合batch可以带来性能上的提升？**

### 3.1 为什么混合batch可以带来性能上的提升

我们来看sarathi-serve做的一个实验（图片来自sarathi-serve论文）  

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-3d7b8c6e3429c86b905bee3c319d934c_1440w.jpg>)

左右两图分别刻画了在不同的batch size下，prefill和decode阶段的吞吐量（tokens per second，每秒能处理的tokens数量）。  

- 观察到，**对于prefill阶段来说，提升batch size时，吞吐量的有增长但不太显著**。甚至当batch size更高时（比如从4～8），还发生了吞吐量的下降。**这是因为prefill阶段是compute-bound的**，也即相比于读数时间，它消耗在计算上的时间更大（由于数据是可以边读边算的，所以我们可以大致认为总时间$T = max(T_{cal}, T_{load})）$。prefill阶段读取数据（例如从显存读取模型权重）的时间成本是固定的，但是计算时间却会随着batch中tokens的数量而增长，因此当gpu的计算单元还没有被打满时，吞吐量还可以上去；被打满时就会下降了。

- **对于decode阶段来说，提升batch size时，吞吐量增长的线性趋势非常明显。这是因为decode是memory-bound的**，也就是它花在读数上的时间更大（回想一下，当你用一个token做decode时，你其实要做的新计算很少，大部分时间你都花在读取KV cache和模型权重上）。decode阶段的算力严重打不满，所以当你增大batch size时，你不仅能多利用算力，也能把多次读取合并成一次读取，吞吐量自然就上升显著了。但是你也不能无止尽地增加batch size，因为gpu的存储是有限的，decode还要读取前面那一长串的KV cache呢。

既然decode和prefill阶段都需要读一些固定的数据（比如模型权重），且decode阶段的算力没有打满，那我们把他们组装在一起，让他们互相搭便车，肯定能取得更好的效果，也即：

- **prefill搭上decode的便车，能用上decode阶段被浪费的算力。**
- **decode搭上prefill的便车，合并数据的读取次数，做到1次读取，大家共享。**

###   3.2 为什么有了selective batching，还需要chunked-prefills

在3.1中，我们介绍了prefill和decode组成混合batch对性能提升的好处：**乍一眼看，既不耽误做prefill（TTFT），也不耽误做decode（TPOT/TBT）。那么目前为止，Orca应该做得挺好了哇，那这个Sarathi-Serve的chunked-prefills，是干什么的呢？**  
  
  
**当你回顾Orca组装batching的过程时，你可能会发现这个过程比较随机**：一个batch中做prefill和做decode的请求有多少条是不确定的，只是大体按照先来后到的原则做动态组装。这就造成了一些问题：  

- **如果一个batch中做prefill的请求非常多，或者做prefill的请求非常长**，那么prefill tokens会占据大量计算资源，使得整个batch变成**compute-bound**。
- **如果一个batch中做decode的请求非常多**（比如当所有的请求都没做完推理时，或者请求队列中没有新序列可以调度时），这个batch就可能变成**memory-bound的**。
- **随机的batch同样可能产生pp并行气泡**。

哦咦，熟悉的感觉，我们再来看看第三点，还是关于pp并行气泡的问题。

我们知道相比于FasterTransformer，Orca已经能在一定程度上改善pp气泡问题了，但是由于其batch组装的随机性，它仍然可能导致气泡问题，我们以下图为例（图片来自Sarathi论文）：  

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-bd27fad6021c1453a8237b61deee5059_1440w.jpg>)

ABCD表示4个队列，下标p表示prefill阶段，di表示decode的第i个阶段。在采用micro-batch的前提下（也是减少pp气泡的一种办法），micro-batch size = 2，AB组成一个小batch，CD组成一个小batch。注**意到这两个batch虽然size一致，但tokens数量更不一致。**  
  
  
**观察到图中一共有3种类型的bubble：**  

- **`PB1`**: 因为micro-batches中prefill序列长度不一致而产生的bubble
- **`PB2`**: 因为prefill和decode阶段计算时间的差异而产生的bubble
- **`PB3`**: 不同micro-batch的decode差异性而产生的bubble，这是因为不同micro-batch在做decode时，要读取的KV cache的长度不一致，这也导致了在读取数据上所花费的时间不一致

基于Orca selective batching的这些缺陷，我们不禁想：**如果我们在保持selective batching这种混合机制的情况下，根据gpu资源的上限（FLOPS/MemBandwidth），找到一个最大batch size，即定义好一个batch内最多能处理的tokens数量，然后在每个batch内，在按照一定比例去分配做prefill的tokens和做decode的tokens，不就既能解决pp并行中的气泡问题，又能让这个batch得到性能最大化吗？**  
  
  
而在这种解决办法下，**一个请求用于做prefill的序列必定是要被拆开的**，所以我们就管这种方法为：**chunked-prefills**

### 3.3 chunked-prefills运作流程

基于pp的chunked-prefills运作流程如下（图片来自Sarathi论文）：  

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-cccd989fe5e0a851c3581032c3344ba6_1440w.jpg>)

- 首先，我们通过3.2中的思路，**从我们所使用的gpu性能出发，确定每个batch中最多能处理的tokens数量**（可以通过profiling做模拟实验得到）。
- **然后，我们在各个batch中进一步确定prefill tokens和decode tokens的比例。确认的原则被称为“decode-maximal batching"**：即优先往batch中添加需要做decode的序列，直到添加不动为止（即我们预留给decode的KV cache空间已经不足了，无法存放新的KV cache了）。然后我们再根据这个batch中剩余的tokens预算，对需要做prefill的序列做chunk切割，把对应的prefill tokens添加进batch中

**最后，Sarathi-Serve依然采用的是iteration-level schedules**，即推理的每一步后，scheduler都会重新组建batch。  
  
【我们会在本章最后一节解读Sarathi-Serve调度器策略的源码，给大家展示更多上述流程的细节，这里大家只需要大致了解chunked-prefills的运作流程即可】  
  
  
**chunked-prefills的额外开销**  
  
  
看完了运作流程，你肯定有这样的疑惑：**原来一条序列做prefill时，我是一起计算的。现在我把它拆成了多个chunk，那么每个chunk去计算时，肯定要去读前一个chunk的KV cache（如下图），那不就增加了IO复杂度了吗？这会影响到prefill计算的性能吗？**  
  
  

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-7d72c6810421bda3a18f0b7cd1bda71c_1440w.jpg>)

这个读取KV cache的额外开销肯定是有的，但它对prefill的影响大吗？基于此，Sarathi-Serve的作者们做了两个实验。  
  

**第一个实验：证明prefill阶段是强compute-bound特性，以及计算attention的时间在总计算时长里占比不高。**  
  
  

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-4575d018731b164a6ca8e7dc557c0b3a_1440w.jpg>)

我们知道KV cache仅用在attention的计算中，所以这里作者把时间消耗拆成了attention和非attention（linear + others）的部分。可以发现：

- 对于prefill的部分，不管prefill tokens数量如何，attention部分的计算时间在总时长里占比并不高。
- 对于prefill部分，随着seq_length的变长，tokens的处理时间也变长。但是在128～512的长度内，tokens的处理时间增长不显著。这是因为在这个范围内，gpu的算力还没有打满。在这之后进入强compute-bound区域，此时读取数据的时间对prefill来说影响更小。

**第二个实验：直接比较chunked-prefills和正常prefill下的延迟**  

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-73036c29eed04dbefbc3c6562a375c3a_1440w.jpg>)

这里以正常prefill为baseline（设其overhead = 1，即没有额外开销），比较不同chunk size下的额外开销。不出意外，prefill chunk分得越细（例如512），开销越大，但是总体来说，开销增长都控制在1.25倍内。稍微影响到TTFT，但是考虑到它对TBT/TPOT的更多提升（可以参见论文别的实验，这里不再写出），这样的开销还是可以接受的。

### 3.4 stall-free schedules

在Sarathi-Serve的设计思想下，**无论是prefill过程还是decode过程，都不会产生停滞（stall）**。以Sarathi-Serve作者的观点来看：在其余的推理架构中（比如vllm，Orca，FasterTransformer），他们都或多或少存在停滞一方以保存另一方的策略，我们来看一个整体流程图（图片来自Sarathi-Serve论文）：  
  
  

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-accc084db2a8e3fc975aac1e7f92d621_1440w.jpg>)

假设最开始有A、B两个序列，他们都处在decode阶段。从上帝视角来看，A和B分别要经过2次、4次decode迭代才能完成推理。  

- 对于这4个框架，A和B首先进入第1次decode迭代（图中第一个红色方块）。到这一步为止这4个框架没有什么差异。
- 当A和B完成第一次decode迭代后。新来了请求C和D。
- **对vllm**，我们在之前的源码解读系列说过，它是prefill优先的，所以它会先处理C和D，这就使得decode暂停了（stall）。这其实是在保吞吐弃延迟（使得TBT增加了）
- **对Orca**，它在硬件资源允许的情况下，是可以让CD做prefill，AB继续做decode的（黄色部分）。**但是由于decode和prefill的完整序列绑定，也使得整个decode的计算时间变长了（特别是在CD是长序列的情况下）**。所以这其实也算是一种decode暂停
- **对于FT**，它是保延迟弃吞吐的。这使得prefill暂停了。
- **对于sarathi-serve**，它和orca一样，也是允许decode和prefill一起做的，但是它通过合理控制每个batch中prefill tokens的数量，使得decode阶段几乎没有延迟（把sarathi的绿色块和FT的红色块相比，可以发现绿色块只长了一点）。这样即保了延迟，又保了吞吐。

###   3.5 Sarathi-Serve调度流程源码解析

**由于Sarathi-Serve论文中的调度流程伪代码，和实际的源码实现存在一定的差异**。所以我这里直接根据源码来分析使用chunked-prefills方法时的调度流程（给出了非常详细的注释，大家可以关注下～）：

```
class SarathiScheduler(BaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SarathiSchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config)
        
        # =================================================================
        # 【固定chunk_size策略】
        # 人为定好的chunk_size。如果你不想动态变更chunk_size大小，你可以固定使用这个。
        # 我们可以通过profiling等方式，在调度开始前确定好能够
        # saturate gpu computation的最大chunk_size
        # （注：在代码中，chunksize不是指prefill的chunksize，是指每次
        #  调度中，整个batch的tokens数量，也包括要做decode的tokens数）
        # =================================================================
        self.chunk_size = self.scheduler_config.chunk_size
        
        # =================================================================
        # 【动态chunk_size策略】
        # 使用动态变化的chunk_size
        # （随着调度次数增加，历史累积的要做decode的序列可能会变多，以及
        # 可能会进来更多的新请求。假设某个序列的prompt特别长，那么它就会持续占据着计算
        # 资源，影响到别的请求。所以对于这样的prompt，我们可以在迭代中逐渐减小它的preill
        # tokens数量）
        # 
        # 为了执行这个chunk_size动态变更的策略，我们需要如下4个参数:
        # 【low_chunk_size】：人为设定的最小chunk_size
        # 【high_chunk_size】: 人为设定的最大chunk_size
        # 【chunk_schedule_stages】：用于刻画调度阶段数。例如该值若等于5，则说明随着
        # 调度次数的增加，我们希望有5种逐步递减的chunk_size可以选择
        # 【chunk_schedule_max_tokens】: 这个变量比较难说明，我们直接看它怎么用。
        # 事实上，在源码中真正有意义的变量是_tokens_per_stage
        # （=chunk_schedule_max_tokens/chunk_schedule_stages）
        # 你可以理解成：对于一个正在做prefill的长序列，我们它的prefill tokens数量
        # 随着迭代阶段（stage）的增加而递减。我们设其做prefill时，每处理_tokens_per_stage
        # 个tokens就算完成了1个stage，然后就要递减一次prefill tokens。简而言之，这些
        # 参数的作用是帮助我们确定某个正在做prefill的序列当前位于哪个stage上
        # =================================================================
        self.enable_dynamic_chunking_schedule = (
            self.scheduler_config.enable_dynamic_chunking_schedule
        )
        # next four params apply only when using dynamic schedule
        self.low_chunk_size = self.scheduler_config.low_chunk_size
        self.high_chunk_size = self.scheduler_config.high_chunk_size
        self.chunk_schedule_max_tokens = self.scheduler_config.chunk_schedule_max_tokens
        self.chunk_schedule_stages = self.scheduler_config.chunk_schedule_stages

        if self.enable_dynamic_chunking_schedule:
            assert self.chunk_schedule_stages > 0
            assert self.chunk_schedule_max_tokens > 0
            assert self.low_chunk_size % 32 == 0
            assert self.high_chunk_size % 32 == 0
            # 计算在动态变更chunk_size的情况下，我们可选的chunk_size列表（详情参见相关函数注释）
            self._chunk_sizes = self._compute_chunk_size_schedule()
            # 用于计算每个stage能处理的token数（详细解释见上）
            self._tokens_per_stage = int(
                np.ceil(self.chunk_schedule_max_tokens / self.chunk_schedule_stages)
            )

    def _compute_chunk_size_schedule(self):
        # =================================================================
        # create num_steps equally spaced chunk sizes 
        # between low_chunk_size and high_chunk_size
        # 
        # self.low_chunk_size = 64
        # self.high_chunk_size = 256
        # self.chunk_schedule_stages = 5
        # 则chunk_sizes = [64, 108, 152, 196, 256]
        # 按照从大到小排序后 = [256, 196, 152, 108, 64]
        # =================================================================
        chunk_sizes = np.linspace(
            self.low_chunk_size,
            self.high_chunk_size,
            self.chunk_schedule_stages,
            dtype=np.int32,
        )[::-1]

        # =================================================================
        # 这里是调整每个备选的分块大小，让其能够被32整除
        # 这样做是考虑到tile-quantization effect，让gpu做gemm时的并行性能最大化
        # =================================================================
        round_of_chunk_sizes = min(32, self.low_chunk_size)
        chunk_sizes = (
            np.round(chunk_sizes / round_of_chunk_sizes) * round_of_chunk_sizes
        )
        chunk_sizes = chunk_sizes.astype(np.int64).tolist()

        return chunk_sizes

    def get_block_space_manager_class(self):
        return SarathiBlockSpaceManager

    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence, num_batched_tokens: int
    ) -> int:
        """
        对于一条还没做完prefill的seq，根据当前batch中已经存放的tokens数量，决定要送
        这个seq的多少tokens去做prefill
        """
        assert not seq.is_finished()
        # =================================================================
        # 如果使用动态chunk_size的方法
        # =================================================================
        if self.enable_dynamic_chunking_schedule:
            # =================================================================
            # 先计算当前seq目前一共处理了多少prefill tokens，然后根据每个阶段其最多能处理
            # 的prefill tokens数量，确定它在第几阶段（stage）
            # =================================================================
            request_stage_idx = int(
                np.ceil(seq.get_num_prompt_tokens_processed() // self._tokens_per_stage)
            )
            # =================================================================
            # 取出这个阶段的chunk_size
            # =================================================================
            assert request_stage_idx < len(self._chunk_sizes)
            chunk_size = self._chunk_sizes[request_stage_idx]
        # =================================================================
        # 如果没有使用动态变更chunk_size的策略，就用固定尺寸的chunk_size
        # （例如代码中的默认值512）
        # =================================================================
        else:
            chunk_size = self.chunk_size

        # =================================================================
        # 对于这个正在做prefill的seq，确定它在下一次迭代中要送去做prefill的tokens数量。
        # 这个数量 = min(该序列还没有做prefill的tokens数，batch中可用的prefill tokens配额)
        # =================================================================
        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_processed(),
            chunk_size - num_batched_tokens,
        )

        return next_num_tokens

    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        running: List[Sequence] = [] # 应该是用来存放确定要被本轮调度的数据
        ignored_seq_ids: List[str] = []
        preempted_seq_ids: List[str] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        num_batched_tokens: int = 0

        ######################################################################
        # Phase 1: Add existing running sequence groups to the batch.
        # There are two cases:
        # 1. The sequence group has incomplete prefill. The routine
        # remains identical to the one in sarathi scheduler for such sequences.
        # 2. The sequence group has completed prefill. In this case, we need to
        # check for memory availability for the next chunk of decode tokens, and preempt
        # some sequence groups if necessary. Note that, the preempted sequence groups
        # might belong to either of the two categories.
        ######################################################################

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        
        # =================================================================
        # 把self.running中的数据按照FCFS原则（先来后到）进行排序
        # =================================================================
        self.running = self.policy.sort_by_priority(now, self.running)

        # in first pass process all the requests with prefill completed
        # this allows us to accurately account for the number of decode tokens
        running_prefills: List[Sequence] = []

        # =================================================================
        # 先去看上一次iteration中被选中的序列
        # =================================================================
        while self.running:
            seq = self.running.pop(0)
            # =================================================================
            # 如果这个seq没有被暂停，那么就把它继续添加到本轮running队列中
            # 
            # （上一轮调度结束后，所有running状态的序列都会被设置为pause状态，
            # 这里可以参考base_sequence_manager的on_step_completed函数，
            # 这个函数是对每轮调度结束后序列的状态和推理结果做处理），
            # 
            # （当然也可能有别的条件会触发pause状态设置，这里没有看完全部源码，所以暂不知道）
            # =================================================================
            if not seq.is_paused():
                running.append(seq)
                continue

            # =================================================================
            # 如果这个seq还没有做完prefill，就把它添加到running_prefill的列表中
            # =================================================================
            if not seq.prompt_processing_finished:
                running_prefills.append(seq)
                continue

            # =================================================================
            # （走到这一步，剩下的都是上一次调度中处于decode阶段的seq了）
            # 如果现在没有足够的空间给处于decode阶段的seq做推理了
            # =================================================================
            while not self.block_manager.can_append_slot():
                # =================================================================
                # 如果self.running队列中有数据，就从running队列中抢占最晚到来的那个
                # sarathi中的抢占是直接做重计算，即把seq重新放回waiting队列中
                # =================================================================
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq = self.running.pop(-1)
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                # =================================================================
                # 如果self.running队列中已经没有数据了，就抢占当前seq
                # =================================================================
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    break
            # =================================================================
            # 如果现在有足够空间给处于decode阶段的seq做推理
            # =================================================================
            else:
                # 给decode阶段的seq分配KV cache空间，并将其添加到本轮的running队列中
                self._append_slot(seq)
                running.append(seq)
                # 当前batch的token数量 += 1
                num_batched_tokens += 1
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )
        
        # =================================================================
        # 接下来处理上一次调度中没有做完prefill的seq
        # 他们的KV cache空间肯定是够的，因为对于一个seq，我们在一开始是根据
        # 它完整的prefill序列长度来分配KV cache，而不是根据prefill chunk大小分配
        # KV cache。所以无论是那一轮iteration，我们都给这个seq的prefill留足了
        # KV cache空间
        # now add the requests with prefill incomplete
        # the memory for all these prefills has already been allocated
        # so we should be able to run all of them
        # =================================================================
        for seq in running_prefills:
            assert not seq.prompt_processing_finished
            # =================================================================
            # 计算对于这个seq，这次调度可以放多少tokens去做prefill
            # =================================================================
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            # as long as the request could fit in the batch previously
            # it should be able to fit in the batch now
            # so in non-pipeline case this condition should always be false
            # however, in pipeline case, the grouping of requests can change
            # between different microbatches, so this is not guaranteed to be always true
            if next_num_prefill_tokens == 0:
                running.append(seq)
                continue

            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)

        ######################################################################
        # Phase 2: Add waiting (new) sequence groups to the batch.
        # This routine is nearly-identical to the one in sarathi scheduler
        # 在phase1中，我们遍历了上一个iteration的batch，来决定有哪些seq可以继续做
        # 这一轮的推理。
        # 在phase2中，我们去waiting队列中继续搜寻，看看是否有新请求能加入这一轮推理
        # 也就是每次调度中，batch = 上一轮batch筛选后的结果 + waiting队列中筛选的结果
        ######################################################################
        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        while self.waiting:
            seq = self.waiting[0]

            # This is required to handle benchmarking where we set request arrival time ahead of time
            if seq.arrival_time > now:
                break

            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                continue

            # =================================================================
            # If the sequence group cannot be allocated, stop.
            # 直接用了vllm的allocate方法，即不是根据seq的prefill chunk大小
            # 预分配物理块的，而是直接根据整个seq的prefill大小分配物理块的
            # =================================================================
            if not self.block_manager.can_allocate(seq):
                # this is different from vllm scheduler
                # even if we cannot allocate this sequence group
                # there might be other sequence groups that can be allocated
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:
                break

            # check if we can fit the prefill in the batch
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            if next_num_prefill_tokens == 0:
                break

            seq = self.waiting.pop(0)
            self._allocate(seq) # 直接为完整的seq prefill（而不是chunk prefill）分配KV cache空间
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)

        # make sure that prefills are at the start of the batch, so that we don't violate assumptions
        # made in the original vllm codebase
        self.running = running

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )

我们可以配合着下面这张图来解读源码：
```

![图](<images/图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills/v2-cccd989fe5e0a851c3581032c3344ba6_1440w.jpg>)

总体来说，Sarathi的源码其实是基于vllm源码框架修改而来的（最新版本的vllm源码中也做了chunked-prefills的优化，等我有时间把这块写进vllm源码解读里）。注释中已经给出了所有的细节，这里额外强调几点：  

- 当整个系统刚启动时，batch中只有做prefill的序列。这时走的是源码中从waiting队列里调度的逻辑。**在sarathi中，我们是根据整个prefill的长度预先分配好KV cache空间（而不是根据prefill chunk长度来分配的）。这确保了在后面所有的iteration中，我们不用再操心这个batch中这条prefill序列的KV cache问题，它一定是留足了空间。**

- **sarathi提供了“固定”和“动态”两种chunk size策略：**
  - **在固定chunk_size策略中**，默认值为512。这是sarathi根据硬件和profiling实验计算出来的能最大化saturate gpu computation的单batch中的tokens数量。从源码中不难知道，在系统刚启动时，每个请求的头512个prefill tokens各组成一个batch（如上图所示），进行前向推理。

  - **随着推理迭代的进行，陆续有请求完成了prefill，进入decode过程**，比如上图中产出了Ad1。那么根据源码，A所在的这个batch，此时要分配1配额的tokens给Ad1继续做decode。同时，它要去waiting队列中按FCFS（先到先服务）的原则找出请求C。由于batch总tokens配额是512，所以它切割了C的511个tokens装进这个新batch中，以此类推。

  - **随着迭代的继续进行，这个batch中总有一些序列是在prefill中，有一些序列是在decode中**。每一次在做新的调度迭代时，**对于正在做decode的策略，我们会先检查当前是否有足够的KV cache空间留给他们做新一轮迭代**，如果没有的话就需要抢占decode序列（细节在源码注释中）。而对于这个batch的prefill序列，正如前文所说，当他进入这个batch的那一刻起，就已经给他分配了完整的KV cache空间，所以它无需再担心这点。

  - 可能在你的印象中，固定大小chunked-prefills意味着每个batch中prefill tokens的数量是不变的，但是通过sarathi的源码解读，你可以发现，尽量保持不变的是batch中的总tokens配额，而prefill tokens数量是随着decode tokens的增减而变动的（只不过decode tokens的数量一般也不多，所以prefill tokens数量和整体batch tokens配额也不会相差很多）

  - **在动态chunk_size策略中，我们希望对于一个请求，它的prefill tokens的数量能随着迭代次数的增加而减少，这主要是为了解决较长序列带来的影响。当一条prompt特别长时，它在每一次迭代中都会占据一定计算资源，导致历史累积的decode序列和新来的请求受到影响**。所以干脆，对于进入这个batch中的请求，在一开始我们多给它一些prefill tokens配额，然后随着迭代次数的增加，递减这个配额，降低它对别人的影响。

论文中其实做了非常多关于性能的实验，篇幅原因这里不再一一给出，大家可以自行阅读论文。

### 3.6 chunked-prefills VS 分离式推理架构

通过以上的介绍，你已经知道，在使用chunked-prefills的策略下，通过合理划分prefill tokens和decode tokens比例，最大化利用好gpu，似乎也能同时保全TTFT和TPOT/TBT。那么在这样的前提下，分离式推理架构还有什么优势呢？  
  
  
其实如果想更好回答这一点，最好的方式是做消融实验并分析。我没有做过相关的实验，所以只能从原理上给出我自己的一些猜想：  
  
  
**我觉得最主要的一点，是chunked-prefills可能还没有完全实现在达到TPOT/TBT SLO的情况下，最大化prefill阶段对GPU FLOPS的利用率（MFU）**。我们从3.3的分析中可以发现，chunked-prefills是会产生额外开销的（overhead），这个开销不仅体现在他需要额外读取KV cache，还体现在prefill chunk size的设定上。我们知道GPU的矩阵计算是存在tile-quantization effect的，也即矩阵是被切分成tiles后送到thread blocks上去做并行计算的。如果你的矩阵尺寸是tiles尺寸的整数倍数，那么就可以最大化并行计算，否则那些除不尽的部分就可能产生额外的开销（Sarathi做过相关实验，在tile_size = 128的前提下，257的矩阵尺寸比256的矩阵尺寸产生的prefill time多了32%）。而在chunk-prefill中，我们只是用profiling估算出在特定设备上一个batch的最大tokens配额而已，这些tokens包括prefill和decode。这个size是对整体的，而不是单独对prefill或decode的。所以仍然存在prefill阶段无法最大化MFU的可能。

**第二个，也是从无法最大化prefill MFU上衍生出来的问题：chunked-prefills对长序列的处理可能还差强人意。从3.5的源码解读中，我们发现在chunked-prefills中，长序列持久地占据着KV cache的存储空间以及gpu的计算资源**。尽管我们可以采用动态减少chunk_size的办法，来减少长序列的影响。但是一来，这个chunk_size递减的策略要怎么设置更合理（而不是像3.5源码中那样可能是自己凭经验拍了一个），还有待研究。二来即使是实现了更好的chunk_size递减策略，但它却使得长序列的TTFT变大了，同样影响用户体验。  
  
  
所以，基于以上这些对chunked-prefills策略缺陷的猜想，或许使用分离式架构，对prefill阶段独立开发一套策略，可能可以更加针对性地解决以上问题。当然，这也取决于各策略的具体实现、业务场景和真实的实验效果。

## 四、参考

1、[https://arxiv.org/abs/2306.02707](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2306.02707)  
2、[https://arxiv.org/abs/2308.16369](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2308.16369)  
3、[https://arxiv.org/abs/2403.02310](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2403.02310)  
4、[https://github.com/microsoft/sarathi-serve](https://link.zhihu.com/?target=https%3A//github.com/microsoft/sarathi-serve)  
5、vllm、FasterTransformer等相关资料，不一一列举
