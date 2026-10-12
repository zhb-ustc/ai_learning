# 图解Mixtral 8 * 7b推理优化原理与源码实现

> 作者：猛猿  
> 来源：知乎专栏（本地网页存档转换）  
> 原文：https://zhuanlan.zhihu.com/p/691066049

大家好，在写这篇文章时，本来是想打算介绍Mixtral 8 * 7b具体模型架构的。但是代码读着读着就发现：

- 最精彩的MoE部分，其相关原理在之前的文章中已经详细介绍过
- 整体来看Mixtral 8 * 7b的模型架构代码，写得非常清楚，几乎没有理解难点。

**就在我以为Mixtral的代码已无更多可写时，我注意到了它在推理时用到的一些加速trick**，具体为：  

- **Sliding Window Attention**(SWA，滑动窗口Attention)
- **Rolling Buffer Cache**（也被称为R**otating Buffer Cache，即旋转式存储的KV cache**）
- **Long-context Chunking**（长上下文场景下的chunking策略，配合前两者食用）

**这些trick的构思比较巧妙，同时代码实现并不好读，（特别是最后两个trick），表现在**：  

- 没有注释。偶有注释举例的地方，例子举得并不好（进入了代码中assert非法分支，不适合用来做代码讲解。所以本文会给出更合适的例子做讲解）
- 变量、class等命名较为晦涩
- 所依赖的外部包（例如Xformers库）的官方文档给的介绍不够清晰
- 逻辑较复杂

所以在这篇文章中，**我们就把焦点放在“Mixtral加速推理”这一块上**，同样通过图解的方式，把代码的运作流程串起来，帮助大家更好阅读源码。在本文的最后一部分，给出一些源码阅读的hint（可能是大部分朋友在读Mixtral代码时感到最痛的点）。

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

---

## 一、LLM推理的两阶段

一个常规的LLM推理过程通常分为两个阶段：**prefill和decode**。  

![图](<images/图解Mixtral 8 _ 7b推理优化原理与源码实现/v2-f8200839801383287291d9af1042458d_1440w.jpg>)

###  1.1 Prefill

**预填充阶段**。在这个阶段中，我们**把整段prompt喂给模型做forward计算**。如果**采用KV cache技术，在这个阶段中我们会把prompt过**$$W_{k}, W_{v}$$**后得到的**$$X_{k}, X_{v$$**保存在cache_k和cache_v中**。这样在对后面的token计算attention时，我们就不需要对前面的token重复计算$$X_{k}, X_{v$$了，可以帮助我们节省推理时间。  
  
在上面的图例中，我们假设prompt中含有3个token，prefill阶段结束后，这三个token相关的KV值都被装进了cache。

### 1.2 Decode

**生成response的阶段**。在这个阶段中，**我们根据prompt的prefill结果，一个token一个token地生成response**。  
同样，如果采用了KV cache，则每走完一个decode过程，我们就把对应response token的KV值存入cache中，以便能加速计算。例如对于图中的t4，它与cache中t0~t3的KV值计算完attention后，就把自己的KV值也装进cache中。对t6也是同理。  
**由于Decode阶段的是逐一生成token的，因此它不能像prefill阶段那样能做大段prompt的并行计算，所以在LLM推理过程中，Decode阶段的耗时一般是更大的**。

## 二、Sliding Window Attention

### 2.1 原理

从第一部分的介绍中，我们应该能感受到一点：**LLM推理中的KV cache加速法，是非常典型的用“空间换时间”的操作**。随着seq_len变长，cache中存储的数据量也越来越大，对显存造成压力。  
  
所以，我们自然而然想问：有什么办法能减缓cache的存储压力呢？  
  
注意到，**cache的存储压力之所以变大，是因为我们的Attention是causal decoder形式的，即每一个token，都要和它之前所有的token做Attention**，所以cache中存储的数据量才和seq_len正相关。如果现在我们转换一下思路，**假设每一个token只和包含其本身在内的前W个token做Attention，这样不就能把cache的容量维持在W吗**？而从直觉上来说，这样的做法也有一定的道理：**对当前token来说，距离越远的token，能提供的信息量往往越低**，所以似乎没有必要浪费资源和这些远距离的token做Attention。  
  
这种Attention思路的改进，就被称为"**Sliding Window Attention"**，其中W表示窗口长度。这也是Mixtral 7b 和Mixtral 8 * 7b采用的方法，我们通过作者论文中的一张图，更清晰地来看下它和传统Attention的区别，这里W=3  
  
  

![图](<images/图解Mixtral 8 _ 7b推理优化原理与源码实现/v2-078cde0db78bcddce2ac624079b9e414_1440w.jpg>)

###   2.2 为什么能用滑动窗口

虽然滑动窗口的策略看起来很不错，**不过你一定有这样的疑惑：虽然距离越远的token涵盖的信息量可能越少，但不意味着它们对当前token一点用处都没有**。在传统的Attention中，我们通过Attention score，或多或少给这些远距离的token一定的参与度；但是在Sliding Window Attention中，却直接杜绝了它们的参与，这真的合理吗？  
  
为了回答这个问题，我们来看一个例子，在本例中W=4，num_layers = 4，num_tokens = 10。  

![图](<images/图解Mixtral 8 _ 7b推理优化原理与源码实现/v2-673c0328df13ab9de3619bc72ccf2fb7_1440w.jpg>)

我们从layer3最后一个位置的token（t9）看起：  

- 对于`layer3 t9`，它是由`layer2 t9`做sliding window attention得来的。也就是`layer3 t9`能看到`layer2 t6 ~ t9`的信息
- 再来看`layer2 t6`，它能看到`layer1 t3 ~ t6`的信息。也就是说对于`layer3 t9`，它最远能看到`layer1 t3`这个位置。
- 以此类推，当我们来到layer0时，不难发现，对于`layer3 t9`，它最远能看到`layer0 t0`这个位置的信息。

欸你发现了吗！**对于`layer3 t9`，虽然在每一层它“最远”只能看到前置序列中部分token，但是只要模型够深，它一定能够在某一层看到所有的前置tokens。**  
  
**如果你还觉得抽象，那么可以想想CNN技术中常谈的“感受野”**。当你用一个固定大小的卷积窗口，对一张原始图片做若干次卷积，得到若干张特征图。越深的特征图，它的每一个像素点看到的原始图片的范围越广。**类比到我们的滑动窗口Attention上，从layer0开始，每往上走一层，对应token的感受野就往前拓宽W。**  
  
**所以，Silding Window Attention并非完全不利用窗口外的token信息，而是随着模型层数的增加，间接性地利用起窗口外的tokens。**

## 三、Rolling Buffer Cache

###  3.1 原理

当我们使用滑动窗口后，KV Cache就不需要保存所有tokens的KV信息了，**你可以将其视为一个固定容量（W）的cache，随着token index增加，我们来“滚动更新” KV Cache。**  
  
下图给出了Rolling Buffer Cache的运作流程:  

![图](<images/图解Mixtral 8 _ 7b推理优化原理与源码实现/v2-f2c610170aa78612c6915dec7ff63651_1440w.jpg>)

在图例中，我们做推理时喂给模型一个batch_size = 3的batch，同时设W = 3。此时KV Cache的容量为`(batch_size, W)`。我们以第1条prompt`This is an example of ...`为例：  

- 在i时刻，我们对`an`做attention，做完后将`an`的KV值更新进cache中
- 在 i + 1时刻，我们对`example`做attention，做完后将`example`的KV值更新进cache中。此时对于第1条prompt，它在KV cache中的存储空间已满。
- 在 i + 2时刻，我们对`of`做attention，由于此时KV cache已满，所以我们将`of`的KV值更新进KV cache的0号位置，替换掉原来`This`的KV值。再后面时刻的token也以此类推。
- **不难发现，prompt中第i个token在KV cache中的存储序号为：`i % W`**

###   3.2 “旋转”从何而来

**如果你读过Mixtral的源码，你可能会记得，在源码中管`Rolling Buffer Cache`叫`Rotary Buffer Cache`**。而“Rotary”这个词很值得我们关注：为什么叫“旋转”呢“  
  
  
我们再回到3.1的图例中：

![图](<images/图解Mixtral 8 _ 7b推理优化原理与源码实现/v2-9f0118745720d2e33e42dead8f360f37_1440w.jpg>)

还是对于第一条数据，我们往上添两个单词，假设其为`This is an example of my last...`。现在来到了单词`last`上，我们需要对它计算Sliding Window Attention。  
  
  
不难理解，在W=4的情况下，`last`的Attention和`example of my last`相关。**现在我们把目光放到图中的KV Cache上：它的存储顺序似乎不太对，如果我们想对`last`做Attention，就要对当前KV Cache中存储的元素做一次“旋转”，将其转回正确的位置。**  
  
  
所以，**Rotary的意思就是：通过某种规则，将Cache中的数据旋转回正确的位置，以便能正确做Attention**。这个规则在Mixtral源码中用一个`unrotate`函数来定义。在后文我们会详细看这个函数的运作方式。

## 四、Chunking

我们回忆一下目前为止Mixtral为了加速模型推理做的操作：

- 使用KV Cache，加速Decode过程
- 使用Sliding Window Attention和Rolling Buffer Cache，降低KV Cache存储压力

你可能已经发现，**这些以“空间换时间”的优化，都是针对Decode过程的。那么对于Prefill过程，我们能做什么优化呢？**  
  
  
相比于更耗时的Decode阶段，**Prefill有一个更加突出的问题：long-context。过长的prompt会给显存带来压力**。一个**符合直觉的解决办法是：把prompt切成若干chunk，每次只喂给模型1个chunk，更新1次KV Cache**。这样我们虽然牺牲了一些Prefill计算的并行性（所有tokens一起计算），却能帮助我们节省显存压力（尤其是在采用sliding window attention的情况下，KV Cache的尺寸是固定的而不是随seq_len增长时）。  
  
  
一般情况下，我们设<strong>`chunk_size = cache_window = sliding_window = W`</strong>，也就是chunk和cache的尺寸都和滑动窗口的尺寸保持一致，都设为W。对这个参数设置我们再说明下：一般满足cache_window = sliding_window，这个不难理解，因为cache中存的是attention感受野范围内的token。而chunk_size可以不等于这两者（源码中也提供了相关处理）。只是chunk_size和这两者相等时，无论是从计算逻辑还是空间利用率上，都是更好的选择（现在觉得抽象没关系，后文会提供具体的图例，大家可以感受下）。  
  
  
好，现在我们来看一个chunking的图例（来自Mixtral论文），假设输入的prompt为`The cat sat on the mat and saw the dog go to`，同时`chunk_size = cache_window = sliding_window = 4`  

![图](<images/图解Mixtral 8 _ 7b推理优化原理与源码实现/v2-e3ca4069c69fa9a54e5bf4bbebf30803_1440w.jpg>)

**假设我们现在来到第三块chunk**，它包含的词为`the dog go to`。我们要对这个chunk中的每一个token计算滑动窗口Attention，同时把每个token的Xk, Xv值更新进KV Cache。

- **图中row方向表示Xq**，即你可以把row方向`the dog go to`的每一个token，当成是这个token过Wq后的Xq值
- **图中col方向表示Xk, Xv**，即你可以把col方向`The cat sat on the mat and saw the dog go to`的每一个token，当成是这个token过Wk，Wv后的Xk，Xv值，这些值存储在KV Cache中
- **图中整个0/1数据块表示mask矩阵**。它表示row方向的Xq应该和col方向的哪些Xk，Xv值做attention。

现在我们已基本能理解这张图的含义，**不过还有一点很奇怪：在这个图下的Past, Cache, Current表示什么意思呢**？  
  
**我们牢记一点：只有1个KV cache（也可以理解成只有1个用于存放Xk值的cache_k，和1个用于存放Xv值的cache_v）**。当我们遍历到某个chunk时，我们取出当前的cache和这个chunk做attention计算，然后再把这个chunk相关的KV值按Rolling Buffer Cache的方式更新进这个cache中。  
  
回到我们的例子上，现在我们位于第3块chunk上，此刻cache中存储的Xk, Xv值，即是上图**中间块**维护的`the mat and saw`，**因此只有中间块的最底下被标上了“cache”，因为它才是此时真正的cache**。而最左侧**past块**维护的则是**前一个时刻**的cache。最右侧的**current块**维护的`the dog go to`是**即将被更新进cache的Xk, Xv值**。这就是past, cache和current的含义。  
  
注意到虽然图中画出了past块，但这并不意味着计算第3块时要把past块也取出（此时past块代表的cache早就被更新了）。论文中这样画只是更方便我们了解cache更新迭代和计算的过程。（悄悄吐槽下，虽然论文中的这些图画得很好很精练，但是少了很多关键信息的文字介绍，容易给人造成似懂非懂的感觉）

## 五、Chunking推理全流程图解

**在我们介绍代码之前，我们先用图解的方式把整个推理流程串一遍，好知道代码在做一件什么事情**（毕竟这块代码确实不好读）。然后我们再来看代码的细节。

### 5.1 输入数据

假设推理时`batch_size = 3`，且有`chunk_size = cache_size = sliding_window = 4`，则这个batch的prompts可表示成下图（每个方块表示1个token，同色方块属于同个prompt）：  

![图](<images/图解Mixtral 8 _ 7b推理优化原理与源码实现/v2-7518b49cc0710750da8372be3e0d60f2_1440w.jpg>)

###  5.2 整体流程

**（1）chunk0**  
  
  

![图](<images/图解Mixtral 8 _ 7b推理优化原理与源码实现/v2-83261e0282c45ca4b1a604cd26320370_1440w.jpg>)

- 我们首先将chunk0送入模型，此时KV cache为空
- **对chunk中的每个token计算Xq，Xk，Xv，用于计算SWA（Sliding Window Attention）。图中刻画了计算时用到的mask矩阵**。在Mixtral源码中使用**Xformers库**的相关API来完成Attention相关的计算（这个库的好处是加速Attention计算）。**BlockDiagonalCausalMask**（全称是BlockDiagonalCausalLocalAttentionMask）是这个库下提供的一种mask方法，它可以这样理解：
  - **block**：将矩阵进行分块（block），之后在每一个块内**单独**做Attention计算
  - **diagonal causal**：每一个block内做对角线mask

Xformers官方文档在这一块的介绍不太全面，对初次使用Xformers的朋友其实不太友好，所以在这里我做了可视化，方便后续大家对代码的理解。

- **chunk0的SWA计算完毕后，我们将每个token对应的Xk, Xv值存入cache**。在源码中，我们会通过一个规则确定每个token的KV值在KV cache中的存储位置，这样也方便我们做unrotate操作（见本文3.2部分）时能把cache中存储的元素旋转回正确的位置。
- 最后，对于KV cache，它的**position序号**的排布顺序是从左至右，从上到下的，即：

```
0 | 1 | 2  | 3
4 | 5 | 6  | 7
8 | 9 | 10 | 11
```

**（2）chunk1**  
  
  

![图](<images/图解Mixtral 8 _ 7b推理优化原理与源码实现/v2-0fdbd610f3b31d5cb77972a611c81476_1440w.jpg>)

- 对于chunk1中维护的tokens，我们正常计算他们的xq，xk，xv。
- **取出当前KV Cache中存储的KV值，和chunk计算出来的KV值进行拼组，计算SWA**（如图所示，mask矩阵的row行，每个色块由两部分组成：当前cache + 当前chunk）
- 在计算SWA的mask矩阵时，我们同样采用Xformers库，这时调用的是<strong>`BlockDiagonalCausalLocalAttentionFromBottomRightMask`</strong>类，和chunk0调用的`BlockDiagonalCausalLocalAttentionMask`相比，它的主要不同在“FromBottomRight”上，也就是**对于每个block，它从右下角开始以窗口长度为W（本例中W=4）的形式设置mask矩阵。**
- 计算完chunk1的SWA后，我们将chunk1的KV值更新进KV Cache中

**（3）chunk2**  
  
  

![图](<images/图解Mixtral 8 _ 7b推理优化原理与源码实现/v2-2ac49063939519f44b05b0a9332b9506_1440w.jpg>)

最后我们来看chunk2，这个chunk比较特殊，因为在这个chunk内，每一个prompt维护的序列长度是不一样的，3个prompt维护的tokens分别为`[[8, 9, 10, 11], [8, 9], [8]]`。  

- 同样，我们计算chunk2的每个tokens的Xq，Xk，Xv
- 取出当前KV cache，与chunk2的相关结果做Attention计算，依然是采用Xformers的<strong>`BlockDiagonalCausalLocalAttentionFromBottomRightMask`</strong>类
- 把chunk2计算的KV结果更新进KV Cache。我**们特别关注第2、3条prompt（绿红色块）更新后的KV cache结果**。按照3.1中rolling buffer cache设置的放置方式，**这两条prompt中KV值是非顺序存放的**。例如对于第2条prompt，它KV值的存放顺序是`[8, 9, 6, 7]`。因此如果我们想继续对它做decode，就要把KV cache的值`unrotate`回`[6, 7, 8, 9]`，以此类推。

**事实上，无论是prefill还是decode，无论是哪个chunk，只要涉及到用当前cache和chunk（在decode阶段则是token）做attention计算，我们都需要把cache中的KV值排布`unrotate`一遍**。`unrotate`的结果就是，如果cache中的值已经是按顺序排布的，那就照常输出；如果是非顺序排布的，那就排好了再输出。由于在Mixtral源码中，这块数据处理逻辑比较复杂，又没有写注释，所以很多朋友读到`unrotate`的部分可能一头雾水。因此这里特地画出，帮助大家做源码解读。

### 5.3 一个新例子：chunk_size != W

在前文我们说过，一般设`chunk_size = cache_window = sliding_window`，我们也说过这个设置并不绝对，一般cache_window和sliding_window相等，但是chunk_size却不一定要和它们相等。  
  
**所以我们来看一个chunk_size和其余两者不等的例子。在这个例子中，`chunk_size = 5, cache_window = sliding_window = 3`**  
  
prompt的划分如下图：  

![图](<images/图解Mixtral 8 _ 7b推理优化原理与源码实现/v2-3d7c1e185567bbbb2f180f47b0f58f9c_1440w.jpg>)

和5.2中的示例一样，**对于每个chunk都主要分成三个阶段：更新前的KV Cache，SWA，更新后的KV cache**。其中前两个阶段和5.2的示例差别不大，**我们主要来关注下第三个阶段：更新KV Cache**。  
  
  
不难理解，对于每个chunk来说，只有倒数W个token的KV值才应该进KV cache。例如**对prompt0的chunk0，我们自然而然会认为用它更新KV cache后，KV cache中token的排布应该是`[2, 3, 4]`，但真的是这样吗？**  
  
  

![图](<images/图解Mixtral 8 _ 7b推理优化原理与源码实现/v2-dcc9d5fbbdb31e9f0859c488c5f6f165_1440w.jpg>)

上图显示了prompt0的不同chunk更新KV cache后的结果，**可以发现，chunk0更新KV cache后，元素的排布方式是`[3,4,2]`（而不是我们认为的`[2,3,4]`）；chunk1更新KV cache后，元素的排布方式是`[9, 7, 8]`（而不是我们认为的`[7, 8, 9]`）。这是因为整个更新过程严格遵循第三部分的Rolling Buffer Cache的更新原则**（这样我们才能使用一套unrotate准则应对chunk_size等于和不等于cache_window/sliding_window的情况）。详细的更新过程已经在图例中画出。  
  
  
**同样，我们每次在使用KV Cache计算Attention时，也要注意用unrotate方法将KV Cache中的元素先按顺序排布好**。

## 六、一些关于源码的hint

在写这篇文章时，本来是打算把源码一起讲的。但是写到这里发现，其实代码中最难理解的部分，在这篇文章中已经做了可视化了，剩下的代码细节对读者们来说应该没难度。在这里再给一些hint（应该也是读者最难理解的part）：

- **代码中的`RotatingBufferCache`类，用来定义一个KV cache**。从始至终只有1个KV cache（或理解成1个cache_k + 1个cache_v），它在prefill和decode阶段不断被更新
- **代码中`CacheView`类，用来操作KV cache（正如它的命名一样，它是cache的视图）**。如果说`RotatingBufferCache`用来管理cache的结构，那么`CacheView`则对cache中的具体数据进行更新、排序等操作。
- **代码中`RotatingCacheInputMetadata`类，用来定义如何生成当前chunk的KV cache信息**。从上面的例子中我们知道，当前chunk计算出的KV值是要被更新进KV cache中的，那么chunk中的哪些token要被更新进KV cache中（例如chunk_size != sliding_window/cache_window时，只有倒数W个token要被更新进KV cache中）？这些token的KV值在cache中要存放在什么位置？诸如此类的信息，我们都在`RotatingCacheInputMetadata`中定义。
- **代码中`unrotate`方法，用来定义如何把KV cache中的元素正确排布，以便做Attention**
- **代码中`interleave_list`方法，用来定义Attention mask矩阵中的col方向元素排布（例如5.2（2）中的中间部分的图**）。interleave是“**交织**”的意思。什么是“交织”呢？就是`prompt0 cache + prompt0 chunk + prompt 1 cache + prompt1 chunk + prompt2 cache + prompt2 chunk`这样插入式交替排布的意思。
