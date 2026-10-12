# 图解大模型计算加速系列：vLLM源码解析3，块管理器BlockManager（上篇）

> 作者：猛猿  
> 来源：知乎专栏（本地网页存档转换）  
> 原文：https://zhuanlan.zhihu.com/p/700780161

大家好，本篇我们进入“有趣的”（ ，反复告诉自己它很有趣，那么它一定能变得有趣起来）且“重要的”（这是真得很重要）的vllm块管理器相关代码解读。  
  
  
vllm块管理器又分成**朴素块管理器（UncachedBlockAllocator**）和p**refix caching型块管理器（CachedBlockAllocator**）。本篇我们先讲比较简单的前者，下篇我们来细看更有趣也是更难的后者。

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

## 一、前情提要

在之前对调度器策略（Scheduler）的讲解中，我们主要说明了以下几点：  

- **从vLLM批处理的入口函数开始，介绍了其推理内核LLMEngine的两个重要函数`add_request()`和`step()`**
- **在LLMEngine开始处理请求前（实例化阶段），它会先做一次模拟实验，来估计gpu上需要预留多少显存给KV Cache block。**
- **当LLMEngine开始处理请求时(add_request)，它会把每个prompt当成一个请求，同时把它包装成一个SequenceGroup对象。**
- **当LLMEngine开始执行1次调度时（step），调度器策略(Scheduler)会选择要送哪些seq_group去做新一轮推理。注意，在1次推理中，所有seq_group要么一起做prefill，要么一起做decode。**

调度器策略流程图清晰版可参见这里：[猛猿：图解大模型计算加速系列：vLLM源码解析2，调度器策略(Scheduler)](https://zhuanlan.zhihu.com/p/692540949)  
  
  
同时，**我们遗留了以下问题**：  

- **问题1：vLLM的物理块管理（block manager）的细节**，包括物理块结构，逻辑块-物理块映射，物理块新增与释放，prefix caching等等
- **问题2：step()其余步骤**：调度器只是决定了要送哪些seq_group去做推理，但是“每1个推理阶段结束后，如何根据推理结果更新seq_group，并将其送入下一次调度”这块不是调度器的职责，这也是后面我们要讲解的“step()的其余步骤”.

今天我们就要对问题1进行解答。问题2我们放在源码解读第四篇进行讲解

## 二、两种不同类型的BlockAllocator

![图](<images/图解大模型计算加速系列：vLLM源码解析3，块管理器BlockManager（上篇）/v2-5065184e3c2d7913856c1268b5c7da9a_1440w.jpg>)

在[源码解读2](https://zhuanlan.zhihu.com/p/692540949)中，我们画过Scheduler的架构图，它的下面维护着今天我们要细讲的块管理器（BlockManager），这也是vLLM自定义的一个class。  
  
  
截止本文写作时，vLLM提供了`BlockSpaceManagerV1`和`BlockSpaceManagerV2`两个版本的块管理器。V1是vLLM默认的版本，V2是改进版本（但还没开发完，例如不支持prefix caching等功能）。所以本文依然基于`BlockSpaceManagerV1`进行讲解。  
  
  
BlockManager这个class下又维护着两个重要属性：

- **`BlockAllocator`：物理块分配者，负责实际为seq做物理块的分配、释放、拷贝等操作**。其下又分成`self.gpu_allocator`和`self.cpu_allocator`两种类型，分别管理gpu和cpu上的物理块。

- <strong>`self.block_tables`：负责维护每个seq下的物理块列表，本质上它是一个字典，形式如`{seq_id: List[PhysicalTokenBlock]}`</strong>。注意，这个字典维护着【所有】seq_group下seq的物理块，而不是单独某一个seq的。因为调度器是全局的，所以它下面的的BlockManager自然也是全局的。

**其中，BlockAllocator又分成两种类型：**  

- **`CachedBlockAllocator`：按照prefix caching的思想来分配和管理物理块**。在原理篇中，我们提过又些prompts中可能含有类似system message（例如，“假设你是一个能提供帮助的行车导航”）E）等prefix信息，带有这些相同prefix信息的prompt完全可以共享用于存放prefix的物理块，这样既节省显存，也不用再对prefix做推理。
- **`UncachedBlockAllocator`：正常分配和管理物理块，没有额外实现prefix caching的功能**。

**在块管理器的上篇中，我们介绍`UncachedBlockAllocator`，在下篇中我们介绍更为复杂的`CachedBlockAllocator`。**

## 三、物理块和逻辑块结构

首先我们来快速回顾下在vllm中一个物理块和一个逻辑块长什么样子。

```
# vllm/block.py
class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
        block_hash: int,
        num_hashed_tokens: int,
    ) -> None:
        # ==============================================================
        # 设备，gpu/cpu
        # ==============================================================
        self.device = device
        # ==============================================================
        # 该物理块在对应设备上的全局block index
        # ==============================================================
        self.block_number = block_number
        # ==============================================================
        # 该物理块的尺寸（即槽位数量，默认为16）
        # ==============================================================
        self.block_size = block_size
        # ==============================================================
        # 该物理块的hash值
        # （在prefix caching场景下使用，非此场景则附值为-1）
        # ==============================================================
        self.block_hash = block_hash 
        # ==============================================================
        # 该物理块的hash值是由多少个前置token计算而来的
        # （prefix caching场景下使用，非此场景则附值为0）
        # ==============================================================
        self.num_hashed_tokens = num_hashed_tokens 
        # ==============================================================
        # 该物理块被多少个逻辑块引用
        # ==============================================================
        self.ref_count = 0
        # ==============================================================
        # 该物理块最后一次被使用的时间
        # （prefix caching场景下使用，非此场景则附值为-1）
        # ==============================================================
        self.last_accessed = DEFAULT_LAST_ACCESSED_TIME
        # ==============================================================
        # 该物理块是否被计算过
        # （prefix caching场景下使用）
        # ==============================================================
        self.computed = False

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'num_hashed_tokens={self.num_hashed_tokens}, '
                f'ref_count={self.ref_count}, '
                f'last_accessed={self.last_accessed}, '
                f'computed={self.computed})')
```

这里有一些和prefix caching相关的物理块属性，大家现在可能还看得一头雾水，不要担心，在块管理器的下篇中我们再来细讲，这里可以忽略。  
  
  
逻辑块结构（一切尽在注释中）：

```
# # vllm/block.py
class LogicalTokenBlock:
    """A block that stores a contiguous chunk of tokens from left to right.

    Logical blocks are used to represent the states of the corresponding
    physical blocks in the KV cache.
    
    KV cache的逻辑块
    """

    def __init__(
        self,
        block_number: int, # 逻辑块的序号
        block_size: int, # 每个逻辑块中有多少个槽位（默认为16）
    ) -> None:
        self.block_number = block_number
        self.block_size = block_size

        # 逻辑块刚初始化时，将其中的每个token_id都初始化为_BLANK_TOKEN_ID（-1）
        self.token_ids = [_BLANK_TOKEN_ID] * block_size 
        # 当前逻辑块中已经装下的token的数量
        self.num_tokens = 0

    def is_empty(self) -> bool:
        """判断当前逻辑块是为空"""
        return self.num_tokens == 0

    def get_num_empty_slots(self) -> int:
        """当前逻辑块的空余槽位"""
        return self.block_size - self.num_tokens

    def is_full(self) -> bool:
        """判断当前逻辑块是否已经被装满"""
        return self.num_tokens == self.block_size

    def append_tokens(self, token_ids: List[int]) -> None:
        """将给定的一些token_ids装入当前逻辑块中"""
        # 给定的token_ids的长度必须 <= 当前逻辑块剩余的槽位
        assert len(token_ids) <= self.get_num_empty_slots()
        # 当前逻辑块第一个空槽的序号
        curr_idx = self.num_tokens
        # 将这些tokens装进去
        self.token_ids[curr_idx:curr_idx + len(token_ids)] = token_ids
        # 更新当前逻辑块中tokens的数量
        self.num_tokens += len(token_ids)

    def get_token_ids(self) -> List[int]:
        """获取当前逻辑块中所有被装满的位置的token_ids"""
        return self.token_ids[:self.num_tokens]

    def get_last_token_id(self) -> int:
        """获取当前逻辑块所所有被装满的位置的最后一个token_id"""
        assert self.num_tokens > 0
        return self.token_ids[self.num_tokens - 1]
```

关于逻辑块，我们已在源码解读2（TODO）的2.3（2）中详细介绍过，它是Sequence实例（seq）下维护的一个属性。我们也提过，在vLLM代码实现中：  

- 每个seq维护自己的一份逻辑块列表，
- BlockManager中的self.block_tables（形式如：{seq_id: List[PhysicalBlock]}）则记录者每个seq下的物理块列表

**通过seq这个中介，我们维护起“逻辑块->物理块”的映射**

## 四、UncachedBlockAllocator

本文我们先来看较为简单的非缓存式BlockAllocator的实现。

### 4.1 在调度器中，什么时候会用到BlockAllocator

在调度器策略（TODO）的讲解中，我们明确了非常重要的一点：**在vllm的1个推理阶段，所有的seq_group要么一起做prefill，要么一起做decode**。这也意味着，**某次调度的结果，要么全部来自waiting队列（等待做prefill的），要么全部来自running或者running + swapped队列（等待做decode的）**。

### 4.2 为waiting队列中的seq_group分配prefill需要的物理块

![图](<images/图解大模型计算加速系列：vLLM源码解析3，块管理器BlockManager（上篇）/v2-acc64a2d59bd05fdf05e28e864e37d28_1440w.jpg>)

如上图，当我们准备从waiting队列中调度seq_group时，我们会依次做两件事：  

- 调用`self.block_manager.can_allocate(seq_group)`方法，**判断当前gpu上是否有充足的空间，能为当下这seq_group的prefill阶段分配充足的物理块**，用于装其KV Cache（**细节我们在源码解读2中已讲过，这里不再赘述**）
- 一旦我们认为当下空间充足，则调用`self._allocate(seq_group)`方法，**为waiting队列中的这个seq_group实际分配物理块**，这时我们就会运用到BlockAllocator，并且BlockAllocator的类型不同（即是否做prefix caching），allocate的方法也会不同。

**所以现在，我们就来看`self._allocate(seq_group)`函数（如何为waiting队列中的seq_group分配物理块做prefill）**  
  
  
`self._allocate(seq_group)`的入口函数如下(一切尽在注释中)：

```
    # vllm/core/scheduler.py
    def _allocate(self, seq_group: SequenceGroup) -> None:
        # ==============================================================
        # block_manager为当前seq_group分配物理块
        # ==============================================================
        self.block_manager.allocate(seq_group)

        # ==============================================================
        # 当前seq_group状态改为running
        # ==============================================================
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING
```

接下来我们看<strong>`self.block_manager.allocate(seq_group)`</strong>实现，如前文所说，本文我们解读的是`BlockSpaceManagerV1`，所以我们就去这个class的顶一下看allocate方法（一切尽在注释中）。

```
# vllm/core/block_manager_v1.py
class BlockSpaceManagerV1(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int, # 每个block的槽位大小，默认为16
        num_gpu_blocks: int, # 当前gpu上最多能分配的block数量
        num_cpu_blocks: int, # 当前cpu上，用于做swap的内存中，最多能分配的block数量
        watermark: float = 0.01, # 内存交换的水位线（阈值）
        sliding_window: Optional[int] = None,  # 滑动窗口的大小
        enable_caching: bool = False, # 是否需要做prefix caching
    ) -> None:

        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        if enable_caching and sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is not allowed with prefix caching enabled!")

        self.block_sliding_window = None
        if sliding_window is not None:
            assert sliding_window % block_size == 0, (sliding_window,
                                                      block_size)
            self.block_sliding_window = sliding_window // block_size

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        # ===========================================================================
        # 水位线block数量：理解成一个阈值，这个阈值决定是否要给当前seq分配block
        # 设置水位线block的目的是不要一下打满设备中的物理块，留一些buffer，避免后续频繁地发生swap
        # ===========================================================================
        self.watermark_blocks = int(watermark * num_gpu_blocks)

        # ===========================================================================
        # 根据是否做了prefix caching限制，来选择不同的allocator
        # ===========================================================================
        if self.enable_caching:
            logger.info("Automatic prefix caching is enabled.")
            self.gpu_allocator = CachedBlockAllocator(Device.GPU, block_size,
                                                      num_gpu_blocks)
            self.cpu_allocator = CachedBlockAllocator(Device.CPU, block_size,
                                                      num_cpu_blocks)
        else:
            self.gpu_allocator = UncachedBlockAllocator(
                Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator = UncachedBlockAllocator(
                Device.CPU, block_size, num_cpu_blocks)
        
        # ===========================================================================
        # 创建block_tables字典，形式如{seq_id: block_table}, 记录每一个序列对应的block table
        # ===========================================================================
        self.block_tables: Dict[int, BlockTable] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        """
        确实是否可以给这个seq_group分配物理块，返回结果有三种情况：
        - AllocStatus.NEVER：不分配；
        - AllocStatus.OK：可以分配；
        - AllocStatus.LATER：延迟分配
        在源码解读2中我们详细讲过这个方法，这里不赘述
        """
        ...

    def allocate(self, seq_group: SequenceGroup) -> None:
        """
        为当前seq_group分配物理块做prefill
        """
        # ==========================================================================
        # NOTE: vllm中有一条重要假设：一个seq_group内的所有seq都共享一个prompt
        #       而我们现在正是要对这个prompt分配物理块。
        # 复习一下，waiting队列中所有的seq_group都没做过prefill，因此每个seq_group下面
        # 只有1条seq，这个seq即位prompt本身，所以我们取[0]即可拿出这个prompt
        # ==========================================================================
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]

        # ==========================================================================
        # 计算该seq的逻辑块数量
        # （prefill阶段，有多少个逻辑块，就应该分配多少个物理块）
        # ==========================================================================
        num_prompt_blocks = len(seq.logical_token_blocks)

        # ==========================================================================
        # 为该seq分配物理块，List[PhysicalTokenBlock]
        # ==========================================================================
        block_table: BlockTable = []
        # 遍历该seq的所有逻辑块
        for logical_idx in range(num_prompt_blocks):
            # ==========================================================================
            # 如果block的滑动窗口长度不为空（可暂时忽略不看）
            # ==========================================================================
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
                # Set the reference counts of the token blocks.
                block.ref_count = seq_group.num_seqs()
            
            # ==========================================================================
            # 如果做了prefix caching，即使用的是CachedBlockAllocator
            # （是下篇要讲解的重点，这里我们用的是UncachedBlockAllocator，所以可忽略不看）
            # ==========================================================================
            elif self.enable_caching:
                block = self.gpu_allocator.allocate(
                    seq.hash_of_block(logical_idx),
                    seq.num_hashed_tokens_of_block(logical_idx))
            # ==========================================================================
            # 其余情况（即UncachedBlockAllocator对应的情况）
            # ==========================================================================
            else:
                # 从空闲物理块中取一块出来，并令其ref_count = 1（表示有1个逻辑块引用它了）
                # 相关代码讲解见下
                block = self.gpu_allocator.allocate()
                # 由于seq_group下的所有seq共享一个prompt，
                # 所以进一步令物理块的ref_count = num_seqs
                # （表示这些seqs的逻辑块都引用它了）
                block.ref_count = seq_group.num_seqs()
            
            block_table.append(block)

        # ==========================================================================
        # prefill阶段，这个seq_group下所有的seq共享一个prompt，也即共享这个prompt代表的物理块
        # ==========================================================================
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            self.block_tables[seq.seq_id] = block_table.copy()
     
     # ... (该class下的其它方法，暂时略过)
```

那现在我们再进一步看下上面代码中`block = self.gpu_allocator.allocate()`的实现（**一切尽在注释中）**：

```
# vllm/core/block_manager_v1.py
class UncachedBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device # 设备：cpu/gpu
        self.block_size = block_size # 该设备上每个物理块的槽位数，默认为16
        self.num_blocks = num_blocks # 该设备上留给KV cache的总物理块数量

        # =================================================================
        # 初始化所有物理块
        # self.free_blocks：List[PhysicalTokenBlock], 用于跟踪该设备上所有
        #                   未被使用过的物理块
        # =================================================================
        self.free_blocks: BlockTable = []
        for i in range(num_blocks):
            # vllm/vllm/block.py
            # 定义物理块
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size,
                                       block_hash=-1,
                                       num_hashed_tokens=0)
            self.free_blocks.append(block)

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1 # 该物理块首次有逻辑块引用了，所以ref_count=1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        """
        释放一条seq对应的物理块
        即将对应物理块的引用-1，如果此时引用数量为0，说明对应物理块完全自由了，
        需要再将其放入自由物理块列表中
        """
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def contains_block(self, block_hash: int) -> bool:
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")
```

好，整个过程代码注释已经说得非常清楚了，这里我们再稍微总结下：  

- waiting队列中的每个seq_group都还未经历过prefill阶段，因此每个seq_group下只有1个seq，这个seq即为prompt
- 在使用`UncachedBlockAllocator`为wating队列中的某个seq_group分配物理块时，我们就是在对初始的这个prompt分配物理块。所以这个prompt有多少个逻辑块，我们就分配多少个可用的空闲物理块，同时注意更新物理块的ref_count。
- 你一定发现了，这里我们做的只是**给定一种“物理块的分配方案”**，我们只是在制定这个seq_group可以使用哪些物理块，**但并没有实际往物理块中添加数据！“添加数据”这一步留到这1步推理实际开始时，由CacheEngine按照这个方案，往物理块中实际添加KV Cache。这个我们留在再后面的系列讲解。**

###   4.3 为running/swapped队列中的seq_group分配decode需要的物理块

![图](<images/图解大模型计算加速系列：vLLM源码解析3，块管理器BlockManager（上篇）/v2-bc060d47b2059b40c0afec78f3757f03_1440w.jpg>)

接下来我们考虑为running/swapped队列中的seq_group分配decode需要的物理块。  
对于每个seq_group，在上1个推理阶段，我们对它下面的每个seq都产出了1个token。所以在这个推理阶段，我们判断能否为这些seq_group分配物理块时，我们也会分成两步：  

- **调用`self.block_manager.can_append_slot(seq_group)`方法**，判断是否至少能为这个seq_group下的每个seq都分配1个空闲物理块。如果可以则认为能调度这个seq_group（原因和代码分析我们在源码解读2中细讲过，这里不赘述）。
- **调用`self._append_slot(seq_group, blocks_to_copy)`方法**，实际分配物理块。我们马上来看细节。

调用入口（一切尽在注释中）：

```
    # vllm/core/scheduler.py
    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]], # {旧物理块id：[由旧物理块copy-on-write而来的新物理块id]}
    ) -> None:
        # =============================================================================
        # 遍历这个seq_group中状态为running的所有seq
        # =============================================================================
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            # ========================================================================
            # 为这个seq分配物理块，代码细节见下
            # ret = None时，说明可以继续使用物理块的空槽位，不需要新分配物理块
            # ret不为空时的结果为：（last_block.block_number, new_block.block_number）
            # 前者表示源物理块，后者表示copy-on-write而来的物理块
            # （有疑惑不要紧，下文我们马上来看代码细节）
            # ========================================================================
            ret = self.block_manager.append_slot(seq)
            # ========================================================================
            # ret非None，说明采用了copy-on-write机制（参见原理篇讲解）
            # 这时我们要记录copy-on-write相关的映射关系
            # ========================================================================
            if ret is not None:
                src_block, dst_block = ret
                # {旧物理块id：[由旧物理块copy-on-write而来的新物理块id]}
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]
```

我们来看`self.block_manager.append_slot(seq)`细节（一切尽在注释中）:

```
# vllm/core/block_manager_v1.py
class BlockSpaceManagerV1(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int, # 每个block的大小
        num_gpu_blocks: int, # 当前gpu上最多能分配的block数量
        num_cpu_blocks: int, # 当前cpu上，用于做swap的内存中，最多能分配的block数量
        watermark: float = 0.01, # 内存交换的水位线（阈值）
        sliding_window: Optional[int] = None,  # 滑动窗口的大小
        enable_caching: bool = False, # 是否需要做prefix caching（目前暂时不支持，所以都设为False）
    ) -> None:

        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        if enable_caching and sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is not allowed with prefix caching enabled!")

        self.block_sliding_window = None
        if sliding_window is not None:
            assert sliding_window % block_size == 0, (sliding_window,
                                                      block_size)
            self.block_sliding_window = sliding_window // block_size

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        # ===========================================================================
        # 水位线block数量：理解成一个阈值，这个阈值决定是否要给当前seq分配block
        # 设置水位线block的目的是不要一下打满设备中的物理块，留一些buffer，避免后续频繁地发生swap
        # ===========================================================================
        self.watermark_blocks = int(watermark * num_gpu_blocks)

        # ===========================================================================
        # 根据是否做了prefix caching限制，来选择不同的allocator
        # ===========================================================================
        if self.enable_caching:
            logger.info("Automatic prefix caching is enabled.")
            self.gpu_allocator = CachedBlockAllocator(Device.GPU, block_size,
                                                      num_gpu_blocks)
            self.cpu_allocator = CachedBlockAllocator(Device.CPU, block_size,
                                                      num_cpu_blocks)
        else:
            self.gpu_allocator = UncachedBlockAllocator(
                Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator = UncachedBlockAllocator(
                Device.CPU, block_size, num_cpu_blocks)
        
        # ===========================================================================
        # 创建block_tables字典，形式如{seq_id: block_table}, 记录每一个序列对应的block table
        # ===========================================================================
        self.block_tables: Dict[int, BlockTable] = {}

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        """
        对于这个seq_group，我们检查对于其中的每一个seq，
        是否能至少分配一个空闲物理块给它
        相关讲解在源码解读2中详细说过，不再赘述
        """
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def _promote_last_block(
        self,
        seq: Sequence,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        assert self.enable_caching

        # Compute a new hash for the block so that it can be shared by other
        # Sequences
        new_hash = seq.hash_of_block(len(seq.logical_token_blocks) - 1)

        # if new_hash is already in the cached table, then free last_block
        # and return the cached version
        if self.gpu_allocator.contains_block(new_hash):
            self.gpu_allocator.free(last_block)
            return self.gpu_allocator.allocate(new_hash)
        else:
            self.gpu_allocator.update_hash(new_hash, last_block)
            return last_block

    def _is_last_block_full(
        self,
        seq: Sequence,
    ) -> bool:
        """
        检查当前这最后一个物理块是不是已经装满了
        """
        # 获取该seq的token数量
        token_ids_len = len(seq.data.get_token_ids())
        # 如果seq的token数量大于0，且token数量能被block整除，说明当前这最后一个物理块是满的
        return token_ids_len > 0 and token_ids_len % seq.block_size == 0

    def _maybe_promote_last_block(
        self,
        seq: Sequence,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        # ===================================================================
        # 检查当前这最后一个物理块是否满了，如果是：
        # ===================================================================
        if self._is_last_block_full(seq):
            return self._promote_last_block(seq, last_block)
        else:
            return last_block

    def _allocate_last_physical_block(
        self,
        seq: Sequence,
    ) -> PhysicalTokenBlock:
        """
        我们在想添加新的物理块之前，调用这个函数，来判断是不是真得有必要添加一个物理块
        """

        # ===================================================================
        # 如果不使用prefix caching，就直接分配物理块（看到这里就可以，下面的不用看）
        # ===================================================================
        if not self.enable_caching:
            return self.gpu_allocator.allocate()
        
        # ===================================================================
        # 使用prefix caching（下篇要讲解的重点，这里可以忽略）
        # ===================================================================
        block_hash: Optional[int] = None
        if (self._is_last_block_full(seq)):
            block_hash = seq.hash_of_block(len(seq.logical_token_blocks) - 1)
        num_hashed_tokens = seq.num_hashed_tokens_of_block(
            len(seq.logical_token_blocks) - 1)

        new_block = self.gpu_allocator.allocate(block_hash, num_hashed_tokens)

        if block_hash is None:
            assert new_block.ref_count == 1
        return new_block

    def append_slot(
        self,
        seq: Sequence,
    ) -> Optional[Tuple[int, int]]:
        """
        为这个seq中的新token分配一个物理槽位
        """
        # ==============================================================
        # 读取这个seq的逻辑块，List[LogicalTokenBlock]
        # ==============================================================
        logical_blocks = seq.logical_token_blocks
        # ==============================================================
        # 读取这个seq的物理块，List[PhysicalTokenBlock]
        # ==============================================================
        block_table = self.block_tables[seq.seq_id]
        
        # ==============================================================
        # 如果物理块数量 < 逻辑块数量（说明此时需要分配新的物理块了）
        # 注：上1个推理阶段完毕后，seq的逻辑块更新了（把最新生成的这个token装进去了）
        #     但物理块还没更新
        # ==============================================================
        if len(block_table) < len(logical_blocks):
            # ==============================================================
            # （需要声明物理块只允许比逻辑块少1块）
            # ==============================================================
            assert len(block_table) == len(logical_blocks) - 1

            # ==============================================================
            # 如果使用滑动窗口时的逻辑（可暂时忽略不看）
            # ==============================================================
            if (self.block_sliding_window
                    and len(block_table) >= self.block_sliding_window):
                # reuse a block
                block_table.append(block_table[len(block_table) %
                                               self.block_sliding_window])
            # ==============================================================
            # 其余情况，直接分配一个新的物理块给当前序列
            # ==============================================================
            else:
                # 如果是UnCachedBlockAllocator，就直接分配一个新的空闲物理块
                new_block = self._allocate_last_physical_block(seq)
                block_table.append(new_block)
                return None

        # ==============================================================
        # 如果物理块数量==逻辑块数量：
        # ==============================================================
        last_block = block_table[-1] # 取出最后一个物理块
        assert last_block.device == Device.GPU # 声明必须是gpu物理块
        
        # ==============================================================
        # 如果最后一个物理块的引用数量为1（只有1个逻辑块引用它）
        # （也就是只有当前这个seq在用它）
        # ==============================================================
        if last_block.ref_count == 1:
            # ==============================================================
            # 如果你是在做prefix caching（暂时不看，下篇再细讲）
            # ==============================================================
            if self.enable_caching:
                maybe_new_block = self._maybe_promote_last_block(
                    seq, last_block)
                block_table[-1] = maybe_new_block
            # ==============================================================
            # 不用prefix caching，此时我们不需要添加新的物理块，所以返回None
            # ==============================================================
            return None
        # ==============================================================
        # 如果最后一个物理块的引用数量为 > 1 (有别的逻辑块在引用它)
        # （也就是有别的seq在用它）
        # ==============================================================
        else:
            # ==============================================================
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            # 触发copy-on-write机制，分配一个新的物理块。机制相关的解释见原理篇讲解
            # ==============================================================
            new_block = self._allocate_last_physical_block(seq)
            block_table[-1] = new_block
            # 从该seq的block_table中释放掉旧的物理块
            # 也即该物理块ref_count -= 1，如果-=1后ref_count=0，说明该物理块彻底自由了，
            # 此时可以把它添加进自由物理块的列表中（细节留给大家自己看源码，不难）
            self.gpu_allocator.free(last_block)
            
            return last_block.block_number, new_block.block_number
```

如果在阅读上述代码中，你感觉有些迷惑，建议先看一下原理篇的copy-on-write机制，以及源码解读2中的相关相关讲解。动手画画图，帮助理清过程。  
  
  
同样，在这里我们依然要强调，调度器中只是给出了物理块的分配方案，并没有实际往物理块中添加数据，添加数据这一步是CacheEngine照着这个方案来实际操作的，这个我们放在后面的文章中讲解。  
  
  
**恭喜你已经了解了非缓存式物理块管理器(`UncachedBlockAllocator`)的全部细节！在块管理器的下篇中，我们将来看一个更有意思，代码上也更有难度的缓存式块管理器`C``achedBlockAllocator`，一起来看看vllm在论文中说的prefix caching是如何实现的。**
