# 图解Vllm V1系列3：KV Cache初始化

> 作者：猛猿  
> 来源：知乎专栏（本地网页存档转换）  
> 原文：https://zhuanlan.zhihu.com/p/1900932850829730567

**提前祝大家五一快乐！节前我们就来学习vllm v1中一个不怎么复杂，但又比较重要的操作：KV Cache的初始化。**  
  
在Vllm V1系列2中，我们以MultiprocExecutor为例（这种类型的Executor适用于单机多卡且不使用ray的场景），详细介绍了Executor->Workers架构，如下图所示：  

![图](<images/图解Vllm V1系列3：KV Cache初始化/v2-9b2d3905dc7e8b4df46a41ce3d7bd4a8_1440w.jpg>)

简单回顾几个重要信息：

- Executor广播数据到各个workers上，各个workers处理数据，并将结果返回给Executor。
- **数据在这里是一个广泛的概念，它可以被抽象理解成`(method, data)`的形式**。其中，data是待处理的数据本身，method表示你期望各个worker调用什么样方法来处理这份数据。**所以广泛上说，这里的数据并不单指要被送去做推理的数据，理论上当你想要所有的workers共同执行某个方法，并且将执行后的结果收集到Executor上时，你都可以通过该操作来完成**。例如我们本文要讨论的在各个workers上做kv cache的初始化，就可以这么做。
- 对于小数据的输入输出（<=10MB），我们采用`rpc_broadcast_mq`和`worker_response_mq`来进行收发管理，这两个队列的本质是shared_memory。对于大数据的输入输出(>10MB)，我们使用zmq sockets进行通信
- 在一个Worker上，它的ModelRunner将实际负责大部分命令的执行，包括load_model、KV cache初始化、推理等。

【系列文章】：

[猛猿：图解Vllm V1系列1：整体流程](https://zhuanlan.zhihu.com/p/1900126076279160869)

[猛猿：图解Vllm V1系列2：Executor-Workers架构](https://zhuanlan.zhihu.com/p/1900613601577899465)

猛猿：【必看】历史技术文章导航

---

##   一、KV Cache初始化的整体流程

> 代码入口：
>
> https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/v1/engine/core.py#L63
>
>

![图](<images/图解Vllm V1系列3：KV Cache初始化/v2-48e6e49435b4d86659ce0d10d1b2ff86_1440w.jpg>)

在我们开始做KV cache初始化之前，已对Executor->Workers架构进行了初始化，同时把模型权重加载到了各个Worker对应的卡上。现在我们直接通过`_initialize_kv_caches`的代码，来看kv cache初始化在做什么事（一切尽在注释中）：

```
    # https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/v1/engine/core.py#L113
    def _initialize_kv_caches(self,
                              vllm_config: VllmConfig) -> tuple[int, int]:
        start = time.time()

        # ===================================================================================
        # 1、计算对于当前模型，各个worker的各个layer上，attn模块相关的kv cache元数据。
        # kv_cache_specs：List[Dict[str, KVCacheSpec]]，每一个dict对应一个worker的返回结果：
        #                - str：模型某一层的layer_name
        #                - KVCacheSpec：维护在该worker上的、这一层layer的kv cache相关元信息
        #                              包括 num_heads, head_size，dtype, use_mla等信息
        #               （注意，这里仅记录元信息，没有实际执行kv cache的分配！）
        # - 执行流程：【Executor.get_kv_cache_specs】
        #           -> 【 MultiProcExecutor.collective_rpc 】：
        #              Executor触发所有worker执行get_kv_cache_specs，并收集相关结果
        #           -> 【 GPUModelRunner.get_kv_cache_specs 】：
        #              Worker上的 ModelRunner 负责实际执行命令
        #           -> 【 MultiProcExecutor.collective_rpc 】：
        #              Executor收集各个workers上的结果
        # ===================================================================================
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        # ===================================================================================
        # 2、profiling：模拟执行1次前向计算，统计各个worker（卡）上有多少空间可以留给kv cache
        # available_gpu_memory：List[float]，每个元素代表一个worker（卡）的返回结果
        # 每块卡上可分配给kv cache的显存 = 该卡总显存 * 用户设置的显存利用率 - fwd推理过程中的峰值显存
        # ===================================================================================
        available_gpu_memory = self.model_executor.determine_available_memory()
        assert len(kv_cache_specs) == len(available_gpu_memory)

        # ===================================================================================
        # 3、计算每块卡上的kv cache配置
        # kv_cache_configs：List[KVCacheConfig]，包含每块gpu上的：
        # - num_blocks：可分配的block数量，floor(可用显存 / 单块缓存大小)
        # - block_size：每块缓存的 token 容量
        # - cache_dtype：数据类型
        # - 等等
        # ===================================================================================
        kv_cache_configs = [
            get_kv_cache_config(vllm_config, kv_cache_spec_one_worker,
                                available_gpu_memory_one_worker)
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in
            zip(kv_cache_specs, available_gpu_memory)
        ]

        # ==================================================================================
        # 4、统一各卡上的kv cache config，例如取 min(num_blocks) 作为最终各卡上可以维护的block数量
        # Since we use a shared centralized controller, we need the
        # `kv_cache_config` to be consistent across all workers to make sure
        # all the memory operators can be applied to all workers.
        # ==================================================================================
        unify_kv_cache_configs(kv_cache_configs)
        # All workers have the same kv_cache_config except layer names, so use
        # an arbitrary one to get the number of blocks.
        assert all([
            cfg.num_blocks == kv_cache_configs[0].num_blocks
            for cfg in kv_cache_configs
        ])
        num_gpu_blocks = kv_cache_configs[0].num_blocks # 最终确定的每张卡上的block数量
        num_cpu_blocks = 0 # 最终确定的cpu上的block数量

        # ==================================================================================
        # 5、在各张卡上实际分配 KV cache（用0张量填充kv cache）
        # Initialize kv cache and warmup the execution
        # ==================================================================================
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)
        return num_gpu_blocks, num_cpu_blocks
```

目前为止，我们通过代码理解了kv cache初始化的大致流程，现在我们就展开这段代码，来看其中的更多细节

## 二、Executor与Workers的配合运作

我们首先来看代码中的这一步:  
`kv_cache_specs = self.model_executor.get_kv_cache_specs()`

在做的事情我们已写在注释中，这里不再赘述。不过在本节中，我们想通过这句代码，来了解Executor分发数据和指令 -> workers执行数据和指令 -> workers返回各自的执行结果 -> Executor收集各个workers结果的整个流程，伪代码如下：

```
1. Executor.get_kv_cache_specs

2. MultiProcExecutor.collective_rpc("get_kv_cache_spec")

(1) 将数据和指令发送给各个workers
rpc_broadcasr_mq.enqueue((method, args, kwargs))
- 在Executor-Workers的初始化阶段，各个workers上已经开启了worker_busy_loop()无限循环，
  在这个循环内，持续监听来自rpc_broadcasr_mq的数据，一旦有数据到来，随即进行dequeue并处理

(2) 收集各个workers的结果
responses = [None] * len(workers)
for worker in workers:
        output = worker.worker_response_mq.dequeue()
        responses[worker.rank] = output

3、在各个worker上，大部分都是由 ModelRunner 来负责实际执行“指令 + 数据”
```

总结来看，Executor下的collective_rpc方法实现了上述的“分发->收集”流程，我们要熟悉这套调度方式，因为在后面各类操作中，我们会经常用到它。

## 三、profiling：计算可用的kv cache的显存

> 代码：
>
> https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/v1/worker/gpu_worker.py#L139
>
>

正如前文所说，这个步骤的最终目的，是去计算一张卡上有多少显存可以分配给kv cache，计算公式如下：  
`每块卡上可分配给kv cache的显存 = 该卡总显存 * 用户设置的显存利用率 - fwd推理过程中的峰值显存`  
  
  
为了达到这个目的，我们可以使用self.model_runner.profile_run()去执行一次模拟的fwd推理，以此得到fwd过程中的峰值显存。在这个fwd中：

- **不使用kv cache**
- **根据配置参数构造出一批模拟的输入数据：**
  - `max_num_tokens`：单次推理中最多可以处理的token数量
  - `max_num_reqs`：单次推理中最多可以处理的请求数量
  - 由`max_num_tokens // max_num_reqs`，我们可以计算单请求最多可以包含的tokens数量，据此我们可以构造一批模拟的输入数据给这次的模拟fwd

**在profiling结束后，我们既知道了“每张卡上可以分配给kv cache的显存”，又知道了“在每张卡上，这个模型的每一层的kv cache元数据”，据此我们就可以计算出每张卡上可以分配多少个blocks了。**  
（更多细节，请大家参见代码）

## 四、初始化kv cache

> 代码：
>
> https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/v1/worker/gpu_model_runner.py#L1537
>
>

**现在，我们已经知道对于模型的每一层，在每张卡上的`[num_blocks, block_size, num_kv_heads, head_size]`，我们就可以根据这个shape，来初始化pytorch kv cache张量，并将他们的初始化值设为0**。如此一来，我们就实际在memory上开辟了一块空间给kv cache，可供后续的推理过程使用。  
  
  
正如本文开篇图例中所说，ModelRunner将实际维护各张卡上的kv cache，它的形式是`List[torch.tensor]`，其中：  

- **list中的各个元素表示各层layer，list中的元素按照layer的顺序排列好了。**
- 每个torch.tensor的尺寸为：`[num_blocks, block_size, num_kv_heads, head_size]`
