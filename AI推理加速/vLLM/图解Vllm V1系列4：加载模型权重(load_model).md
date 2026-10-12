# 图解Vllm V1系列4：加载模型权重(load_model)

> 作者：猛猿  
> 来源：知乎专栏（本地网页存档转换）  
> 原文：https://zhuanlan.zhihu.com/p/1908151478557839879

【vllm v1系列】

[猛猿：图解Vllm V1系列1：整体流程](https://zhuanlan.zhihu.com/p/1900126076279160869)

[猛猿：图解Vllm V1系列2：Executor-Workers架构](https://zhuanlan.zhihu.com/p/1900613601577899465)

[猛猿：图解Vllm V1系列3：KV Cache初始化](https://zhuanlan.zhihu.com/p/1900932850829730567)

【历史文章汇总】

猛猿：【必看】历史技术文章导航

---

按照原定计划，这篇文章应该要开始进入Scheduler的讲解了。但是我突然想起，在之前的文章中，漏掉了一个看似简单，但是十分重要的细节：**vllm是如何加载模型权重的**？在现在非常热门的rlhf训练中，避不开的一点是Actor和推理引擎间是需要做权重通信和更新的。所以，了解vllm load_model()的运作流十分重要。  
  
在系列2中我们提过，vllm首次加载模型，是在EngineCore首次初始化Executor->Workers架构的时候，这个初始化的过程做的事情如下图所示：

![图](<images/图解Vllm V1系列4：加载模型权重(load_model)/v2-f5426f986e9d1e83a40f0bbd6e0608ca_1440w.jpg>)

我们不再赘述上图中的全部细节（大家可自行阅读系列2），只看其中和load_model()相关的部分：

> （1）当MultiProcExecutor创建其下的若干Workers时，它会让每个worker都执行一次self.worker.load_model()
>
> 代码参见：
>
> https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/v1/executor/multiproc_executor.py#L245
>
> （2）而self.worker.load_model()，实际执行的是self.model_runner.load_model()。正如前面的系列文章所说，worker上真正负责计算干活的，是ModelRunner。
>
> 代码参见：
>
> https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/v1/worker/gpu_worker.py#L136
>
> （3）ModelRunner上的load_model()，是本文要探讨的重点。
>
> 代码参见：
>
> https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/v1/worker/gpu_model_runner.py#L1173
>
>

**所以load_model的整体流程是：Executor -> Worker.load_model() -> ModelRunner.load_model()**

## 一、入口函数

我们直接来看代码：

```
    # https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/v1/worker/gpu_model_runner.py#L1173
    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            time_before_load = time.perf_counter()
            # =================================================================
            # 核心代码：在这一步完成了：
            # （1）权重下载（如有必要）
            # （2）模型架构初始化（每个ModelRunner上维护着自己的切片架构）
            # （3）权重注入模型架构（一般权重是完整的，在注入模型架构的时候，每个ModelRunner
            #                    读取自己需要的那部分切片）
            # 最终返回的结果，self.model = model.eval()
            # =================================================================
            self.model = get_model(vllm_config=self.vllm_config)
            # lora部分的处理，可以暂时忽略不看
            if self.lora_config:
```

再来看核心函数`get_model`

```
# https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/model_executor/model_loader/__init__.py#L12
def get_model(*, vllm_config: VllmConfig) -> nn.Module:
    # =================================================================
    # 默认情况下(LoadFormat.Auto)，loader = DefaultModelLoader
    # =================================================================
    loader = get_model_loader(vllm_config.load_config)
    # 返回的是model.eval()，此时已将权重加载完毕
    return loader.load_model(vllm_config=vllm_config)
```

**loader的类型由LoaderFormat决定，其默认值为Auto，在接下来的讲解中，我们假设使用的是`DefaultModelLoader`**，其余类型的loader留给读者们自己探索。

## 二、DefaultModelLoader

### 2.1 整体流程

回顾一下当前的路径：  
Executor -> Worker.load_model() -> ModelRunner.load_model() -> DefaultModelLoader.load_model()，  
本节中我们就来看DefaultModelLoader.load_model()的细节：

```
# https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/model_executor/model_loader/loader.py#L435
    def load_model(self, vllm_config: VllmConfig) -> nn.Module:
        # ==========================================================================
        # 获取设备配置
        # ==========================================================================
        device_config = vllm_config.device_config
        # ==========================================================================
        # 获取模型配置
        # ==========================================================================
        model_config = vllm_config.model_config
        # ==========================================================================
        # 确定目标设备（如cuda:0）
        # ==========================================================================
        target_device = torch.device(device_config.device)
        # ==========================================================================
        # 设置默认的pytorch数据类型（如torch.float16）
        # ==========================================================================
        with set_default_torch_dtype(model_config.dtype):
            # ==========================================================================
            # 在目标设备上初始化模型权重
            # ==========================================================================
            with target_device:
                # ==========================================================================
                # 1. 构建模型架构（每张卡上维护自己的那部分模型架构切片，但还没有实际装载模型）
                # ==========================================================================
                model = _initialize_model(vllm_config=vllm_config)
            # ==========================================================================
            # 收集这张卡上所有需要加载的模型参数名称
            # ==========================================================================
            weights_to_load = {name for name, _ in model.named_parameters()}
            
            # ==========================================================================
            # 2. 实际加载权重
            # (1) _get_all_weights：生成权重迭代器，形式如（权重名称 ,tensor）
            #     - 下载权重到本地
            #     - 生成权重迭代器，形式如（权重名称 ,tensor），迭代器的作用是，先不去加载权重，
            #       到第二步 model.loads_weights时，遍历到哪一块权重，再具体去加载      
            # (2) model.load_weights：真正将模型权重注入本卡上所维护的模型切片中，在注入的过程中，
            #                         如有需要，会对送来的这部分权重进行切片
            # ==========================================================================
            loaded_weights = model.load_weights(
                self._get_all_weights(model_config, model))
            # ==========================================================================
            # 记录加载权重的耗时
            # ==========================================================================
            self.counter_after_loading_weights = time.perf_counter()
            logger.info(
                "Loading weights took %.2f seconds",
                self.counter_after_loading_weights -
                self.counter_before_loading_weights)
            # ==========================================================================
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            # 检查权重完整性
            # ==========================================================================
            if model_config.quantization is None and loaded_weights is not None:
                # ==========================================================================
                # 计算未成功加载的权重，如果存在没有成功加载的情况，直接报错
                # ==========================================================================
                weights_not_loaded = weights_to_load - loaded_weights
                if weights_not_loaded:
                    raise ValueError(
                        "Following weights were not initialized from "
                        f"checkpoint: {weights_not_loaded}")
            # ==========================================================================
            # 后处理：量化权重处理或者特定层调整
            # ==========================================================================
            _process_weights_after_loading(model, model_config, target_device)

        return model.eval()
```

为了简化讨论，这里我们假设仅使用tp的方式做分布式推理，由上述代码可知，ModelRunner加载模型主要分成两步：**初始化模型架构**和**实际加载权重**。我们来看这两者细节。

### 2.2 初始化模型架构

**这一步的目标是在各个ModelRunner（各张卡）上初始化模型架构分片，但不会涉及权重的实际装载**。例如你使用tp做分布式推理，那么每个ModelRunner上只维护部分模型，所以我们称呼其为“分片”。我们来看具体代码：

```
# https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/model_executor/model_loader/loader.py#L110
def _initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
) -> nn.Module:
    """Initialize a model with the given configurations."""
    # ==============================================================================
    # (vllm类， hf类名)
    # 例如 (<class 'vllm.model_executor.models.qwen.QWenLMHeadModel'>, "QWenLMHeadModel")
    # ==============================================================================
    model_config = vllm_config.model_config
    model_class, _ = get_model_architecture(model_config)

    # ==============================================================================
    # 如果配置了量化（如AWQ/GPTQ），动态修改 model_class 的层定义（例如将 Linear 替换为 QuantLinear)
    # ==============================================================================
    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)
    
    # ==============================================================================
    # 检查model_class是否支持新版vllm（要求接受vllm_config和prefix）
    # ==============================================================================
    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    
    # ==============================================================================
    # 新版模型初始化（推荐路径）
    # ==============================================================================
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        with set_current_vllm_config(vllm_config, check_compile=True):
            # ==============================================================================
            # 真正执行初始化模型实例（这时每张卡上维护的就是自己的那部分模型切片了，只不过还没有实际装载模型）
            # ==============================================================================
            return model_class(vllm_config=vllm_config, prefix=prefix)

    # ==============================================================================
    # 旧版模型初始化（略去不看）
    # ==============================================================================
    msg = ("vLLM model class should accept `vllm_config` and `prefix` as "
           "input arguments. Possibly you have an old-style model class"
           " registered from out of tree and it is used for new vLLM version. "
           "Check https://docs.vllm.ai/en/latest/design/arch_overview.html "
           "for the design and update the model class accordingly.")
    ...
```

我们先来看函数`get_model_architecture()`，它的返回值包含2个元素：  

- **vllm类(model_class)**：本质是一个python class，形式如<class 'vllm.model_executor.models.qwen.QWenLMHeadModel'>，最终vllm将使用它来初始化ModelRunner上维护的模型架构
- **hf类**：本质是一个string，形式如"QWenLMHeadModel"

你一定很好奇，这两个东西都是什么意思呢？简单来说，在vllm中，对于各种模型(qwen/llama/etc)，vllm会重写这些模型的架构，以便更好实现分布式推理，以及为推理做特定的优化等。**你可以将hf类理解成原始模型，vllm类理解成是这个模型在vllm中的重写实现。**  
  
  
更具体地来说，假设你现在想使用hf的Qwen-7B模型，当你不用vllm，而使用`transformers`做原生推理时。`transformers`首先要做的也是去找应该用哪个python class来初始化Qwen-7B实例，这时，它会去读`config.json`中的`architectures`字段：  

![图](<images/图解Vllm V1系列4：加载模型权重(load_model)/v2-b0d4b5ce6b8639185a810dac53112af9_1440w.jpg>)

这个字段告诉`transformers`，它应该用`QwenLMHeadModel`做实例化。  
**那么到了vllm中，它就会把这个字段值当一个key（hf类），然后它会去找这个key在vllm中对应的python class实现（vllm类），最终用这个vllm类做实例化。**  
  
  
**诶，那么如果存在这样一个key->value的mapping关系，那么这个mapping关系是在哪里构建的呢**？在vllm中，通过一个`registry.py`文件，注册了所有vllm支持的模型，并构建了上述的映射关系，如果你也想往vllm中注册新模型，那么你也需要操作这个文件，相关的入口代码在这里：[https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/model_executor/models/registry.py#L528](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/model_executor/models/registry.py%23L528)，这边留给读者自行阅读，我就不补充细节了（不过大家可以多关注下，为什么这边要使用惰性注册（_LazyRegisteredModel）的方式，也就是把实际的import延迟到各个worker真正需要import的时候再执行，答案让代码告诉你 ）  
  
&**lt;br/>总结来说，在这一步，vllm根据你要使用的模型，找到这个模型在vllm中的实现（python class），并根据这个实现做实例化，这样各个ModelRunner上就有了各自的模型架构切片。**

### 2.3 加载模型权重

回顾一下2.1过程中关于这一块的代码：

```
            # ==========================================================================
            # 2. 实际加载权重
            # (1) _get_all_weights：生成权重迭代器，形式如（权重名称 ,tensor）
            #     - 下载权重到本地
            #     - 生成权重迭代器，形式如（权重名称 ,tensor），迭代器的作用是，先不去加载权重，
            #       到第二步 model.loads_weights时，遍历到哪一块权重，再具体去加载      
            # (2) model.load_weights：真正将模型权重注入本卡上所维护的模型切片中，在注入的过程中，
            #                         如有需要，会对送来的这部分权重进行切片
            # ==========================================================================
            loaded_weights = model.load_weights(
                self._get_all_weights(model_config, model))
```

### （1）_get_all_weights

代码入口：[https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/model_executor/model_loader/loader.py#L406](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/model_executor/model_loader/loader.py%23L406)

这一步返回结果是一个权重迭代器，形式如`(权重名称, tensor)`，比如`("model.layers.0.self_attn.q_proj.weight", tensor)`，需要注意的是，这是一个不切割的完整权重，它将会被完整地交给这个ModelRunner上维护的model，然后model会根据分布式配置，决定要怎么读取这个模型。  
**所以在rlhf中，如果你想把Actor的权重更新给vllm，方法之一就是把Actor权重构建成这样一种迭代器的形式，直接传给model_runner.model.load_weights()即可，由于这边我们需要这个权重迭代器里包含完整的、不做切片的权重，所以对于采用分布式训练的Actor，我们还需要想办法拿回完整的权重，然后让每个vllm实例的每个tp rank都取得完整的权重即可。当然，如果你有更精细的方法，你可以顺藤摸瓜，按需修改这块代码。总之，掌握了原理，就可以按照你的需求慢慢调整。**

### （2）model.load_weights

这里我假设我们使用的模型是Qwen-7b，那么其对应的model.load_weights的代码入口就在：  
[https://github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/model_executor/models/qwen.py#L297](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/blob/refs/tags/v0.8.2/vllm/model_executor/models/qwen.py%23L297)  

这块代码主要做了2件事情：

- **首先，根据权重名字，决定那些权重应该要被读取**。例如在pp并行中，每个ModelRunner可能维护不同layer的数据，所以当传递过来的权重不是这个ModelRunner所维护的layers范围内时，这个ModelRunner就不会使用它。
- **其次，根据分布式配置，决定要读取这个权重的哪一部分**。例如在tp并行中，每个ModelRunner只维护一个权重块，所以要先对传过来的权重做切割，然后读取自己所维护的那块。

这部分代码比较好读，我就不再展开说明了。  
  
  
**好，关于vllm load_model方法，就介绍到这里了。其实不要看本文篇幅很短，但是在读代码的过程中，我还是做了好多细节笔记的，只是没法一一展示在这里。本文提供一个大致的实现框架，方便有需要对load_model做操作的朋友快速上手修改，更多细节留给读者们自行探索吧！**
