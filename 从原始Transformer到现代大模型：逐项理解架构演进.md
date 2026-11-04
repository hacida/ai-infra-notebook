# 从原始 Transformer 到现代大模型：逐项理解架构演进

> 这是一组由浅入深的学习笔记。先用[原始 Transformer](transform.md)建立 Q/K/V、Encoder–Decoder、FFN 与 Add & Norm 的基线，再按下面的顺序阅读各项变化。每篇都围绕一件事说明：**遇到了什么问题、怎样改变计算或数据流、为什么有效、代价是什么**。

现代大模型没有一条统一的“升级路线”。整体架构、Attention 头与缓存、位置表示、FFN、MoE、归一化和推理系统是可以组合的不同设计轴。比如 GQA 减少历史 KV 头数，RoPE 让位移参与 Q/K 匹配，SwiGLU 改 FFN 内部计算；三者互不替代。

## 如果你第一次接触这些词

先把一个 Token 想成句子中的一个小片段。它进入某一层时带着一个数字向量。**Attention** 让这个位置读取其他允许看到的位置；当前 Token 提出的问题叫 Query，历史位置提供匹配用的 Key 和被取回的 Value。**FFN** 则在一个位置内部加工已经汇集好的信息。模型把这两类子层重复堆叠；**残差**让原向量可以直接加回，**Norm**控制向量尺度。生成文字时，模型一次选出下一个 Token；**KV Cache** 留下已经算好的历史 Key/Value，供下一轮继续使用。

这组文章就围绕一次“读历史、加工信息、生成下一词”的旅程展开。每篇先用具体场景说明问题，再看方案改变了哪条数据路径，最后讨论收益与代价。读到图时，建议先只追一枚 Token 的箭头，不急着记缩写。公式只保留用于核对关键关系的少数几条。

## 建议阅读顺序

| 顺序 | 专题 | 核心问题 | 读完应能解释 |
| :--- | :--- | :--- | :--- |
| 1 | [整体架构与可见范围](transformer-evolution/01-整体架构与可见范围.md) | 一个 Token 能看见谁？ | Encoder-only、Encoder–Decoder、Decoder-only 为何适合不同任务；为什么生成必须因果 |
| 2 | [MHA、MQA、GQA](transformer-evolution/02-MHA-MQA-GQA.md) | 多个 Query 头为什么共享 KV？ | MQA 怎样减缓存；GQA 为什么又保留多组 KV；头数与窗口的区别 |
| 3 | [MLA](transformer-evolution/03-MLA.md) | 为什么改存低维 latent？ | 历史内容怎样压缩；如何避免每步展开全部 K/V；为何单存位置 Key |
| 4 | [位置编码](transformer-evolution/04-位置编码.md) | 顺序与距离怎样进入 Attention？ | 绝对位置、T5 Bias、RoPE、ALiBi 各解决了什么，又留下什么限制 |
| 5 | [FFN 与门控](transformer-evolution/05-FFN与门控.md) | 每个 Token 的特征怎样变换？ | ReLU/GELU 与 SwiGLU 的结构差别；门控宽度为何不能机械照搬 |
| 6 | [MoE](transformer-evolution/06-MoE.md) | 为什么不让每个 Token 用全部 FFN 参数？ | Router、Top-$k$、Expert 合并，以及总参数、激活计算、跨卡通信的关系 |
| 7 | [归一化与残差](transformer-evolution/07-归一化与残差.md) | 深层网络怎样传递表示与梯度？ | Pre/Post-Norm 与 LayerNorm/RMSNorm 是两项独立选择 |
| 8 | [KV Cache](transformer-evolution/08-KV-Cache.md) | 生成下一词时为什么不用重算整个前缀？ | Prefill/Decode、容量估算、缓存复用和系统管理 |
| 9 | [按模型组合架构](transformer-evolution/09-按模型组合架构.md) | 这些部件放回模型后怎样一起工作？ | 沿 T5、Llama、Mistral、Mixtral、DeepSeek 的数据流读配置 |

## 一张问题地图

| 读模型论文时遇到的说法 | 先问什么 | 去读哪篇 |
| :--- | :--- | :--- |
| “这是 Decoder-only 模型” | 输入和输出是否在同一条因果序列？ | [整体架构](transformer-evolution/01-整体架构与可见范围.md) |
| “采用 GQA / MLA” | 每层每历史 Token 究竟存几组 KV，或存什么 latent？ | [头共享](transformer-evolution/02-MHA-MQA-GQA.md)、[MLA](transformer-evolution/03-MLA.md) |
| “支持长上下文” | 位置规则能外推吗？单层能看到多远？Cache 如何增长？ | [位置](transformer-evolution/04-位置编码.md)、[KV Cache](transformer-evolution/08-KV-Cache.md) |
| “采用 SwiGLU / MoE” | 改的是一个 FFN 内部通道，还是按 Token 选择 Expert？ | [FFN](transformer-evolution/05-FFN与门控.md)、[MoE](transformer-evolution/06-MoE.md) |
| “采用 Pre-Norm RMSNorm” | Norm 放在哪里，具体又怎么算？ | [归一化](transformer-evolution/07-归一化与残差.md) |

这些专题聚焦**模型内部原理**。如果进一步关心 TP、PP、EP 如何把它们放到多张 GPU，以及通信为何成为瓶颈，可接着读[分布式并行策略](大模型分布式并行策略_DP_TP_SP_PP_EP%20%281%29.md)与[GPU 通信和 NCCL](大模型_GPU通信方式_NCCL与性能优化.md)。

## 这组笔记的读法

图先帮助建立数据流，文字解释因果与边界，少量公式用于检验关键计算。示意图中的头数和路由选择只代表教学例子；具体模型配置以相应论文和实现为准。每篇结尾设有几个检查问题，可以先自己回答，再回看该篇的图和例子。
