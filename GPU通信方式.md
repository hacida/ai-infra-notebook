# 大模型 GPU 通信：通信原语、硬件链路与性能优化

> 从 P2P 和 Collective 基础，到 NCCL、GPU 互联拓扑及 LLM 推理性能分析\
> 适用对象：希望理解多 GPU 通信机制与性能瓶颈的工程师、研究人员

------------------------------------------------------------------------

## 1. 为什么 GPU 通信如此重要？

分布式大模型把权重、激活值、数据或专家分配到不同 GPU 后，GPU
之间必须交换信息。

例如：

-   TP：多个 GPU 计算同一层的不同分片，需要汇总或重新分布结果；
-   PP：前一个 Stage 把激活值发送给下一个 Stage；
-   EP：Token 根据 Router 结果发送到持有目标 Expert 的 GPU；
-   DP 训练：不同 GPU 计算不同数据的梯度，需要同步。

并行化带来计算和容量收益，但也引入通信成本。若通信等待超过并行计算节省的时间，多
GPU 反而可能更慢。

分析 GPU 通信，需要同时看：

1.  **通信语义**：要执行什么数据操作？
2.  **通信算法**：数据通过什么步骤传输？
3.  **硬件路径**：数据经过 NVLink、PCIe 还是网络？
4.  **性能指标**：延迟、带宽、数据量、拥塞和负载均衡如何？

------------------------------------------------------------------------

## 2. 两大类通信：P2P 与 Collective

### 2.1 Point-to-Point：点对点通信

点对点通信在指定的发送方和接收方之间传输数据。

概念上：

\[ GPU_0`\xrightarrow{Send}`{=tex}GPU_1 \]

典型接口：

-   Send：发送；
-   Receive：接收。

典型场景：

-   PP 的相邻 Pipeline Stage 传递激活值；
-   显式 GPU P2P 数据交换；
-   某些 MoE 或分布式算子中的定向传输。

P2P 的参与方和数据目的地较明确。它不要求所有 GPU 都执行同一种集合操作。

### 2.2 Collective：集合通信

集合通信由一个通信组中的多个 Rank 按照约定共同参与。

常见操作：

-   Broadcast
-   Reduce
-   All-Reduce
-   All-Gather
-   Reduce-Scatter
-   All-to-All

在 NCCL 中，通信组内的 GPU 通常以 Rank 标识。集合通信要求各参与 Rank
的调用顺序和参数满足接口约定，否则可能发生挂起或错误。

------------------------------------------------------------------------

## 3. 六种核心集合通信原语

以下统一假设有 4 张 GPU：GPU 0、GPU 1、GPU 2、GPU 3。

### 3.1 Broadcast：广播

**作用：把某个 Rank 的数据发送给通信组内所有 Rank。**

假设 GPU 0 持有：

\[ X=\[1,2,3,4\] \]

Broadcast 后，所有 GPU 都持有相同的数据：

  GPU     结果
------- -------------
  GPU 0   \[1,2,3,4\]
  GPU 1   \[1,2,3,4\]
  GPU 2   \[1,2,3,4\]
  GPU 3   \[1,2,3,4\]

常见用途：

-   分发公共参数或元数据；
-   同步某些控制信息；
-   模型初始化或特定状态分发。

Broadcast 不等同于 DP 训练中的梯度同步。梯度同步通常使用 All-Reduce
或分片优化方案。

### 3.2 Reduce：归约

**作用：将多个 Rank 的同形状数据按指定运算归约到一个目标 Rank。**

例如：

\[ X_0=\[1,2\],`\quad `{=tex}X_1=\[3,4\] \] \[
X_2=\[5,6\],`\quad `{=tex}X_3=\[7,8\] \]

执行 SUM Reduce：

\[ Y=X_0+X_1+X_2+X_3=\[16,20\] \]

结果被放到指定的目标 Rank，其他 Rank 不保证获得完整结果。

常见归约操作包括：

-   SUM；
-   MAX；
-   MIN；
-   PROD。

### 3.3 All-Reduce：全归约

**作用：归约所有 Rank 的数据，并让所有 Rank 都得到完整归约结果。**

沿用上例，SUM All-Reduce 后，每张 GPU 都得到：

\[ Y=\[16,20\] \]

概念上：

\[ AllReduce=Reduce+将结果分发给所有Rank \]

典型场景：

-   DP 训练中的梯度同步；
-   TP 中需要得到完整归约输出的算子；
-   分布式统计量计算。

注意：TP 并非每个算子都使用 All-Reduce。根据输出布局，可能使用
All-Gather 或 Reduce-Scatter。

### 3.4 All-Gather：全收集

**作用：每个 Rank 提供自己的数据分片，所有 Rank 收集并拼接全部分片。**

例如：

\[ X_0=\[1,2\], X_1=\[3,4\],\
X_2=\[5,6\], X_3=\[7,8\] \]

All-Gather 后，每个 Rank 都获得：

\[ Y=Concat(X_0,X_1,X_2,X_3) \]

\[ Y=\[1,2,3,4,5,6,7,8\] \]

典型用途：

-   TP 中恢复完整张量布局；
-   SP 中恢复某些算子所需的完整 Sequence；
-   分布式特征或参数分片收集。

All-Gather 的核心是拼接，不做数值求和。

### 3.5 Reduce-Scatter：归约后散播

**作用：先对各 Rank 的数据执行归约，再把归约结果切分，每个 Rank
只保留其中一个分片。**

例如：

\[ X_0=\[1,2,3,4\] \] \[ X_1=\[10,20,30,40\] \] \[
X_2=\[100,200,300,400\] \] \[ X_3=\[1000,2000,3000,4000\] \]

先求和：

\[ Y=\[1111,2222,3333,4444\] \]

再切分：

  GPU     得到的分片
------- ------------
  GPU 0   \[1111\]
  GPU 1   \[2222\]
  GPU 2   \[3333\]
  GPU 3   \[4444\]

因此：

\[ AllReduce=ReduceScatter+AllGather \]

这里是概念上的分解关系，实际算法可能采用不同的调度和通信实现。

Reduce-Scatter 常用于：

-   TP 输出分片；
-   SP 中激活值重新分布；
-   分片优化器或梯度同步方案。

### 3.6 All-to-All：全交换

**作用：每个 Rank 将不同数据块发送给不同 Rank，同时接收其他 Rank
发来的数据块。**

可把输入理解为一个发送矩阵：

\[ A=
```{=tex}
\begin{bmatrix}
A_{00}&A_{01}&A_{02}&A_{03}\\
A_{10}&A_{11}&A_{12}&A_{13}\\
A_{20}&A_{21}&A_{22}&A_{23}\\
A_{30}&A_{31}&A_{32}&A_{33}
\end{bmatrix}
```
\]

其中 (A\_{ij}) 表示 GPU i 发给 GPU j 的数据。

通信后，GPU j 收到第 j 列中的数据块：

\[ GPU_j:{A\_{0j},A\_{1j},A\_{2j},A\_{3j}} \]

All-to-All 与 All-Gather 的关键区别：

-   All-Gather：每个 Rank 的数据都被所有 Rank 收集；
-   All-to-All：每个 Rank 按目的 Rank 对数据分块，进行定向交换。

典型用途是 MoE 的 Token Dispatch 和 Combine。

------------------------------------------------------------------------

## 4. 六种原语的对照

---------------------------------------------------------------------------------
  原语              输入分布                   输出分布           是否做数值归约
----------------- -------------------------- ------------------ -----------------
  Broadcast         一个源 Rank 有数据         所有 Rank          否
                                               有完整副本         

  Reduce            所有 Rank 有局部数据       目标 Rank          是
                                               有归约结果         

  All-Reduce        所有 Rank 有局部数据       所有 Rank          是
                                               有归约结果         

  All-Gather        各 Rank 有不同分片         所有 Rank          否
                                               有完整拼接结果     

  Reduce-Scatter    所有 Rank 有局部数据       各 Rank            是
                                               有归约结果分片     

  All-to-All        各 Rank                    各 Rank 收到其他   通常否
                    有面向不同目的地的数据块   Rank               
                                               发来的对应数据块   
  ---------------------------------------------------------------------------------

记忆方式：

-   Reduce：多对一归约；
-   All-Reduce：归约后人人都有；
-   All-Gather：收集并拼接；
-   Reduce-Scatter：归约后分片；
-   All-to-All：按目的地互换数据；
-   Broadcast：一份数据发给所有人。

------------------------------------------------------------------------

## 5. GPU 之间通过什么硬件链路通信？

通信原语定义"做什么"，硬件链路决定"数据怎么走"。

### 5.1 PCIe

PCIe 是通用高速互联总线，常用于：

-   CPU 与 GPU；
-   GPU 与 PCIe Switch；
-   支持条件下的 GPU P2P。

带宽取决于 PCIe 代际、Lane 数量和实际拓扑。例如 PCIe 5.0 x16
的单方向理论链路带宽约为 64
GB/s；实际应用吞吐量会受到协议开销、拓扑和设备能力影响。

### 5.2 NVLink

NVLink 是 NVIDIA 的 GPU 高速互联技术。

它用于提高 GPU 之间的带宽、降低通信延迟，尤其适合 TP
等需要频繁交换张量的场景。

具体带宽取决于 GPU 型号、NVLink
代际、链路数量和系统设计。不要把某一代产品的指标套用到所有 GPU。

### 5.3 NVSwitch

NVSwitch 是 GPU 高速互联交换结构，可将多张 GPU 连接成高带宽互联域。

它有助于多 GPU 之间进行高效通信，常见于 NVIDIA HGX/DGX 等系统设计。

NVSwitch 并不意味着所有通信都没有成本；消息大小、拥塞、GPU
负载和通信算法仍会影响性能。

### 5.4 InfiniBand 与 RoCE

多服务器 GPU 集群需要节点间网络。

-   **InfiniBand**：面向高性能计算和数据中心的网络技术；
-   **RoCE**：在以太网上承载 RDMA；
-   **RDMA**：Remote Direct Memory Access，远程直接内存访问；
-   **GPUDirect RDMA**：在硬件、驱动和网络配置支持时，让网络设备直接访问
    GPU 显存，减少主机内存中转。

典型多节点路径：

\[
GPU`\rightarrow `{=tex}NIC`\rightarrow `{=tex}Network`\rightarrow `{=tex}NIC`\rightarrow `{=tex}GPU
\]

实际路径还受 PCIe/NVLink 拓扑、NIC 绑定、NUMA 和驱动配置影响。

### 5.5 节点内与节点间通信

  范围           常见链路                 主要关注
-------------- ------------------------ ----------------------------
  单 GPU 内部    GPU 本地显存与计算单元   Kernel 与显存访问
  单节点多 GPU   NVLink/NVSwitch、PCIe    GPU P2P、拓扑、带宽
  跨节点         InfiniBand、RoCE         网络带宽、延迟、拥塞、RDMA

常见部署原则是：尽量将通信频繁的 TP
组放在高速节点内互联域；跨节点通信则需要认真评估网络成本。但最终应以实测拓扑和性能为准。

------------------------------------------------------------------------

## 6. 通信性能模型：Latency 与 Bandwidth

一个常见的简化模型：

\[ T`\approx`{=tex}`\alpha`{=tex}+`\frac{M}{B}`{=tex} \]

其中：

-   (T)：通信时间；
-   (`\alpha`{=tex})：启动延迟；
-   (M)：通信数据量；
-   (B)：有效带宽。

这个模型忽略了协议、同步、拥塞、分块和多跳路径等细节，但有助于理解主要瓶颈。

### 6.1 Latency-bound

当数据量较小，(`\alpha`{=tex}) 可能占主导。

例如小 Tensor
的通信，即使链路带宽很高，启动、同步和调度开销也可能占据主要时间。

优化方向：

-   减少通信调用次数；
-   合并小消息；
-   减少不必要的同步；
-   采用更适合小消息的算法或协议。

### 6.2 Bandwidth-bound

当数据量很大，(M/B) 占主导。

优化方向：

-   使用更高带宽的链路；
-   减少传输数据量；
-   优化数据布局；
-   使用合适的分块和通信算法；
-   避免跨节点传输不必要的数据。

### 6.3 All-to-All 的额外复杂性

All-to-All 不仅取决于总数据量，还取决于流量矩阵：

-   每个 Rank 发出多少数据；
-   每个 Rank 接收多少数据；
-   数据目的地是否集中；
-   是否存在热点链路；
-   是否有 Rank 因负载不均而拖慢整体完成时间。

因此，MoE 的 All-to-All 性能不能只用总字节数解释。

------------------------------------------------------------------------

## 7. Ring 与 Tree：常见集合通信算法

集合通信原语是逻辑语义，Ring 和 Tree
是实现这些语义的算法结构。不能把两者混为一谈。

### 7.1 Ring

Ring 将 N 个 Rank 按环连接。以 Ring All-Reduce 为例：

1.  Reduce-Scatter：经过 (N-1) 轮传递与归约，每个 Rank
    得到一个归约分片；
2.  All-Gather：再经过 (N-1) 轮交换，使每个 Rank 获得完整结果。

对于每个 Rank 原始数据量为 (M)，理想 Ring All-Reduce 中，每个 Rank
的发送量约为：

\[ V=`\frac{2(N-1)}{N}`{=tex}M \]

优点：

-   大消息场景下能较好地利用链路带宽；
-   各 Rank 通信负载相对均匀。

限制：

-   通信轮数随 Rank 数量线性增长；
-   小消息时多轮启动成本可能突出；
-   性能依赖环路链路质量和拓扑映射。

### 7.2 Tree

Tree 将 Rank 组织成树状结构。

Reduce 阶段，子节点向父节点汇总；Broadcast 阶段，根节点向下分发。

理想平衡树的通信深度约为：

\[ O(`\log `{=tex}N) \]

优点：

-   对小消息或延迟敏感场景可能有优势；
-   通信步骤数可随 Rank 数量按对数增长。

限制：

-   上层节点或链路可能形成瓶颈；
-   性能依赖树结构与硬件拓扑。

### 7.3 Ring 与 Tree 如何选择？

  维度           Ring             Tree
-------------- ---------------- ------------------------
  结构           环               树
  理想通信深度   (O(N)) 轮        (O(`\log `{=tex}N)) 层
  常见优势       大消息带宽利用   小消息延迟
  风险           多轮传递         上层节点/链路热点

实际 NCCL 会依据消息大小、GPU
拓扑、协议和版本选择算法。不能简单规定"小消息必用 Tree、大消息必用
Ring"。

------------------------------------------------------------------------

## 8. NCCL：GPU 集合通信库

NCCL（NVIDIA Collective Communications Library）为 NVIDIA GPU
提供集合通信和 P2P 通信能力。

常见操作包括：

-   `ncclSend` / `ncclRecv`
-   `ncclBroadcast`
-   `ncclReduce`
-   `ncclAllReduce`
-   `ncclAllGather`
-   `ncclReduceScatter`
-   All-to-All 能力取决于 NCCL 版本和所使用的 API/封装。

PyTorch Distributed、Megatron 等框架可以通过 NCCL 后端执行 GPU 通信。

### 8.1 一次 NCCL 操作背后可能发生什么？

以 All-Reduce 为例，底层实现可能涉及：

1.  识别通信组和 Rank；
2.  根据拓扑选择通信路径；
3.  根据消息大小选择算法；
4.  选择通信协议；
5.  将数据切分为 Chunk；
6.  调度 GPU 间传输和归约；
7.  处理 Stream 依赖与完成状态。

所以，一次高层 `all_reduce()` 调用并不意味着底层只有一次简单的数据复制。

### 8.2 常见 NCCL Protocol

常见协议名称：

-   **LL**：Low Latency，面向低延迟；
-   **LL128**：低延迟与带宽之间的一种折中设计；
-   **Simple**：适合较大数据块的传输路径。

协议选择与 NCCL 版本、硬件、消息大小和算法有关，通常由 NCCL 自动决定。

------------------------------------------------------------------------

## 9. 通信与计算重叠

通信和计算重叠（Communication-Compute Overlap）是提高 GPU
利用率的重要手段。

假设：

-   计算耗时 10 ms；
-   通信耗时 5 ms。

如果完全串行：

\[ T=10+5=15ms \]

如果理想地完全重叠：

\[ T=`\max`{=tex}(10,5)=10ms \]

真实情况通常介于两者之间。

常见技术：

-   CUDA Streams；
-   异步 P2P；
-   NCCL 异步通信；
-   Tensor 分块；
-   Double Buffering；
-   Pipeline Scheduling。

需要注意：

-   CPU API 返回不等于 GPU 通信完成；
-   通信结果被后续 Kernel 使用前，必须建立正确的 Stream 依赖；
-   过度重叠也可能争用 GPU、显存带宽或互联资源。

------------------------------------------------------------------------

## 10. CUDA Stream 与分布式通信的同步

在 PyTorch 中，示意代码：

``` python
import torch
import torch.distributed as dist

tensor = torch.randn(1024, device="cuda")

dist.all_reduce(
    tensor,
    op=dist.ReduceOp.SUM
)
```

如果使用 NCCL 后端，通信操作会被提交到相应的 GPU 执行路径。

要区分：

  概念            含义
--------------- --------------------------------
  Host API 返回   CPU 端调用返回
  通信任务完成    GPU 端已完成通信
  Stream 同步     等待指定 Stream 上相关操作完成

在异步执行环境中，必须正确管理 Stream
间依赖。否则可能出现读取未完成通信结果、隐式同步或性能下降。

### 10.1 Collective 调用一致性

同一通信组中的 Rank 必须按照兼容的顺序参与 Collective。

例如所有 Rank 都执行同一组 All-Reduce：

``` text
Rank 0: AllReduce(A)
Rank 1: AllReduce(A)
Rank 2: AllReduce(A)
Rank 3: AllReduce(A)
```

若部分 Rank 执行不同的
Collective，或调用顺序不一致，可能导致通信不匹配、挂起或错误。

排查时检查：

-   Rank 与 Process Group 是否正确；
-   Collective 调用顺序是否一致；
-   Tensor Shape、Dtype 和操作是否符合约定；
-   Stream 依赖是否正确；
-   是否有 Rank 提前退出或异常。

------------------------------------------------------------------------

## 11. 通信原语与并行策略的对应关系

  并行策略           主要通信                                 典型通信内容
------------------ ---------------------------------------- -----------------------------------
  DP（训练）         All-Reduce、Reduce-Scatter 等            梯度或梯度分片
  DP（推理副本）     通常不需要副本间梯度通信                 请求由调度器分发
  TP                 All-Reduce、All-Gather、Reduce-Scatter   层内张量和局部计算结果
  Megatron 经典 SP   All-Gather、Reduce-Scatter               Sequence 分片激活值
  PP                 P2P Send/Recv                            Stage 边界激活值
  EP                 All-to-All 或分层交换                    Token Dispatch 与 Combine
  CP                 K/V 交换、Ring Attention 等              长上下文分片及 Attention 所需数据

### 11.1 DP

训练时不同 GPU 处理不同
Batch，梯度需要同步。推理时多个独立副本通常各自处理请求，不需要梯度
All-Reduce。

### 11.2 TP

TP Rank
共同完成层内计算，频繁使用张量归约或数据重分布。它对节点内高速互联尤其敏感。

### 11.3 SP

经典 SP 通过 All-Gather 和 Reduce-Scatter
等操作调整激活值布局，使部分算子只需处理本地 Sequence 分片。

### 11.4 PP

相邻 Stage 通过 P2P 传递激活值。优化重点包括 Stage
均衡、消息大小、Micro-batch 调度和通信计算重叠。

### 11.5 EP

MoE 中 Token 按 Router 结果被发送到 Expert 所在 GPU。Dispatch 和 Combine
通常涉及 All-to-All 或分层交换，需重点分析负载均衡与跨节点流量。

------------------------------------------------------------------------

## 12. 如何判断通信是否成为推理瓶颈？

不要只看 GPU 利用率或单一带宽数字。建议从端到端指标和时间线两方面判断。

### 12.1 先定义服务指标

常见 LLM 推理指标：

-   **TTFT**：Time To First Token，首 Token 延迟；
-   **TPOT**：Time Per Output Token，输出 Token 平均间隔；
-   **Throughput**：单位时间处理的 Token 数或请求数；
-   **Latency**：请求端到端延迟。

Prefill 通常对 TTFT 影响明显，Decode 通常对 TPOT
和持续吞吐量影响明显，但具体关系依调度与模型实现而变。

### 12.2 常见症状与排查方向

----------------------------------------------------------------------------------
  现象                    可能原因                           排查方向
----------------------- ---------------------------------- -----------------------
  增大 TP 后反而变慢      通信成本超过计算收益               对比不同
                                                             TP、检查节点拓扑

  多节点 TP 明显退化      跨节点通信频繁                     检查网络和 Rank 映射

  All-Reduce 耗时高       带宽不足或算法/拓扑不匹配          NCCL Tests、拓扑检查

  EP 吞吐低               All-to-All、负载不均或小消息延迟   Token 分布、网络拥塞

  GPU 利用率低            等待通信、调度或负载不均           Nsight Systems 时间线

  PP 吞吐低               Pipeline Bubble 或 Stage 不均衡    Micro-batch 与 Stage
                                                             负载

  Decode 延迟高           小消息同步、TP 通信、KV Cache 压力 TPOT 分解、通信时间线
  ----------------------------------------------------------------------------------

### 12.3 计算通信占比

可先做粗略估算：

\[ R\_{comm}=`\frac{T_{comm}}{T_{total}}`{=tex} \]

但若通信与计算重叠，不能简单把两者耗时相加。应结合时间线，识别关键路径上的等待时间和资源争用。

------------------------------------------------------------------------

## 13. 常用性能分析工具

  工具               用途
------------------ --------------------------------------------------------
  NCCL Tests         测试 All-Reduce、All-Gather、Reduce-Scatter 等集合通信
  Nsight Systems     查看 Kernel、通信、CPU 调度与 Stream 时间线
  Nsight Compute     分析 GPU Kernel 的计算和内存性能
  `nvidia-smi`       查看 GPU 利用率、显存、功耗等
  PyTorch Profiler   分析 PyTorch 算子和通信调用

### 13.1 NCCL Tests 示例

All-Reduce Benchmark：

``` bash
./build/all_reduce_perf \
    -b 8M \
    -e 1G \
    -f 2 \
    -g 8
```

参数：

-   `-b 8M`：最小消息大小；
-   `-e 1G`：最大消息大小；
-   `-f 2`：消息大小按 2 倍递增；
-   `-g 8`：每个进程使用的 GPU 数量，具体运行方式还需结合 NCCL Tests
    的进程启动配置。

重点观察：

-   不同消息大小的延迟；
-   Algorithm Bandwidth；
-   Bus Bandwidth；
-   GPU 数量变化后的扩展效率；
-   节点内与跨节点性能差异。

注意：NCCL Tests 的带宽指标有特定统计口径，不能直接等同于模型的有效
Token 吞吐量。

------------------------------------------------------------------------

## 14. 工程优化 checklist

### 硬件与拓扑

-   [ ] 确认 GPU 型号、NVLink 代际和链路数量；
-   [ ] 确认 NVSwitch、PCIe Switch 和 NUMA 拓扑；
-   [ ] 确认 NIC 与 GPU 的亲和关系；
-   [ ] 确认跨节点网络是 InfiniBand 还是 RoCE；
-   [ ] 验证 GPUDirect RDMA 是否真正启用。

### 通信算法与数据布局

-   [ ] 确认 Collective 的消息大小和调用频率；
-   [ ] 检查 TP 是否跨节点；
-   [ ] 检查 All-to-All 的发送/接收流量分布；
-   [ ] 检查是否存在不必要的 All-Gather 或 All-Reduce；
-   [ ] 评估分块、融合和通信计算重叠。

### 运行时与同步

-   [ ] 确认所有 Rank 的 Collective 顺序一致；
-   [ ] 检查 CUDA Stream 依赖；
-   [ ] 排查隐式同步；
-   [ ] 检查通信失败、超时和 Rank 异常；
-   [ ] 使用 Profiling 验证实际关键路径。

------------------------------------------------------------------------

## 15. 总结：建立正确的 GPU 通信心智模型

GPU 通信可以分成三个层次：

**第一层：通信语义**

-   Broadcast：一份数据发给所有人；
-   Reduce：多份数据归约到一个 Rank；
-   All-Reduce：归约后所有 Rank 都得到结果；
-   All-Gather：收集并拼接所有分片；
-   Reduce-Scatter：归约后分片；
-   All-to-All：按目的地交换数据；
-   Send/Recv：指定 Rank 之间传输。

**第二层：通信实现**

-   Ring、Tree 等算法；
-   NCCL 的算法和协议选择；
-   Chunk、异步执行和 Stream 同步；
-   通信计算重叠。

**第三层：硬件路径**

-   节点内：NVLink、NVSwitch、PCIe；
-   节点间：InfiniBand、RoCE、RDMA、GPUDirect RDMA。

把这三个层次与 DP、TP、SP、PP、EP
对应起来，才能真正理解一个分布式推理配置的性能表现。

最重要的工程结论：

1.  TP 通常受层内高频通信影响，优先考虑高速节点内互联；
2.  EP 通常需要重点优化 Token Dispatch/Combine、All-to-All 和负载均衡；
3.  PP 需要关注 Stage 负载与流水线 Bubble；
4.  小消息更容易受延迟影响，大消息更容易受带宽影响；
5.  最终必须通过 NCCL Tests、Nsight Systems
    和真实推理指标验证，不能只凭理论带宽判断。
