# TurboQuant KV Cache Quantization — 系统测试计划

## 背景

TurboQuant 实现了基于 Walsh-Hadamard Transform + Lloyd-Max 最优标量量化的 KV Cache 压缩方案（arXiv 2504.19874），支持 2/3/4-bit 量化及 K/V 非对称量化。核心技术特性：

- **Query Rotation**：将 WHT 逆变换从 KV dequant 路径移至 Q 端，decode 阶段 dequant 从 7+ ops 降至 3 ops
- **Fused Triton Decode Kernel**：直接读取 packed uint8 KV 做 attention，零 dequant buffer，2 kernel launches
- **Shared Dequant Buffer**：extend 路径共享单对 bf16 buffer（替代 per-layer），节省 ~97% dequant 内存
- **True 3-bit Packing**：10 values per int32（30/32 bits），4.3x 压缩比
- **CUDA Graph 兼容**：4-bit symmetric decode 路径全部操作固定 shape，支持 CUDA graph capture

## 测试矩阵

### 模型选择

| 模型 | 参数量 | K/V Norm Ratio | 预期难度 | 选择理由 |
|------|--------|---------------|---------|---------|
| Mistral-7B-Instruct-v0.3 | 7B | ~1.3x | 低 | K/V norm 接近，量化误差小，最佳场景 |
| Llama-3.1-8B-Instruct | 8B | ~2x | 低 | 论文主测模型，社区基准 |
| Qwen2.5-7B-Instruct | 7B | ~106x | 高 | K norm 远大于 V，量化压力大，压力测试 |

### 量化配置

| 配置 | Decode 路径 | CUDA Graph | 说明 |
|------|------------|-----------|------|
| `bf16` | 标准 | 支持 | Baseline |
| `fp8_e5m2` | 标准（fused dequant） | 支持 | 竞品 baseline |
| `turboquant_4bit` | Fused Triton kernel | 支持 | 主推配置 |
| `turboquant_3bit` | Dequant buffer fallback | 不支持 | 真 3-bit packing |
| `turboquant_k4v2` | Dequant buffer fallback | 不支持 | K/V 非对称 |

---

## 一、精度测试

### 1.1 Perplexity（核心精度指标）

| 项目 | 说明 |
|------|------|
| 工具 | lm-eval-harness |
| 数据集 | WikiText-2 |
| 测试矩阵 | 3 模型 × 5 量化配置 = 15 组 |
| 报告指标 | PPL 绝对值 + 相对 bf16 的退化百分比 |
| 通过条件 | turboquant_4bit PPL 退化 < 2% vs bf16 |

**结果记录表：**

| 模型 | bf16 | fp8_e5m2 | TQ-4bit | TQ-3bit | TQ-k4v2 |
|------|------|---------|---------|---------|---------|
| Mistral-7B | | | | | |
| Llama-3.1-8B | | | | | |
| Qwen2.5-7B | | | | | |

### 1.2 标准 Benchmark

| 项目 | 说明 |
|------|------|
| 工具 | lm-eval-harness |
| 模型 | Llama-3.1-8B-Instruct + Qwen2.5-7B-Instruct |
| 配置 | bf16 / fp8 / turboquant_4bit |

| Benchmark | 测试能力 | 通过条件 |
|-----------|---------|---------|
| MMLU (5-shot) | 通用知识与推理 | 退化 < 1% vs bf16 |
| GSM8K (8-shot, CoT) | 数学推理 | 退化 < 1% vs bf16 |
| HumanEval (pass@1) | 代码生成 | 退化 < 2% vs bf16 |
| GPQA (0-shot) | 科学推理 | 记录退化幅度（PR #21419 报告 ~10% 退化，重点关注） |

**结果记录表：**

| Benchmark | 模型 | bf16 | fp8 | TQ-4bit | 退化 |
|-----------|------|------|-----|---------|------|
| MMLU | Llama-3.1-8B | | | | |
| MMLU | Qwen2.5-7B | | | | |
| GSM8K | Llama-3.1-8B | | | | |
| GSM8K | Qwen2.5-7B | | | | |
| HumanEval | Llama-3.1-8B | | | | |
| GPQA | Llama-3.1-8B | | | | |
| GPQA | Qwen2.5-7B | | | | |

### 1.3 长上下文精度

| 项目 | 说明 |
|------|------|
| 模型 | Llama-3.1-8B-Instruct（原生 128K 上下文） |
| 配置 | bf16 / turboquant_4bit |

| Benchmark | 上下文长度 | 通过条件 |
|-----------|-----------|---------|
| NIAH（Needle-in-a-Haystack） | 4K / 16K / 64K / 128K | recall ≥ 0.99 |
| LongBench-E | 混合 6 类任务 | avg ≥ 49.5（论文 3.5bit 基线 50.06） |

**LongBench-E 结果记录表：**

| 方法 | SingleQA | MultiQA | Summarization | Few-shot | Synthetic | Code | Avg |
|------|----------|---------|---------------|----------|-----------|------|-----|
| bf16 | | | | | | | |
| TQ-4bit | | | | | | | |

---

## 二、性能测试

### 2.1 Decode 吞吐与延迟

| 项目 | 说明 |
|------|------|
| 模型 | Llama-3.1-8B-Instruct |
| 硬件 | 记录 GPU 型号、Driver 版本、CUDA 版本 |
| 工具 | sglang bench_serving |
| 参数 | num-prompts=200, input-len=512, output-len=128 |
| 配置 | bf16 / fp8 / turboquant_4bit（均开 CUDA graph） |

| 指标 | bf16 | fp8 | TQ-4bit | TQ/bf16 比 |
|------|------|-----|---------|-----------|
| Throughput (output tok/s) | | | | |
| TPOT median (ms) | | | | |
| TPOT P99 (ms) | | | | |
| TTFT median (ms) | | | | |
| TTFT P99 (ms) | | | | |

通过条件：TQ-4bit throughput ≥ 80% of bf16

### 2.2 Prefill 吞吐

| 项目 | 说明 |
|------|------|
| 参数 | num-prompts=100, input-len=4096, output-len=1 |
| 配置 | bf16 / turboquant_4bit |

| 指标 | bf16 | TQ-4bit |
|------|------|---------|
| Prefill throughput (tok/s) | | |

### 2.3 CUDA Graph 验证

| 检查项 | 预期 | 实际 |
|--------|------|------|
| Server log 中 CUDA graph capture 成功 | 无 fallback warning | |
| 7 decode batch size capture | 全部成功 | |
| Piecewise capture (extend) | 全部成功 | |
| Decode 吞吐 graph ON vs OFF | graph ON 更快 | |

---

## 三、内存测试

### 3.1 KV Cache 压缩比

| 项目 | 说明 |
|------|------|
| 模型 | Llama-3.1-8B-Instruct |
| 参数 | mem-fraction-static=0.85 |

| 配置 | max_total_tokens | 压缩比 vs bf16 |
|------|-----------------|---------------|
| bf16 | | 1.0x |
| fp8_e5m2 | | |
| turboquant_4bit | | |
| turboquant_3bit | | |
| turboquant_k4v2 | | |

通过条件：turboquant_4bit ≥ 3.0x

### 3.2 峰值 GPU 内存

| 配置 | nvidia-smi 峰值 (MiB) |
|------|----------------------|
| bf16 | |
| turboquant_4bit | |

---

## 四、稳定性测试

### 4.1 持续负载

| 项目 | 说明 |
|------|------|
| 模型 | Llama-3.1-8B-Instruct + turboquant_4bit |
| 参数 | num-prompts=500, request-rate=5 |
| 持续时间 | ~15-20 分钟 |

| 检查项 | 通过条件 | 实际 |
|--------|---------|------|
| Throughput 波动 | < 10% | |
| GPU 内存是否稳定 | 不持续增长 | |
| 请求成功率 | 100% | |

### 4.2 Chunked Prefill

| 项目 | 说明 |
|------|------|
| 参数 | chunked-prefill-size=4096, 输入 >8K tokens |
| 通过条件 | 不 crash，输出连贯 |

### 4.3 高并发

| 项目 | 说明 |
|------|------|
| 参数 | num-prompts=100, request-rate=50 |
| 通过条件 | 不 crash，请求全部返回 |

---

## 五、对比基准（PR #21419 参考数据）

| 指标 | PR #21419 | 本实现 |
|------|----------|--------|
| Decode 吞吐 vs bf16 | 38-71% | 85.4% |
| 内存压缩比 | 3.37x | 3.41x |
| CUDA Graph | 不支持 | 支持 |
| 稳定性 | 多个 crash | 200 req 零 crash |
| 3-bit packing | 假 3-bit（=4-bit 存储） | 真 3-bit（10/int32） |
| K/V 非对称量化 | 无 | 有 |
| GSM8K 精度 | 与 fp8 相当 | 待测 |
| GPQA 精度 | ~10% 退化 | 待测 |

---

## 六、测试环境记录

| 项目 | 值 |
|------|------|
| GPU | |
| GPU 数量 | |
| Driver 版本 | |
| CUDA 版本 | |
| PyTorch 版本 | |
| sglang commit | |
| 测试日期 | |

---

## 七、测试优先级

| 优先级 | 测试项 | 阻塞关系 |
|--------|--------|---------|
| P0 | 1.1 PPL + 1.2 MMLU/GSM8K | 不通过则不提交 |
| P0 | 2.1 Decode 吞吐 + 2.3 CUDA Graph | 不通过则不提交 |
| P1 | 1.3 NIAH/LongBench + 3.1 内存压缩 | 影响 PR 说服力 |
| P2 | 4.1-4.3 稳定性 | 上线前必须通过 |

**执行顺序：P0 → P1 → P2。P0 不通过，后续不用跑。**

---

## 八、PR 报告格式建议

PR description 中至少包含以下表格：

1. **Accuracy Table** — PPL + MMLU + GSM8K（bf16 / fp8 / TQ-4bit，2-3 个模型）
2. **Performance Table** — throughput + TPOT + memory（bf16 / fp8 / TQ-4bit）
3. **Long Context Table** — NIAH recall + LongBench-E avg
4. **Configuration** — GPU 型号、driver 版本、CUDA 版本、sglang commit

**重要原则：好的数字和差的数字都要报，展示 tradeoff 而非只展示优势。**
