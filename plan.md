# Project 3 方案计划：基于实体感知段落记忆的上下文压缩

---

## 1. 背景与问题定义

### 1.1 核心问题

多轮对话随着轮次增加，上下文长度持续增长。LLM的context window是有限的，
必须在某个时刻决定"保留什么、丢弃什么"。目前主流的三种策略都有明显缺陷：

- **全量输入**：把所有历史都塞进prompt，计算成本高，超出window后直接崩
- **滑动窗口**：只保留最近N轮，简单但早期信息全丢
- **Summarization**：用LLM改写旧的对话，压缩成摘要，但会丢失具体事实

这三种策略的共同失败场景是：**答案在很早的对话里，而且是一个具体的细节
（人名、日期、某个决定）**。这正是LoCoMo benchmark里multi-hop和temporal
类问题的核心挑战。

### 1.2 现有工作的空白

近期最相关的工作是**SeCom**（微软，ICLR 2025），它用RAG的思路解决了这个问题：
不压缩全部历史，而是把对话切成话题段落（segment），存入memory bank，
有新query时检索最相关的段落注入prompt。

SeCom的创新在于发现了**segment粒度**（比session粗，比turn细）是最优的检索单元。
但SeCom继承了其压缩组件LLMLingua-2的一个已知缺陷：

> **基于困惑度的token删除对语义重要性是盲目的。**
> 低频但关键的token（专有名词、日期、数字）因为罕见而困惑度高，
> 但这些恰恰是multi-hop和temporal问题最需要的信息。

LongLLMLingua（微软，ACL 2024）已经承认了这个问题，并提出了**事后**的
子序列恢复算法（压完再修补）。我们认为**事前保护**更简洁、更根本。

### 1.3 我们的方案

在SeCom基础上，增加一个**context map**（对话地图）——一个固定大小、
增量维护的结构化记录，包含对话中的关键实体、决定和时间线。

Map在pipeline里有两个接入点：

1. **压缩时**：把高优先级实体作为`force_tokens`传给LLMLingua-2，防止被删除
2. **检索时**：用map改写模糊的query（解决"他后来怎么样了？"这类指代不清的问题），
   提升BM25和dense retrieval的recall

整个方案**完全不需要训练**，只需要API调用。

---

## 2. 故事线（Paper-Ready版本）

*为什么这样选择，一个连贯的叙事：*

长期对话中积累的内容可以分为两类：
- **闲聊内容**：话题性强，可以摘要，丢了影响不大
- **Commitments（承诺/事实）**：人名、日期、决定、事件，答案依赖于此，
  丢了直接答错

现有压缩方法的根本失败在于**把所有token等同对待**，偏向高频和最近的内容，
系统性地丢弃低频但commitment-carrying的token。

RAG类方法（SeCom）部分解决了这个问题——只把相关的历史注入prompt。
但检索质量依赖两件事：
- (a) 存进去的内容是否保真（compression fidelity）
- (b) 查询是否清晰（query clarity）

SeCom的LLMLingua-2威胁(a)——会删掉实体token；
"他后来怎么样了？"这类模糊query威胁(b)——BM25不知道"他"是谁。

我们的context map同时解决这两个问题。它不是要替代SeCom的segment检索，
而是一个轻量级的全局索引，在两个关键瓶颈处提供精准干预。

**核心实验问题**：这个干预有没有用？在哪类问题上有用？

基于SmartSearch（2026）的分析，我们预期：
- entity protection和query rewriting主要对**multi-hop和temporal**有帮助，
  因为这类问题一个实体miss就彻底答错
- 对**single-hop**问题，SeCom baseline已经够用，map增益接近零

这个**分类别的差异预测**在LoCoMo的标注上直接可验，
而这个差异本身就是贡献——它告诉我们compression策略的选择应该依赖于
query类型，而不是一刀切。

---

## 3. 数据集：LoCoMo

**来源**：`github.com/snap-research/locomo`（Maharana et al., ACL 2024）

**数据规模**：
- 10条超长对话，每条最多35个session，平均300轮，约9K tokens
- 每个QA对包含：`question`、`answer`、`category`、`evidence_dialog_ids`

**问题分类**（直接作为分析维度使用）：

| 类别 | 含义 | 对我们的意义 |
|------|------|------------|
| `single_hop` | 答案在单个session中明确出现 | 控制组，各方法应该差不多 |
| `multi_hop` | 需要跨多个session综合信息 | 核心测试，预期SeCom-MAP赢 |
| `temporal` | 需要时间推理（日期、顺序、间隔） | 核心测试，预期SeCom-MAP赢 |
| `open_domain` | 需要对话外的知识 | 可以包含但不是重点 |
| `adversarial` | 故意误导，正确答案是"不知道" | **过滤掉**，不适合compression评估 |

**采样策略**：
- 过滤掉adversarial类
- 重点采样multi_hop + temporal（各15条左右）
- single_hop作为对照组（约10条）
- 总计：30-40个QA对

**为什么选LoCoMo**：
- 对话足够长（必须压缩）
- QA对有ground truth的evidence turn IDs
- 问题类别有标注（不需要额外打标）
- SeCom（ICLR 2025）也在LoCoMo上评测过，有直接可比的baseline数字

---

## 4. 三种方法详解

### 4.1 Baseline 1 — 滑动窗口（Sliding Window）

**做什么**：只保留最近的N轮对话，丢弃更早的内容。

**控制方式**：扫描 N 轮数（N = [25, 50, 100, 200, 300, 500]），得到不同压缩率。
跑完后计算 `token_reduction = 1 - tokens_used / full_tokens` 作为统一横轴。

`run_all.py` 单点评估默认：`n_turns=150`（约74%压缩率）。

**为什么选这个baseline**：
- 最简单的策略，代表了"不做任何智能处理"的下界
- 预期在single-hop（答案是最近的信息）上表现尚可
- 预期在multi-hop/temporal（答案在很早以前）上直接失败

**实现复杂度**：约20行Python。

---

### 4.2 Baseline 2 — 递归摘要（Recursive Summarization / RecurSum）

**做什么**：维护**一个** running summary，模拟流式处理：
- 每处理一批新 turns（每10轮一批），若 summary + 新 turns 超过 `max_summary_tokens`，
  则调用 LLM 将 summary + 新 turns 压缩成新的 summary（上限 `max_summary_tokens`）；
  否则直接拼接，不调用 LLM。
- 最后一批 turns 保持原文（verbatim）不压缩。

**控制方式**：扫描 `max_summary_tokens` = [100, 300, 800, 2000, 5000, 10000]。
  值越大 → summary 越详细 → 压缩率越低 → accuracy 越高。
  跑完后计算 `token_reduction` 作为统一横轴。

`run_all.py` 单点评估默认：`max_summary_tokens=5000`（约70%压缩率）。

**参考论文**：RecurSum（Wang et al., Neurocomputing 2025, arXiv:2308.15022）

**为什么选这个baseline**：
- 代表了生产环境最常见的做法（Claude的`/compact`、ChatGPT的内部压缩）
- 不需要预知总轮数，可以流式运行（与滑动窗口一样现实可行）
- 预期保留全局主题但丢失具体事实——正是我们要研究的失败模式

**实现复杂度**：约70行Python。每次超出 max_summary_tokens 才触发一次 LLM 调用。

---

### 4.3 方法3 — SeCom-MAP（我们的方法）

**完整pipeline**：

```
离线阶段（每个session处理完后执行）：
  Step 1: 话题切分
    → GPT-4 zero-shot prompt
    → 把session切成topically coherent的segments
    → 参考SeCom §2.2的实现

  Step 2: 更新context map
    → 从新segments中提取：实体、决定、时间线
    → 按重要性打分（出现次数 × 最近性权重）
    → 超出token预算（~300 tokens）时驱逐低分实体

  Step 3: 压缩segments（entity-aware）
    → 获取map中高优先级实体列表
    → 动态分配压缩率：
        含重要实体的segment → 保留率0.85（少压）
        普通segment → 保留率0.65（多压）
    → 调用LLMLingua-2，force_tokens=高优先级实体
    → 存入memory bank（BM25索引）

在线阶段（有新query时）：
  Step 4: Query改写
    → 用context map解析指代词和省略
    → "他后来怎么样了？" → "Alex的PhD申请结果如何？"

  Step 5: 检索
    → 用改写后的query检索top-k segments

  Step 6: 组装prompt
    → [context map摘要] + [检索到的segments] + [query]
    → 送给LLM生成答案
```

**每个组件的选择理由**：

| 组件 | 来自 | 理由 |
|------|------|------|
| Segment粒度 | SeCom (ICLR 2025) | Turn太碎，Session太粗；segment是最优粒度 |
| LLMLingua-2压缩 | SeCom (ICLR 2025) | 去除冗余，提升检索信噪比 |
| **Entity protection** | **我们的贡献** | 防止LLMLingua-2删掉关键实体（LongLLMLingua承认的失败模式） |
| **动态压缩率** | **我们的贡献** | 含重要实体的segment应该少压缩 |
| **Query改写** | **我们的贡献** | 解决指代模糊的query，提升检索recall |
| Context map概念 | PEEK (Gu et al., 2026) | 固定大小的全局导航索引，从语料库级迁移到对话级 |

**不需要训练**：GPT-4/Claude API做切分和改写；LLMLingua-2是预训练模型直接推理。

---

### 4.4 消融实验设计（Ablation）

为了定量区分两个map组件的贡献：

| 变体 | Entity Protection | Query Rewriting |
|------|------------------|-----------------|
| SeCom-only | ✗ | ✗ |
| SeCom + EP | ✓ | ✗ |
| SeCom + QR | ✗ | ✓ |
| SeCom-MAP | ✓ | ✓ |

这个2×2消融设计可以清晰回答：是哪个组件带来了增益？

---

## 5. 评估方案（严格对应Assignment 3要求）

### 5.1 核心指标

**指标1 — Token Reduction（token压缩率）**

```
token_reduction = 1 - (放入prompt的tokens / 完整历史的tokens)
```

- 每个样本独立计算，报告均值 ± 标准差
- 注意：SeCom是检索式的，"放入prompt的tokens"定义为检索到的top-k segments的tokens

**指标2 — Answer Correctness（答案正确率，LLM-as-Judge）**

Judge prompt（binary 0/1）：
```
参考答案（来自完整上下文）：{reference}
候选答案（来自压缩上下文）：{candidate}
问题：{question}

候选答案是否正确回答了问题？
只回答0（错误）或1（正确），不需要解释。
```

- Judge模型：GPT-4o（与生成答案的模型分开）
- 每个样本独立打分
- 聚合：accuracy = 所有样本的0/1均值
- 参考文献：MT-Bench（Zheng et al., NeurIPS 2023）

### 5.2 Cliff Point分析（Assignment明确要求）

**设计**：每个方法扫描自己的**内部控制参数**，跑完后统一换算为 `token_reduction`
作为横轴，三条曲线画在同一张图上。

| 方法 | 扫描的参数 | 参数值 |
|------|----------|--------|
| Sliding Window | N（保留轮数）| [25, 50, 100, 200, 300, 500] |
| Summarization | max_summary_tokens | [100, 300, 800, 2000, 5000, 10000] |
| SeCom-MAP | token_budget（检索填充量）| [1000, 3000, 5000, 7000, 9000, 12000] |

每个点的横轴 = `token_reduction = 1 - tokens_used / full_tokens`（跑完后计算）。
覆盖约 40–95% 压缩率范围。

**额外产出——内部参数图**：
除统一对比图外，额外画三张子图，各方法用自己的内部参数作横轴，展示各方法
自身的 cliff 在什么参数值附近发生。

画图：
1. `pareto_frontier.png`：x=token_reduction，y=accuracy，三条曲线对比（主图）
2. `internal_params.png`：三个子图，各方法内部参数 vs accuracy（辅助图）

**Cliff point定义**：曲线斜率变化最大的点，即quality下降速度开始超过token节省速度的位置。

```
answer correctness
    |
100%|  ★ full context
    |
 80%|  ●—— SeCom-MAP
    |    ╲  ●——●
 70%|        ╲     ●—— Summarization
    |          ╲  ●——●
 60%|            ╲       ●—— Sliding Window
    |         cliff╲  ●——●
 40%|                ╲●
    |__________________________________
    0%   20%   40%   60%   80%  100%
              token reduction
```

理想情况下SeCom-MAP的曲线**整体更靠右上**——同样压缩率下质量更高，cliff point出现得更晚。

### 5.3 分类别分析（我们的核心贡献）

在LoCoMo的question category标签上分层报告：

```
              | Sliding Window | Summarization | SeCom-only | SeCom-MAP |
single_hop    |                |               |            |           |
multi_hop     |                |               |            |           |
temporal      |                |               |            |           |
```

**预设假设（可被实验验证或证伪）**：

- H1：SeCom-MAP > SeCom-only on multi_hop & temporal（map有效）
- H2：SeCom-MAP ≈ SeCom-only on single_hop（map中性）
- H3：所有检索方法 > summarization on multi_hop（RAG paradigm优势）
- H4：Sliding window在非常近的single_hop问题上反而最好

### 5.4 Failure Case分析（Assignment明确要求）

找一个SeCom-MAP仍然答错、但full context答对的具体对话样本，解释机制。

预期的失败模式：
```
对话中有个人物X只在session 2提到了一次
session 8的问题用代词指代X
→ X在map里只有1次mention，重要性分低
→ map的token预算不够时X被驱逐
→ query改写失败，仍然是模糊指代
→ BM25检索到错误的segment
→ 答案错误
```

这个failure case很自然地引出"下一步"：重要性评分应该考虑
"该实体是否后来成为了问题的主体"——但这个信号在inference time没有，
只有在offline分析时才能看到evidence_turn_ids。

---

## 6. 参考论文

| 论文 | 发表地点 | 在本项目的作用 |
|------|---------|--------------|
| Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM Agents" | ACL 2024 | 数据集（LoCoMo） |
| Pan et al., "SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents" | ICLR 2025 | 方法3的核心基础 |
| Pan et al., "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression" | ACL 2024 | 压缩组件 |
| Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference of LLMs" | EMNLP 2023 | 压缩背景 |
| Jiang et al., "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios" | ACL 2024 | Entity protection的motivation来源 |
| Wang et al., "Recursively Summarizing Enables Long-Term Dialogue Memory in LLMs" | Neurocomputing 2025 | Baseline 2的参考实现 |
| Gu et al., "PEEK: Context Map as an Orientation Cache for Long-Context LLM Agents" | arXiv 2026 | Context map概念来源 |
| Derehag et al., "SmartSearch: How Ranking Beats Structure for Conversational Memory Retrieval" | arXiv 2026 | 批判性分析：结构化map什么时候没用 |
| Xiao et al., "StreamingLLM: Efficient Streaming Language Models with Attention Sinks" | ICLR 2024 | 滑动窗口的理论背景 |
| Zheng et al., "Judging LLM-as-a-Judge with MT-Bench" | NeurIPS 2023 | LLM-as-judge评估方法论 |

---

## 7. 代码实现步骤

### Step 0：环境配置（30分钟）

```bash
git clone https://github.com/snap-research/locomo
git clone https://github.com/microsoft/SeCom
pip install llmlingua python-dotenv openai faiss-cpu rank_bm25 tiktoken
```

配置`.env`：
```
OPENAI_API_KEY=你的key
```

---

### Step 1：数据加载与采样（1小时）

**文件**：`data/load_locomo.py`

任务：
- 加载`locomo10.json`
- 解析QA对，提取：question, answer, category, evidence_dialog_ids
- 过滤掉adversarial类
- 采样30-40个QA对（multi_hop + temporal为主）
- 对每个QA对，提取该问题之前的完整对话历史

输出格式`data/samples.json`：
```json
{
  "id": "conv1_q3",
  "question": "...",
  "reference_answer": "...",
  "category": "multi_hop",
  "full_history": [["用户说的", "agent回答"], ...],
  "evidence_turn_ids": [3, 17]
}
```

---

### Step 2：滑动窗口Baseline（1小时）

**文件**：`methods/sliding_window.py`

```python
def sliding_window(history, token_budget=2000):
    """
    从最新的turn开始往前数，
    直到超过token_budget为止，
    返回能放下的turns和实际用掉的token数
    """
```

**文件**：`eval/run_sliding_window.py`

- 对每个样本跑sliding_window
- 调用LLM得到答案
- 保存结果到`results/sliding_window.json`

---

### Step 3：Summarization Baseline（2小时）

**文件**：`methods/summarization.py`

```python
def recursive_summarize(history, token_budget=2000, summarize_every=10):
    """
    把历史分成chunks，
    对旧的chunks调用LLM做摘要，
    最近的turns verbatim保留，
    返回：[running_summary] + [recent_verbatim_turns]
    """
```

摘要prompt：
```
请将以下对话轮次总结成2-3句话，
必须保留：人名、日期、数字、具体决定。
不要添加原文没有的信息。
```

---

### Step 4：Context Map实现（2小时）

**文件**：`methods/context_map.py`

```python
class ContextMap:
    def __init__(self, token_budget=300):
        self.entities = {}    # 实体名 -> {类型, 出现次数, 最近session}
        self.decisions = []   # 确认的事实/决定
        self.timeline = []    # (时间, 事件) 列表
        self.token_budget = token_budget

    def update(self, new_turns):
        # 调用LLM从new_turns中提取实体/决定/时间线
        # 合并到现有map
        # 超出budget时驱逐低重要性实体

    def get_high_priority_entities(self, top_k=20):
        # 按重要性排序返回实体字符串列表
        # 重要性 = 出现次数 × 最近性权重

    def rewrite_query(self, query):
        # 调用LLM，输入：map内容 + 原始query
        # 解析代词和省略
        # 返回改写后的query

    def to_prompt_string(self):
        # 把map序列化成紧凑的文本，用于注入prompt
```

Map更新prompt：
```
根据以下对话，提取：
1. 命名实体（人名、地点、日期、组织）
2. 确认的事实或决定
3. 带时间标记的事件

对话内容：{turns}

以JSON格式返回，不要有其他内容：
{"entities": [...], "decisions": [...], "timeline": [...]}
```

---

### Step 5：SeCom-MAP集成（2小时）

**文件**：`methods/secom_map.py`

```python
from secom import SeCom
from llmlingua import PromptCompressor
from methods.context_map import ContextMap

class SeComMAP:
    def __init__(self):
        self.segmenter = SeCom(granularity="segment")
        self.compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True
        )
        self.context_map = ContextMap(token_budget=300)
        self.memory_bank = []  # 存compressed segments

    def build_memory(self, history):
        # Step 1: 切分segments
        segments = self.segmenter.segment(history)
        # Step 2: 更新map
        self.context_map.update(history)
        # Step 3: entity-aware压缩
        protected_entities = self.context_map.get_high_priority_entities()
        for seg in segments:
            importance = self._calc_importance(seg, protected_entities)
            rate = 0.85 if importance > 0.5 else 0.65
            compressed = self.compressor.compress_prompt(
                seg,
                rate=rate,
                force_tokens=protected_entities
            )
            self.memory_bank.append(compressed)

    def answer(self, query):
        # Step 4: 改写query
        rewritten_query = self.context_map.rewrite_query(query)
        # Step 5: 检索
        retrieved = self.segmenter.get_memory(
            [rewritten_query],
            [self.memory_bank]
        )
        # Step 6: 组装prompt
        prompt = self._assemble_prompt(query, retrieved)
        return prompt, rewritten_query

    def _assemble_prompt(self, query, retrieved):
        map_str = self.context_map.to_prompt_string()
        return f"[对话背景]\n{map_str}\n\n[相关历史]\n{retrieved}\n\n问题：{query}"
```

---

### Step 6：评估Harness（2小时）

**文件**：`eval/judge.py`

```python
def llm_judge(question, reference_answer, candidate_answer):
    """
    返回0或1
    0 = 候选答案不正确
    1 = 候选答案正确
    """
    prompt = f"""
参考答案：{reference_answer}
候选答案：{candidate_answer}
问题：{question}

候选答案是否正确回答了问题？只回答0或1。
"""
    response = call_gpt4o(prompt)
    return int(response.strip())
```

**文件**：`eval/run_all.py`

```python
methods = {
    "sliding_window": run_sliding_window,
    "summarization": run_summarization,
    "secom_only": run_secom_only,      # SeCom不带map
    "secom_ep": run_secom_ep,          # 只加entity protection
    "secom_qr": run_secom_qr,          # 只加query rewriting
    "secom_map": run_secom_map,        # 完整SeCom-MAP
}

for method_name, method_fn in methods.items():
    results = []
    for sample in load_samples():
        answer, tokens_used = method_fn(sample)
        correctness = llm_judge(
            sample["question"],
            sample["reference_answer"],
            answer
        )
        results.append({
            "id": sample["id"],
            "category": sample["category"],
            "correctness": correctness,
            "tokens_used": tokens_used,
            "full_tokens": count_tokens(sample["full_history"]),
        })
    save_json(results, f"results/{method_name}.json")
```

---

### Step 7：Cliff Point实验（1小时）

**文件**：`eval/cliff_analysis.py`

```python
# 统一token budget作为控制变量，三个方法共用同一组budget值
TOKEN_BUDGETS = [200, 400, 600, 800, 1000, 1500, 2000, 3000]

# 每个方法在给定budget下贪心地填充最有用的内容
# sliding_window: 从最近turn往前填
# summarization:  调整摘要长度fit到budget
# secom_map:      压缩率固定，检索阶段贪心加segment直到用完budget
#                 (pipeline build_memory只构建一次，跨budget复用)

# 对每个(method, budget)跑一遍评估
# 记录 (token_reduction, accuracy) 点对
# 三条曲线画在同一张Pareto frontier图上
```

**输出**：`results/pareto_frontier.png`

---

### Step 8：结果分析与README（2小时）

**文件**：`analysis/plot_results.py`

产出图表：
1. **主结果表**：每个方法的token_reduction + overall accuracy
2. **分类别表**：3×6的矩阵（3个category × 6个方法）
3. **Pareto frontier曲线**：三个方法画在同一张图，x轴统一用token_reduction（由token budget控制）
4. **消融实验表**：2×2 grid（EP × QR）

**文件**：`README.md`

必须包含：
- 为什么选这个assignment
- 数据集选择和采样说明
- 每个方法的决策和排除的替代方案
- Headline数字
- 一个具体的failure case + 机制解释
- 如果再有一周会做什么

**文件**：`Makefile`

```makefile
run:
    python data/load_locomo.py
    python eval/run_all.py
    python eval/cliff_analysis.py
    python analysis/plot_results.py
```

---

## 8. 时间规划

| 时间 | 任务 | 估计代码量 |
|------|------|---------|
| Day 1 上午 | Step 0-2：环境、数据、滑动窗口 | ~80行 |
| Day 1 下午 | Step 3-4：summarization、context map | ~150行 |
| Day 2 上午 | Step 5：SeCom集成 | ~150行 |
| Day 2 下午 | Step 6-7：评估harness、cliff分析 | ~150行 |
| Day 2 晚上 | Step 8：画图、README、Makefile | ~80行 |

**总计**：约600行Python + 若干prompt文本。全部无训练，纯API调用。

---

## 9. 预期结果与结论

**预测数字**（这些是预期方向，实际跑出来以实验为准）：

| 方法 | Token压缩率 | 总体准确率 | Multi-hop准确率 | Temporal准确率 |
|------|-----------|---------|----------------|--------------|
| Full context | 0% | ~85% | ~75% | ~70% |
| Sliding window | ~70% | ~60% | ~35% | ~40% |
| Summarization | ~75% | ~65% | ~50% | ~45% |
| SeCom-only | ~65% | ~72% | ~65% | ~62% |
| SeCom-MAP（我们的） | ~65% | ~76% | ~72% | ~70% |

**核心结论（写进README的那段话）**：

> 基于检索的方法（SeCom）在multi-hop和temporal问题上显著优于
> 基于压缩的方法（summarization、sliding window），
> 这证实了query-specific的上下文选择比均匀压缩在事实密集型问题上更鲁棒。
> 在检索范式内部，entity-aware的context map在LLMLingua-2
> 最容易丢失关键token的地方提供了有针对性的保护，
> 验证了事前保护相对于LongLLMLingua事后修复的改进。
> 然而，这个增益在single-hop问题上几乎为零，
> 说明压缩策略的选择应该依赖于预期的query类型。

**Failure case（必须写）**：

> 在对话X中，一个次要人物只在session 2被提及一次，
> session 8的问题用代词指代他。
> 由于单次提及导致重要性分数低，该实体在map token预算压缩时被驱逐。
> query改写因此失败，BM25检索到了错误的segment，最终答错。
> 根本原因：我们的重要性评分无法预测"这个实体未来会被问到"。
> 解决方向：如果能在offline阶段利用evidence_turn_ids反向标注重要实体，
> 可以提升map的coverage——但这个信号在inference time不可用，
> 是一个有趣的train/test asymmetry问题。

---

*本方案满足Assignment 3的全部要求：
3种压缩策略、30+对话样本、token reduction + answer correctness双指标、
cliff point分析、策略推荐与论证、failure case机制解释。*