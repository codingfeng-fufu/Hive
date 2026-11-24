# HiVG-RAG 后续研究计划 · IEEE ICME

当前路线、历次停损原因及重开条件见 [研究路径图](research-path-map.md)。下文保留原 HiVG-RAG 的实验安排与历史记录。

按 **IEEE ICME 2027 Regular Paper** 排期。官网列出的截稿日是 **2026 年 12 月 30 日（AoE）**。初稿最多 6 页，包含图表和参考文献，采用双盲评审。安排 11 月 23 日至 12 月 20 日为四周执行窗口，12 月 21–30 日用于补跑关键实验、核对结果和提交。 [官方重要日期](https://2027.ieeeicme.org/about/important-dates) · [作者指南](https://2027.ieeeicme.org/authors/initial-author-instructions)

## 当前证据与论文主线

HiVG-RAG 已跑通 MinerU、SigLIP、图检索和 vLLM 上的 7B 与 32B AWQ。本地已核实 MMLongBench-Doc 135 份 PDF、1,091 题和 DocBench 229 份 PDF、1,102 题。7 份 MMLongBench-Doc 与 4 份 DocBench 开发文档已有 7B 对照和五项消融。32B 已完成 HiVG-RAG 与 Direct 对照。 [先导报告](../../experiments/vger-pilot/report.md)

在留出实体对上，局部裁剪将同名异物误合并从 4/4 降到 2/4，整图亦为 2/4。生产索引只正确识别并定位 2/101 个标注端点。先在开发文档上改善名称和区域覆盖，测两端同时定位的比例，再用未参与调试的文档检验增益。QAHT 当前在两组 7B 开发文档的 ANLS 均低于全图 Micro-only，后续需在同图和固定证据预算下测父节点、已检索区域和实际送入模型的证据覆盖。先导数据仍保留作开发记录，保持现有默认配置直到开发协议冻结。 [标注与结果](../../experiments/vger-pilot/report.md)

## 四周安排

| 阶段 | 工作 | 交付与验收 |
| --- | --- | --- |
| 11 月 23–29 日 · 核验 VGER | 建立真实文档实体对标注集。复查 MG²-RAG、MMGraphRAG 和 RAG-Anything 的视觉关联、实体合并与检索粒度，记录与 VGER、QAHT 重叠的操作和区别。并行补齐数据、MinerU 批处理与基线环境。 | 交付带页码、BBox 和实体 ID 的标注集，以及三组配对结果。首轮能定位问题发生在视觉打分、区域提取还是候选筛选。 |
| 11 月 30 日–12 月 6 日 · 冻结协议，跑 7B | 在开发文档上确定阈值、候选规则、Macro/Micro 标注和证据预算。先验证生产索引能找到标注区域，再跑 MMLongBench-Doc 和 DocBench 的 7B 主对比及五项消融。 | 交付冻结 manifest、YAML、模型版本和主结果表。所有方法使用共同的题目 ID、参考答案和生成后端，日志能追到每道题的失败或得分。 |
| 12 月 7–13 日 · 补强机制证据 | 完成干扰页实验、阈值扫描、拓扑诊断、效率测量与案例。用冻结配置在 SlideVQA、E-VQA 上测试 7B 与 32B 的迁移。 | 交付鲁棒性曲线、敏感性图、准确率–token 曲线和案例追踪。迁移结果注明数据、模型和覆盖，外部依赖未齐时记录具体缺口。 |
| 12 月 14–20 日 · 写作与投稿检查 | 依据冻结结果写方法、实验、局限和摘要。同步整理代码入口、配置、数据来源和复现命令。按 IEEE ICME 2027 作者指南核对模板、页数、匿名和提交要求。 | 交付论文初稿、可复核的主表与图、匿名复现材料。摘要和结论中的每项定量主张都能定位到结果文件。 |

## 第 1 周优先实验

1. 从约 **12–20 份**有重复部件标签、图注编号或明确图文引用的真实文档中，预标 **80–120 对**。覆盖同名异物、同名同物、异名同物和异名异物。记录两端原始表述、实体身份、页码、对象 BBox 和身份依据。复核边界样本以及视角变化、同图多对象等情况，保留原始分歧。文档按来源去重，并以文档为单位划分开发集和留出集。
2. 在完全相同的实体对上比较 **纯文本、整图视觉、人工核实的局部裁剪**，共享编码器、文本候选和上下文规则。记录每对的文本与视觉相似度。用同名异物误合并率和同名同物保留率评价拆分，用异名同物链接的精确率、召回率评价桥接。另测生产索引对标注提及的定位覆盖率和候选召回，给出各组样本数，避免把漏掉的正例排除在分母之外。
3. 用开发集选工作点，留出集只用于比较。按文档报告配对差异和置信区间。局部裁剪若能降低误合并且保住同名同物，同时改善别名链接，就进入端到端归因实验。人工框有效而自动定位覆盖低时，先修布局和对象定位。若两种视觉粒度均无稳定收益，就先检查编码器与标注误差，再决定 VGER 的主张范围。

## 正式实验清单

| 实验 | 对照与指标 |
| --- | --- |
| 主对比 | MMLongBench-Doc 与 DocBench。对比 Direct MLLM、VisRAG、ColPali、RAG-Anything、MMGraphRAG、MG²-RAG。报告 Overall、Macro 和 Micro 的 Accuracy、ANLS、样本数和失败率。MMLongBench-Doc 的原版评分和 DocBench 官方评审分数在各自协议可复现时单列。 |
| 跨数据集与模型 | SlideVQA、E-VQA，分别使用 7B 与 32B AWQ，沿用冻结阈值和检索配置。为 E-VQA 固定候选证据池和查询图片。记录官方分区、样本 ID 与模型版本。 |
| 五项消融 | 分别关闭 VGER-Split、VGER-Bridge、原生裁剪，并对比固定 Macro-Only 和全图 Micro-Only。VGER 消融同时做实体级及 QA 指标。QAHT 对照同时报告 Macro/Micro 的证据召回、输入 token 和问答得分。关闭原生裁剪沿用相同的证据节点排序。 |
| 鲁棒性与敏感性 | 在 100 份符合条件的文档上做 0、1、2、4 页同名异物干扰。同一文档的干扰页集合按数量嵌套，复核参考答案在原文中的依据和干扰页外观。另在同一个冻结评估子集上报告 `tau_split = 0.4, 0.5, 0.6, 0.7, 0.8` 和五档 `tau_zoom` 的敏感性。 |
| 效率与拓扑 | 在相同 vLLM 版本、硬件、图像处理约束与题目集合下报告索引耗时、查询延迟与服务端多模态输入 token。记录 MinerU 解析耗时和预热条件。给出准确率–token 曲线、super-hub 数量、实体合并和别名链接质量。 |
| 案例分析 | 选同名异物、异名同物、微观证据和宏观整合案例，包含失败案例。展示文本/视觉相似度、实体 ID、路由分数、父节点、裁剪区域和最终答案。 |

## 运行顺序与资源依赖

- 机制诊断先做编码和检索，7B 端到端比较再加入生成。发布表使用按文档配对的置信区间。主比较冻结共同题集、生成设置、上下文控制和错误计数。需要跨方法限制预算时，同时保留实际输入 token 和超限率。
- SlideVQA 的官方 GitHub 测试题已到位，共 2,215 题、400 套幻灯片。Hugging Face 图片分片返回 403，SlideShare CDN 抽查连接失败。E-VQA 的验证、测试、Lens CSV 和官方知识库已下载并校验。以 Lens 前五个候选为固定上游证据池，5,750 个测试问图对中有 4,790 个至少匹配一篇文章，2,956 个覆盖金标文章。该设置保留全部问图对，适用于给定候选池的评测。查询原图仍待获取，来源数据集的索引已挂后台。MG²-RAG 固定版本已到位，SAM3 权重请求返回 403。100 文档干扰实验依赖数量足够且已核查的目标与干扰素材。 [SlideVQA 数据说明](https://github.com/nttmdlab-nlp/SlideVQA) · [E-VQA 候选覆盖](../../../data/upstream/encyclopedic-vqa/lens-top5/summary.json) · [SAM3 权重](https://huggingface.co/facebook/sam3)
- 长实验以 `nohup` 运行并保留 PID、日志和退出状态。GPU 1 排程先做 RAG-Anything 开发集，再补两库 MinerU 布局。RAG-Anything 的 12K 请求曾超限，之后的 32K 单并发服务出现超时。当前运行使用 32K 上下文，最多 4 个并行请求，每次索引和检索的辅助生成上限为 2,048 token。布局作业跳过可读缓存，记录每份 PDF 的解析状态和耗时，两库解析完成后写出带布局路径的 manifest。 [GPU 队列脚本](run_gpu_queue.sh) · [RAG-Anything PID](../../../runs/vger-pilot/full-document-raganything-bounded/pid) · [全量布局 PID](../../../runs/full-corpus-layouts/pid) · [解析脚本](parse_corpora.sh)
- QAHT 等证据项检索诊断已完成。两层各限制 6 个证据项，默认路由将 6/30 道 Micro 题转入微观层，全证据页覆盖为 14/50，全图 Micro-only 为 19/50。完整文档上增加到四父节点后为 15/50。先在相同证据项限制下识别路由遗漏，再比较不同实际输入 token 下的 QA 结果。 [等证据项检索记录](../../../runs/vger-pilot/full-routing-matched-k6.json) · [诊断源码](../../experiments/vger-pilot/source/routing_probe.py)
- GPU 队列接续运行五道先导案例、两库全量默认配置对照、开发集 25 点阈值扫描和 MMGraphRAG 开发对照。全量 Direct 采用最多 24 页的位置均匀采样，记录实际页数与截页比例。两库分别有 690 和 465 道题来自超过 24 页的文档，发布表需注明页预算。全量结果使用保守的临时粒度标注，当前 MMLongBench-Doc 的 Macro/Micro/未标注为 130/277/684，DocBench 为 5/67/1030。正式分层评测需要复核。 [后续评测脚本](run_eval_queue.sh) · [队列日志](../../../runs/hivg-rag-7b/launch.log)
- 全量布局完成后生成 100 文档干扰实验的候选池。仅从题目标注的证据页选择目标，并记录每张干扰图的来源。候选池后续需核实实体身份与答案证据。 [候选池准备日志](../../../runs/polysemy-stress-prep/launch.log)
- GPU 作业结束后接续重跑两库的 VisRAG 与 ColPali。先保存实际逐文档索引和逐题检索耗时，再在 7B 上复用排名生成问答。 [页检索基线日志](../../../runs/full-page-baselines-7b/launch.log)
- MinerU 对未缓存文档按 48 页分段，校验各段页数及图片路径，合并时恢复原 PDF 页号。评测每完成一份文档就保存检查点，中断后使用同一配置恢复。已用真实 MinerU 缓存验证页号和裁剪文件的合并，并通过模拟残留半行 JSONL 验证评测恢复。 [分段脚本](parse_mineru_document.py) · [统一评测入口](../../../eval/run_benchmark.py)
- PDF 文字坐标定位在 101 个标注端点上得到 11 个首选框命中、15 个前五框命中，命中集中在一份论文。跨区域的正确别名仍缺自动定位支持。别名窗口扩大到 8 页后开发集增加两个真别名，留出集的跨区域召回仍为 1/9。 [定位记录](../../../runs/vger-pilot/layout-grounding.json) · [窗口对照逐对结果](../../../runs/vger-pilot/alias-context/graphs.json)
- RAG-Anything 已完成两库共 88 道先导题。受小裁剪图像描述错误影响的 7 道题已在修复后复跑。旧队列接续失败，提前启动的 7B 评测产生显存错误。恢复队列用同一把 GPU 锁串行执行余下各阶段，将 7B 服务显存比例调到 0.74。PDFium 无法读取个别 PDF 时重建副本并验证页数，继续保留原文档。 [复跑结果](../../../runs/vger-pilot/raganything-crop-repair/results/summary.json) · [恢复队列](run_remaining.sh) · [解析进度](../../../runs/full-corpus-layouts/progress.tsv)
- CCFA 分工按任务切换。实验设计负责冻结比较与判读结果，科研工程负责补齐 pairwise 指标、批处理、真实基线和运行排障，论文写作与投稿检查负责稿件及官方格式。沿用现有配置、评测入口和 JSON/CSV 日志。 [实验运行说明](../../../README.md#experiments)
