# HiVG-RAG 先导实验 · IEEE ICME 2027

自动实体定位是当前 VGER 的主要瓶颈。手工核实的裁剪在实体对上减少了误合并，但生产索引只在 101 个标注端点中正确识别并定位了 2 个。端到端的 HiVG-RAG 在两组开发文档上减少了生成端输入 token。当前 QAHT 在两组 7B 对照中都低于全图 Micro-only 的 ANLS。

## 实体消歧与定位

从 12 份真实 PDF 标注 101 个实体对，按文档分出开发集 51 对、留出集 50 对。标注由单人完成并目视核实，未测标注者间一致性。各方法共享文本候选与上下文规则。图像条件使用人工提供的区域位置，以下是默认阈值下的留出结果。 [标注](source/annotations.yaml) · [逐对结果](../../../runs/vger-pilot/pairs.csv) · [阈值和分组](../../../runs/vger-pilot/summary.json)

| 视觉证据 | 同名异物误合并 | 同名同物保留 | 异名同物召回（不同区域） | 别名精确率（不同区域） |
| --- | ---: | ---: | ---: | ---: |
| 纯文本 | 4/4 | 7/7 | 0/9 | 无预测 |
| 整图 | 2/4 | 6/7 | 2/9 | 2/5 |
| 局部裁剪 | 2/4 | 7/7 | 3/9 | 3/6 |

局部裁剪相对整图多保留 1 个同名同物，并多连回 1 个不同区域的别名。留出集只有 4 个同名异物，估计对单个样本敏感。按文档重采样的裁剪减整图总体别名召回差 95% 区间为 [0, 0.273]。案例展示同名面板拆分、跨页别名找回，以及因外形相似产生的误连。 [配对结果](../../../runs/vger-pilot/summary.json) · [案例图](entity-cases.pdf) · [绘图源码](source/plot_entity_cases.py)

59 页 MinerU 生产索引中，101 个端点里有 20 个找到 IoU ≥ 0.5 的区域，有 8 个找到同名提及，两者同时成立的有 2 个。关键实体对中，两端同时正确定位的数量为 0。布局规则找到了 42/101 个同名候选。给定 18 个真实名称，在人工整图、MinerU 最佳重叠块和整页中使用 Grounding DINO，Top-5 分别命中 12/18、4/18 和 9/18。自动布局名称 + DINO 命中 4/101，Qwen 抽取名称 + DINO 命中 1/101。生产索引需要同时提升候选名称覆盖和名称到区域的定位精度。 [生产索引覆盖](../../../runs/vger-pilot/coverage.log) · [布局名称探针](../../../runs/vger-pilot/layout-name-probe.json) · [人工整图](../../../runs/vger-pilot/grounding-dino.json) · [MinerU 块](../../../runs/vger-pilot/grounding-dino-mineru-figure.json) · [整页](../../../runs/vger-pilot/grounding-dino-page.json) · [自动布局名称](../../../runs/vger-pilot/automatic-grounding.json) · [Qwen 名称](../../../runs/vger-pilot/vlm-automatic-grounding.json)

新增一个使用 PDF 文字坐标的候选定位诊断。它将短标题、编号和图注与同页的 MinerU 框及原生图片框按几何位置关联。101 个端点中，44 个有同名候选，11 个在首个同名框达到 IoU ≥ 0.5，15 个在前五个同名框内达到该要求。23 个跨区域的同名异物或异名同物对中，有 2 对的两端同时命中，均为同名异物。15 个命中中有 11 个来自同一篇论文的分图，说明收益集中在文本与分图直接相邻的布局。该诊断沿用已分析过的 12 份文档，核对候选定位后仍需在新文档上验证最终选框与图级消歧。 [逐端点结果](../../../runs/vger-pilot/layout-grounding.json) · [诊断源码](source/layout_grounding.py)

将这些端点和所属整图交给生产版 GraphBuilder，沿用实际的上下文约束和完整链接合并规则。留出图中对带金标的预测实体对，文本、整图和裁剪条件分别命中 12/39、29/59 和 30/48。三个条件的 super-hub 数均为 0。不同区域的九个别名中，整图与裁剪各直接连回一个。实体级精确率的增益来自带标注名称和框的受控索引。 [图级受控探针](../../../runs/vger-pilot/oracle-graph.json)

同一图级探针将别名窗口由 1 页放宽到 2、4 或 8 页，固定文本、视觉与父节点相似度阈值。8 页窗口在开发集多连出 2 个跨区域真别名，留出集的跨区域召回在四种窗口下均为 1/9，带金标的别名精确率均为 18/24。这批文档上的开发增益没有迁移到留出集。 [别名窗口图级诊断](../../../runs/vger-pilot/alias-context/graphs.json)

### 加入 PDF 局部文字的成对对照

复用原有 12 份 PDF 和 SigLIP 图像相似度。去掉两端共用同一框的配对后，开发和留出各 41 对。每个提及以名称、距标注框最近的五行 PDF 文字，以及距所属整图 36 点内的至多三行邻近文字组成上下文。文本组用冻结 BGE-small-en-v1.5 的余弦相似度。另两组取相同文本分数与 SigLIP 整图或局部裁剪分数的等权平均。对规范化名称相同与不同的配对，分别在开发集以平衡准确率和别名 F1 定阈值。 [评测代码](source/scope_pair_probe.py) · [阈值与逐对结果](../../../runs/vger-pilot/scope-pair/results.json)

| 证据 | 同名异物误合并 ↓ | 同名同物保留 ↑ | 异名同物召回 ↑ | 别名精确率 ↑ |
| --- | ---: | ---: | ---: | ---: |
| PDF 局部文字 | 1/4 | 4/7 | 9/9 | 9/24 |
| PDF 局部文字 + 整图 | 1/4 | 3/7 | 8/9 | 8/13 |
| PDF 局部文字 + 裁剪 | 1/4 | 7/7 | 9/9 | 9/20 |

视觉输入在当前阈值下未减少残留的误合并。裁剪组多保留 3 对同名同物，整图组的别名精确率最高，少找回 1 对别名。四对同名异物只来自两份 PDF，同一对 Camry 在三组均误合并。结果来自已有标注的二次分析，名称和框由标注提供。无文字层页面、自动候选和新文档仍需另测。CPU 复跑用时 16 秒，结果与初次运行逐字节一致。 [复跑退出码](../../../runs/vger-pilot/scope-pair/recheck.exit) · [耗时](../../../runs/vger-pilot/scope-pair/time)

另在此前未标注的冰箱手册 PDF 第 28 页人工核对 4 对，用旧开发阈值跑三组。三组都区分了冷冻室和冷藏室的同名上铰链，也都找回了图 2 与图 3 中不同描述的同一个冷冻室插头。三组均误连了图 2 的铰链盖与图 3 的电插头。整图组还漏合并了正文与图 2 指向同一个冷冻室铰链的配对。该页同时提供明确图号、正文步骤和视觉标线，适合继续核对同一组件图内的易混部件。 [标注及依据](source/reference-pair-seed.yaml) · [手册逐对结果](../../../runs/vger-pilot/scope-pair/reference-seed.json)

## 完整文档开发集

MMLongBench-Doc 采用 7 份 PDF、68 题，其中 Macro 20、Micro 30、未标粒度 18。DocBench 采用 4 份 PDF、20 题，其中 Macro 5、Micro 14、未标粒度 1。选取的文档附有 MinerU 全文布局，各方法使用同一组题目与同一骨干模型上的公共生成提示。粒度标签为本次开发集标注。ANLS 是两组共用指标，另行保存 MMLongBench-Doc 原版分数。DocBench 官方模型评审尚未运行。 [队列与标签构建](source/prepare_full_cohorts.py) · [逐分组聚合结果](../../../runs/vger-pilot/full-document-results.json)

各单元格给出总体 ANLS / vLLM 记录的平均多模态输入 token。32B 使用 Qwen2.5-VL-32B-Instruct-AWQ。

| 方法 | MMLongBench-Doc 7B | DocBench 7B | MMLongBench-Doc 32B | DocBench 32B |
| --- | ---: | ---: | ---: | ---: |
| HiVG-RAG | 0.243 / 450 | 0.069 / 300 | 0.274 / 450 | 0.031 / 300 |
| Direct | 0.226 / 4,328 | 0.159 / 1,345 | 0.196 / 4,328 | 0.291 / 1,345 |
| VisRAG | 0.221 / 1,621 | 0.209 / 963 | — | — |
| ColPali | 0.250 / 1,621 | 0.195 / 963 | — | — |
| RAG-Anything | 0.264 / 22,293 | 0.345 / 16,222 | — | — |
| Macro-only | 0.273 / 440 | 0.000 / 307 | — | — |
| Micro-only | 0.318 / 650 | 0.151 / 432 | — | — |
| w/o VGER-Split | 0.257 / 450 | 0.069 / 299 | — | — |
| w/o VGER-Bridge | 0.242 / 447 | 0.069 / 300 | — | — |
| w/o Native Crop | 0.243 / 382 | 0.069 / 288 | — | — |

与 Direct 相比，HiVG-RAG 的平均输入 token 在两个开发集分别降低约 90% 和 78%。MMLongBench-Doc 上 HiVG-RAG 减 Direct 的 ANLS 差在 7B 和 32B 的文档级 bootstrap 区间分别为 [−0.121, 0.168] 和 [−0.044, 0.194]。在 DocBench 上两组点估计均下降，其中 32B 项来自 4 份开发 PDF。五项消融尚未表明拆分、桥接、原生裁剪或当前路由有稳定问答收益。 [ANLS–token 图](pilot-pareto.pdf) · [绘图源码](source/plot_pareto.py)

按相同题目配对，7B 默认路由减全图 Micro-only 的总体 ANLS 差在 MMLongBench-Doc 为 −0.075，文档 bootstrap 95% 区间 [−0.126, −0.026]，在 DocBench 为 −0.082 [−0.194, −0.004]。Micro 子集分别下降 0.143 和 0.118。两组 Macro 问题上，默认路由与固定 Macro-only 的逐题 ANLS 均相同。区间描述的是少量开发文档上的差异。 [按粒度配对结果](../../../runs/vger-pilot/full-document-results.json)

7B 的 HiVG-RAG 平均索引耗时分别为 14.48、5.05 秒/文档，计时从文档载入后开始，复用已有 MinerU 输出。平均查询耗时分别为 0.323 和 0.189 秒。super-hub 数按固定度数阈值计，MMLongBench-Doc 从 57 增至 61，DocBench 为 13 → 13。两组图分开了 1 和 2 对候选，加入了 48 和 12 条别名边。开发集的 QA manifest 没有带金标身份的图节点，图级实体精确率为空。 [MMLongBench-Doc 逐文档诊断](../../../runs/vger-pilot/full-document/mmlongbench-doc-full-document-pilot/7b/hivg-rag/default/documents.json) · [DocBench 逐文档诊断](../../../runs/vger-pilot/full-document/docbench-full-document-pilot/7b/hivg-rag/default/documents.json)

早期 14 题页片段探针中，将父节点数从 2 调到 4 后，全证据页召回由 8/14 升至 11/14，原版评分由 0.348 降至 0.256。后续应在共同图与固定证据预算下同时测父节点覆盖、送入模型的证据覆盖及问答得分。Micro-only 首轮中有 6 题因小裁剪被 Qwen 拒收，在保持原像素的上传填充后按原题 ID 补跑，聚合结果使用补跑成功值。 [片段检索探针](../../../runs/vger-pilot/routing-probe.json) · [补跑逐题结果](../../../runs/vger-pilot/full-document-padded/summary.json)

完整文档上以每条路径至多六个证据项重放了 50 道有金标证据页的 MMLongBench-Doc 开发题。默认路由将 6/30 道 Micro 题和 8/20 道 Macro 题送入微观层。全部证据页覆盖由默认路由的 14/50 升到四父节点的 15/50，Macro-only 为 15/50，全图 Micro-only 为 19/50。Micro 题的平均图像证据页召回为 13.3%（默认路由）和 26.7%（全图 Micro-only）。这些结果测到的是等证据项条件下的页覆盖。更新后的敏感性作业同时记录真实输入 token。 [完整文档等证据项诊断](../../../runs/vger-pilot/full-routing-matched-k6.json)

同一批题的 Micro 识别 AUROC 为词法先验 0.350、熵集中度 0.630、默认组合 0.392。仅移除 `how many` 触发时，8 道被送入微观层的 Macro 题降到 1 道，Micro 题由 6 道降到 2 道。以原有 14 道的路由量作对照，熵集中度排名前 14 道中有 11 道 Micro 和 3 道 Macro。该阈值从本开发队列估计，后台已排入使用六个证据项的问答对照。 [路由分量探针](../../../runs/vger-pilot/routing-score-components.json)

五道问答案例已获得三种路由的完整轨迹，共 15 条。问及示例中的图表类型时，默认路由的 γ 为 0.036，选择宏观层，ANLS 为 0。强制全图微观检索的 ANLS 为 1.000，输入从 952 增至 1,102 token。DocBench 听力损失人数案例中，默认路由选中微观层并取得 0.625，另外两种固定路由均为 0。五题按已观察的路由分歧选取，包含一次服务重启后的单题补跑。 [并排案例与原始轨迹](../../../runs/vger-pilot/qaht-case-retry/cases.html)

## 与近期方法的重叠

MG²-RAG 已做多粒度视觉定位，按文本名建立实体节点，后续使用带种子的全图 PPR。MMGraphRAG 对近邻描述聚类，并让 LLM 对齐图片实体。RAG-Anything 从模态描述构造跨模态关系，使用混合检索。HiVG-RAG 需要以同一标注身份在不同页面的提及为单位，验证视觉约束对误合并和别名链接的影响，并在同预算下检验对遍历层的限制。 [MG²-RAG 源码](https://github.com/Daboolu/MG2-RAG) · [MMGraphRAG 源码](https://github.com/wanxueyao/MMGraphRAG) · [RAG-Anything 源码](https://github.com/HKUDS/RAG-Anything)

全量 MinerU、7B 两库默认配置对照、五个问答案例和开发集阈值扫描已排入后台。正式实验仍需要其余基线、100 文档干扰和跨数据集迁移。阈值冻结依据开发集的自动定位和固定预算检索结果。 [ICME 执行计划](../../planning/hivg-rag-icme/plan.md) · [方法与实验段落草稿](../../writing/hivg-rag-icme/method-and-pilot.md)

RAG-Anything 已完成两库的 88 道先导题。其中一份 PDF 的五个裁剪短边低于 28 像素，造成索引图像描述返回 HTTP 500。经原像素填充后，该文档 7 道题已完整重跑。聚合时按文档和题目 ID 替换，保留原始输出。RAG-Anything 使用 32K 服务，其他 7B 先导运行使用 12K 服务，图中报告各方法实际输入 token。 [先导基线日志](../../../runs/vger-pilot/full-document-raganything-bounded/benchmark.log) · [复跑结果](../../../runs/vger-pilot/raganything-crop-repair/results/summary.json) · [准确率–token 图](pilot-accuracy-pareto.pdf)
