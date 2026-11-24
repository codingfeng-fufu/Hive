# 同预算下的图表标签交换实验

从 MMLongBench-Doc 取四份原始 PDF 的标注页。每页选两个可交换的标签，将两个等尺寸的像素块原位互换，保留数值、柱形或曲线和其他标签。每份文档构造两道受交换影响的题和一道无关项题，原图与编辑图使用同一问句。受影响的两个答案应互换，无关项答案应保持。四份文档为人工选例，12 道题据原图编写。 [坐标、题目与参考答案](source/cases.json)

用冻结的 Qwen2.5-VL-7B-Instruct（NF4）读图，每次只给一张固定大小的输入图。整页包含原有的图外文字，编辑仅改变指定图表标签。人工图表裁剪聚焦目标图。自动裁剪从给定证据页的 MinerU 布局取最上方图表块及与它竖向重叠的相邻图表块。补标题组使用同一个自动框，将上边界额外上移 5% 页高。这一调整在查看自动裁剪遗漏标题后选定，用原样本复测。

| 每次视觉 token | 视图 | 受影响题原图答对 | 受影响题编辑图答对 | 受影响题两图均对 | 无关项两图均对 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 432 | 整页 | 8/8 | 7/8 | 7/8 | 4/4 |
| 432 | 人工图表框 | 8/8 | 8/8 | 8/8 | 4/4 |
| 216 | 整页 | 2/8 | 1/8 | 0/8 | 1/4 |
| 216 | 人工图表框 | 6/8 | 6/8 | 6/8 | 3/4 |
| 216 | 自动图表框 | 7/8 | 7/8 | 6/8 | 3/4 |
| 216 | 自动图表框补标题 | 7/8 | 8/8 | 7/8 | 3/4 |
| 108 | 整页 | 0/8 | 0/8 | 0/8 | 1/4 |
| 108 | 人工图表框 | 4/8 | 5/8 | 4/8 | 3/4 |

432 token 时，两种视图的原图答案均正确。交换「要求出示照片身份证明」和「自动登记合资格选民」的标题后，整页组在前一个问题给出 38，读取了同一组别的总支持比例。正确的「强烈支持」比例是 14，图表框组读对了。216 token 的自动裁剪中，原始版面框切掉了面板标题。补上方上下文后，两图均对增加一道，但同图上的无关项仍错。 [432 token 逐题数据](../../../runs/chart-evidence-intervention/results.json) · [216 token 逐题数据](../../../runs/chart-evidence-intervention/216/results.json) · [108 token 逐题数据](../../../runs/chart-evidence-intervention/108/results.json) · [自动裁剪实例](../../../runs/chart-evidence-intervention/216/images/924-edited-layout.png)

216 和 108 token 的整页原图成绩已很低，配对差距含有可读性效应。432 token 的单例提供了原图已答对时的编辑敏感性证据。四份文档和单一 7B 骨干用于确认协议能否捕捉此类错误。上移裁剪边界的结果是同题诊断。近期 FaithChart 也用局部像素干预评价图表解释，已有指标针对引用区域的充分性与全面性。潜在区别需要落到文档检索中，评测候选身份在图内重新绑定后，取证和回答是否共同随之变化。 [FaithChart 预印本](https://www.researchsquare.com/article/rs-11135102/v1)

## 接入全文检索

在选举政策案例上，从 24 页原始 PDF 重建 ColPali 排名。三道题均将标注页排在第 1，随后固定原图的候选排名进行编辑对照。每题共 432 个视觉 token，由两张 216 token 的图像组成。两种裁剪方式共用候选及同一生成模型。空白第二图对照保持输入尺寸、第一张图和生成提示一致，测试额外检索页的影响。

| 两图输入 | 受影响题两图均对 | 无关项两图均对 |
| --- | ---: | ---: |
| 两张检索整页 | 0/2 | 0/1 |
| 两张检索页分别自动裁剪并保留上方上下文 | 0/2 | 0/1 |
| 第 1 页整图加空白图 | 0/2 | 0/1 |
| 第 1 页自动裁剪并保留上方上下文，加空白图 | 1/2 | 1/1 |

将相同协议扩展至另外三份 PDF。四份文档的 12 道题里，目标页在前两位出现 11 次，位列第一 7 次。每次均以同样的 432 个视觉 token 输入。

| 四份全文 PDF 的两图输入 | 受影响题两图均对 | 无关项两图均对 |
| --- | ---: | ---: |
| 两张检索整页 | 0/8 | 0/4 |
| 两张检索页分别自动裁剪并保留上方上下文 | 3/8 | 1/4 |
| 第 1 页整图加空白图 | 1/8 | 2/4 |
| 第 1 页自动裁剪并保留上方上下文，加空白图 | 4/8 | 3/4 |

前两页用自动裁剪后，新增的三道成对正确题来自两份 PDF。额外页面对照中，仅看目标页已排第一的题，可排除把目标页替换为空白的情况。该子集中，保留裁剪并用空白替换第二页，使受影响题由 3/5 到 4/5，无关项由 0/2 到 2/2。两道无关项和一道受影响题的改善来自两份 PDF。其余三道目标页未排第一的受影响题，保留两张裁剪也全部失败。第二页的去留因此同时受召回与读图影响。决定保留**视觉预算内的图表证据粒度选择**为主线，将额外页面干扰作为受控诊断。实验固定原图排名，样本来自同一轮设计的四份 PDF。 [政策报告对照](../../../runs/chart-evidence-intervention/retrieval-924/blank-second/results.json) · [国际合作报告](../../../runs/chart-evidence-intervention/retrieval-353/results.json) · [族裔认同报告](../../../runs/chart-evidence-intervention/retrieval-497/results.json) · [电信幻灯片](../../../runs/chart-evidence-intervention/retrieval-257/results.json)

## 四份新文档的粒度验证

从 MMLongBench-Doc 另取四份 PDF，包含三个调查报告和一套幻灯片。根据原图人工编写八道受交换影响的题和四道无关项题。冻结 ColPali 原图排名和选区规则后再进行生成。两张输入图共 432 个视觉 token。标注页在前两页出现 12/12 次，排第一 11/12 次。 [冻结协议与标注](source/heldout-protocol.md)

| 选图方法 | 受影响题两图均对 | 无关项两图均对 |
| --- | ---: | ---: |
| 两张整页 | 1/8 | 1/4 |
| 两页各取顶部图表组，固定扩框 | 4/8 | 3/4 |
| 按问题选择图表组，固定扩框 | 6/8 | 3/4 |
| 按问题选择图表组，动态扩框 | 6/8 | 3/4 |

相对顶部图表组，按问题选图增加的两道正确题都来自同一份 PDF。两题指向下面一张图，原有策略裁取了上方图表。另一个幻灯片案例的检索页没有 PDF 文字层，MinerU 图表块也没有图注。按问题选图退回整页，两个受影响题均失败。冻结协议要求增益至少来自两份 PDF，本轮达不到该条件。动态扩框和固定扩框的逐题正确性相同。 [双图报告逐题数据](../../../runs/chart-evidence-heldout/118/results.json) · [幻灯片逐题数据](../../../runs/chart-evidence-heldout/174/results.json)

将前后两批共八份 PDF 的固定扩框结果合并作描述性统计。16 道受影响题中，整页有 3 道原图答对，其中 1 道编辑后也答对。固定扩框有 10 道原图答对，其中 7 道编辑后也答对。整页有 13 道原图即错。按方法分别报告原图可答数，可区分原有的读图或取证错误与原图答对后出现的编辑错误。两种方法条件分母对应的题目不同，条件正确率适合诊断各自失败，比较整体表现仍用共同的 16 道题。 [首批结果](#接入全文检索) · [本批结果](#四份新文档的粒度验证)

用相同排名和选区，将每次的图像预算从 432 提到 864，单张图从 336 × 504 到 504 × 672。整页原图答对从 1/8 到 5/8，联合正确从 1/8 到 2/8，三道原图答对的题在交换后出错，分别来自三份 PDF。例如将共和党和民主党标签交换后，整页组仍回答 15%，原图为 15%，编辑图按原样保留的数值归属应为 42%。固定扩框仍为 4/8，查询选图固定扩框仍为 6/8，三组无关项均为 4/4。查询选图超过顶部裁剪的两道题依旧来自同一份文档。因此预算放宽后仍能定位到具体的交换错误，同时复现了方法增益集中于单份文档的边界。 [864 token 的双图报告](../../../runs/chart-evidence-heldout/118/864/results.json) · [算法调查](../../../runs/chart-evidence-heldout/819/864/results.json) · [对华观感调查](../../../runs/chart-evidence-heldout/926/864/results.json) · [幻灯片](../../../runs/chart-evidence-heldout/174/864/results.json)

## 原始问题与预算分配

复用 ColPali 粒度试验的六份 PDF 和 30 道 MMLongBench-Doc 原题。对每道题共用 ColPali 前两页和 432 个视觉 token。另做一组只改变两张整页分辨率的实验。均分和不均分条件重跑后，均分组与裁剪实验的整页组 30/30 预测一致。

| 输入 | 精确匹配 | ANLS | 图、表或图表题 ANLS（19 题） |
| --- | ---: | ---: | ---: |
| 整页，216 + 216 | 6/30 | 0.338 | 0.297 |
| 顶部图表组，固定扩框 | 3/30 | 0.212 | 0.213 |
| 查询选择图表组，固定扩框 | 5/30 | 0.274 | 0.253 |
| 查询选择图表组，动态扩框 | 5/30 | 0.274 | 0.253 |
| 整页，336 + 96 | 3/30 | 0.244 | 0.251 |
| 整页，96 + 336 | 3/30 | 0.223 | 0.311 |

查询裁剪改变了 13 道题的输入，其中 3 道的 ANLS 下降。比如问第二页第一张图中的指挥官姓名时，整页组答出了姓名，裁剪组只答出职务。两种固定的非均匀分配均降低总体 ANLS。逐题事后取三种整页分配的最高分，上限是 0.384，比均分高 0.046。该上限使用答案得分选条件，只反映选择空间。六份 PDF 来自此前的留出队列，本轮复用它们检查原始问题下的表现。 [裁剪逐题数据](../../../runs/chart-evidence-heldout/benchmark-results.json) · [分配逐题数据](../../../runs/chart-evidence-heldout/page-budget-results.json)

结合两次结果，暂停当前词面裁剪、动态扩框和按固定名次偏置分辨率。保留图表标签交换评测与均分整页对照。后续候选需要同时覆盖答案所在的局部与必要文本上下文，先用成对错误确认机制，再用新的原题队列确认覆盖与得分。

为检查保留整页缩略图能否缓解裁剪损失，在同一组 30 道原题上补做三组比较。每组仍用 432 视觉 token。两张整页均分时 ANLS 为 **0.338**。每页同时输入一张整页缩略图和一张查询裁剪图，将预算分成四张各 108 token 的图时，ANLS 为 **0.244**。同样用四张图，把查询裁剪换成等尺寸整页副本，ANLS 也是 **0.244**。两张整页均分组的 30 道题预测与此前一致。两种四图输入相对均分均有 1 道 ANLS 上升、4 道下降。每个局部占用的视觉 token 从 216 降到 108，同时改变了视图数和整页分辨率。这次对照能判定当前组合策略的表现，无法单独估计这两个因素各自的影响。 [逐题结果](../../../runs/chart-evidence-heldout/context-pair-results.json) · [汇总](../../../runs/chart-evidence-heldout/context-pair-summary.json)

随后保持两张图和 432 个视觉 token，给整页组和查询裁剪组附上同一段 PDF 文本。每个检索页最多取三个问题匹配的原始文本块、800 个字符。裁剪组 ANLS 从 **0.274 到 0.331**，三道题改善，包含前面丢掉指挥官姓名的案例。整页组从 **0.338 到 0.343**。两组补文本后的平均总输入都是 694 token，约比纯图输入多 191 token。在相同文本输入下，裁剪组有两道题比整页差，其余持平。该对照仅在原图原题上执行，文本补充可以恢复部分裁剪损失，但这批数据仍支持把整页加同样文本作为对照。 [逐题结果](../../../runs/chart-evidence-heldout/text-context-results.json) · [汇总](../../../runs/chart-evidence-heldout/text-context-summary.json)

## 保持答案不变的局部编辑对照

在四份新文档的已标注矩形内，将两个标签各水平移动 3 像素。标签仍属于原来的行或柱，所有参考答案保持原值。像素差分确认矩形外像素一致。继续使用原始 ColPali 排名、相同选区和 864 个视觉 token。重新运行的原图预测与标签交换实验逐条一致，共 36 条。

| 方法 | 8 道交换题中原图正确 | 原图正确且交换图正确 | 原图正确且平移图正确 | 4 道无关项在三种图上都正确 |
| --- | ---: | ---: | ---: | ---: |
| 两张整页 | 5 | 2 | 5 | 4 |
| 顶部图表组固定扩框 | 4 | 4 | 4 | 4 |
| 查询选择图表组固定扩框 | 6 | 6 | 6 | 4 |

整页组在三份 PDF 上出现原图正确、平移后仍正确、交换后出错的情况。三次错误中有一次保留了原答案，另两次给出了不同的错误值。以整页缩放后的 504 × 672 输入图计数，平移改变的像素数为交换的 92% 至 119%。交换还改变了内容在两个框之间的分布，平移检验的是相近改动面积下的局部位置扰动。 [平移结果 118](../../../runs/chart-evidence-heldout/118/shift/results.json) · [819](../../../runs/chart-evidence-heldout/819/shift/results.json) · [926](../../../runs/chart-evidence-heldout/926/shift/results.json) · [174](../../../runs/chart-evidence-heldout/174/shift/results.json)

根据现有实验清单筛出 25 份新 PDF 和 43 道单页图表候选。先核对两份新文档，分别是图像型市场分析幻灯片和带文字层的调查报告。已按原图核对两组交换标签、数值、保义平移及无关项。在给定证据页、单图 432 视觉 token 的条件下，两份文档的四道交换题和两道无关项，用整页和人工图表框都在原图、交换图、平移图上答对。48 条记录含 12 条重复运行的原图预测，两次原图答案逐项一致。这个小试记录了能够正确跟随标签的案例，也验证了新增标签模板可以沿用现有代码。 [候选池](source/expansion-candidates.json) · [已核图样本](source/expansion-pilot.json) · [交换预测](../../../runs/chart-evidence-expansion-pilot/swap/results.json) · [平移预测](../../../runs/chart-evidence-expansion-pilot/shift/results.json)

## 复现

在现有安装了 PyMuPDF、Transformers、PyTorch 和 bitsandbytes 的环境，从项目根目录运行。各次实测均使用 nohup 并保留 PID、日志和退出码。 [执行脚本](source/probe.py)

```bash
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
probe=ccfa-workfiles/experiments/chart-evidence-intervention/source/probe.py
out=runs/chart-evidence-intervention
python "$probe" --output "$out"
python "$probe" --output "$out/216" --size 336 504 --views page figure layout layout-context
python "$probe" --output "$out/108" --size 252 336
```

全文检索小试使用本地 ColPali v1.3 Transformers 权重，CPU BF16 autocast 编码页图，生成使用同一个本地 7B。 [执行代码](source/retrieval_probe.py)

```bash
export HIVG_BASELINE_DEVICE=cpu HIVG_COLPALI_FORMAT=transformers
export HIVG_COLPALI_MODEL=data/models/colpali-v1.3-hf
retrieve=ccfa-workfiles/experiments/chart-evidence-intervention/source/retrieval_probe.py
python "$retrieve" --anchor 924 --output "$out/retrieval-924"
python "$retrieve" --anchor 924 --output "$out/retrieval-924/blank-second" \
  --rank-cache "$out/retrieval-924/rankings.json" --second-page blank
```

其余文档使用同样两条命令，将锚点和输出目录对应改为 `353`、`497` 或 `257`。

本轮新增配对和原题实验可从项目根目录运行。已有 ColPali 原题排名由前次实验留下。所有执行均以 nohup 保存日志、PID 和退出状态。

```bash
export PYTHONPATH=".:data/tools${PYTHONPATH:+:$PYTHONPATH}"
export HIVG_BASELINE_DEVICE=cpu HIVG_COLPALI_FORMAT=transformers
export HIVG_COLPALI_MODEL=data/models/colpali-v1.3-hf
src=ccfa-workfiles/experiments/chart-evidence-intervention/source
out=runs/chart-evidence-heldout
for anchor in 118 819 926 174; do
  python "$src/retrieval_probe.py" --cases "$src/heldout-cases.json" \
    --anchor "$anchor" --output "$out/$anchor" \
    --views page layout-context query-fixed query-adaptive
  python "$src/retrieval_probe.py" --cases "$src/heldout-cases.json" \
    --anchor "$anchor" --output "$out/$anchor/864" \
    --rank-cache "$out/$anchor/rankings.json" --size 504 672 \
    --views page layout-context query-fixed
  python "$src/retrieval_probe.py" --cases "$src/heldout-cases.json" \
    --anchor "$anchor" --output "$out/$anchor/shift" \
    --rank-cache "$out/$anchor/rankings.json" --size 504 672 \
    --views page layout-context query-fixed --edit-mode shift
done
python "$src/benchmark_probe.py" --output "$out"
python "$src/benchmark_probe.py" --output "$out" --page-budget
python "$src/benchmark_probe.py" --output "$out" --context-pair
python "$src/benchmark_probe.py" --output "$out" --text-context
```

两份扩展样本使用给定页的读取对照。

```bash
out=runs/chart-evidence-expansion-pilot
python "$src/probe.py" --cases "$src/expansion-pilot.json" --output "$out/swap"
python "$src/probe.py" --cases "$src/expansion-pilot.json" \
    --output "$out/shift" --edit-mode shift
```
