# 跨页视觉证据绑定小试

2026-11-23。使用 MMLongBench-Doc 的 5 道真实题，每道来自一份 PDF。题 295 来自前期开发集，418、641、348、838 来自留出文档复核和定向查找。本轮用人工标注的证据页、图表框和首跳子问，测试固定证据范围内的读图与取值。自动定位、自动分解及候选页重排留待独立验证。 [题目、框与评分配置](source/cases.json)

## 对照与结果

冻结 Qwen2.5-VL-7B-Instruct，使用 NF4、BF16 计算和贪心解码。自由文本两跳把首跳的自然语言说明交给模型，在目标图上直接回答。显式绑定用同一首跳得到的键，在模型抽取的目标候选中查找唯一匹配。两种方法每题均用两次模型调用，图像和视觉 token 数完全相同。题 641 在显式匹配时使用了人工确认的 `FEVEROUS-S` 到 `FEVEROUS` 别名，年份匹配允许从年月标签提取年份。 [执行代码](source/run_probe.py)

| 输入与提示 | 首跳键正确 | 自由文本答案 | 显式绑定答案 | 金标键下的显式绑定答案 |
| --- | ---: | ---: | ---: | ---: |
| 初版，人工裁剪 | 2/5 | 2/5 | 1/5 | 2/5 |
| 初版，证据整页 | 1/5 | 1/5 | 1/5 | 2/5 |
| 聚焦子问，人工裁剪 | 5/5 | 5/5 | 5/5 | 5/5 |
| 聚焦子问，证据整页 | 3/5 | 3/5 | 3/5 | 4/5 |

初版的第二次调用同时输出候选与自由文本选择。查看错误后，在同一批题上改用单独的首跳子问，兼容列表和日期形式，并让自由文本和绑定分别执行第二次调用。复跑的结果属于同题开发反馈。聚焦版在裁剪图上每题平均使用 815 个视觉 token。自由文本平均生成 70 个 token、用时 2.51 秒，显式绑定平均生成 122 个 token、用时 3.84 秒。时间为单卡模型生成时间，包含两次调用。 [初版逐题记录](../../../runs/visual-binding-pilot/results.json) · [聚焦版表格](../../../runs/visual-binding-pilot/focused/results.csv) · [提示和原始输出](../../../runs/visual-binding-pilot/focused/trace.jsonl)

### 两个可定位的失败

- 题 348 的整页首跳把 2017 读成 2019。目标页已经正确提取了 `2017 → 49%` 和 `2019 → 29%`，两种方法都输出 29%。只换成金标键即可读回 49%。
- 题 641 的整页首跳误选 `HOVER (4-hop)`，次跳还把表格值移到了错误的列。提供金标键也无法恢复答案。裁剪输入上，两种方法都得到正确数值 56.82，自由文本却把列写成 `HOVER`，显式绑定返回 `FEVEROUS`。聚焦版 5 道裁剪题中，自由文本有 3 道同时报对键和答案，另外两道分别报了错误列名和指标名。显式绑定报告匹配键是流程的一部分，该分数衡量输出能否核查到对应行列。 [逐题输出](../../../runs/visual-binding-pilot/focused/results.json)

## 替换中间键

固定同一张目标裁剪图，对每题分别提供正确键与替换键。自由文本对照收到明确的键和取值指令，重新读取目标图。显式绑定在已抽取的候选表上重放。显式绑定在两种键条件下均得到预期结果的题数为 **5/5**，自由文本为 **3/5**。自由文本在题 295 的正确键条件下把 `OPEN` 读成 0.970，预期值为 0.833。在题 838 中，目标图仅显示 2013 和 2014 年。键换成 2015 后，自由文本仍输出 6，显式绑定因缺少该年份而返回空值。该对照测量给定中间键后的取值行为。 [替换键对照代码](source/replay_key.py) · [全部响应](../../../runs/visual-binding-pilot/focused/key-replay.json)

## ICME 方向判断

现有小试支持把显式绑定作为可追溯的取值约束。与聚焦后的自由文本两跳相比，原题准确率持平，候选枚举增加了生成时间。题 348 的整页首跳误读和题 641 的整页列错配表明，自动定位和结构识读会直接限制绑定。考虑到五题为人工选例，且采用人工框、分解提示和一处人工别名，目前对原方案作为独立 ICME 方法贡献的证据不足。

## 复现

在项目根目录运行。环境使用 Python 3.11、PyTorch 2.4.0、Transformers 4.51.3、bitsandbytes 0.46.1、Pillow 和 PyMuPDF。两轮及替换键对照均通过 `nohup` 执行，PID、日志及退出码在各运行目录中。

```bash
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python ccfa-workfiles/experiments/visual-binding-pilot/source/run_probe.py \
  --protocol initial --output runs/visual-binding-pilot/initial-reproduction
python ccfa-workfiles/experiments/visual-binding-pilot/source/run_probe.py \
  --protocol focused --output runs/visual-binding-pilot/focused
python ccfa-workfiles/experiments/visual-binding-pilot/source/replay_key.py \
  --focused runs/visual-binding-pilot/focused --output runs/visual-binding-pilot/focused/key-replay.json
```
