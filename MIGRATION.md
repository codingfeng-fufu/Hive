# 迁移到新实验服务器

待运行的是 VGER 成对诊断，冻结了 49 对实体、6 种输入条件和判断标准，共 294 次 Qwen 请求。启动状态为暂停。 [研究路径图](ccfa-workfiles/planning/hivg-rag-icme/research-path-map.md#待运行的批次)

## 1. 克隆和建环境

在新 Linux 服务器上克隆 GitHub 仓库，然后在仓库根目录执行。vLLM 按新机 GPU 架构和 NVIDIA 驱动选择官方支持的版本。旧机使用 vLLM 0.8.5.post1，批处理入口也接受新版 CLI 的图像参数。 [vLLM GPU 安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[encoder,evaluation,test]'
.venv/bin/python -m pytest -q

python3.11 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install vllm
.venv-vllm/bin/vllm --version
nvidia-smi
```

## 2. 搬运模型、PDF 和历史记录

Git 仓库含标注和待运行批次使用的三份小型结果文件。批次还依赖 14 份原始 PDF 和三套模型，总大小约 19 GiB。资源检查校验 PDF 的 SHA-256、权重字节数及非空的配置和分词文件。下面任选一种方式准备。

**从旧机搬运。** 在旧机仓库根目录执行，将 SSH 主机和克隆路径改为实际值。资源直接流到新机。

```bash
set -o pipefail
python scripts/reference_assets.py files |
  tar -cf - -T - |
  ssh new-server 'cd ~/HiVG-RAG && tar -xf -'
```

**在新机下载。** 下载器按原数据版本取 PDF，模型使用公开仓库当前快照。如需复用旧机权重和配置的确切字节，选择服务器间传输。

```bash
.venv/bin/python scripts/reference_assets.py download
```

两种方式都在新机执行 `.venv/bin/python scripts/reference_assets.py check`。仅运行 VGER 批次时，到此已齐备。

**历史实验和完整 benchmark。** 如需复用旧解析和结果，另同步下表中的目录并保持相对路径。报告的历史链接指向这些本地文件。

| 路径 | 用途 |
| --- | --- |
| `data/upstream/` | MMLongBench-Doc、DocBench、SlideVQA、E-VQA 的原始下载 |
| `data/models/` | 其他模型，含 32B AWQ、ColPali、VisRAG 和 MinerU |
| `data/mineru-full/`、`data/mineru-docbench/`、`data/mineru-output/`、`data/rendered/` | 已解析布局和页面 |
| `data/*.json`、`data/*.yaml`、`data/tools/`、`data/vger-screen/` | 既有 manifest、原始评分器、队列和辅助工具 |
| `data/upstream-code/` | 上游源码快照。也可在新机按同一版本重新检出 |
| `runs/` | 旧预测、索引、检查点和图像证据 |

重新创建 MinerU 和各基线的 Python 环境。早期队列脚本里的 `data/serve-mirror-env/bin/vllm` 等路径属于旧机布局，继续使用这些脚本时需按新机环境调整。

## 3. 恢复批处理

确认 GPU 可用后，在新机仓库根目录执行。120 GB 主机内存可用于 PDF 处理和 CPU 编码。按一张约 33 GB 显存的卡安排首批，SigLIP 与 Qwen2.5-VL-7B 顺序使用同一卡。32B AWQ 后续单独运行，启动参数依据目标卡的实际空闲显存验证。

```bash
run=runs/vger-pilot/reference-batch
mkdir -p "$run"
VGER_GPU=0 \
VGER_PYTHON="$PWD/.venv/bin/python" \
VGER_VLLM="$PWD/.venv-vllm/bin/vllm" \
VGER_VLLM_LIMIT_STYLE=dotted \
nohup bash ccfa-workfiles/experiments/vger-pilot/source/run_reference_batch.sh \
  > "$run/launch.log" 2>&1 < /dev/null &
```

若用 vLLM 0.8.5.post1，设 `VGER_VLLM_LIMIT_STYLE=json`。接口绑定回环地址，默认端口 8147。批次结束后检查 `exit`、`summary.json` 和 `predictions.jsonl`。`exit=0` 代表脚本完成。中断后用同一配置和输出目录续跑。模型调用阶段超时为 40 分钟，实际用时以新机为准。
