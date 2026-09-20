# HCMT: Hierarchical Clinical-Markov Transformer

HCMT 是面向围术期稀疏临床事件序列的连续时间预测模型。这个仓库只包含训练所需的模型、事件构建、采样和训练入口；数据、checkpoint、外部验证、基线实验和历史代码均不随仓库发布。

## 方法概览

每位患者被表示为按时间排序的稀疏 token 序列

\[
\mathcal{S}_p = \{(x_i, t_i, v_i, q_i, f_i)\}_{i=1}^{n_p},
\]

其中 `x_i` 是事件/上下文 token，`t_i` 是相对 episode 起点的分钟数，`v_i` 是可选连续值，`q_i` 是 token 类型/观测状态，`f_i` 是临床事件 family。预处理只从输入数据中生成有时间证据的事件；稳定期通过 clock token 显式表示，而不是伪造五分钟网格事件。

模型在最多 256 个短期 token 上工作，并可加入 phase summary token，以保留 pre-op、induction、surgery、emergence、PACU、ICU 等阶段信息。每个位置的表示融合：

- token、事件 family、token kind 和静态患者 embedding；
- absolute time、相邻 gap 和 pairwise relative-time attention bias；
- measurement intensity 与 time-since-last-measurement；
- phase summary embedding 和可选连续测量值。

同一时间戳的并发事件采用 block-causal mask：同一时间 block 内的不同 token 不能互相注意，防止由文件排序或 token 排序造成的信息泄漏；不同时间 block 仍按因果方向可见。

## 预测头与目标函数

主任务同时预测 exact event/severity、event family 和等待时间：

\[
\hat{p}(e_{i+1}|h_i),\quad
\hat{p}(f_{i+1}|h_i),\quad
\hat{\mu}_{i+1},\hat{\log\sigma}_{i+1},
\]

其中时间头按 event family 条件化，等待时间使用 log-normal 似然，并对右删失 episode 使用 survival-style censoring 项。训练总损失为

\[
\mathcal{L} = \mathcal{L}_{event} + \lambda_f\mathcal{L}_{family}
 + \lambda_t\mathcal{L}_{time} + \lambda_{traj}\mathcal{L}_{trajectory}
 + \lambda_m\mathcal{L}_{masked-event}
 + \lambda_v\mathcal{L}_{masked-value}.
\]

连续值辅助任务对有值的 MAP、Hb、lactate、SpO₂ 等观测进行 masked-value reconstruction；它们不会把正常且不变的测量强行变成事件类别。训练窗口使用动态随机起点和随机 context truncation，避免每个 epoch 重复固定窗口集合。

## 主要创新点

1. 无人为 100 类上限：候选事件按时间戳可靠性、临床可区分性、至少 100 次 occurrences 和至少 50 位独立患者筛选。
2. 同时间 block-causal mask，消除并发事件的排序泄漏。
3. exact event + semantic family 的 hierarchical event head。
4. clock/no-event token 与 phase summary token，同时表达稳定期流逝和长期手术阶段。
5. relative-time attention bias、event-conditioned time head、measurement-intensity context。
6. patient-level validation、validation early stopping、平台期学习率衰减和模型/词表消融。

## 输入数据契约

训练目录必须由事件预处理器生成，至少包含：

```text
event_sequence_meta.json
admissions.csv
sequence_ptr.npy
token_id.bin
token_kind.bin
outcome_class.bin
time_min.bin
value.bin
has_value.bin
static_baseline.npy
token_vocabulary.json
```

原始训练数据通过 `scripts/preprocess_event_sequences.py` 转换。外部 MIMIC/MOVER 数据不进入训练数据或训练词表。

## 运行方式

```bash
python3 scripts/preprocess_event_sequences.py \
  --input /path/to/train_data \
  --output /path/to/perioperative_event_sequences_v5_full

python3 scripts/train.py \
  --data_dir /path/to/perioperative_event_sequences_v5_full \
  --output_dir outputs/event_hcmt_v5_full \
  --epochs 1000 --block_size 256 \
  --same_time_block_causal \
  --relative_time_attention \
  --event_conditioned_time_head \
  --phase_memory --observation_intensity --dynamic_windows \
  --early_stopping_patience 25 --min_lr 1e-6
```

`best_model.pt` 只能由独立 patient-level validation loss 选择；`checkpoint_latest.pt` 用于断点续训。训练输出还包括 `run_config.json`、`validation_history.jsonl` 和日志。

## 代码边界

- `hcmt/data/`：临床事件本体、事件抽取、序列数据集、采样器和预处理依赖。
- `hcmt/models/`：HCMT 主模型、时间似然和 masked-task losses。
- `scripts/train.py`：唯一主训练入口。
- `scripts/preprocess_event_sequences.py`：训练事件序列构建入口。
- `scripts/run_ablation_matrix.py`：模型规模/词表训练编排。
- `scripts/monitor_progress.py`：日志状态查看。

外部验证、baseline comparison、autoregressive evaluation、linear probe、报告生成和测试不包含在这个训练发布仓库中。

## 研究注意事项

HCMT 的事件定义是带阈值的临床状态转变，并不等价于未经验证的因果标签；药物后的变化只能解释为观察关联。跨数据库比较应同时报告 zero-shot 与 calibration→sealed-test 结果，并区分 HCMT-Core 与 HCMT-Full。该仓库不包含任何患者级数据。
