# 希腊语评测套件

> English version: [`GREEK_EVAL.md`](GREEK_EVAL.md) · 一页速查（任务表 + 命令）：[`GREEK_BENCHMARKS.md`](GREEK_BENCHMARKS.md)

基于 lm-evaluation-harness 搭建的完整希腊语 LLM 评测栈：**13 个 benchmark，分布在 16 个任务目录里，共 264 个叶子任务**，外加一个规则式的生成质量诊断。数据集缓存好之后全部可离线运行，每个任务都记录了它与参考实现的差异。

**状态：已建成并验证。** 下面每个任务都用真实模型端到端跑过（`ilsp/Meltemi-7B-Instruct-v1.5`，第一批另用了 `models/e1_qwen3.5-4b-base_*_merged`）—— 不是只验证能加载。真正跑评测是另一件事，外层训练仓库的 `evaluation/TASKS_TODO.md` 记录了哪些是刻意没做的以及原因。

## 目录结构

```
lm_eval/tasks/
├── greekmmlu/          GreekMMLU —— 原生希腊语，自 PR #3581 起已在上游
└── ilspgreek*/         这里新增的 11 个任务族，每个目录带自己的 README
```

每个任务目录的 README 记录了该任务与参考实现的逐项差异，**包括在参考实现里发现的 bug**。

本套件有三样东西不在 harness 里，在外层的训练仓库：

- `src/greekllm/eval/failure_profile.py` —— 下文的生成诊断脚本。
- `evaluation/ilsp-assets/` —— ILSP 的希腊语 MT-Bench judge prompt，留给装不进 lm-eval 的 judge 脚本。
- `evaluation/TASKS_TODO.md` —— 还剩什么、卡在哪、以及评估后被否掉的方案。

## 安装

```bash
pip install -e .                    # 在本目录下执行，新增约 28 个包，不动 torch/transformers
pip install langdetect immutabledict # ilspgreekifeval 需要
python -c "import nltk; nltk.download('punkt_tab')"  # 希腊语断句，约 11 MB
```

`ilspgreektruthfulqa_gen` 另需 `sacrebleu` 和 `rouge_score`（上面的安装已带上）。

## 这里建的任务

原生 = 希腊语母语者用希腊语写的。机翻 = 从英语机器翻译而来，翻译痕迹和污染风险同时存在。

| 任务 | 数据集 | 类型 | 评测集 | 来源性质 | ILSP few-shot |
|---|---|---|---|---|---|
| `greekmmlu` | `dascim/GreekMMLU` | 选择题 | 公开 16,857，45 学科 | **原生** | — |
| `greekmmlu_gen` | `dascim/GreekMMLU` | 生成式，模型写字母；prompt 与 `greekmmlu` 逐字节相同 | 16,632，45 学科 | **原生** | 5 |
| `greekmmlu_gen_boxed` | `dascim/GreekMMLU` | 生成式，带格式指令，抽 `\boxed{}` 里的字母 | 16,632，45 学科 | **原生** | 0 |
| `ilspgreekmmlu` | `ilsp/mmlu_greek` | 选择题 | 14,042，57 学科 | 机翻 | 5 |
| `ilspgreekmmlu_gen` | `ilsp/mmlu_greek` | 生成式，模型写字母；prompt 与 `ilspgreekmmlu` 逐字节相同 | 14,042，57 学科 | 机翻 | 5 |
| `ilspgreekmmlupro` | `ilsp/MMLU-Pro_greek` | 生成式，10 选 1 | 12,032 | 机翻 | 0 |
| `ilspgreekarc_easy` / `_challenge` | `ilsp/arc_greek` | 选择题 | 2,376 / 1,168 | 机翻 | 25 |
| `ilspgreekhellaswag` | `ilsp/hellaswag_greek` | 选择题 | 10,024 | 机翻 | 10 |
| `ilspgreektruthfulqa_mc1` / `_mc2` | `ilsp/truthful_qa_greek` | 选择题 | 817 | 机翻，经人工校对 | 0 |
| `ilspgreektruthfulqa_gen` | `ilsp/truthful_qa_greek` | 生成式 | 817 | 机翻，经人工校对 | 0 |
| `ilspgreekmedicalmcqa` | `ilsp/medical_mcqa_greek` | 选择题，5 选 1 | 1,602 | **原生**（DOATAP 医师考试） | 15 |
| `ilspgreekasep` | `ilsp/mcqa_greek_asep` | 选择题 | 2,346 | **原生**（ASEP 公务员考试） | — |
| `ilspgreekifeval` | `ilsp/ifeval_greek` | 生成式，规则判分 | 541 | 翻译 + 本地化改编 | 0 |
| `ilspgreekmgsm` | `ilsp/mgsm_greek` | 生成式，CoT | 250 | 机翻 | 8 |
| `ilspgreekcivicsqa` | `ilsp/greek_civics_qa` | 生成式，BLEU | 407 | **原生**（教科书） | 0 |
| `ilspgreekflores_en_el` / `_el_en` | `ilsp/flores200_en-el` | 翻译 | 各 1,012 | 专业人工翻译 | — |

组与标签：`ilspgreekmmlu` 聚合它的 57 个学科；`greekmmlu` 聚合 45 个学科，另有 `greekmmlu_stem` / `_humanities` / `_social_sciences` / `_other` 四个分类组。标签 `ilspgreekarc`、`ilspgreektruthfulqa`、`ilspgreekflores` 可以把各自的成员一起跑。

按 benchmark 算是 **7 个选择题 : 6 个生成式** —— 比任务数看起来均衡得多，因为两个 MMLU 自己就占了 116 个叶子任务里的 102 个。算上下面的 `belebele_ell_Grek` 就是 8:6。`belebele_ell_Grek_gen`（在 `belebele_gen/`）是 Belebele 的同款写字母打分；英文防遗忘那侧由上游自带的 `mmlu_generative` 承担。GreekMMLU 另外有两种生成式打分（`greekmmlu_gen` 用于 5-shot，`greekmmlu_gen_boxed` 用于 0-shot），用来判断 log-likelihood 的变化是校准还是知识。

## lm-eval 自带的希腊语任务

不是这里建的，但在同一个 harness 里可用，值得知道。以下都已在本 fork 中确认存在。

| 任务 | 数据集 | 类型 | 说明 |
|---|---|---|---|
| `belebele_ell_Grek` | `facebook/belebele` | 选择题 | 阅读理解，**专业人工翻译** —— 本套件里翻译噪声最低的一个。ILSP 按 5-shot 报告 |
| `global_mmlu_full_el` | `CohereForAI/Global-MMLU` | 选择题 | 另一个希腊语 MMLU，部分经人工校验，带「文化敏感 / 文化无关」子集标注可分开报告。57 学科 + 4 个分类组 |
| `include_base_44_greek` | `CohereForAI/include-base-44` | 选择题 | **原生希腊语考试**，分类里包含 `medical_license` 和 `professional_certification` —— **同时报告前先查是否与 `ilspgreekmedicalmcqa`、`ilspgreekasep` 重叠**。有 `default`、`few_shot_en`、`few_shot_og` 三个变体 |
| `global_piqa_*_ell_grek` | `mrlbenchmarks/global-piqa-*` | 选择题 / 生成式 | 物理常识，parallel 与 nonparallel、cloze 与 generation 四种组合 |
| `xnli_el` | `facebook/xnli` | 选择题 | 自然语言推理，专业人工翻译 |
| `xquad_el` | `google/xquad` | 生成式 | 抽取式问答，专业人工翻译 |
| `arc_challenge_mt_el` | `LumiOpen/arc_challenge_mt` | 选择题 | ARC-Challenge 的**另一个机翻版本** —— 可以和 `ilspgreekarc_challenge` 对照，用来估计翻译带来的方差 |
| `multiblimp_ell` / `multiblimp_grc` | `jumelet/multiblimp` | 选择题 | 最小对立对语法性判断。`grc` 是**古希腊语** |

英文的 `mmlu`、`mmlu_pro`、`winogrande`、`mgsm`、`ifeval` 作为防遗忘的对照。

## 怎么跑

```bash
lm-eval --model hf --model_args pretrained=<model>,dtype=bfloat16 \
        --tasks greekmmlu --num_fewshot 5 --batch_size 4 --device cuda:0 \
        --log_samples --output_path runs/eval/<name>
```

**务必加 `--log_samples`**：下面的生成诊断要读这些 dump，而且它们是模型到底说了什么的唯一记录。

### 两种 prompt 格式

- **裸 few-shot + likelihood** —— 公开的希腊语数字用的就是这个设置（Meltemi、Llama-Krikri、Open LLM Leaderboard）。
- **chat template**（`--apply_chat_template`，few-shot 生成式任务还要加 `--fewshot_as_multiturn`）—— 部署后的 instruct 模型实际看到的格式。

**instruct 模型拿到裸 prompt 会怎样：生成式任务读 0。** 不是模型答错了，而是它立刻吐 EOS，harness 记录成空字符串，所有指标把空字符串判为失败。实测 Meltemi-7B-Instruct-v1.5：IFEval 15 条里 8 条空输出、instruction-level 0.087，MGSM 40 条里 36 条空、exact_match 0.0；套上模板后分别是 0.565 和 0.25。**所以生成式任务上看到 `0.000`，先去查生成诊断的 `empty` 那一行再下结论。**

### 选协议时要权衡什么

这里没有唯一正确答案，而且不同任务类型的取舍不一样。值得想清楚的几点：

**选择题任务一个 token 都不生成。** 打分是比较给定选项的 log-likelihood，所以空输出不可能发生，chat template 只是改变了周围的上下文。公开的希腊语数字都是裸 few-shot likelihood，所以「裸」买到的是可比性。在这里套模板也说得过去 —— 如果你关心的是模型部署形态下的表现 —— 但要预期到数字会变，并且明说变了。

**生成式任务才是取舍真正咬人的地方。** 一个在 chat template 背后微调出来的模型，会把裸 prompt 当成分布外输入，经常立刻吐 EOS —— harness 记成空串、所有指标判失败，于是任务读 0，而这跟能力毫无关系。反过来也成立：一个从没见过模板的 base checkpoint，套上模板也会被拖累。所以「base 和微调后走完全相同的路径」与「各自走原生格式」在生成式任务上是真冲突，**选哪个取决于你要下什么结论**。两种都报、并注明哪个是哪个，成本是多跑一次，换来的是没有歧义。

**先查一件事：你的 base 模型是不是已经认识自己的模板。** 有些 checkpoint 认识，通常是因为 mid-training 阶段混了对话数据 —— 训练仓库里的 `greekllm.train.probe_chat` 测的就是这个。如果 base 认识，上面那个冲突就不存在了：两边都可以套模板，既同路径又原生。

**空输出本身就是一个发现。** 微调后的模型还答不答裸 prompt，是指令微调改变的东西之一，不只是个要靠配置绕开的 artifact。生成诊断的 `terminated` 和 `empty` 两行直接测它 —— 所以即便裸 prompt 跑出来的数字不作为主结果，留一份也有价值。

### few-shot 设置

要和 ILSP 公布的数字可比时，用上面任务表里的 ILSP 那一列。`--num_fewshot` 是全局参数，few-shot 数不同的任务需要分开调用。

### 生成诊断

选择题 benchmark 排序的是给定选项的 log-likelihood，模型一个 token 都不生成，所以它们看不见模型会不会停、是否用希腊语作答、有没有原地打转。下面这个脚本读你已经产出的 sample dump：

```bash
python -m greekllm.eval.failure_profile --runs base=runs/eval/base sft=runs/eval/sft \
    --tokenizer <model> [--json profile.json]
```

报告：终止率、空输出率、语言一致性（GlotLID）、复读、乱码、长度合理性，以及 IFEval 的格式约束通过率。它不增加任何 prompt、不增加任何标注，**所以它是诊断不是 benchmark，报告里不要叫它 benchmark**。细节与阈值标定见训练仓库的 `evaluation/TASKS_TODO.md` §5。

## 坑

**ROUGE 对希腊语静默地恒为 0。** `rouge_score` 的分词器把 `[a-z0-9]` 之外的字符全部替换成空格，于是希腊语文本分词后是空列表，**两句完全相同的希腊语也得 0 分**。`ilspgreektruthfulqa_gen` 已经换上 Unicode 分词器修掉了；但 **lm-eval 里任何其他用 ROUGE 给希腊语生成打分的地方都有同一个洞**。BLEU 不受影响（sacrebleu 用 `tokenize="intl"`）。

**数字与 ILSP 协议可比，但不会逐位相同。** prompt 跟着他们的 lighteval fork，但我们没有复现他们实现里的机械缺陷（Python 三引号字符串带进来的缩进伪影、`Aπάντηση` 里的拉丁字母 `A`、被截断的 primer 句子），而且 lm-eval 把 TruthfulQA 的 mc1/mc2 当两个独立任务算，lighteval 是合并成一个候选列表算的。对照验证：MMLU EL `anatomy` 全量 135 题、5-shot，lighteval 0.3556 ± 0.0414，本移植 0.3926 ± 0.0422。

**希腊字母不是拉丁字母。** `ilspgreekmmlu` 和 `greekmmlu` 打分用的都是希腊字母 `Α/Β/Γ/Δ`（U+0391–0394），不是拉丁 `A/B/C/D` —— 它们是不同的 token，先验也不同。如果你要新增选择题任务，**跟上现有约定**，否则两者之间的比较会被标签选择污染。

**`ilspgreekmedicalmcqa` 评的是 `train` 分割。** 这是 ILSP 自己的配置这么做的（尽管分割名字看着反直觉），他们公布的 48.0% 指的也是这个。如果你更想用留出的那一半，在 yaml 里把 `test_split` 和 `fewshot_split` 对调 —— 然后就别再拿它和他们的数字比了。

**长 few-shot 上下文在大 batch 下会 OOM。** 4B 模型跑 5-shot MMLU、`--batch_size 16` 时试图分配 37 GiB 的 logits，而卡只有 48 GiB。用 4，或者 `--batch_size auto`。

**只有 `ilspgreekifeval` 能测终止率**，因为它是唯一配置了 `until: []` 的任务。其他任务 lm-eval 都会在停止串处截断生成。

**benchmark 数据绝不能进训练集。** 任何由强模型生成的合成数据，都要重跑一遍评测黑名单 —— 它可能逐字复现 benchmark 里的题目。

## 没有包含什么，以及为什么

- **`ilsp/winogrande_greek`** —— 该数据集的 `sentence`、`option1`、`option2` 仍然是**英文**，希腊语只存在于 `multiple_choice_targets`，而且它的两个候选是各自独立机翻的（抽样 100 条：14% 整句被翻成了不同版本，7% 只在一个分支里留着拉丁字母人名，模型可以靠这个表面线索作弊）。ILSP 自己的套件里也没有 winogrande 任务。
- **MT-Bench Greek、Arena-Hard Greek** —— 多轮对话和成对判分装不进 lm-eval。希腊语 judge prompt 存在训练仓库的 `evaluation/ilsp-assets/`。
- **`ilsp/greek_lyceum_mathematics`** —— 答案是完整的解题过程，所以在成为任务之前需要先定打分方式（judge，还是从最后一行抽取答案）。
- **把 ILSP 的 lighteval fork 直接当 runner 用** —— 测过，否掉了。它把 torch 钉在 `<2.5`，导致 transformers 被压到 4.46，根本加载不了 Qwen3.5-4B 或 K2；而且要改六处才能跑通一次。值得拿的是它的希腊语 IFEval 约束库，其余每个任务不过是一个约 20 行的 prompt 函数。
