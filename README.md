<div align= "center">
    <h1>AgenticDataBench: A Comprehensive Benchmark for Data Agents</h1>
</div>


<br>

<div align="center">
<img src="docs/img/example.png" width="600px">
</div>

<br>

AgenticDataBench is a comprehensive benchmark for evaluating LLM-based data agents that automate real-world data science workflows. It addresses the lack of rigorous evaluation by providing diverse, realistic tasks with fine-grained ground-truth labels.

The benchmark spans 15 domains, including real B2B fintech use cases, and is structured around reusable data science skills—core operational patterns extracted from large-scale task solutions (see [`skill_cluster`](./skill_cluster)). It combines curated real-world tasks with systematically generated ones (see [`generator`](./generator)) to ensure broad coverage and minimal redundancy.

AgenticDataBench enables detailed evaluation of data agents, offering both overall accuracy and fine-grained, skill-level performance insights.

## Community

We deeply appreciate the invaluable effort contributed by our dedicated team of developers, supportive users, and esteemed industry partners.

- [Tsinghua University](https://www.tsinghua.edu.cn/en)
- [Ant Digital Technologies, Ant Group](https://intl.antdigital.com/en)

## 📁 Benchmark Data

- **Datasets**: [Download from HuggingFace](https://huggingface.co/datasets/shawnzzzh/AgenticDataBench) → `testbed/datasets/`
- **Skills**: [`skill_cluster/data/skill-descriptions.jsonl`](./skill_cluster/data/skill-descriptions.jsonl)
- **Tasks**: [`testbed/tasks`](./testbed/tasks)
- **Ground-truth**: [`testbed/gold`](./testbed/gold)

For leaderboard integrity, we withhold 100 tasks as a private test set. These tasks will be publicly released once the benchmark loses its evaluation significance.

<span id="-quickstart"></span>

## Quickstart

### 🔑 Set API Keys

Configure your API keys in a `.env` file:

```bash
# For Qwen models (DashScope)
echo "DASHSCOPE_API_KEY=your_key_here" > .env
```

### 🔧 Installation

```bash
pip install -r testbed/requirements.txt
```

### 🚀 Run Benchmark

You can also explore task generation and skill construction via [`generator`](./generator) and [`skill_cluster`](./skill_cluster).

```bash
# For da-agent
cd testbed && ./run_da_agent.sh

# For smolagents
cd testbed && ./run_smolagents.sh
```

After running, evaluate the results:

```bash
cd testbed
python3 evaluate.py --output_dir output/da-agent-qwen-{experiment_id}
```

<div align="center">
<img src="docs/img/pipeline.png" width="400px">
</div>

## 📊 Result Uploading

Benchmark results are stored in [`testbed/results`](./testbed/results). Initial experiments cover [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5), and [Claude Sonnet 4.6](https://platform.claude.com/docs/en/about-claude/models/overview) across [DA-Agent](https://github.com/yiyihum/da-code/tree/main) and [Smolagents](https://github.com/huggingface/smolagents) frameworks.

<div align="center">
<img src="docs/img/cost_score_scatter.png" width="400px">
</div>