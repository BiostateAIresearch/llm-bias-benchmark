# LLM Bias Benchmark

A benchmark for evaluating large language model (LLM) calibration and decision-making bias, as described in:

> **Failure to Mix: Large language models struggle to answer according to desired probability distributions**  
> Yang & Zhang, 2025

## Quick Start

```bash
# Install dependencies
pip install aiohttp pandas numpy tqdm nest_asyncio

# Run benchmark
python benchmark.py --model "google/gemini-2.5-pro" --api-key "YOUR_OPENROUTER_KEY"
```

## Usage

### Single API Key
```bash
python benchmark.py -m "openai/gpt-4" -k "sk-or-v1-xxx"
```

### Multiple API Keys (for load balancing)
```bash
python benchmark.py -m "anthropic/claude-3.5-sonnet" --api-keys "key1,key2,key3"
```

### Custom Output Directory
```bash
python benchmark.py -m "google/gemini-2.5-pro" -k "YOUR_KEY" -s "./my_results"
```

## Experiments

The benchmark consists of 5 experiments (50 points total):

| Experiment | Description | Max Score |
|------------|-------------|-----------|
| Exp1 | Single binary flip calibration | 20 |
| Exp2 | Two flips per call (D=2) | 20 |
| Exp3 | Three-outcome distribution (3 outcomes) | 20 |
| Exp4 | Decision problems (Bioinformatics, Game Theory) | 20 |
| Exp5 | Word choice and position bias | 20 |

## Scoring

### Step-Likeness Score (S)

For experiments 1-3, we compute S = 4 × ∫|r(p) - p| dp, where:
- r(p) = observed response rate
- p = requested probability
- S = 0: Perfect calibration
- S = 1: Perfect step function

### Experiment Scores

- **Exp1**: Score = max(0, 20 × (1 - S))
- **Exp2**: Score_S2 (10 pts) + Score_Savg (10 pts)
- **Exp3**: Score_S0 + Score_S1 + Score_S2 (20/3 pts each)
- **Exp4**: Bioinformatics (8 pts) + MatchingPennies (8 pts) + Luxury (4 pts)
- **Exp5**: Position bias (10 pts) + Semantic bias (10 pts)

## Output

Results are saved to `./results/<model>_<timestamp>/`:
- `exp1_raw.csv` - Raw API responses for Exp1
- `exp2_raw.csv` - Raw API responses for Exp2
- `exp3_raw.csv` - Raw API responses for Exp3
- `exp5_raw.csv` - Raw API responses for Exp4
- `exp6_raw.csv` - Raw API responses for Exp5
- `summary.json` - Scores and metrics

## API

This benchmark uses [OpenRouter](https://openrouter.ai/) API. Supported models include:
- `google/gemini-2.5-pro`
- `openai/gpt-4`
- `openai/gpt-5`
- `anthropic/claude-3.5-sonnet`
- `anthropic/claude-sonnet-4`
- `moonshotai/kimi-k2`
- `qwen/qwen3-235b-a22b`

## Citation

```bibtex
@article{yang2025failure,
  title={Failure to Mix: Large language models struggle to answer according to desired probability distributions},
  author={Yang, Ivy Yuqian and Zhang, David Yu},
  journal={Nature Methods},
  year={2025}
}
```

## License

MIT License
