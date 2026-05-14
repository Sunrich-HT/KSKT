# KSKT

**Know Thyself, Know Thy User: Intrinsic Dual-Perspective Reasoning for Role-Playing LLMs**

ICML 2026 · First author: Haotong Sun

> Code, configs, and adapters will be released here upon publication.

## TL;DR

An intrinsic dual-stream reasoning framework for role-playing LLMs — combining dual-stream axial attention, mutual-understanding positional encoding, an adaptive deliberation budget, and a self-aware MoE.

- Backbone: Qwen3-4B-Thinking
- Parameter efficiency: **1/18** of Qwen2-72B yet stronger on role-playing benchmarks
- Chinese role-playing: surpasses Claude-3-opus
- SOTOPIA Relationship: **+19.3%**

Concurrent submission with [BRIDGE](https://github.com/Sunrich-HT/BRIDGE) — KSKT targets intra-turn dual-perspective reasoning, BRIDGE targets cross-turn long-horizon consistency. Together they form a complete persona-consistency research matrix.

## Status

🚧 Placeholder — code release planned post camera-ready.

## Citation

```bibtex
@inproceedings{sun2026kskt,
  title={Know Thyself, Know Thy User: Intrinsic Dual-Perspective Reasoning for Role-Playing LLMs},
  author={Sun, Haotong and others},
  booktitle={ICML},
  year={2026}
}
```
