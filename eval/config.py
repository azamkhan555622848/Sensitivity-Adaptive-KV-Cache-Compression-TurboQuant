"""YAML config loader for experiments."""
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    name: str
    dtype: str = "bfloat16"
    max_seq_len: int = 8192

@dataclass
class MethodConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SweepConfig:
    output_dir: str
    models: List[ModelConfig]
    methods: List[MethodConfig]
    benchmarks: List[BenchmarkConfig]

def load_sweep(path: str) -> SweepConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    models = []
    for m in raw["models"]:
        if isinstance(m, str):
            with open(m) as f:
                m = yaml.safe_load(f)
        models.append(ModelConfig(**m))
    methods = [MethodConfig(**m) for m in raw["methods"]]
    benchmarks = [BenchmarkConfig(**b) for b in raw["benchmarks"]]
    return SweepConfig(output_dir=raw["output_dir"], models=models, methods=methods, benchmarks=benchmarks)
