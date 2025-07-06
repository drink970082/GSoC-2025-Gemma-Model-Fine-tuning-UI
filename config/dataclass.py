from dataclasses import dataclass
from typing import Literal, Optional, Union


@dataclass
class LoraParams:
    lora_rank: int


@dataclass
class DpoParams:
    dpo_beta: float


@dataclass
class MethodConfig:
    name: Literal["Standard", "LoRA", "DPO"]
    parameters: Optional[Union[LoraParams, DpoParams]] = None


@dataclass
class DataConfig:
    source: str
    dataset_name: str
    split: str
    shuffle: bool
    batch_size: int
    seq2seq_in_prompt: str
    seq2seq_in_response: str
    seq2seq_max_length: int
    seq2seq_truncate: bool
    config: Optional[str] = None


@dataclass
class ModelConfig:
    model_variant: str
    epochs: int
    learning_rate: float


@dataclass
class TrainingConfig:
    model_name: str
    method_config: MethodConfig
    data_config: DataConfig
    model_config: ModelConfig
