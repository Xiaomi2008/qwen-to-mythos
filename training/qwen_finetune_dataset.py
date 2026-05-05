"""
Streaming dataset wrapper for QwenRecurrentModel fine-tuning.

Uses HuggingFace's AutoTokenizer (Qwen3) with the same packing/sharding
pattern as FineWebEduDataset. Supports any HF dataset with a text column.
"""

from torch.utils.data import IterableDataset, get_worker_info
from datasets import load_dataset
from transformers import AutoTokenizer
import torch


class QwenStreamingDataset(IterableDataset):
    """
    Streaming dataset loader yielding fixed-length (input, target) pairs.

    Uses two-dimensional sharding (world_size ranks x num_workers DataLoader
    workers) for disjoint coverage without cross-process coordination.

    Documents are concatenated into a rolling buffer and sliced into fixed-length
    chunks, packing short docs together and splitting long ones.
    """

    def __init__(
        self,
        dataset_name: str,
        seq_len: int,
        rank: int,
        world_size: int,
        tokenizer_name: str = "Qwen/Qwen3-4B",
        dataset_config: str = None,
        text_column: str = "text",
    ):
        """
        Args:
            dataset_name   -- HuggingFace dataset name (e.g. "roneneldan/TinyStories")
            seq_len        -- context length; every yielded pair has this many tokens
            rank           -- global rank of this process
            world_size     -- total number of distributed processes
            tokenizer_name -- HuggingFace tokenizer repo (default: Qwen/Qwen3-4B)
            dataset_config -- HF dataset config name (e.g. "sample-10BT") or None
            text_column    -- name of the text column in the dataset
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, legacy=False)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.text_column = text_column
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 1

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        if self.dataset_config:
            ds = load_dataset(
                self.dataset_name,
                name=self.dataset_config,
                split="train",
                streaming=True,
            ).shard(num_shards=total_shards, index=shard_index)
        else:
            ds = load_dataset(
                self.dataset_name,
                split="train",
                streaming=True,
            ).shard(num_shards=total_shards, index=shard_index)

        buf = []
        for sample in ds:
            tokens = self.tokenizer.encode(sample[self.text_column], add_special_tokens=False)
            buf.extend(tokens)
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
