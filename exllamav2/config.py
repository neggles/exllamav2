import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from exllamav2.fasttensors import STFile


@dataclass
class ExLlamaV2Config:
    debug_mode: bool = False
    # Directory containing model files
    model_dir: Optional[Path] = None
    # Maximum sequence length. Sequences longer than this will throw an exception
    max_seq_len: int = 2048
    # Maximum size of batches to process
    max_batch_size: int = 1
    # Maximum length of input IDs in a single forward pass. Sequences longer than this will be processed in multiple steps
    max_input_len: int = 2048
    # Sequences will be processed in chunks to keep the size of the attention weights matrix <= this
    max_attention_size: int = 2048**2
    # Factor by which to scale positional embeddings, e.g. for 4096-token sequence use a scaling factor of 2.0, requires finetuned model or LoRA
    scale_pos_emb: float = 1.0
    # Alpha value for NTK RoPE scaling. Similar to compress_pos_emb but works without finetuned model
    scale_alpha_value: float = 1.0
    # Implementation will automatically use flash-attn-2 when available
    no_flash_attn: bool = False

    # Loaded/set by .prepare():
    architecture: str = field(init=False)

    model_config: str = field(init=False)
    tensor_file_map: dict = field(init=False)
    tensor_files: list = field(init=False)

    tokenizer_path: str = field(init=False)

    bos_token_id: int = field(init=False)
    eos_token_id: int = field(init=False)
    pad_token_id: int = field(init=False)

    hidden_size: int = field(init=False)
    initializer_range: int = field(init=False)
    intermediate_size: int = field(init=False)
    num_attention_heads: int = field(init=False)
    num_key_value_heads: int = field(init=False)
    num_key_value_groups: int = field(init=False)
    num_hidden_layers: int = field(init=False)
    rms_norm_eps: float = field(init=False)
    vocab_size: int = field(init=False)
    # Constant for all Llama models, nodified by .prepare() if scale_alpha_value != 1.0
    rotary_embedding_base: float = field(default=10000.0, init=False)
    # Constant for all Llama models, except 3b
    head_dim: int = field(default=128, init=False)
    num_experts: int = None
    num_experts_per_token: int = None
    attention_bias_qkv: bool = False
    attention_bias_o: bool = False
    checkpoint_fused_mlp: bool = False

    # Experimental, Linux only
    fasttensors: bool = False

    def set_low_mem(self):
        """Set low-memory options for processing large sequences."""
        self.max_input_len = 1024
        self.max_attention_size = 1024**2

    def prepare(self, no_tensors=False):
        """Populate config with required files from model_dir"""
        if self.model_dir is None:
            raise ValueError(" ## No model_dir specified in ExLlamaV2Config")
        self.model_dir = Path(self.model_dir).resolve()

        if not self.model_dir.exists() and self.model_dir.is_dir():
            raise FileNotFoundError(f" ## Can't find model dir: {self.model_dir}")

        # Load config.json
        self.model_config = self.model_dir / "config.json"
        if not self.model_config.is_file():
            raise FileNotFoundError(f" ## Can't find model config: {self.model_config}")

        with self.model_config.open(encoding="utf8") as f:
            read_config = json.load(f)

            layer_keys = []
            expect_keys = []
            layer_keys_llama_norms = [["input_layernorm"], ["post_attention_layernorm"]]
            layer_keys_yi_norms = [["ln1", "input_layernorm"], ["ln2", "post_attention_layernorm"]]
            layer_keys_llama_attn = [
                ["self_attn.q_proj"],
                ["self_attn.k_proj"],
                ["self_attn.v_proj"],
                ["self_attn.o_proj"],
            ]
            layer_keys_llama_mlp = [["mlp.down_proj"], ["mlp.gate_proj"], ["mlp.up_proj"]]
            layer_keys_llama_mlp_swiglu = [["mlp.swiglu.w12"], ["mlp.swiglu.w3"]]
            expect_keys_llama = [["lm_head"], ["model.norm"], ["model.embed_tokens"]]
            expect_keys_gemma = [["model.norm"], ["model.embed_tokens"]]

            if "LlamaForCausalLM" in read_config["architectures"]:
                self.architecture = "Llama"
                layer_keys += layer_keys_llama_norms + layer_keys_llama_attn + layer_keys_llama_mlp
                expect_keys += expect_keys_llama

            elif "MistralForCausalLM" in read_config["architectures"]:
                self.architecture = "Llama"
                layer_keys += layer_keys_llama_norms + layer_keys_llama_attn + layer_keys_llama_mlp
                expect_keys += expect_keys_llama

            elif "YiForCausalLM" in read_config["architectures"]:
                self.architecture = "Yi"
                layer_keys += layer_keys_yi_norms + layer_keys_llama_attn + layer_keys_llama_mlp
                expect_keys += expect_keys_llama

            elif "MixtralForCausalLM" in read_config["architectures"]:
                self.architecture = "Mixtral"
                self.num_experts = read_config["num_local_experts"]
                self.num_experts_per_token = read_config["num_experts_per_tok"]
                layer_keys += (
                    layer_keys_llama_norms
                    + layer_keys_llama_attn
                    + [[f"block_sparse_moe.experts.{e}.w{w}" for e in range(8) for w in range(3)]]
                    + [["block_sparse_moe.gate"]]
                )
                expect_keys += expect_keys_llama

            elif "OrionForCausalLM" in read_config["architectures"]:
                self.architecture = "Orion"
                layer_keys += layer_keys_llama_norms + layer_keys_llama_attn + layer_keys_llama_mlp
                expect_keys += expect_keys_llama

            elif "Qwen2ForCausalLM" in read_config["architectures"]:
                self.architecture = "Qwen2"
                layer_keys += layer_keys_llama_norms + layer_keys_llama_attn + layer_keys_llama_mlp
                expect_keys += expect_keys_llama
                self.attention_bias_qkv = True
                self.attention_bias_o = False

            elif "GemmaForCausalLM" in read_config["architectures"]:
                self.architecture = "Gemma"
                layer_keys += layer_keys_llama_norms + layer_keys_llama_attn + layer_keys_llama_mlp
                expect_keys += expect_keys_gemma

            else:
                print(f" !! Warning, unknown architecture: {repr(read_config['architectures'])}")
                print(" !! Loading as LlamaForCausalLM")
                self.architecture = "Llama"
                layer_keys += layer_keys_llama_norms + layer_keys_llama_attn + layer_keys_llama_mlp
                expect_keys += expect_keys_llama

            self.bos_token_id = read_config["bos_token_id"] if "bos_token_id" in read_config else 1
            self.eos_token_id = read_config["eos_token_id"] if "eos_token_id" in read_config else 2
            self.pad_token_id = read_config["pad_token_id"] if "pad_token_id" in read_config else 0

            self.hidden_size = read_config["hidden_size"]
            self.initializer_range = read_config["initializer_range"]
            self.intermediate_size = read_config["intermediate_size"]
            self.num_attention_heads = read_config["num_attention_heads"]
            self.num_hidden_layers = read_config["num_hidden_layers"]
            self.rms_norm_eps = read_config["rms_norm_eps"]
            self.vocab_size = read_config["vocab_size"]
            if read_config.get("attention_bias", False):
                self.attention_bias_qkv = True
                self.attention_bias_o = True

            self.rotary_embedding_base = read_config["rope_theta"] if "rope_theta" in read_config else 10000.0

            if "num_key_value_heads" in read_config:
                self.num_key_value_heads = read_config["num_key_value_heads"]
                self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
            else:
                self.num_key_value_heads = self.num_attention_heads
                self.num_key_value_groups = 1

            if "max_sequence_length" in read_config:
                self.max_seq_len = read_config["max_sequence_length"]
            elif "max_position_embeddings" in read_config:
                self.max_seq_len = read_config["max_position_embeddings"]

            rs = read_config.get("rope_scaling", None)
            if rs and "factor" in rs:
                factor = rs["factor"]
                scaling_type = rs.get("type", None)
                if scaling_type == "linear":
                    self.scale_pos_emb = factor
                # elif scaling_type == "yarn":
                #     self.scale_alpha_value = factor

        # Model dimensions

        if "head_dim" in read_config:
            self.head_dim = read_config["head_dim"]
        else:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Create map of model tensors

        if no_tensors:
            return

        self.tensor_file_map = {}
        self.tensor_files = [x for x in self.model_dir.iterdir() if x.suffix.lower() == ".safetensors"]

        if len(self.tensor_files) == 0:
            raise ValueError(f" ## No .safetensors files found in {self.model_dir}")

        for st_file in self.tensor_files:
            f = STFile.open(st_file, fast=self.fasttensors)
            for key in f.get_dict():
                self.tensor_file_map[key] = st_file

        # For loading checkpoints with fused MLP layers

        if self.architecture == "Llama" or self.architecture == "Yi":
            if (
                "model.layers.0.mlp.down_proj.weight" not in self.tensor_file_map
                and "model.layers.0.mlp.swiglu.w12.weight" in self.tensor_file_map
            ):
                for x in layer_keys_llama_mlp:
                    layer_keys.remove(x)
                layer_keys += layer_keys_llama_mlp_swiglu
                self.checkpoint_fused_mlp = True

        # Make sure we found all the layers we need

        for layer_idx in range(self.num_hidden_layers):
            for ks in layer_keys:
                prefixes = [f"model.layers.{layer_idx}.{k}" for k in ks]
                expect_keys.append(prefixes)

        for prefixes in expect_keys:
            for prefix in prefixes:
                if any(key.startswith(prefix) for key in self.tensor_file_map):
                    break
            else:
                raise ValueError(f" ## Could not find {prefix}.* in model")
