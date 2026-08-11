"""
VLM Inference Pipeline for ViCSynAD.

Handles Qwen2-VL-7B 4-bit quantized loading and feature extraction.
The vision encoder is FROZEN — we only extract embeddings for fusion training.

VRAM Budget: ~5.5 GB for Qwen2-VL-7B 4-bit quantized model.
"""

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import Optional, List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLMFeatureExtractor:
    """
    Load quantized Qwen2-VL and extract vision features from time series images.

    Uses:
    - Qwen2-VL-7B-Instruct with 4-bit NF4 quantization
    - Vision tower only (text generation head not loaded/inactive)
    - Target VRAM: ~5.5 GB
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        use_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4",
    ):
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading VLM: {model_id}")
        logger.info(f"  4-bit quantization: {use_4bit}")
        logger.info(f"  Device: {self.device}")

        self.model = self._load_model(
            use_4bit, device_map, torch_dtype,
            bnb_4bit_compute_dtype, bnb_4bit_quant_type
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Cache vision encoder reference
        self.vision_tower = self.model.visual

        # Extract vision feature dimension
        self.vision_dim = self.model.config.vision_config.hidden_size
        logger.info(f"  Vision feature dim: {self.vision_dim}")
        logger.info(f"  Model loaded. VRAM used: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")

    def _load_model(
        self, use_4bit, device_map, torch_dtype,
        bnb_4bit_compute_dtype, bnb_4bit_quant_type
    ):
        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=True,
            )
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
        model.eval()
        return model

    @torch.no_grad()
    def extract_vision_features(
        self,
        images: List[Image.Image],
        batch_size: int = 4,
    ) -> torch.Tensor:
        """
        Extract vision features for a list of time series images.

        Args:
            images: List of PIL Images (TS visualizations)
            batch_size: batch size for feature extraction

        Returns:
            features: (N, vision_dim) tensor of vision embeddings
        """
        all_features = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]

            # Process inputs through processor
            inputs = self.processor(
                text=["Describe the temporal patterns in this time series."] * len(batch_imgs),
                images=batch_imgs,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward through vision encoder and get pooled features
            pixel_values = inputs.get("pixel_values")
            image_grid_thw = inputs.get("image_grid_thw")

            if pixel_values is not None:
                # Extract vision features directly
                vision_outputs = self.vision_tower(
                    pixel_values,
                    grid_thw=image_grid_thw
                )
                # vision_outputs is (sum(T*H*W), vision_dim) for all images in batch
                # Use mean pooling over spatial positions
                features = vision_outputs.mean(dim=0, keepdim=True)  # (1, vision_dim)
                # Actually we need per-image features, so handle grid_thw properly
                if image_grid_thw is not None:
                    # Split by grid_thw: each image has thw[0] * thw[1] * thw[2] tokens
                    split_sizes = (image_grid_thw.prod(dim=-1) // 4).tolist()  # divided by spatial merge
                    features_list = torch.split(vision_outputs, split_sizes)
                    features = torch.stack([f.mean(dim=0) for f in features_list])
                else:
                    features = vision_outputs.mean(dim=0, keepdim=True)
            else:
                # Fallback: use model's internal representation
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                )
                # Use last hidden state of vision tokens
                hidden = outputs.hidden_states[0]  # First layer hidden states
                # Pool: take mean over non-text tokens
                features = hidden.mean(dim=1)  # (B, hidden_dim)

            all_features.append(features.cpu())

        result = torch.cat(all_features, dim=0)  # (N, vision_dim)
        return result

    @torch.no_grad()
    def generate_explanation(
        self,
        image: Image.Image,
        causal_info: Dict,
        prompt_template: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate a natural language explanation using the full VLM (vision + language).

        Args:
            image: Anomaly window visualization
            causal_info: dict with causal graph, attribution scores, etc.
            prompt_template: CoT prompt template string
            max_new_tokens: max generation length
            temperature: sampling temperature

        Returns:
            Generated explanation text
        """
        # Format prompt with causal information
        prompt = prompt_template.format(
            causal_graph=causal_info.get("graph_summary", ""),
            attribution_scores=causal_info.get("attribution", ""),
            root_cause=causal_info.get("root_cause", ""),
            propagation_path=causal_info.get("propagation", ""),
        )

        # Process
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
        )

        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        explanation = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return explanation

    @property
    def vram_usage_gb(self) -> float:
        return torch.cuda.max_memory_allocated() / 1024**3


# ── Feature Cache ───────────────────────────────────────────────────

class FeatureCache:
    """
    Cache pre-extracted vision features to avoid redundant VLM forward passes.
    Uses a dictionary keyed by dataset + window index.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache = {}
        self.cache_dir = cache_dir

    def _make_key(self, dataset: str, window_idx: int) -> str:
        return f"{dataset}__{window_idx:06d}"

    def get(self, dataset: str, window_idx: int) -> Optional[torch.Tensor]:
        key = self._make_key(dataset, window_idx)
        return self.cache.get(key)

    def set(self, dataset: str, window_idx: int, features: torch.Tensor):
        key = self._make_key(dataset, window_idx)
        self.cache[key] = features.cpu()

    def extract_or_cache(
        self,
        extractor: VLMFeatureExtractor,
        images: List[Image.Image],
        dataset: str,
        start_idx: int = 0,
        batch_size: int = 4,
    ) -> torch.Tensor:
        """
        Extract features, caching results.
        """
        # Check cache first
        cached = []
        uncached_indices = []
        for i, img in enumerate(images):
            widx = start_idx + i
            feat = self.get(dataset, widx)
            if feat is not None:
                cached.append(feat)
            else:
                uncached_indices.append(i)
                cached.append(None)

        # Extract uncached
        if uncached_indices:
            uncached_imgs = [images[i] for i in uncached_indices]
            new_features = extractor.extract_vision_features(
                uncached_imgs, batch_size=batch_size
            )
            for j, idx in enumerate(uncached_indices):
                self.set(dataset, start_idx + idx, new_features[j])
                cached[idx] = new_features[j]

        return torch.stack([f for f in cached if f is not None])


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing VLM pipeline (requires model download on first run)...")
    print("Skipping auto-test to avoid large model download.")
    print("Run with --download to trigger model loading.")
