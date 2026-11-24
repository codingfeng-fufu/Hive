"""A shared Qwen reader for the ColPali evidence-selection comparison."""

from time import perf_counter

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

PROMPT = (
    "Answer the question using the supplied document evidence. "
    "Return the shortest complete answer without explanation. "
    "If the evidence cannot answer the question, return Not answerable."
)


class Reader:
    def __init__(self, image_size, *, variable_size=False):
        from os import environ

        import torch

        self.torch = torch
        device = environ.get("HIVG_READER_DEVICE", "cuda")
        patch_area = image_size[0] * image_size[1]
        path = "data/models/qwen2.5-vl-7b"
        self.processor = AutoProcessor.from_pretrained(
            path,
            min_pixels=56 * 56 if variable_size else patch_area,
            max_pixels=patch_area,
            local_files_only=True,
        )
        options = {}
        if device == "cuda":
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            local_files_only=True,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            **options,
        ).eval()
        # Qwen's SDPA builds a dense mask across all images even though the
        # vision tower's attention is confined to each image. Keep the decoder
        # input and its positions intact while reducing the peak vision memory.
        vision = self.model.visual
        encode = vision.forward

        def encode_images(pixels, grid_thw):
            if len(grid_thw) <= 2:
                return encode(pixels, grid_thw)
            lengths = grid_thw.prod(dim=1).tolist()
            pieces = pixels.split(lengths)
            return torch.cat(
                [encode(piece, grid[None]) for piece, grid in zip(pieces, grid_thw, strict=True)]
            )

        vision.forward = encode_images
        # Generation only consumes the final sequence position.
        head = self.model.lm_head
        project = head.forward

        def last_token_logits(hidden):
            return project(hidden[:, -1:, :])

        head.forward = last_token_logits

    def answer(self, question, images, evidence=None, *, instruction=PROMPT):
        context = f"Document evidence:\n{evidence}\n" if evidence is not None else ""
        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": f"{context}Question: {question}\nShort answer:"},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(text=[text], images=images or None, return_tensors="pt").to(
            self.model.device
        )
        start = perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**batch, do_sample=False, max_new_tokens=64)
        seconds = perf_counter() - start
        length = batch.input_ids.shape[-1]
        return {
            "answer": self.processor.decode(output[0, length:], skip_special_tokens=True).strip(),
            "prompt_tokens": length,
            "visual_tokens": int(batch.input_ids.eq(self.model.config.image_token_id).sum()),
            "generation_seconds": seconds,
        }
