import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

system_prompt = "write answer inside <answer></answer>\n"
# (prompt, completion) target pairs.
DATASET = [
    ("What is 1 + 1?", "<answer>2</answer>"),
    ("What is 2 + 3?", "<answer>5</answer>"),
    ("What is 4 + 4?", "<answer>8</answer>"),
    ("What is 5 + 7?", "<answer>12</answer>"),
    ("What is 6 + 9?", "<answer>15</answer>"),
    ("What is 3 + 8?", "<answer>11</answer>"),
]


class SFT:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seqlen = 64

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)

    def _encode(self, prompts, completions):
        # Tokenize each (prompt, completion). Mask = 1 only on completion tokens.
        ids_list, masks_list = [], []
        for p, c in zip(prompts, completions):
            # Tokenize the two halves separately: BPE can merge across the
            # boundary (e.g. "?<" is one token), which would shift the mask.
            prompt_ids = self.tokenizer(p, return_tensors='pt').input_ids[0]
            completion_ids = self.tokenizer(c, return_tensors='pt', add_special_tokens=False).input_ids[0]
            full_ids = torch.cat([prompt_ids, completion_ids])
            prompt_len, seqlen = len(prompt_ids), len(full_ids)
            full_ids = F.pad(full_ids, (0, self.max_seqlen - seqlen), value=self.tokenizer.pad_token_id)
            mask = torch.zeros_like(full_ids)
            mask[prompt_len:seqlen] = 1
            ids_list.append(full_ids)
            masks_list.append(mask)
        return torch.stack(ids_list).to(DEVICE), torch.stack(masks_list).to(DEVICE)

    def step(self, ids, mask):
        # Cross-entropy loss on completion tokens only.
        log_probs = F.log_softmax(self.model(ids).logits, -1)
        shift_logp = log_probs[:, :-1, :]
        shift_targets = ids[:, 1:, None]
        shift_mask = mask[:, 1:].float()
        token_logp = shift_logp.gather(-1, shift_targets).squeeze(-1)  # N, T-1
        denom = shift_mask.sum().clamp(min=1.0)
        loss = -(token_logp * shift_mask).sum() / denom

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        seq_logp = (token_logp * shift_mask).sum(dim=-1)  # N (per-example sum)
        metrics = dict(
            loss=loss.detach(),
            seq_logp=seq_logp.detach().mean(),
            grad_norm=grad_norm,
        )
        return metrics


sft = SFT()

for step in range(10):
    batch = random.sample(DATASET, k=1)
    prompts = [system_prompt + p for p, c in batch]
    completions = [c for p, c in batch]
    ids, mask = sft._encode(prompts, completions)
    metrics = sft.step(ids, mask)

    print('-' * 10, 'step', step, '-' * 10)
    print(f"prompt: {batch[0][0]}  completion: {batch[0][1]}")
    print(" ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
