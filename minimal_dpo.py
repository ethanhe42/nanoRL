import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

system_prompt = "write answer inside <answer></answer>\n"
# (prompt, chosen, rejected) preference triples.
DATASET = [
    ("What is 1 + 1?", "<answer>2</answer>",  "<answer>3</answer>"),
    ("What is 2 + 3?", "<answer>5</answer>",  "<answer>4</answer>"),
    ("What is 4 + 4?", "<answer>8</answer>",  "<answer>9</answer>"),
    ("What is 5 + 7?", "<answer>12</answer>", "<answer>13</answer>"),
    ("What is 6 + 9?", "<answer>15</answer>", "<answer>14</answer>"),
    ("What is 3 + 8?", "<answer>11</answer>", "<answer>10</answer>"),
]


class DPO:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seqlen = 64
        self.beta = 0.1

        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

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

    def _seq_logprob(self, m, ids, mask):
        # Sum of log probs of completion tokens under model m. (N, T) → (N,)
        log_probs = F.log_softmax(m(ids).logits, -1)
        shift_logp = log_probs[:, :-1, :]
        shift_targets = ids[:, 1:, None]
        shift_mask = mask[:, 1:].float()
        token_logp = shift_logp.gather(-1, shift_targets).squeeze(-1)
        return (token_logp * shift_mask).sum(dim=-1)

    def step(self, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
        # One forward pass on concat([chosen, rejected]) — current model has grad, ref doesn't.
        N = chosen_ids.size(0)
        ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        mask = torch.cat([chosen_mask, rejected_mask], dim=0)

        logp = self._seq_logprob(self.model, ids, mask)
        with torch.no_grad():
            ref_logp = self._seq_logprob(self.ref_model, ids, mask)

        # DPO loss: -log σ(β · (log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l))).
        log_ratio = logp - ref_logp                # 2N
        margin = log_ratio[:N] - log_ratio[N:]     # N (chosen − rejected)
        loss = -F.logsigmoid(self.beta * margin).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        metrics = dict(
            loss=loss.detach(),
            chosen_logp=logp[:N].detach().mean(),
            rejected_logp=logp[N:].detach().mean(),
            reward_gap=(self.beta * margin).detach().mean(),
            accuracy=(margin.detach() > 0).float().mean(),
            grad_norm=grad_norm,
        )
        return metrics


dpo = DPO()

for step in range(30):
    batch = random.sample(DATASET, k=1)
    prompts = [system_prompt + p for p, c, r in batch]
    chosens = [c for p, c, r in batch]
    rejecteds = [r for p, c, r in batch]

    chosen_ids, chosen_mask = dpo._encode(prompts, chosens)
    rejected_ids, rejected_mask = dpo._encode(prompts, rejecteds)

    metrics = dpo.step(chosen_ids, chosen_mask, rejected_ids, rejected_mask)

    print('-' * 10, 'step', step, '-' * 10)
    print(f"prompt: {batch[0][0]}  chosen: {batch[0][1]}  rejected: {batch[0][2]}")
    print(" ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
