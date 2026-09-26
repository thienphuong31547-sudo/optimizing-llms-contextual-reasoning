import torch
import torch.nn as nn

class ContextualReasoningModel(nn.Module):
    """LLM wrapper for contextual reasoning in multi-task environments."""

    def __init__(self, model_name="bert-base-uncased", num_tasks=5):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        self.task_heads = nn.ModuleList([nn.Linear(self.encoder.config.hidden_size, 2) for _ in range(num_tasks)])

    def forward(self, input_ids, attention_mask, task_id=0):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.task_heads[task_id](outputs.last_hidden_state[:, 0, :])
        return logits
