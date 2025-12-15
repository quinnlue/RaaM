import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model import get_model
from training.data import CoTDataset, BucketBatchSampler
from training.utils import Metrics, LRScheduler
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=12
    )

dataset = CoTDataset(
    data_root="data/",
    batch_column="bucket",
    data_column="seq",
    ext=".parquet"
)


bucket_sizes = {0: 512, 1: 1024, 2: 2048, 3: 4096}

loader = DataLoader(
    dataset,
    batch_sampler=BucketBatchSampler(
        dataset,
        tokens_per_batch=8192,
        bucket_sizes=bucket_sizes,
        shuffle=True

    ),
    num_workers=0,
    pin_memory=True
)

print(f"Number of batches: {len(loader)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NUM_EPOCHS = 1
MAX_LR = 1e-4
MIN_LR = 1e-6
WARMUP_STEPS = 100

model, tokenizer, config = get_model()

total_steps = NUM_EPOCHS * len(loader)


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.00,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
)

model = get_peft_model(model, lora_config)

emb = model.get_input_embeddings()
for param in emb.parameters():
    param.requires_grad = True

model.config.use_cache = False
model.gradient_checkpointing_enable()
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)
scheduler = LRScheduler(
    optimizer,
    max_lr=MAX_LR,
    total_steps=total_steps,
    warmup_steps=WARMUP_STEPS,
    min_lr=MIN_LR
)
model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
model.print_trainable_parameters()



if __name__ == "__main__":
    metrics = Metrics(
        num_epochs=NUM_EPOCHS,
        dataloader_length=len(loader),
        log_frequency=100,
        training_log_path="training.log"
    )
    
    pbar = tqdm(total=total_steps, desc="Training")
    
    for epoch in range(NUM_EPOCHS):
        for batch, mask in loader:
            with accelerator.accumulate(model):
                mask = mask.to(dtype=torch.bool)
                batch = batch.to(dtype=torch.long)
                optimizer.zero_grad()
                outputs = model(input_ids=batch, attention_mask=mask, labels=batch)
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                metrics.update(loss.item(), optimizer, pbar, scheduler)
    
    pbar.close()
    accelerator.print(f"Training complete. Final step: {metrics.pbar_manager.current_step}")
