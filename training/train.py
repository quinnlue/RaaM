import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model import get_model
from training.data import CoTDataset, BucketBatchSampler
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")

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
        tokens_per_batch=16384,
        bucket_sizes=bucket_sizes,
        shuffle=True

    ),
    num_workers=0,
    pin_memory=True
)

print(f"Number of batches: {len(loader)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model, tokenizer, config = get_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

if __name__ == "__main__":
    for epoch in range(10):
        total_loss = 0.0

        for batch, mask in tqdm(loader, desc=f"Epoch {epoch+1}"):
            mask = mask.to(dtype=torch.bool)
            batch = batch.to(dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(input_ids=batch, attention_mask=mask, labels=batch)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()
            print(loss.item())

        # Print after epoch, not every batch
        accelerator.print(f"Epoch {epoch+1} - Avg Loss: {total_loss / len(loader):.4f}")
