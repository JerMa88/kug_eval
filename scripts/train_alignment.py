import os
import json
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from kug_eval.data.dataset import get_dataloader
from kug_eval.models.hooks import RepresentationCache, register_hooks
from kug_eval.losses import rep_distill_loss, contrastive_loss, hybrid_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Alignment-Aware SFT model.")
    parser.add_argument("--model_id", type=str, default="sapientinc/HRM-Text-1B", help="Base model ID or path")
    parser.add_argument("--data_path", type=str, default="data/tasks/sota_generalization_benchmark.jsonl", help="JSONL dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--loss_variant", type=str, default="hybrid", choices=["ce_only", "rep_distill", "contrastive", "hybrid"], help="Loss variant")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alignment loss weight")
    parser.add_argument("--out_dir", type=str, default="outputs/train_run", help="Output directory")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading tokenizer & model {args.model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, local_files_only=True)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)
    model.to(device)

    l_t = 10
    l_s = 24
    cache = RepresentationCache()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = get_dataloader(args.data_path, tokenizer, batch_size=2, shuffle=True)

    metrics = []
    model.train()
    for epoch in range(args.epochs):
        epoch_loss, epoch_ce, epoch_align = 0.0, 0.0, 0.0
        for step, batch in enumerate(loader):
            optimizer.zero_grad()

            mem_ids = batch["mem_input_ids"].to(device)
            gen_ids = batch["gen_input_ids"].to(device)
            mem_span = batch["mem_span"].tolist()
            gen_span = batch["gen_span"].tolist()

            # Forward P_mem
            cache.clear()
            handles_mem = register_hooks(model, [l_s], cache, mem_span)
            with torch.no_grad():
                model(mem_ids)
            for h in handles_mem: h.remove()
            h_mem = cache[l_s]

            # Forward P_gen
            cache.clear()
            handles_gen = register_hooks(model, [l_t], cache, gen_span)
            labels = gen_ids.clone()
            for b_idx in range(len(gen_span)):
                s_start, s_end = int(gen_span[b_idx][0]), int(gen_span[b_idx][1])
                labels[b_idx, :s_start] = -100
                labels[b_idx, s_end:] = -100

            outputs = model(gen_ids, labels=labels)
            ce_loss = outputs.loss
            for h in handles_gen: h.remove()
            h_gen = cache[l_t]

            if args.loss_variant == "ce_only":
                align_l = torch.tensor(0.0, device=device)
            elif args.loss_variant == "rep_distill":
                align_l = rep_distill_loss(h_mem, h_gen)
            elif args.loss_variant == "contrastive":
                align_l = contrastive_loss(h_mem, h_gen)
            elif args.loss_variant == "hybrid":
                align_l = hybrid_loss(h_mem, h_gen, ce_loss, alpha=0.5)

            total_loss = ce_loss + args.alpha * align_l
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_ce += ce_loss.item()
            epoch_align += align_l.item()

            if step >= 10:
                break

        avg_loss = epoch_loss / (step + 1)
        logger.info(f"Epoch {epoch} | Total Loss: {avg_loss:.4f}")
        metrics.append({"epoch": epoch, "total_loss": avg_loss, "ce_loss": epoch_ce / (step + 1), "align_loss": epoch_align / (step + 1)})

    out_metrics = os.path.join(args.out_dir, "train_metrics.json")
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics to {out_metrics}")


if __name__ == "__main__":
    main()
