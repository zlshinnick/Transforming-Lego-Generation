from pathlib import Path
import re
import typing as T
from concurrent.futures import ProcessPoolExecutor
from loguru import logger
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    get_scheduler,
    EarlyStoppingCallback,
)
import typer
torch.cuda.empty_cache()

FN_P = "([-+]?(?:\d*\.*\d+))"
LDR_INSTRUCTION_REGEX_PATTERN = re.compile(
    rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)"
)

def load_all_ldrs(
    root_dir: Path,
    decimals: int = 2,
):
    """
    This reads all LDR files from the specified directory
    and rounds up all numeric entries to the specified number of decimals;
    the rounding part works well for synthetic data, use with care on
    real models.
    """
    
    src_files = sorted(root_dir.glob("*.ldr"))
    all_lines = []
    for src_file in src_files:
        # Skip meta data files
        if src_file.name.startswith('._'):
            print(f"Skipping macOS metadata file: {src_file.name}")
            continue

        print(f"processing {src_file.name}")
        file_lines = []
        for line in src_file.read_text(encoding="utf-8").splitlines():
            m = LDR_INSTRUCTION_REGEX_PATTERN.findall(line)
            if len(m) != 1:
                continue
            processed = []
            for numeric_entry in m[0][:-1]:
                if int(float(numeric_entry)) == float(numeric_entry):
                    processed.append(str(int(float(numeric_entry))))
                else:
                    processed.append(
                        str(np.round(float(numeric_entry), decimals=decimals))
                    )
            processed.append(m[0][-1])  # part ID
            file_lines.append(" ".join(processed))
        all_lines.append("\n".join(file_lines))
    return all_lines

class LDRTextDataset(Dataset):
    def __init__(
        self,
        lines,
        tokenizer,
        max_length: int 
    ):
        # Ensure that all examples are truncated to the max_length
        self.examples = tokenizer.batch_encode_plus(
            lines,
            max_length=max_length, 
            truncation=True,  
            padding='max_length', 
        ).input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_tokenizer(
    tokenizer_params_path: Path = Path("./tokenizer/"),
    default_tokenizer: str = "gpt2",
    vocab_size: int = 52000,
    corpus=None,
    model_max_length: int = 1024,
):
    if tokenizer_params_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_params_path)
        logger.info(f"Loaded tokenizer from {tokenizer_params_path}")
        return tokenizer
    logger.info(
        f"Could not load tokenizer from {tokenizer_params_path}. Training a new one"
    )
    old_tokenizer = AutoTokenizer.from_pretrained(
        default_tokenizer, model_max_length=model_max_length
    )
    tokenizer = old_tokenizer.train_new_from_iterator(
        corpus,
        vocab_size,
        new_special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
    )
    tokenizer.save_pretrained(tokenizer_params_path)
    return tokenizer


def main(
    ldr_root_dir: Path,
    output_dir: Path = Path("./final_normal_logs/"),
    checkpoint_dir: T.Optional[Path] = None,
    tokenizer_dir: Path = Path("/omr50-tokenizer"),
    n_positions: int = 1536,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 4,
    logging_steps: int = 1000,
    save_steps: int = 1000,
    eval_steps: int = 1000,
    fp16: bool = True,
    save_total_limit: int = 3,
    learning_rate: float = 5e-5,
    vocab_size: int = 10000,
):
    train_lines = load_all_ldrs(ldr_root_dir / "train")
    eval_lines = load_all_ldrs(ldr_root_dir / "test")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    train_dataset = LDRTextDataset(train_lines, tokenizer, max_length=n_positions)
    eval_dataset = LDRTextDataset(eval_lines, tokenizer, max_length=n_positions)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=n_positions,
        dropout=0.1 
    )
    
    if checkpoint_dir and checkpoint_dir.exists():
        model = AutoModelForCausalLM.load_pretrained(checkpoint_dir)
    else:
        model = AutoModelForCausalLM.from_config(config)
    logger.info(
        f"# trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        fp16=fp16,
        save_total_limit=save_total_limit,
        push_to_hub=False,
        learning_rate=learning_rate,
        weight_decay=0.01,  
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        logging_dir='./logs', 
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        lr_scheduler_type="cosine" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )

    trainer.train()
    # Save Final Model
    final_model_directory = Path(output_dir, "final_model")
    model.save_pretrained(final_model_directory)

if __name__ == "__main__":
    typer.run(main)
