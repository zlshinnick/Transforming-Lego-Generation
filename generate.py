from pathlib import Path
import os
import torch
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
import typer
import re

quartonian = True

if quartonian:
    FN_P = r"([-+]?(?:\d*\.*\d+))"
    LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")
else:
    FN_P = r"([-+]?(?:\d*\.*\d+))"
    LDR_INSTRUCTION_REGEX_PATTERN = re.compile(rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)")

def main(
    checkpoint_dir: Path,
    tokenizer_params_path: Path,
    max_new_tokens: int = 1536,
    top_k: int = 51,
    top_p: float = 0.85,
    do_sample: bool = True,
    repetition_penalty=1.3,
    temp_file_path: Path = Path("outputs/a_out.ldr"),
    output_file_path: Path = Path("outputs/a_gen.ldr"),
    n_positions: int = 1536,
):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_params_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.model_max_length = n_positions
    print("Tokenizer Loaded")

    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir).eval()
    generation_config = GenerationConfig(
        max_length=model.config.n_positions,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    print("Model Loaded")

    prompt = torch.as_tensor([tokenizer.encode("1")]).to(device)
    out = model.generate(prompt, generation_config=generation_config)
    print("Output Generated")
    
    decoded = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    temp_file_path.write_text(decoded)
    with open(temp_file_path, 'r') as infile:
        lines = infile.readlines()

    filtered_lines = ["0 FILE LDR_TRANSFORMER_OUTPUT.ldr\n", "0 Main\n", "0 Name: TEST_OUTPUT_ASSEMBLY.ldr\n", "0 Author: WillZach_MODEL\n"]
    for line in lines:
        entries = line.split()
        if len(filtered_lines) == 11 and entries[0] == '1' and entries[14].endswith('.dat'):
            final_line = " ".join(entries[:15]) + "\n"
            filtered_lines.append(final_line)
            break
        elif len(entries) == 15 and entries[0] == '1' and entries[-1].endswith('.dat'):
            filtered_lines.append(line)

    try:
        with open(output_file_path, 'w') as outfile:
            outfile.writelines(filtered_lines)
        print(f"Output successfully saved to {output_file_path}")
    except Exception as e:
        print(f"Failed to save output to {output_file_path}: {e}")

if __name__ == "__main__":
    typer.run(main)
