from pathlib import Path
import typer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from scipy.linalg import logm
from scipy.spatial.transform import Rotation as R
import re
import itertools
from multiset import Multiset
import pandas as pd

FN_P = "([-+]?(?:\d*\.*\d+))"
LDR_INSTRUCTION_REGEX_PATTERN = re.compile(
    rf"(1)\s+(\d+)\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+{FN_P}\s+(.*)"
)

def parse_ldr_lines(lines, decimals=2):
    assembly = {
        'shape': [],
        'color': [],
        'position': [],
        'orientation': [],
        'pose': [],
        'edges': ([], [])
    }

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15 or parts[0] != '1':
            continue

        color = int(parts[1])
        shape_file = parts[-1]
        position = np.array(list(map(float, parts[2:5])))
        orientation_matrix = np.array(list(map(float, parts[5:14]))).reshape((3, 3))

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = orientation_matrix
        pose_matrix[:3, 3] = position

        assembly['color'].append(color)
        assembly['shape'].append(shape_file)
        assembly['position'].append(position)
        assembly['orientation'].append(orientation_matrix)
        assembly['pose'].append(pose_matrix)

    assembly['pose'] = np.array(assembly['pose'])
    return assembly

def round_line(line, decimals=2):
    m = LDR_INSTRUCTION_REGEX_PATTERN.findall(line)
    if len(m) != 1:
        return line
    processed = []
    for numeric_entry in m[0][:-1]:
        if int(float(numeric_entry)) == float(numeric_entry):
            processed.append(str(int(float(numeric_entry))))
        else:
            processed.append(str(np.round(float(numeric_entry), decimals=decimals)))
    processed.append(m[0][-1])  # part ID
    return " ".join(processed)

def position_accuracy(predicted, ground_truth):
    try:
        if len(predicted['position']) == 0 or len(ground_truth['position']) == 0:
            return None
        predicted_positions = np.array(predicted['position'])
        ground_truth_positions = np.array(ground_truth['position'])
        mse = np.mean((predicted_positions - ground_truth_positions) ** 2)
        return mse
    except:
        return None

def geodesic_distance(R1, R2):
    return np.linalg.norm(logm(np.dot(R1.T, R2)), 'fro')

def rotation_matrix_to_quaternion(matrix):
    rotation = R.from_matrix(matrix)
    return rotation.as_quat()

def quaternion_distance(Q1, Q2):
    dot_product = np.dot(Q1, Q2)
    return 1 - np.abs(dot_product)

def orientation_accuracy(predicted, ground_truth):
    try:
        if len(predicted['orientation']) == 0 or len(ground_truth['orientation']) == 0:
            return None
        predicted_orientations = np.array(predicted['orientation'])
        ground_truth_orientations = np.array(ground_truth['orientation'])
        distances = [
            geodesic_distance(pred, gt)
            for pred, gt in zip(predicted_orientations, ground_truth_orientations)
        ]
        avg_distance = np.mean(distances)
        return avg_distance
    except:
        return None

def quaternion_orientation_accuracy(predicted, ground_truth):
    try:
        if len(predicted['orientation']) == 0 or len(ground_truth['orientation']) == 0:
            return None
        predicted_orientations = np.array(predicted['orientation'])
        ground_truth_orientations = np.array(ground_truth['orientation'])
        distances = [
            quaternion_distance(
                rotation_matrix_to_quaternion(pred),
                rotation_matrix_to_quaternion(gt)
            )
            for pred, gt in zip(predicted_orientations, ground_truth_orientations)
        ]
        avg_distance = np.mean(distances)
        return avg_distance
    except:
        return None

def color_accuracy(predicted, ground_truth):
    try:
        if len(predicted['color']) == 0 or len(ground_truth['color']) == 0:
            return None
        predicted_colors = np.array(predicted['color'])
        ground_truth_colors = np.array(ground_truth['color'])
        correct_colors = np.sum(predicted_colors == ground_truth_colors)
        total_colors = len(ground_truth['color'])
        accuracy = correct_colors / total_colors
        return accuracy
    except:
        return None

def shape_accuracy(predicted, ground_truth):
    try:
        if len(predicted['shape']) == 0 or len(ground_truth['shape']) == 0:
            return None
        predicted_shapes = np.array(predicted['shape'])
        ground_truth_shapes = np.array(ground_truth['shape'])
        correct_shapes = np.sum(predicted_shapes == ground_truth_shapes)
        total_shapes = len(ground_truth['shape'])
        accuracy = correct_shapes / total_shapes
        return accuracy
    except:
        return None

def color_set_accuracy(predicted, ground_truth):
    try:
        predicted_colors = set(predicted['color'])
        ground_truth_colors = set(ground_truth['color'])
        correct_colors = len(predicted_colors & ground_truth_colors)
        total_colors = len(ground_truth['color'])
        accuracy = correct_colors / total_colors
        return accuracy
    except:
        return None

def shape_set_accuracy(predicted, ground_truth):
    try:
        predicted_shapes = set(predicted['shape'])
        ground_truth_shapes = set(ground_truth['shape'])
        correct_shapes = len(predicted_shapes & ground_truth_shapes)
        total_shapes = len(ground_truth['shape'])
        accuracy = correct_shapes / total_shapes
        return accuracy
    except:
        return None

def shape_color_set_accuracy(predicted, ground_truth):
    try:
        predicted_shape_color_pairs = set(zip(predicted['shape'], predicted['color']))
        ground_truth_shape_color_pairs = set(zip(ground_truth['shape'], ground_truth['color']))
        correct_pairs = len(predicted_shape_color_pairs & ground_truth_shape_color_pairs)
        total_pairs = len(ground_truth['shape'])
        accuracy = correct_pairs / total_pairs
        return accuracy
    except:
        return None

def f1b(predicted, ground_truth):
    try:
        predicted_bricks = Multiset(zip(predicted['shape'], predicted['color']))
        if (0, 0) in predicted_bricks:
            predicted_bricks.remove((0, 0))
        ground_truth_bricks = Multiset(zip(ground_truth['shape'], ground_truth['color']))
        if (0, 0) in ground_truth_bricks:
            ground_truth_bricks.remove((0, 0))
        tp = len(predicted_bricks & ground_truth_bricks)
        fp = len(predicted_bricks - ground_truth_bricks)
        fn = len(ground_truth_bricks - predicted_bricks)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1_score
    except:
        return None

def evaluate_file(file_path, tokenizer, model, device, repetition_penalty):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            all_lines = [round_line(line) for line in file.readlines()]

        if len(all_lines) < 12:
            print(f"The file {file_path} does not contain enough lines for the specified prompt and evaluation.")
            return None

        prompt_bricks = all_lines[7:27]
        eval_bricks = all_lines[27:32]
        prompt_text = "\n".join(prompt_bricks)

        print(f"File: {file_path}")
        print("Prompt Bricks:")
        for line in prompt_bricks:
            print(line)
        
        print("Eval Bricks:")
        for line in eval_bricks:
            print(line)

        prompt = tokenizer(prompt_text, return_tensors='pt')
        outputs = model.generate(
            prompt.input_ids.to(device),
            attention_mask=prompt.attention_mask.to(device),
            max_length=1516,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_lines = [round_line(line.strip()) for line in decoded_output.split("\n")]

        if len(generated_lines) < 25:
            print(f"The generated output from file {file_path} does not contain enough lines.")
            return None

        generated_bricks = generated_lines[20:25]

        print("Generated Bricks:")
        for line in generated_bricks:
            print(line)

        generated_assembly = parse_ldr_lines(generated_bricks)
        target_assembly = parse_ldr_lines(eval_bricks)

        position_acc = position_accuracy(generated_assembly, target_assembly)
        orientation_acc = orientation_accuracy(generated_assembly, target_assembly)
        quaternion_orientation_acc = quaternion_orientation_accuracy(generated_assembly, target_assembly)
        color_acc = color_accuracy(generated_assembly, target_assembly)
        shape_acc = shape_accuracy(generated_assembly, target_assembly)
        color_set_acc = color_set_accuracy(generated_assembly, target_assembly)
        shape_set_acc = shape_set_accuracy(generated_assembly, target_assembly)
        shape_color_set_acc = shape_color_set_accuracy(generated_assembly, target_assembly)
        f1b_score = f1b(generated_assembly, target_assembly)

        return {
            'position_acc': position_acc,
            'orientation_acc': orientation_acc,
            'quaternion_orientation_acc': quaternion_orientation_acc,
            'color_acc': color_acc,
            'shape_acc': shape_acc,
            'color_set_acc': color_set_acc,
            'shape_set_acc': shape_set_acc,
            'shape_color_set_acc': shape_color_set_acc,
            'f1b_score': f1b_score
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def main(checkpoint_dir: str, tokenizer_params_path: str, target_dir: str, csv_filename: str, repetition_penalty: float = 1.0):
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer_params_path = Path(tokenizer_params_path)
    target_dir = Path(target_dir)

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Target directory {target_dir} does not exist or is not a directory.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load custom tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_params_path.resolve())
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if not tokenizer.is_fast:
        raise ValueError("Tokenizer must be a fast tokenizer.")
    tokenizer.model_max_length = 1536

    # Load model
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir.resolve()).to(device)
    model.resize_token_embeddings(len(tokenizer))

    all_results = []
    for file_path in itertools.islice(target_dir.glob("*.ldr"), 100):
        result = evaluate_file(file_path, tokenizer, model, device, repetition_penalty)
        if result:
            all_results.append(result)

    if not all_results:
        print("No valid files found for evaluation.")
        return

    avg_position_acc = np.mean([r['position_acc'] for r in all_results if r['position_acc'] is not None])
    avg_orientation_acc = np.mean([r['orientation_acc'] for r in all_results if r['orientation_acc'] is not None])
    avg_quaternion_orientation_acc = np.mean([r['quaternion_orientation_acc'] for r in all_results if r['quaternion_orientation_acc'] is not None])
    avg_color_acc = np.mean([r['color_acc'] for r in all_results if r['color_acc'] is not None])
    avg_shape_acc = np.mean([r['shape_acc'] for r in all_results if r['shape_acc'] is not None])
    avg_color_set_acc = np.mean([r['color_set_acc'] for r in all_results if r['color_set_acc'] is not None])
    avg_shape_set_acc = np.mean([r['shape_set_acc'] for r in all_results if r['shape_set_acc'] is not None])
    avg_shape_color_set_acc = np.mean([r['shape_color_set_acc'] for r in all_results if r['shape_color_set_acc'] is not None])
    avg_f1b_score = np.mean([r['f1b_score'] for r in all_results if r['f1b_score'] is not None])

    print(f"Average Position Accuracy (MSE): {avg_position_acc:.4f}")
    print(f"Average Orientation Accuracy (Geodesic Distance): {avg_orientation_acc:.4f}")
    print(f"Average Quaternion Orientation Accuracy: {avg_quaternion_orientation_acc:.4f}")
    print(f"Average Color Accuracy: {avg_color_acc:.2f}")
    print(f"Average Shape Accuracy: {avg_shape_acc:.2f}")
    print(f"Average Color Set Accuracy: {avg_color_set_acc:.2f}")
    print(f"Average Shape Set Accuracy: {avg_shape_set_acc:.2f}")
    print(f"Average Shape-Color Set Accuracy: {avg_shape_color_set_acc:.2f}")
    print(f"Average F1B Score: {avg_f1b_score:.4f}")

    # Save average results to a CSV file
    averages = {
        'Metric': [
            'Position Accuracy (MSE)', 
            'Orientation Accuracy (Geodesic Distance)', 
            'Quaternion Orientation Accuracy', 
            'Color Accuracy', 
            'Shape Accuracy', 
            'Color Set Accuracy', 
            'Shape Set Accuracy', 
            'Shape-Color Set Accuracy', 
            'F1B Score'
        ],
        'Average': [
            avg_position_acc, 
            avg_orientation_acc, 
            avg_quaternion_orientation_acc, 
            avg_color_acc, 
            avg_shape_acc, 
            avg_color_set_acc, 
            avg_shape_set_acc, 
            avg_shape_color_set_acc, 
            avg_f1b_score
        ]
    }

    averages_df = pd.DataFrame(averages)
    averages_df.to_csv(csv_filename, index=False)
    print(f"Average results saved to {csv_filename}")

if __name__ == "__main__":
    typer.run(main)
