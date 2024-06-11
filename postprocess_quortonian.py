import numpy as np
input_file_path = "outputs/a_out.ldr"
output_file_path = "post_processed_gen.ldr"

# Function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

# Function to clean a single LDR file by ensuring the first 50 lines have 10 entries
def clean_ldr_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        # Read the first 50 lines only
        lines = file.readlines()

    # Filter lines with exactly 10 space-separated entries
    clean_lines = [line for line in lines if len(line.strip().split()) == 10][:50]
    original_line_count = len(clean_lines)
    print(f"Original line count: {original_line_count}")
    # Get the unique lines
    unique_lines = set(clean_lines)
    final_lines = set()
    for line in unique_lines:
        parts = line.strip().split()
        quaternion_elements = list(map(float, parts[5:9]))
        matrix = quaternion_to_rotation_matrix(quaternion_elements)
        matrix_str = ' '.join(map(str, matrix.flatten()))
        new_line = ' '.join(parts[:5]) + ' ' + matrix_str + ' ' + ' '.join(parts[9:]) + '\n'
        final_lines.add(new_line)

    # Write the unique lines to the output file
    with open(output_file_path, 'w') as file:
        file.writelines(final_lines)
    
    print(f"Found {len(final_lines)} unique lines, outputted to {output_file_path}.")

clean_ldr_file(input_file_path, output_file_path)