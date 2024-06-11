input_file_path = "outputs/a_out.ldr"
output_file_path = "post_processed_gen.ldr"

# Function to clean a single LDR file by ensuring the first 50 lines have 15 entries
def clean_ldr_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        # Read the first 50 lines only
        lines = file.readlines()


    # Filter lines with exactly 15 space-separated entries
    clean_lines = [line for line in lines if len(line.strip().split()) == 15]
    original_line_count = len(clean_lines)
    print(f"Original line count: {original_line_count}")
    # Get the unique lines
    unique_lines = set(clean_lines)

    # Write the unique lines to the output file
    with open(output_file_path, 'w') as file:
        file.writelines(unique_lines)
    
    print(f"Found {len(unique_lines)} unique lines, outputted to {output_file_path}.")

clean_ldr_file(input_file_path, output_file_path)
