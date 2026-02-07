import csv
import os

# Input and output file paths  
input_file = r"d:\Uni\GAS\data\bignet\legit.csv"
output_file = r"d:\Uni\GAS\data\bignet\vpn.csv"

# Number of records to extract
num_records = 2_200_000

print(f"Extracting up to {num_records:,} records from Time column...")
print(f"Input: {input_file}")
print(f"Output: {output_file}")

extracted_count = 0

with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    
    # Write header with column name
    writer.writerow(['IPDs'])
    
    for row in reader:
        if extracted_count >= num_records:
            break
        
        try:
            # Get the Time value and format with FULL precision
            # Using Python's repr() gives full float64 precision
            # This preserves scientific notation for very small numbers (e.g., 2.86102294921875e-06)
            time_value = float(row['Time'])
            # Format with 16 significant digits, use scientific notation when appropriate
            if time_value == 0:
                formatted_value = "0"
            elif abs(time_value) < 1e-4:
                # Use scientific notation for very small numbers
                formatted_value = f"{time_value:.16e}".rstrip('0').rstrip('.')
                # Clean up the exponent format (e.g., e-05 instead of e-005)
                if 'e' in formatted_value:
                    mantissa, exp = formatted_value.split('e')
                    exp_val = int(exp)
                    formatted_value = f"{mantissa}e{exp_val:+03d}".replace('+0', '-0').replace('e+', 'e-') if exp_val < 0 else f"{mantissa}e-{abs(exp_val):02d}" if exp_val < 0 else f"{mantissa}e+{exp_val:02d}"
                    # Simplify: just use standard format
                    formatted_value = f"{time_value:.16g}"
            else:
                # Use decimal notation for larger numbers
                formatted_value = f"{time_value:.16g}"
            
            writer.writerow([formatted_value])
            extracted_count += 1
            
            # Progress indicator every 500,000 records
            if extracted_count % 500_000 == 0:
                print(f"Processed {extracted_count:,} records...")
                
        except (ValueError, KeyError) as e:
            print(f"Warning: Skipping row due to error: {e}")
            continue

print(f"\nDone! Extracted {extracted_count:,} records to {output_file}")

# Show file size
file_size = os.path.getsize(output_file)
print(f"Output file size: {file_size / (1024*1024):.2f} MB")

# Show first 30 lines of output
print("\nFirst 30 lines of output:")
with open(output_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 30:
            break
        print(line.strip())
