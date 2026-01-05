import re
import sys
import os

def parse_spice_params(text):
    """
    Parses the SPICE-like text to extract parameter values for each corner.
    Returns a dictionary: {corner_name: {param_name: value}}
    
    Note: Dictionary keys in Python 3.7+ preserve insertion order.
    This ensures we can generate outputs in the same order they appeared in the input file.
    """
    data = {}
    current_corner = None
    
    # Regex patterns
    lib_pattern = re.compile(r'\.lib\s+(\w+)')
    endl_pattern = re.compile(r'\.endl\s+(\w+)')
    # Regex to capture key and value (handling scientific notation like 1.2e-5)
    param_pattern = re.compile(r'(\w+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 1. Detect start of a corner block
        lib_match = lib_pattern.match(line)
        if lib_match:
            raw_corner = lib_match.group(1)
            # Normalize corner names to standard keys
            if "ff_g" in raw_corner: current_corner = "ff_g"
            elif "ss_g" in raw_corner: current_corner = "ss_g"
            elif "fnsp_g" in raw_corner: current_corner = "fnsp_g"
            elif "snfp_g" in raw_corner: current_corner = "snfp_g"
            
            if current_corner not in data:
                data[current_corner] = {}
            continue

        # 2. Detect end of a corner block
        if endl_pattern.match(line):
            current_corner = None
            continue

        # 3. Extract parameters if inside a known corner
        if current_corner:
            matches = param_pattern.findall(line)
            for key, val in matches:
                data[current_corner][key] = float(val)
                
    return data

def generate_content(data):
    """
    Generates the equation block content based on parsed data.
    """
    # Define mapping groups (suffix definitions)
    groups = [
        {
            "alpha": "alpha_n_pg_svtp8_6tp069",
            "input_suffix": "_n_pgp086_svt_g",
            "output_suffix": "_n_pgp069_svt_g"
        },
        {
            "alpha": "alpha_n_pd_svtp8_6tp069",
            "input_suffix": "_n_pdp086_svt_g",
            "output_suffix": "_n_pdp069_svt_g"
        },
        {
            "alpha": "alpha_p_l_svtp8_6tp069",
            "input_suffix": "_p_lp086_svt_g",
            "output_suffix": "_p_lp069_svt_g"
        }
    ]

    output = []
    
    # 1. Add fixed headers
    headers = [
        "+alpha_n_pg_svtp8_6tp069 = 1.0",
        "+alpha_n_pd_svtp8_6tp069 = 1.0",
        "+alpha_p_l_svtp8_6tp069 = 1.0",
        "+a1_np_svtp8_6tp069_xx =  agauss(0, sigma, 3)",
        "+a2_np_svtp8_6tp069_xx =  agauss(0, sigma, 3)",
        "+a1_np_svtp8_6tp069 =  a1_np_svtp8_6tp069_xx",
        "+a2_np_svtp8_6tp069 =  a2_np_svtp8_6tp069_xx",
        "+ffg_ssg_corner_flag = '(a1_np_svtp8_6tp069 > 0) ? 1 : 0'",
        "+fsg_sfg_corner_flag = '(a2_np_svtp8_6tp069 > 0) ? 1 : 0'"
    ]
    output.extend(headers)

    # 2. Determine reference keys for ordering
    # We use 'ff_g' as the reference for parameter order.
    reference_corner = "ff_g"
    if reference_corner not in data and data:
        # Fallback to the first available corner if ff_g is missing
        reference_corner = next(iter(data)) 
        
    if not data or reference_corner not in data:
        return "* No valid corner data found to generate equations."

    # Get all keys in the order they appeared in the file
    reference_keys = list(data[reference_corner].keys())

    # 3. Generate equations for each group
    for group in groups:
        alpha = group["alpha"]
        in_suf = group["input_suffix"]
        out_suf = group["output_suffix"]
        
        # Filter keys that match this group's suffix
        # Iterating through reference_keys ensures we respect the ORIGINAL ORDER
        params_in_order = []
        for key in reference_keys:
            if key.endswith(in_suf):
                base_name = key[:-len(in_suf)]
                params_in_order.append(base_name)
        
        # Process parameters (No sorting applied)
        for param in params_in_order:
            param_full_name = param + in_suf
            
            try:
                # Extract values and apply the logic (divide by 3.0)
                val_ff   = data.get("ff_g",   {}).get(param_full_name, 0.0) / 3.0
                val_ss   = data.get("ss_g",   {}).get(param_full_name, 0.0) / 3.0
                val_fnsp = data.get("fnsp_g", {}).get(param_full_name, 0.0) / 3.0
                val_snfp = data.get("snfp_g", {}).get(param_full_name, 0.0) / 3.0
                
                # Format the equation string
                # using :g to format float compactly
                eq = (
                    f"+{param}{out_suf} = '{alpha}*("
                    f"{val_ff:g}*a1_np_svtp8_6tp069*ffg_ssg_corner_flag+"
                    f"{val_ss:g}*a1_np_svtp8_6tp069*(ffg_ssg_corner_flag-1)+"
                    f"{val_fnsp:g}*a2_np_svtp8_6tp069*fsg_sfg_corner_flag+"
                    f"{val_snfp:g}*a2_np_svtp8_6tp069*(fsg_sfg_corner_flag-1))'"
                )
                output.append(eq)
                
            except Exception as e:
                print(f"Warning: Could not process {param_full_name}: {e}")

    return "\n".join(output)

def process_file(input_filename, output_filename):
    print(f"Reading from: {input_filename}")
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"Error: File {input_filename} not found.")
        return

    # 1. Parse Data
    spice_data = parse_spice_params(raw_text)
    
    # 2. Generate the inner content (the params)
    generated_block = generate_content(spice_data)
    
    # 3. Construct the final file content with the requested wrapper
    final_content = (
        ".lib statistical_p_svtp8_6tp086\n"
        ".param\n"
        f"{generated_block}\n"
        ".endl statistical_p_svtp8_6tp086\n"
    )
    
    # 4. Write to output file
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_content)
        print(f"Successfully wrote to: {output_filename}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    # Define filenames
    in_file = "xxx.lib"
    out_file = "xxx_top.lib"
    
    # Check for command line arguments
    if len(sys.argv) >= 3:
        in_file = sys.argv[1]
        out_file = sys.argv[2]
        
    process_file(in_file, out_file)
