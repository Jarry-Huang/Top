import re
import os

# Set file names
INPUT_FILE = "/nfs/site/disks/tadm_work_19/Jarry/West/Project/Study/00_TOP/00_Case1_P069/l14ffc_mm_svtp8_6tp069_v102.lib"
OUTPUT_FILE = "/nfs/site/disks/tadm_work_19/Jarry/West/Project/Study/00_TOP/00_Case1_P069/l14ffc_mm_svtp8_6tp069_v102_top.lib"
TARGET_LIB_NAME = "statistical_p_svtp8_6tp069"

def parse_spice_data(text):
    """
    Parse SPICE content to extract parameter values for corners like ff_g, ss_g.
    Use dictionary to preserve insertion order (Python 3.7+).
    """
    data = {}
    current_corner = None
    
    # Used to capture corner names
    lib_pattern = re.compile(r'\.lib\s+(\w+)')
    endl_pattern = re.compile(r'\.endl\s+(\w+)')
    # Used to capture parameter key=val
    param_pattern = re.compile(r'([\+\s]*)(\w+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

    lines = text.split('\n')
    
    for line in lines:
        line_strip = line.strip()
        if not line_strip or line_strip.startswith('*'):
            continue
            
        # Check for Corner start
        lib_match = lib_pattern.match(line_strip)
        if lib_match:
            raw_corner = lib_match.group(1)
            # Simple normalization of corner key
            if "ff_g" in raw_corner: current_corner = "ff_g"
            elif "ss_g" in raw_corner: current_corner = "ss_g"
            elif "fnsp_g" in raw_corner: current_corner = "fnsp_g"
            elif "snfp_g" in raw_corner: current_corner = "snfp_g"
            else: current_corner = raw_corner
            
            if current_corner not in data:
                data[current_corner] = {}
            continue

        # Check for Corner end
        if endl_pattern.match(line_strip):
            current_corner = None
            continue

        # Extract parameters
        if current_corner:
            matches = param_pattern.findall(line_strip)
            for _, key, val in matches:
                data[current_corner][key] = float(val)
                
    return data

def generate_equations(data):
    """
    Generate equation strings based on parsed data.
    Strictly generate in the order parameters appear in ff_g, without sorting.
    """
    # Define suffix mapping rules
    suffix_rules = [
        # (Input Suffix, Output Suffix, Alpha Variable)
        ("_n_pgp069_svt_g", "_n_pgp069_svt_g", "alpha_n_pg_svtp8_6tp069"),
        ("_n_pdp069_svt_g", "_n_pdp069_svt_g", "alpha_n_pd_svtp8_6tp069"),
        ("_p_lp069_svt_g",  "_p_lp069_svt_g",  "alpha_p_l_svtp8_6tp069")
    ]

    output_lines = []
    
    # 1. Write fixed Header
    output_lines.append("+alpha_n_pg_svtp8_6tp069 = '1.0/1.22'")
    output_lines.append("+alpha_n_pd_svtp8_6tp069 = '1.0/1.22'")
    output_lines.append("+alpha_p_l_svtp8_6tp069 = '1.0/1.22'")
    output_lines.append("+a1_np_svtp8_6tp069_xx =  agauss(0, sigma, 3)")
    output_lines.append("+a2_np_svtp8_6tp069_xx =  agauss(0, sigma, 3)")
    output_lines.append("+a1_np_svtp8_6tp069 =  a1_np_svtp8_6tp069_xx")
    output_lines.append("+a2_np_svtp8_6tp069 =  a2_np_svtp8_6tp069_xx")
    output_lines.append("+ffg_ssg_corner_flag = '(a1_np_svtp8_6tp069 > 0) ? 1 : 0'")
    output_lines.append("+fsg_sfg_corner_flag = '(a2_np_svtp8_6tp069 > 0) ? 1 : 0'")

    ref_corner = "ff_g"
    if ref_corner not in data:
        return "\n".join(output_lines) + "\n* Warning: No ff_g data found"

    # 2. Scan parameters according to original file order (Single Pass)
    # data[ref_corner] is a dict, Python 3.7+ guarantees iteration order equals insertion order
    for full_key in data[ref_corner]:
        
        # Check if this key matches any defined suffix rules
        matched_rule = None
        for in_suf, out_suf, alpha in suffix_rules:
            if full_key.endswith(in_suf):
                matched_rule = (in_suf, out_suf, alpha)
                break
        
        # If it doesn't match any rule, skip it (e.g., irrelevant parameters)
        if not matched_rule:
            continue

        in_suf, out_suf, alpha = matched_rule
        base_name = full_key[:-len(in_suf)]

        # Get value (default to 0.0 if missing)
        val_ff   = data.get("ff_g",   {}).get(full_key, 0.0)
        val_ss   = data.get("ss_g",   {}).get(full_key, 0.0)
        val_fnsp = data.get("fnsp_g", {}).get(full_key, 0.0)
        val_snfp = data.get("snfp_g", {}).get(full_key, 0.0)

        # Calculate coefficients
        t_ff   = val_ff / 3.0
        t_ss   = val_ss / 3.0
        t_fnsp = val_fnsp / 3.0
        t_snfp = val_snfp / 3.0

        # Generate equation
        # Use :g to format numerical values
        line = (
            f"+{base_name}{out_suf} = '{alpha}*("
            f"{t_ff:g}*a1_np_svtp8_6tp069*ffg_ssg_corner_flag+"
            f"{t_ss:g}*a1_np_svtp8_6tp069*(ffg_ssg_corner_flag-1)+"
            f"{t_fnsp:g}*a2_np_svtp8_6tp069*fsg_sfg_corner_flag+"
            f"{t_snfp:g}*a2_np_svtp8_6tp069*(fsg_sfg_corner_flag-1))'"
        )
        output_lines.append(line)

    return "\n".join(output_lines)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found")
        return

    print(f"Reading {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # 1. Parse parameters
    print("Parsing parameters...")
    data = parse_spice_data(original_content)
    
    # 2. Generate new equation block content
    print("Generating equations...")
    new_equations = generate_equations(data)
    
    # Assemble the complete .lib block string
    # Note: Added .lib / .endl wrappers here
    new_lib_block = (
        f".lib {TARGET_LIB_NAME}\n"
        f".param\n"
        f"{new_equations}\n"
        f".endl {TARGET_LIB_NAME}"
    )

    # 3. Replace or add to file content
    # Use Regex to find existing block
    # Pattern explanation:
    # \.lib\s+TARGET...  Find start
    # (?:.|\n)*?         Non-greedy match for all content in between
    # \.endl\s+TARGET... Find end
    pattern = re.compile(
        rf"(\.lib\s+{re.escape(TARGET_LIB_NAME)}\s*[\r\n]+(?:.|\n)*?\.endl\s+{re.escape(TARGET_LIB_NAME)})",
        re.IGNORECASE
    )

    match = pattern.search(original_content)
    
    if match:
        print(f"Found existing .lib {TARGET_LIB_NAME} block, replacing...")
        # Replace the found old block with the new block
        final_content = original_content.replace(match.group(1), new_lib_block)
    else:
        print(f"Did not find .lib {TARGET_LIB_NAME} block, appending to end of file...")
        # Ensure there is a newline before
        if not original_content.endswith('\n'):
            final_content = original_content + "\n\n" + new_lib_block
        else:
            final_content = original_content + "\n" + new_lib_block

    # 4. Write new file
    print(f"Writing {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
