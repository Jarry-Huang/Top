import matplotlib.pyplot as plt
import subprocess
import re
import numpy as np

# Configuration
DEVICES = ['PG', 'PD', 'PL']
# 假設 Vtlin 的檔案命名與關鍵字邏輯與 Idlin/Idsat 相同，若不同請在此修改
FILE_TEMPLATES = {
    'idlin': 'idlin_monte_{}.lis',
    'idsat': 'idsat_monte_{}.lis',
    'vtlin': 'vthlin_monte_{}.lis'
}
SEARCH_PATTERNS = {
    'idlin': 'idlin_2d002_01',
    'idsat': 'idsat_2d002_01',
    'vtlin': 'vthlin_2d002_01'
}

def extract_data(file_name, search_pattern):
    """
    Extract data from specified file using grep and regex
    """
    try:
        # Use grep command to extract data (Linux/Unix)
        # 如果是在 Windows 執行且沒有 grep，這段會報錯，需改用純 Python 讀檔
        result = subprocess.run(['grep', search_pattern, file_name], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Warning: No data found for {search_pattern} in {file_name}")
            return []
        
        # Extract numerical values
        lines = result.stdout.strip().split('\n')
        values = []
        
        for line in lines:
            if line.strip():
                match = re.search(r'=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
                if match:
                    values.append(float(match.group(1)))
        return values
    
    except FileNotFoundError:
        print(f"Error: File {file_name} not found.")
        return []
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return []

def load_all_device_data():
    """
    Load all parameter data for all devices into a dictionary.
    Structure: data['PG']['idlin'] = [...]
    """
    data = {}
    for device in DEVICES:
        data[device] = {}
        for param, file_template in FILE_TEMPLATES.items():
            file_name = file_template.format(device)
            pattern = SEARCH_PATTERNS[param]
            values = extract_data(file_name, pattern)
            data[device][param] = np.array(values) # Convert to numpy array for easier math
            print(f"Loaded {len(values)} points for {device} - {param}")
    return data

def add_correlation_text(ax, x_data, y_data):
    """
    Calculate Pearson correlation and add text to the plot
    """
    if len(x_data) > 1 and len(y_data) > 1:
        # Calculate Correlation
        corr_matrix = np.corrcoef(x_data, y_data)
        r_value = corr_matrix[0, 1]
        
        # Add text box
        text_str = f'R = {r_value:.4f}\nN = {len(x_data)}'
        ax.text(0.05, 0.95, text_str, 
                transform=ax.transAxes, 
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_scatter(ax, x, y, title, xlabel, ylabel, color='blue'):
    """
    Helper function for consistent scatter plots
    """
    if len(x) == 0 or len(y) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
        ax.set_title(f"{title} (No Data)")
        return

    # Ensure lengths match (truncate to minimum)
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    ax.scatter(x, y, alpha=0.6, s=20, c=color, edgecolors='black', linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    add_correlation_text(ax, x, y)

def main():
    plt.rcParams['font.size'] = 12
    
    # 1. Load Data
    print("Loading data...")
    all_data = load_all_device_data()
    
    # --- Figure 1: Intrinsic Correlations (Same Device) ---
    # Row 1: Idlin vs Idsat (Original)
    # Row 2: Vtlin vs Idsat (Requirement #1)
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
    fig1.suptitle('Figure 1: Intrinsic Device Correlations', fontsize=16)
    
    colors = {'PG': 'blue', 'PD': 'red', 'PL': 'green'}
    
    for i, device in enumerate(DEVICES):
        # Row 1: Idlin vs Idsat
        plot_scatter(
            axes1[0, i], 
            all_data[device].get('idlin', []), 
            all_data[device].get('idsat', []),
            title=f'{device}: Idlin vs Idsat',
            xlabel='Idlin (A)',
            ylabel='Idsat (A)',
            color=colors[device]
        )
        
        # Row 2: Vtlin vs Idsat (Requirement #1)
        # X-axis: Vtlin, Y-axis: Idsat (通常 Vt 是自變數)
        plot_scatter(
            axes1[1, i], 
            all_data[device].get('vtlin', []), 
            all_data[device].get('idsat', []),
            title=f'{device}: Vtlin vs Idsat',
            xlabel='Vtlin (V)',
            ylabel='Idsat (A)',
            color=colors[device]
        )
        
    plt.tight_layout()
    plt.savefig('monte_carlo_intrinsic.png', dpi=300, bbox_inches='tight')
    print("Saved 'monte_carlo_intrinsic.png'")

    # --- Figure 2: Cross-Device Correlations (Tracking) ---
    # Requirement #2: Vtlin (PL) vs Vtlin (PG/PD)
    # Requirement #3: Idsat (PL) vs Idsat (PG/PD)
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Figure 2: Cross-Device Tracking (Reference: PL)', fontsize=16)
    
    # Check if PL data exists as reference
    if len(all_data['PL'].get('vtlin', [])) > 0:
        
        # Req #2: Vtlin Tracking
        # Plot 1: PL vs PG
        plot_scatter(
            axes2[0, 0],
            all_data['PL']['vtlin'],
            all_data['PG'].get('vtlin', []),
            title='Vtlin Tracking: PL vs PG',
            xlabel='Vtlin PL (V)',
            ylabel='Vtlin PG (V)',
            color='purple'
        )
        
        # Plot 2: PL vs PD
        plot_scatter(
            axes2[0, 1],
            all_data['PL']['vtlin'],
            all_data['PD'].get('vtlin', []),
            title='Vtlin Tracking: PL vs PD',
            xlabel='Vtlin PL (V)',
            ylabel='Vtlin PD (V)',
            color='purple'
        )
    else:
        print("Warning: Missing PL Vtlin data for cross-correlation")

    if len(all_data['PL'].get('idsat', [])) > 0:
        # Req #3: Idsat Tracking
        # Plot 3: PL vs PG
        plot_scatter(
            axes2[1, 0],
            all_data['PL']['idsat'],
            all_data['PG'].get('idsat', []),
            title='Idsat Tracking: PL vs PG',
            xlabel='Idsat PL (A)',
            ylabel='Idsat PG (A)',
            color='orange'
        )
        
        # Plot 4: PL vs PD
        plot_scatter(
            axes2[1, 1],
            all_data['PL']['idsat'],
            all_data['PD'].get('idsat', []),
            title='Idsat Tracking: PL vs PD',
            xlabel='Idsat PL (A)',
            ylabel='Idsat PD (A)',
            color='orange'
        )
    else:
        print("Warning: Missing PL Idsat data for cross-correlation")

    plt.tight_layout()
    plt.savefig('monte_carlo_tracking.png', dpi=300, bbox_inches='tight')
    print("Saved 'monte_carlo_tracking.png'")
    
    plt.show()

if __name__ == "__main__":
    main()
