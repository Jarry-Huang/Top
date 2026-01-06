import matplotlib.pyplot as plt
import subprocess
import re
import numpy as np

# Configuration
DEVICES = ['PG', 'PD', 'PL']
CORNER_NAMES = ['TT', 'FF_G', 'SS_G', 'FNSP_G', 'SNFP_G']
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

def load_all_corner_data():
    """
    Load corner data for all devices and parameters.
    Corner files follow naming: xxx.lis -> xxx_global_corner.lis
    Each corner file contains 5 values corresponding to: TT, FF_G, SS_G, FNSP_G, SNFP_G
    Structure: corner_data['PG']['idlin'] = [5 values]
    """
    corner_data = {}
    for device in DEVICES:
        corner_data[device] = {}
        for param, file_template in FILE_TEMPLATES.items():
            # Generate corner filename: xxx.lis -> xxx_global_corner.lis
            original_file = file_template.format(device)
            corner_file = original_file.replace('.lis', '_global_corner.lis')
            
            pattern = SEARCH_PATTERNS[param]
            values = extract_data(corner_file, pattern)
            
            # Should have exactly 5 values (one for each corner)
            if len(values) == 5:
                corner_data[device][param] = np.array(values)
                print(f"Loaded {len(values)} corner points for {device} - {param}")
            else:
                corner_data[device][param] = np.array([])
                if len(values) > 0:
                    print(f"Warning: Expected 5 corner values but got {len(values)} for {device} - {param}")
    return corner_data

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

def plot_scatter(ax, x, y, title, xlabel, ylabel, color='blue', corner_x=None, corner_y=None, corner_labels=None):
    """
    Helper function for consistent scatter plots
    Parameters:
        corner_x: array of x-coordinates for corner points (optional)
        corner_y: array of y-coordinates for corner points (optional)
        corner_labels: list of labels for corner points (optional)
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
    
    # Plot corner points if provided
    if corner_x is not None and corner_y is not None and len(corner_x) > 0 and len(corner_y) > 0:
        plot_corners(ax, corner_x, corner_y, corner_labels)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    add_correlation_text(ax, x, y)

def plot_corners(ax, corner_x, corner_y, corner_labels):
    """
    Plot corner points on the scatter plot with labels.
    Parameters:
        ax: matplotlib axes object
        corner_x: array of x-coordinates for corner points
        corner_y: array of y-coordinates for corner points
        corner_labels: list of labels for corner points
    """
    # Ensure we have matching lengths
    num_corners = min(len(corner_x), len(corner_y))
    if corner_labels is None:
        corner_labels = [f'C{i}' for i in range(num_corners)]
    else:
        corner_labels = corner_labels[:num_corners]
    
    corner_x = corner_x[:num_corners]
    corner_y = corner_y[:num_corners]
    
    # Plot corner points with distinctive marker
    ax.scatter(corner_x, corner_y, 
               s=100, c='red', marker='D', 
               edgecolors='darkred', linewidth=2, 
               alpha=0.8, zorder=5, label='Corners')
    
    # Add text labels for each corner
    for i, (cx, cy, label) in enumerate(zip(corner_x, corner_y, corner_labels)):
        # Position text slightly offset from the point
        ax.annotate(label, 
                   xy=(cx, cy), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=8, 
                   fontweight='bold',
                   color='darkred',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='darkred'))

def main():
    plt.rcParams['font.size'] = 12
    
    # 1. Load Data
    print("Loading data...")
    all_data = load_all_device_data()
    
    # 2. Load Corner Data
    print("\nLoading corner data...")
    corner_data = load_all_corner_data()
    
    # --- Figure 1: Intrinsic Correlations (Same Device) ---
    # Row 1: Idlin vs Idsat (Original)
    # Row 2: Vtlin vs Idsat (Requirement #1)
    # Figure 1 uses only first 3 corners: TT, FF_G, SS_G
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
    fig1.suptitle('Figure 1: Intrinsic Device Correlations', fontsize=16)
    
    colors = {'PG': 'blue', 'PD': 'red', 'PL': 'green'}
    
    # Corner labels for Figure 1 (first 3 corners)
    fig1_corner_labels = CORNER_NAMES[:3]  # ['TT', 'FF_G', 'SS_G']
    
    for i, device in enumerate(DEVICES):
        # Row 1: Idlin vs Idsat
        idlin_corners = corner_data[device].get('idlin', np.array([]))[:3]
        idsat_corners = corner_data[device].get('idsat', np.array([]))[:3]
        
        plot_scatter(
            axes1[0, i], 
            all_data[device].get('idlin', []), 
            all_data[device].get('idsat', []),
            title=f'{device}: Idlin vs Idsat',
            xlabel='Idlin (A)',
            ylabel='Idsat (A)',
            color=colors[device],
            corner_x=idlin_corners,
            corner_y=idsat_corners,
            corner_labels=fig1_corner_labels
        )
        
        # Row 2: Vtlin vs Idsat (Requirement #1)
        # X-axis: Vtlin, Y-axis: Idsat (通常 Vt 是自變數)
        vtlin_corners = corner_data[device].get('vtlin', np.array([]))[:3]
        
        plot_scatter(
            axes1[1, i], 
            all_data[device].get('vtlin', []), 
            all_data[device].get('idsat', []),
            title=f'{device}: Vtlin vs Idsat',
            xlabel='Vtlin (V)',
            ylabel='Idsat (A)',
            color=colors[device],
            corner_x=vtlin_corners,
            corner_y=idsat_corners,
            corner_labels=fig1_corner_labels
        )
        
    plt.tight_layout()
    plt.savefig('monte_carlo_intrinsic.png', dpi=300, bbox_inches='tight')
    print("Saved 'monte_carlo_intrinsic.png'")

    # --- Figure 2: Cross-Device Correlations (Tracking) ---
    # Requirement #2: Vtlin (PL) vs Vtlin (PG/PD)
    # Requirement #3: Idsat (PL) vs Idsat (PG/PD)
    # Figure 2 uses all 5 corners: TT, FF_G, SS_G, FNSP_G, SNFP_G
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Figure 2: Cross-Device Tracking (Reference: PL)', fontsize=16)
    
    # Corner labels for Figure 2 (all 5 corners)
    fig2_corner_labels = CORNER_NAMES  # ['TT', 'FF_G', 'SS_G', 'FNSP_G', 'SNFP_G']
    
    # Check if PL data exists as reference
    if len(all_data['PL'].get('vtlin', [])) > 0:
        
        # Req #2: Vtlin Tracking
        # Plot 1: PL vs PG
        vtlin_pl_corners = corner_data['PL'].get('vtlin', np.array([]))
        vtlin_pg_corners = corner_data['PG'].get('vtlin', np.array([]))
        
        plot_scatter(
            axes2[0, 0],
            all_data['PL']['vtlin'],
            all_data['PG'].get('vtlin', []),
            title='Vtlin Tracking: PL vs PG',
            xlabel='Vtlin PL (V)',
            ylabel='Vtlin PG (V)',
            color='purple',
            corner_x=vtlin_pl_corners,
            corner_y=vtlin_pg_corners,
            corner_labels=fig2_corner_labels
        )
        
        # Plot 2: PL vs PD
        vtlin_pd_corners = corner_data['PD'].get('vtlin', np.array([]))
        
        plot_scatter(
            axes2[0, 1],
            all_data['PL']['vtlin'],
            all_data['PD'].get('vtlin', []),
            title='Vtlin Tracking: PL vs PD',
            xlabel='Vtlin PL (V)',
            ylabel='Vtlin PD (V)',
            color='purple',
            corner_x=vtlin_pl_corners,
            corner_y=vtlin_pd_corners,
            corner_labels=fig2_corner_labels
        )
    else:
        print("Warning: Missing PL Vtlin data for cross-correlation")

    if len(all_data['PL'].get('idsat', [])) > 0:
        # Req #3: Idsat Tracking
        # Plot 3: PL vs PG
        idsat_pl_corners = corner_data['PL'].get('idsat', np.array([]))
        idsat_pg_corners = corner_data['PG'].get('idsat', np.array([]))
        
        plot_scatter(
            axes2[1, 0],
            all_data['PL']['idsat'],
            all_data['PG'].get('idsat', []),
            title='Idsat Tracking: PL vs PG',
            xlabel='Idsat PL (A)',
            ylabel='Idsat PG (A)',
            color='orange',
            corner_x=idsat_pl_corners,
            corner_y=idsat_pg_corners,
            corner_labels=fig2_corner_labels
        )
        
        # Plot 4: PL vs PD
        idsat_pd_corners = corner_data['PD'].get('idsat', np.array([]))
        
        plot_scatter(
            axes2[1, 1],
            all_data['PL']['idsat'],
            all_data['PD'].get('idsat', []),
            title='Idsat Tracking: PL vs PD',
            xlabel='Idsat PL (A)',
            ylabel='Idsat PD (A)',
            color='orange',
            corner_x=idsat_pl_corners,
            corner_y=idsat_pd_corners,
            corner_labels=fig2_corner_labels
        )
    else:
        print("Warning: Missing PL Idsat data for cross-correlation")

    plt.tight_layout()
    plt.savefig('monte_carlo_tracking.png', dpi=300, bbox_inches='tight')
    print("Saved 'monte_carlo_tracking.png'")
    
    plt.show()

if __name__ == "__main__":
    main()
