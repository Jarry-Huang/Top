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

def extract_data(file_name, search_pattern, directory='.'):
    """
    Extract data from specified file using grep and regex
    Parameters:
        file_name: Name of the file to extract data from
        search_pattern: Pattern to search for
        directory: Directory path where the file is located (default: current directory)
    """
    import os
    full_path = os.path.join(directory, file_name)
    try:
        # Use grep command to extract data (Linux/Unix)
        # 如果是在 Windows 執行且沒有 grep，這段會報錯，需改用純 Python 讀檔
        result = subprocess.run(['grep', search_pattern, full_path], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Warning: No data found for {search_pattern} in {full_path}")
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
        print(f"Error: File {full_path} not found.")
        return []
    except Exception as e:
        print(f"Error processing {full_path}: {e}")
        return []

def load_all_device_data(directory='.'):
    """
    Load all parameter data for all devices into a dictionary.
    Structure: data['PG']['idlin'] = [...]
    Parameters:
        directory: Directory path where the data files are located (default: current directory)
    """
    data = {}
    for device in DEVICES:
        data[device] = {}
        for param, file_template in FILE_TEMPLATES.items():
            file_name = file_template.format(device)
            pattern = SEARCH_PATTERNS[param]
            values = extract_data(file_name, pattern, directory)
            data[device][param] = np.array(values) # Convert to numpy array for easier math
            print(f"Loaded {len(values)} points for {device} - {param} from {directory}")
    return data

def load_all_corner_data(directory='.'):
    """
    Load corner data for all devices and parameters.
    Corner files follow naming: xxx.lis -> xxx_global_corner.lis
    Each corner file contains values corresponding to: TT, FF_G, SS_G, FNSP_G, SNFP_G
    Structure: corner_data['PG']['idlin'] = [values]
    Parameters:
        directory: Directory path where the data files are located (default: current directory)
    """
    corner_data = {}
    expected_corners = len(CORNER_NAMES)
    for device in DEVICES:
        corner_data[device] = {}
        for param, file_template in FILE_TEMPLATES.items():
            # Generate corner filename: xxx.lis -> xxx_global_corner.lis
            original_file = file_template.format(device)
            corner_file = original_file.replace('.lis', '_global_corner.lis')
            
            pattern = SEARCH_PATTERNS[param]
            values = extract_data(corner_file, pattern, directory)
            
            # Should have exactly expected_corners values (one for each corner)
            if len(values) == expected_corners:
                corner_data[device][param] = np.array(values)
                print(f"Loaded {len(values)} corner points for {device} - {param} from {directory}")
            else:
                corner_data[device][param] = np.array([])
                if len(values) > 0:
                    print(f"Warning: Expected {expected_corners} corner values but got {len(values)} for {device} - {param} from {directory}")
    return corner_data

def add_correlation_text(ax, x_data, y_data, label='', y_position=0.95):
    """
    Calculate Pearson correlation and add text to the plot
    Parameters:
        ax: matplotlib axes object
        x_data: x-axis data
        y_data: y-axis data
        label: Label for the dataset (e.g., 'TOP', 'PCA')
        y_position: Vertical position for text (0-1, default 0.95)
    """
    if len(x_data) > 1 and len(y_data) > 1:
        # Calculate Correlation
        corr_matrix = np.corrcoef(x_data, y_data)
        r_value = corr_matrix[0, 1]
        
        # Add text box
        text_str = f'{label}R = {r_value:.4f}\nN = {len(x_data)}'
        ax.text(0.05, y_position, text_str, 
                transform=ax.transAxes, 
                verticalalignment='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_scatter(ax, x, y, title, xlabel, ylabel, color='blue', corner_x=None, corner_y=None, corner_labels=None):
    """
    Helper function for consistent scatter plots (legacy single dataset support)
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
    
    # Plot corner points if provided and valid
    if (corner_x is not None and corner_y is not None and 
        len(corner_x) > 0 and len(corner_y) > 0):
        plot_corners(ax, corner_x, corner_y, corner_labels)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    add_correlation_text(ax, x, y)

def plot_scatter_comparison(ax, datasets, title, xlabel, ylabel):
    """
    Plot multiple datasets on the same axes for comparison
    Parameters:
        ax: matplotlib axes object
        datasets: list of dicts, each containing:
            - 'x': x-axis data
            - 'y': y-axis data
            - 'label': dataset label (e.g., 'TOP', 'PCA')
            - 'color': color for the dataset
            - 'corner_x': corner x-coordinates (optional)
            - 'corner_y': corner y-coordinates (optional)
            - 'corner_labels': corner labels (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    has_data = False
    
    for idx, dataset in enumerate(datasets):
        x = dataset.get('x', [])
        y = dataset.get('y', [])
        label = dataset.get('label', f'Dataset {idx+1}')
        color = dataset.get('color', 'blue')
        corner_x = dataset.get('corner_x', None)
        corner_y = dataset.get('corner_y', None)
        corner_labels = dataset.get('corner_labels', None)
        
        if len(x) == 0 or len(y) == 0:
            continue
        
        has_data = True
        
        # Ensure lengths match (truncate to minimum)
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        
        # Plot scatter points
        ax.scatter(x, y, alpha=0.5, s=20, c=color, edgecolors='black', 
                   linewidth=0.5, label=label)
        
        # Plot corner points if provided and valid
        if (corner_x is not None and corner_y is not None and 
            len(corner_x) > 0 and len(corner_y) > 0):
            plot_corners(ax, corner_x, corner_y, corner_labels, color=color, marker_size=80)
        
        # Add correlation text (stacked)
        y_position = 0.95 - (idx * 0.15)
        add_correlation_text(ax, x, y, label=f'{label}: ', y_position=y_position)
    
    if not has_data:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
        ax.set_title(f"{title} (No Data)")
        return
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    ax.legend(loc='upper right', fontsize=9)

def plot_corners(ax, corner_x, corner_y, corner_labels, color='red', marker_size=100):
    """
    Plot corner points on the scatter plot with labels.
    Parameters:
        ax: matplotlib axes object
        corner_x: array of x-coordinates for corner points
        corner_y: array of y-coordinates for corner points
        corner_labels: list of labels for corner points
        color: color for corner markers (default: 'red')
        marker_size: size of corner markers (default: 100)
    """
    # Ensure we have matching lengths
    num_corners = min(len(corner_x), len(corner_y))
    if num_corners == 0:
        return  # Nothing to plot
    
    if corner_labels is None:
        corner_labels = [f'C{i}' for i in range(num_corners)]
    else:
        corner_labels = corner_labels[:num_corners]
    
    corner_x = corner_x[:num_corners]
    corner_y = corner_y[:num_corners]
    
    # Determine edge color based on marker color
    edge_colors = {'red': 'darkred', 'blue': 'darkblue'}
    edge_color = edge_colors.get(color, 'black')
    
    # Plot corner points with distinctive marker
    ax.scatter(corner_x, corner_y, 
               s=marker_size, c=color, marker='D', 
               edgecolors=edge_color, linewidth=2, 
               alpha=0.8, zorder=5)
    
    # Add text labels for each corner
    for i, (cx, cy, label) in enumerate(zip(corner_x, corner_y, corner_labels)):
        # Position text slightly offset from the point
        ax.annotate(label, 
                   xy=(cx, cy), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=7, 
                   fontweight='bold',
                   color=edge_color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor=edge_color))

def main():
    plt.rcParams['font.size'] = 12
    
    # 1. Load Data from both directories
    print("Loading TOP data from sp_top directory...")
    all_data_top = load_all_device_data('sp_top')
    
    print("\nLoading PCA data from sp_pca directory...")
    all_data_pca = load_all_device_data('sp_pca')
    
    # 2. Load Corner Data from both directories
    print("\nLoading TOP corner data from sp_top directory...")
    corner_data_top = load_all_corner_data('sp_top')
    
    print("\nLoading PCA corner data from sp_pca directory...")
    corner_data_pca = load_all_corner_data('sp_pca')
    
    # --- Figure 1: Intrinsic Correlations (Same Device) ---
    # Row 1: Idlin vs Idsat (Original)
    # Row 2: Vtlin vs Idsat (Requirement #1)
    # Figure 1 uses only first 3 corners: TT, FF_G, SS_G
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
    fig1.suptitle('Figure 1: Intrinsic Device Correlations (TOP vs PCA)', fontsize=16)
    
    # Corner labels for Figure 1 (first 3 corners)
    fig1_corner_labels = CORNER_NAMES[:3]  # ['TT', 'FF_G', 'SS_G']
    
    for i, device in enumerate(DEVICES):
        # Row 1: Idlin vs Idsat
        datasets_idlin_idsat = [
            {
                'x': all_data_top[device].get('idlin', []),
                'y': all_data_top[device].get('idsat', []),
                'label': 'TOP',
                'color': 'blue',
                'corner_x': corner_data_top[device].get('idlin', np.array([]))[:3],
                'corner_y': corner_data_top[device].get('idsat', np.array([]))[:3],
                'corner_labels': fig1_corner_labels
            },
            {
                'x': all_data_pca[device].get('idlin', []),
                'y': all_data_pca[device].get('idsat', []),
                'label': 'PCA',
                'color': 'red',
                'corner_x': corner_data_pca[device].get('idlin', np.array([]))[:3],
                'corner_y': corner_data_pca[device].get('idsat', np.array([]))[:3],
                'corner_labels': fig1_corner_labels
            }
        ]
        
        plot_scatter_comparison(
            axes1[0, i],
            datasets_idlin_idsat,
            title=f'{device}: Idlin vs Idsat',
            xlabel='Idlin (A)',
            ylabel='Idsat (A)'
        )
        
        # Row 2: Vtlin vs Idsat (Requirement #1)
        # X-axis: Vtlin, Y-axis: Idsat (通常 Vt 是自變數)
        datasets_vtlin_idsat = [
            {
                'x': all_data_top[device].get('vtlin', []),
                'y': all_data_top[device].get('idsat', []),
                'label': 'TOP',
                'color': 'blue',
                'corner_x': corner_data_top[device].get('vtlin', np.array([]))[:3],
                'corner_y': corner_data_top[device].get('idsat', np.array([]))[:3],
                'corner_labels': fig1_corner_labels
            },
            {
                'x': all_data_pca[device].get('vtlin', []),
                'y': all_data_pca[device].get('idsat', []),
                'label': 'PCA',
                'color': 'red',
                'corner_x': corner_data_pca[device].get('vtlin', np.array([]))[:3],
                'corner_y': corner_data_pca[device].get('idsat', np.array([]))[:3],
                'corner_labels': fig1_corner_labels
            }
        ]
        
        plot_scatter_comparison(
            axes1[1, i],
            datasets_vtlin_idsat,
            title=f'{device}: Vtlin vs Idsat',
            xlabel='Vtlin (V)',
            ylabel='Idsat (A)'
        )
        
    plt.tight_layout()
    plt.savefig('monte_carlo_intrinsic.png', dpi=300, bbox_inches='tight')
    print("Saved 'monte_carlo_intrinsic.png'")

    # --- Figure 2: Cross-Device Correlations (Tracking) ---
    # Requirement #2: Vtlin (PL) vs Vtlin (PG/PD)
    # Requirement #3: Idsat (PL) vs Idsat (PG/PD)
    # Figure 2 uses all 5 corners: TT, FF_G, SS_G, FNSP_G, SNFP_G
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Figure 2: Cross-Device Tracking (Reference: PL) - TOP vs PCA', fontsize=16)
    
    # Corner labels for Figure 2 (all 5 corners)
    fig2_corner_labels = CORNER_NAMES  # ['TT', 'FF_G', 'SS_G', 'FNSP_G', 'SNFP_G']
    
    # Check if PL data exists as reference for both datasets
    if len(all_data_top['PL'].get('vtlin', [])) > 0 or len(all_data_pca['PL'].get('vtlin', [])) > 0:
        
        # Req #2: Vtlin Tracking
        # Plot 1: PL vs PG
        datasets_vtlin_pl_pg = [
            {
                'x': all_data_top['PL'].get('vtlin', []),
                'y': all_data_top['PG'].get('vtlin', []),
                'label': 'TOP',
                'color': 'blue',
                'corner_x': corner_data_top['PL'].get('vtlin', np.array([])),
                'corner_y': corner_data_top['PG'].get('vtlin', np.array([])),
                'corner_labels': fig2_corner_labels
            },
            {
                'x': all_data_pca['PL'].get('vtlin', []),
                'y': all_data_pca['PG'].get('vtlin', []),
                'label': 'PCA',
                'color': 'red',
                'corner_x': corner_data_pca['PL'].get('vtlin', np.array([])),
                'corner_y': corner_data_pca['PG'].get('vtlin', np.array([])),
                'corner_labels': fig2_corner_labels
            }
        ]
        
        plot_scatter_comparison(
            axes2[0, 0],
            datasets_vtlin_pl_pg,
            title='Vtlin Tracking: PL vs PG',
            xlabel='Vtlin PL (V)',
            ylabel='Vtlin PG (V)'
        )
        
        # Plot 2: PL vs PD
        datasets_vtlin_pl_pd = [
            {
                'x': all_data_top['PL'].get('vtlin', []),
                'y': all_data_top['PD'].get('vtlin', []),
                'label': 'TOP',
                'color': 'blue',
                'corner_x': corner_data_top['PL'].get('vtlin', np.array([])),
                'corner_y': corner_data_top['PD'].get('vtlin', np.array([])),
                'corner_labels': fig2_corner_labels
            },
            {
                'x': all_data_pca['PL'].get('vtlin', []),
                'y': all_data_pca['PD'].get('vtlin', []),
                'label': 'PCA',
                'color': 'red',
                'corner_x': corner_data_pca['PL'].get('vtlin', np.array([])),
                'corner_y': corner_data_pca['PD'].get('vtlin', np.array([])),
                'corner_labels': fig2_corner_labels
            }
        ]
        
        plot_scatter_comparison(
            axes2[0, 1],
            datasets_vtlin_pl_pd,
            title='Vtlin Tracking: PL vs PD',
            xlabel='Vtlin PL (V)',
            ylabel='Vtlin PD (V)'
        )
    else:
        print("Warning: Missing PL Vtlin data for cross-correlation")

    if len(all_data_top['PL'].get('idsat', [])) > 0 or len(all_data_pca['PL'].get('idsat', [])) > 0:
        # Req #3: Idsat Tracking
        # Plot 3: PL vs PG
        datasets_idsat_pl_pg = [
            {
                'x': all_data_top['PL'].get('idsat', []),
                'y': all_data_top['PG'].get('idsat', []),
                'label': 'TOP',
                'color': 'blue',
                'corner_x': corner_data_top['PL'].get('idsat', np.array([])),
                'corner_y': corner_data_top['PG'].get('idsat', np.array([])),
                'corner_labels': fig2_corner_labels
            },
            {
                'x': all_data_pca['PL'].get('idsat', []),
                'y': all_data_pca['PG'].get('idsat', []),
                'label': 'PCA',
                'color': 'red',
                'corner_x': corner_data_pca['PL'].get('idsat', np.array([])),
                'corner_y': corner_data_pca['PG'].get('idsat', np.array([])),
                'corner_labels': fig2_corner_labels
            }
        ]
        
        plot_scatter_comparison(
            axes2[1, 0],
            datasets_idsat_pl_pg,
            title='Idsat Tracking: PL vs PG',
            xlabel='Idsat PL (A)',
            ylabel='Idsat PG (A)'
        )
        
        # Plot 4: PL vs PD
        datasets_idsat_pl_pd = [
            {
                'x': all_data_top['PL'].get('idsat', []),
                'y': all_data_top['PD'].get('idsat', []),
                'label': 'TOP',
                'color': 'blue',
                'corner_x': corner_data_top['PL'].get('idsat', np.array([])),
                'corner_y': corner_data_top['PD'].get('idsat', np.array([])),
                'corner_labels': fig2_corner_labels
            },
            {
                'x': all_data_pca['PL'].get('idsat', []),
                'y': all_data_pca['PD'].get('idsat', []),
                'label': 'PCA',
                'color': 'red',
                'corner_x': corner_data_pca['PL'].get('idsat', np.array([])),
                'corner_y': corner_data_pca['PD'].get('idsat', np.array([])),
                'corner_labels': fig2_corner_labels
            }
        ]
        
        plot_scatter_comparison(
            axes2[1, 1],
            datasets_idsat_pl_pd,
            title='Idsat Tracking: PL vs PD',
            xlabel='Idsat PL (A)',
            ylabel='Idsat PD (A)'
        )
    else:
        print("Warning: Missing PL Idsat data for cross-correlation")

    plt.tight_layout()
    plt.savefig('monte_carlo_tracking.png', dpi=300, bbox_inches='tight')
    print("Saved 'monte_carlo_tracking.png'")
    
    plt.show()

if __name__ == "__main__":
    main()
