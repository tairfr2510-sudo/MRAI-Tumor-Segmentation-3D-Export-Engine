import os
import time
import datetime
import threading
import traceback

import numpy as np
import nibabel as nib
import pydicom
import stl.mesh as stl_mesh
from scipy.ndimage import gaussian_filter, binary_closing
from skimage import measure
from skimage.filters import unsharp_mask
from fpdf import FPDF

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches

# Third-party AI/DSP integration
from fastai.callback.hook import hook_output
import ai_engine 
import DSP_Engine 

# ======================
# GLOBAL CONFIG & THEME
# ======================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['toolbar'] = 'None'

THEME = {
    'bg': '#121212',
    'panel_bg': '#1E1E1E',
    'fg': '#E0E0E0',
    'accent': '#00E5FF',
    'button_bg': '#333333',
    'button_hover': '#4D4D4D',
    'slider_track': '#404040',
    'success': '#00C853',
}

voxel_size_mm = 1.0  
diagnosis_info = 0.0
clinical_volume_cm3 = 0.0


# ======================
# DATA LOADING MODULE
# ======================
def select_mri_data():
    """
    Creates a styled file selection dialog allowing the user to load 
    either DICOM series directories or NIfTI files.
    """
    selected_data = {"value": None}
    
    selector = tk.Tk()
    selector.title("MRI Data Loader")
    selector.geometry("350x200")
    selector.configure(bg=THEME['panel_bg'])
    
    # Center window on screen
    screen_width = selector.winfo_screenwidth()
    screen_height = selector.winfo_screenheight()
    x = (screen_width // 2) - (350 // 2)
    y = (screen_height // 2) - (200 // 2)
    selector.geometry(f'+{x}+{y}')

    label = tk.Label(selector, text="Select Input Format:", 
                     bg=THEME['panel_bg'], fg=THEME['accent'], 
                     font=("Segoe UI", 12, "bold"))
    label.pack(pady=20)

    def load_dicom():
        folder = filedialog.askdirectory(title="Select DICOM Folder")
        if not folder:
            selector.destroy()
            return

        print(f"Scanning directory: {folder}")
        series_dict = {}
        
        # Group by series UID
        for f in os.listdir(folder):
            file_path = os.path.join(folder, f)
            if os.path.isfile(file_path):
                try:
                    ds = pydicom.dcmread(file_path, force=True)
                    if hasattr(ds, 'PixelData'):
                        uid = getattr(ds, 'SeriesInstanceUID', 'unknown')
                        if uid not in series_dict: 
                            series_dict[uid] = []
                        series_dict[uid].append(ds)
                except Exception:
                    continue
        
        if not series_dict:
            print("Error: No valid DICOM files found.")
            selector.destroy()
            return

        largest_series = max(series_dict.values(), key=len)
        print(f"Loading main series with {len(largest_series)} slices...")

        # Advanced spatial sorting
        try:
            iop = largest_series[0].ImageOrientationPatient
            row_cosine = np.array(iop[:3])
            col_cosine = np.array(iop[3:])
            slice_normal = np.cross(row_cosine, col_cosine)
            largest_series.sort(key=lambda ds: np.dot(slice_normal, np.array(ds.ImagePositionPatient)))
        except AttributeError:
            largest_series.sort(key=lambda ds: int(getattr(ds, 'InstanceNumber', 0)))

        # Extraction and calibration
        pixel_data = []
        for ds in largest_series:
            img = ds.pixel_array.astype(float)
            img = (img * getattr(ds, 'RescaleSlope', 1.0)) + getattr(ds, 'RescaleIntercept', 0.0)
            pixel_data.append(img)
        
        pixel_data = np.stack(pixel_data)
        
        # Safe Aspect Ratio Correction
        try:
            pixel_spacing = getattr(largest_series[0], 'PixelSpacing', [1.0, 1.0])
            try:
                spacing_y = float(pixel_spacing[0])
                spacing_x = float(pixel_spacing[1])
            except Exception:
                spacing_y, spacing_x = 1.0, 1.0

            global voxel_size_mm
            voxel_size_mm = spacing_x 
                
            slice_thickness = getattr(largest_series[0], 'SpacingBetweenSlices', 
                                      getattr(largest_series[0], 'SliceThickness', 1.0))
            try:
                spacing_z = float(slice_thickness)
            except Exception:
                spacing_z = 1.0

            zoom_z = spacing_z / spacing_x
            zoom_y = spacing_y / spacing_x
            zoom_x = 1.0

            if abs(zoom_z - 1.0) > 0.05 or abs(zoom_y - 1.0) > 0.05:
                from scipy.ndimage import zoom
                pixel_data = zoom(pixel_data, (zoom_z, zoom_y, zoom_x), order=1)
        except Exception as e:
            print(f"Warning: Could not correct physical proportions: {e}")

        # Axis alignment
        try:
            iop = np.round(largest_series[0].ImageOrientationPatient, 4)
            row_cosine = np.array(iop[:3])
            col_cosine = np.array(iop[3:])
            slice_normal = np.cross(row_cosine, col_cosine)
            dominant_axis = np.argmax(np.abs(slice_normal))
            
            if dominant_axis == 2:
                vol = np.transpose(pixel_data, (2, 1, 0))
            elif dominant_axis == 1:
                vol = np.transpose(pixel_data, (2, 0, 1)) 
            elif dominant_axis == 0:
                vol = np.transpose(pixel_data, (0, 2, 1))
            else:
                vol = np.transpose(pixel_data, (2, 1, 0))
        except Exception:
            vol = np.transpose(pixel_data, (2, 1, 0))
            
        # Invert Y-axis for correct anatomical orientation
        vol = np.flip(vol, axis=1) 

        # Intensity normalization
        v_min, v_max = vol.min(), vol.max()
        if v_max > v_min:
            vol = (vol - v_min) / (v_max - v_min)
        else:
            vol = np.zeros_like(vol)
            
        selected_data["value"] = vol
        print(f"Volume loaded successfully. Dimensions: {vol.shape}")
        selector.destroy()

    def load_nifti():
        file_path = filedialog.askopenfilename(title="Select NIfTI File", 
                                              filetypes=(("NIfTI", "*.nii *.nii.gz"), ("All files", "*.*")))
        if file_path:
            nii = nib.load(file_path)
            selected_data["value"] = nii.get_fdata()
            
            # Extract physical dimensions from NIfTI header
            global voxel_size_mm
            try:
                zooms = nii.header.get_zooms() 
                voxel_size_mm = float(zooms[0]) 
            except Exception as e:
                print(f"Warning: Could not read NIfTI spacing: {e}")
                
        selector.destroy()

    style = ttk.Style()
    btn_dicom = tk.Button(selector, text="DICOM FOLDER", command=load_dicom,
                          bg=THEME['button_bg'], fg=THEME['fg'], width=20, relief='flat')
    btn_dicom.pack(pady=10)

    btn_nifti = tk.Button(selector, text="NIfTI FILE (.nii / .gz)", command=load_nifti,
                          bg=THEME['button_bg'], fg=THEME['fg'], width=20, relief='flat')
    btn_nifti.pack(pady=5)

    selector.mainloop()
    return selected_data["value"]

# Execute data loader
raw_data = select_mri_data()

if raw_data is not None:
    # Normalize entire volume for the AI model pipeline
    data = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min())
else:
    print("Warning: No data selected. Generating noise fallback.")
    data = np.random.rand(100, 100, 100)

sx, sy, sz = data.shape

# ======================
# BOOT ANIMATION
# ======================
def show_splash_screen():
    """Displays a stylized loading screen while initializing core components."""
    splash_bg = '#121212'
    accent_color = '#00E5FF'
    text_color = '#E0E0E0'

    splash = tk.Tk()
    splash.overrideredirect(True) 

    width, height = 500, 250
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    splash.geometry(f'{width}x{height}+{x}+{y}')
    splash.configure(bg=splash_bg)

    lbl_title = tk.Label(splash, text="BIOMED MRI VIEWER", font=("Segoe UI", 20, "bold"),
                         bg=splash_bg, fg=text_color)
    lbl_title.pack(pady=(50, 10))

    lbl_subtitle = tk.Label(splash, text="Loading Neuro-Imaging Modules...", font=("Consolas", 10),
                            bg=splash_bg, fg=accent_color)
    lbl_subtitle.pack(pady=5)

    style = ttk.Style()
    style.theme_use('default')
    style.configure("Cyan.Horizontal.TProgressbar", background=accent_color, troughcolor='#333333', thickness=6)

    progress = ttk.Progressbar(splash, style="Cyan.Horizontal.TProgressbar", orient="horizontal", length=400,
                               mode="determinate")
    progress.pack(pady=30)

    loading_steps = ["Initializing GUI...", "Loading 3D Engine...", "Calibrating Sensors...", "Ready."]
    for i in range(1, 101):
        progress['value'] = i
        if i % 25 == 0:
            lbl_subtitle.config(text=loading_steps[i // 25 - 1])
        splash.update_idletasks()
        splash.update()
        time.sleep(0.015) 

    time.sleep(0.5)
    splash.destroy() 

show_splash_screen()

# ======================
# CORE STATE VARIABLES
# ======================
ax_idx, cor_idx, sag_idx = sz // 2, sy // 2, sx // 2
rot = {"axial": 3, "coronal": 3, "sagittal": 3}
initial_state = {
    'rot': {"axial": 3, "coronal": 3, "sagittal": 3},
    'mask_min': 0.25,
    'mask_max': 0.60
}

mask_on = False
mask_min_val = initial_state['mask_min']
mask_max_val = initial_state['mask_max']

cmap_name = 'gray'
sharpen_on = False
sharpen_radius = 1.0
sharpen_amount = 1.0

# ======================
# IMAGE LOGIC FUNCTIONS
# ======================
def transform_coords(r, c, h, w, k):
    """Calculates crosshair coordinates relative to slice rotation."""
    k = k % 4
    if k == 0: return r, c
    elif k == 1: return w - 1 - c, r
    elif k == 2: return h - 1 - r, w - 1 - c
    elif k == 3: return c, h - 1 - r

def get_slice_dims(axis_name):
    if axis_name == 'axial': return sx, sy
    elif axis_name == 'coronal': return sx, sz
    elif axis_name == 'sagittal': return sy, sz

def axial_slice(z): return data[:, :, z]
def coronal_slice(y): return data[:, y, :]
def sagittal_slice(x): return data[x, :, :]
def rot90(img, k): return np.rot90(img, k=k)

def apply_sharpen(slice_img):
    if sharpen_on:
        sharpened = unsharp_mask(slice_img, radius=sharpen_radius, amount=sharpen_amount)
        return np.clip(sharpened, 0, 1)
    return slice_img

def get_range_mask(slice_data):
    return ((slice_data >= mask_min_val) & (slice_data <= mask_max_val)).astype(float)


# ======================
# GUI LAYOUT SETUP
# ======================
fig = plt.figure(figsize=(16, 9), facecolor=THEME['bg'])

gs = fig.add_gridspec(2, 2, width_ratios=[0.14, 0.86], height_ratios=[0.80, 0.20],
                      left=0.02, right=0.98, bottom=0.02, top=0.92, wspace=0.05, hspace=0.1)

# Images Area
gs_images = gs[0, 1].subgridspec(1, 3, wspace=0.15)
axA = fig.add_subplot(gs_images[0])
axC = fig.add_subplot(gs_images[1])
axS = fig.add_subplot(gs_images[2])
axes = [axA, axC, axS]

imA = axA.imshow(rot90(axial_slice(ax_idx), rot["axial"]), cmap=cmap_name, origin='lower', aspect='equal')
imC = axC.imshow(rot90(coronal_slice(cor_idx), rot["coronal"]), cmap=cmap_name, origin='lower', aspect='equal')
imS = axS.imshow(rot90(sagittal_slice(sag_idx), rot["sagittal"]), cmap=cmap_name, origin='lower', aspect='equal')

for ax, title in zip(axes, ['AXIAL', 'CORONAL', 'SAGITTAL']):
    ax.axis('off')
    ax.set_title(title, color=THEME['accent'], fontsize=14, fontweight='bold', pad=10)

# Sidebar Area
ax_sidebar = fig.add_subplot(gs[0, 0])
ax_sidebar.set_facecolor(THEME['bg'])
ax_sidebar.axis('off')

# ======================
# GRAPHICS OVERLAYS
# ======================
mask_collections = {ax: None for ax in axes}
cross_lines = {ax: [] for ax in axes}
colors = ['#00E5FF', '#FF4081', '#76FF03'] 

def clear_masks():
    for ax in axes:
        if mask_collections[ax] is not None:
            try:
                mask_collections[ax].remove()
            except:
                for c in mask_collections[ax].collections: c.remove()
            mask_collections[ax] = None

def update_masks():
    clear_masks()
    m_a = rot90(get_range_mask(axial_slice(ax_idx)), rot['axial'])
    m_c = rot90(get_range_mask(coronal_slice(cor_idx)), rot['coronal'])
    m_s = rot90(get_range_mask(sagittal_slice(sag_idx)), rot['sagittal'])

    props = dict(colors='#FF5722', linewidths=1.5)
    if m_a.max() > 0: mask_collections[axA] = axA.contour(m_a, levels=[0.5], **props)
    if m_c.max() > 0: mask_collections[axC] = axC.contour(m_c, levels=[0.5], **props)
    if m_s.max() > 0: mask_collections[axS] = axS.contour(m_s, levels=[0.5], **props)

def draw_cross(ax, r, c, col_h, col_v):
    for line in cross_lines[ax]: line.remove()
    cross_lines[ax] = []
    l1 = ax.axhline(r, color=col_h, lw=1, alpha=0.8)
    l2 = ax.axvline(c, color=col_v, lw=1, alpha=0.8)
    cross_lines[ax].extend([l1, l2])

def update_ai_mask():
    """
    Dynamically renders the AI-generated tumor contours.
    Slices the 3D segmentation volume in real-time based on the user's current Z-axis depth.
    """
    global ai_contour_coll
    
    # Aggressive cleanup of previous matplotlib collections to prevent memory leaks
    if ai_contour_coll is not None:
        try:
            for c in ai_contour_coll.collections: 
                c.remove()
        except AttributeError:
            try: ai_contour_coll.remove()
            except: pass
        except Exception:
            pass
        ai_contour_coll = None

    # Volumetric slicing and rendering
    if show_ai_mask and ai_tumor_mask is not None:
        # Extract the specific 2D cross-section corresponding to the current slider position
        current_ai_slice = ai_tumor_mask[:, :, ax_idx]
        
        # Render the contour boundary only if tumor pixels exist in the current slice
        if current_ai_slice.max() > 0:
            ai_contour_coll = axA.contour(current_ai_slice, levels=[0.5], colors='#FF4081', linewidths=1.5, linestyles='dashed')


def update_all_crosshairs():
    h, w = get_slice_dims('axial')
    r_new, c_new = transform_coords(sag_idx, cor_idx, h, w, rot['axial'])
    draw_cross(axA, r_new, c_new, colors[0], colors[1])

    h, w = get_slice_dims('coronal')
    r_new, c_new = transform_coords(sag_idx, ax_idx, h, w, rot['coronal'])
    draw_cross(axC, r_new, c_new, colors[0], colors[2])

    h, w = get_slice_dims('sagittal')
    r_new, c_new = transform_coords(cor_idx, ax_idx, h, w, rot['sagittal'])
    draw_cross(axS, r_new, c_new, colors[1], colors[2])

def zoom_all_to_cross(factor):
    for ax in axes:
        if ax is axA:
            h, w = get_slice_dims('axial'); r_c, c_c = transform_coords(sag_idx, cor_idx, h, w, rot['axial'])
        elif ax is axC:
            h, w = get_slice_dims('coronal'); r_c, c_c = transform_coords(sag_idx, ax_idx, h, w, rot['coronal'])
        elif ax is axS:
            h, w = get_slice_dims('sagittal'); r_c, c_c = transform_coords(cor_idx, ax_idx, h, w, rot['sagittal'])
        else:
            continue

        x1, x2 = ax.get_xlim(); y1, y2 = ax.get_ylim()
        w_new = (x2 - x1) / factor; h_new = (y2 - y1) / factor
        ax.set_xlim(c_c - w_new / 2, c_c + w_new / 2)
        ax.set_ylim(r_c - h_new / 2, r_c + h_new / 2)
    fig.canvas.draw_idle()

# ======================
# MASTER RENDER PIPELINE
# ======================
def refresh(reset_view=False):
    saved_lims = {}
    if not reset_view:
        for ax in axes: saved_lims[ax] = (ax.get_xlim(), ax.get_ylim())

    sl_a = apply_sharpen(axial_slice(ax_idx))
    sl_c = apply_sharpen(coronal_slice(cor_idx))
    sl_s = apply_sharpen(sagittal_slice(sag_idx))

    img_a = rot90(sl_a, rot['axial'])
    img_c = rot90(sl_c, rot['coronal'])
    img_s = rot90(sl_s, rot['sagittal'])

    imA.set_data(img_a)
    imC.set_data(img_c)
    imS.set_data(img_s)

    # Extents updating (Critical for precise mapping)
    rows_A, cols_A = img_a.shape; imA.set_extent([-0.5, cols_A - 0.5, -0.5, rows_A - 0.5])
    rows_C, cols_C = img_c.shape; imC.set_extent([-0.5, cols_C - 0.5, -0.5, rows_C - 0.5])
    rows_S, cols_S = img_s.shape; imS.set_extent([-0.5, cols_S - 0.5, -0.5, rows_S - 0.5])

    img_dims = {axA: (rows_A, cols_A), axC: (rows_C, cols_C), axS: (rows_S, cols_S)}
    for ax in axes:
        rows, cols = img_dims[ax]
        if reset_view:
            ax.set_xlim(-0.5, cols - 0.5)
            ax.set_ylim(-0.5, rows - 0.5)
        elif ax in saved_lims:
            ax.set_xlim(saved_lims[ax][0])
            ax.set_ylim(saved_lims[ax][1])

    if mask_on: update_masks()
    else: clear_masks()

    update_all_crosshairs()
    update_ai_mask() 
    update_info()
    fig.canvas.draw_idle()


# ======================
# UI CONTROLS & WIDGETS
# ======================
def style_slider(slider, label_text):
    slider.label.set_text(label_text)
    slider.label.set_color(THEME['fg'])
    slider.label.set_fontsize(9)
    slider.valtext.set_color(THEME['accent'])
    slider.poly.set_facecolor(THEME['accent'])
    slider.track.set_facecolor(THEME['slider_track'])

def style_button(btn, bg_color=THEME['button_bg'], text_color=THEME['fg'], bold=False):
    btn.label.set_color(text_color)
    btn.label.set_fontsize(9)
    if bold: btn.label.set_weight('bold')
    btn.color = bg_color
    btn.hovercolor = THEME['button_hover']
    btn.ax.set_facecolor(bg_color)
    for spine in btn.ax.spines.values(): spine.set_visible(False)

# Color Map Selection
ax_cmap = plt.axes([0.02, 0.72, 0.10, 0.15], facecolor=THEME['bg'])
ax_cmap.set_title("Color Map", color=THEME['accent'], fontsize=10, loc='left')
cmap_radio = RadioButtons(ax_cmap, ('gray', 'hot', 'viridis', 'inferno'), activecolor=THEME['accent'])
for label in cmap_radio.labels: label.set_color(THEME['fg'])
cmap_radio.on_clicked(
    lambda l: (globals().update(cmap_name=l), [im.set_cmap(l) for im in [imA, imC, imS]], fig.canvas.draw_idle()))

# View Controls
fig.text(0.02, 0.65, "View Controls", color=THEME['accent'], fontsize=10, weight='bold')

bZin = Button(plt.axes([0.02, 0.60, 0.04, 0.04]), "Z+")
style_button(bZin)
bZin.on_clicked(lambda e: zoom_all_to_cross(1.5))

bZout = Button(plt.axes([0.07, 0.60, 0.04, 0.04]), "Z-")
style_button(bZout)
bZout.on_clicked(lambda e: zoom_all_to_cross(1 / 1.5))

bReset = Button(plt.axes([0.02, 0.55, 0.09, 0.04]), "Reset View")
style_button(bReset)
bReset.on_clicked(lambda e: refresh(reset_view=True))


# ======================
# AI INTEGRATION LOGIC
# ======================
ai_tumor_mask = None
ai_peak_z = -1
ai_contour_coll = None
show_ai_mask = True

def finalize_full_scan(has_tumor, confidence, t_range, peak_slice, t_mask, center_coords):
    global ai_tumor_mask, ai_peak_z
    global ax_idx, cor_idx, sag_idx 
    
    if has_tumor:
        range_text = f"{t_range[0]}-{t_range[1]}" if t_range else "Unknown"
        status = f"AI DETECTION: TUMOR FOUND | Confidence: {confidence:.1f}% \n Z-Slices: {range_text} | Peak at Z={peak_slice}"
        global diagnosis_info
        diagnosis_info = confidence
        txt_ai_info.set_color('#FF4081')
        ai_peak_z = peak_slice
        ai_tumor_mask = t_mask 
        
        # Navigate UI to AI anchor coordinates
        if center_coords and center_coords[0] != -1:
            new_x = int(center_coords[0])
            new_y = int(center_coords[1])
            
            def move_ui_elements():
                slS.set_val(new_x)
                slC.set_val(new_y)
                refresh()

            fig.canvas.get_tk_widget().after(10, move_ui_elements)
            
    else:
        status = "AI ANALYSIS: SCAN COMPLETE \n NO TUMOR DETECTED"
        txt_ai_info.set_color(THEME['success'])
        ai_peak_z = -1
        ai_tumor_mask = None
        diagnosis_info = 0
        
    txt_ai_info.set_text(status)
    slA.set_val(peak_slice) 
    refresh() 

def ai_worker_thread(volume_to_scan, current_rot):
    num_total = volume_to_scan.shape[2]
    prog_win, prog_bar, prog_lbl = create_progress_window(num_total)
    
    def update_progress_ui(current, total):
        def update():
            prog_bar["value"] = current
            prog_bar["maximum"] = total
            prog_lbl.config(text=f"Analyzing Slice: {current} / {total}")
            prog_win.update_idletasks()
        fig.canvas.get_tk_widget().after(0, update)

    try:
        has_tumor, conf, t_range, peak_slice, t_mask, center_coords = ai_engine.scan_full_volume(volume_to_scan, current_rot, update_progress_ui)
        fig.canvas.get_tk_widget().after(0, prog_win.destroy)
        fig.canvas.get_tk_widget().after(0, finalize_full_scan, has_tumor, conf, t_range, peak_slice, t_mask, center_coords)
    except Exception as e:
        print(f"Error during AI scan: {e}")
        fig.canvas.get_tk_widget().after(0, prog_win.destroy)

def create_progress_window(num_slices):
    prog_win = tk.Toplevel()
    prog_win.title("AI Volume Scanning...")
    prog_win.geometry("400x150")
    prog_win.configure(bg=THEME['panel_bg'])
    prog_win.attributes("-topmost", True)
    
    x = (prog_win.winfo_screenwidth() // 2) - 200
    y = (prog_win.winfo_screenheight() // 2) - 75
    prog_win.geometry(f"+{x}+{y}")

    lbl = tk.Label(prog_win, text="Analyzing Brain Slices...", bg=THEME['panel_bg'], fg=THEME['accent'], font=("Segoe UI", 10))
    lbl.pack(pady=20)

    progress = ttk.Progressbar(prog_win, orient="horizontal", length=300, mode="determinate")
    progress.pack(pady=10)
    progress["maximum"] = num_slices
    
    return prog_win, progress, lbl

def run_ai_analysis(event):
    txt_ai_info.set_text("Scanning 3D Volume... Please wait.")
    txt_ai_info.set_color(THEME['accent'])
    fig.canvas.draw_idle()
    
    threading.Thread(
        target=ai_worker_thread, 
        args=(data, rot['axial']), 
        daemon=True
    ).start()

# AI Control Buttons
ax_ai = plt.axes([0.02, 0.35, 0.09, 0.06]) 
bAI = Button(ax_ai, "RUN AI\nANALYSIS")
style_button(bAI, bg_color='#6200EA', text_color='white', bold=True) 
bAI.on_clicked(run_ai_analysis)

ax_ai_toggle = plt.axes([0.12, 0.35, 0.05, 0.06]) 
bAIToggle = Button(ax_ai_toggle, "Hide\nAI")
style_button(bAIToggle, bg_color=THEME['button_bg'], text_color=THEME['fg'])

def toggle_ai_mask(event):
    global show_ai_mask
    show_ai_mask = not show_ai_mask
    bAIToggle.label.set_text("Hide\nAI" if show_ai_mask else "Show\nAI")
    bAIToggle.color = THEME['button_bg'] if show_ai_mask else '#555555'
    refresh()

bAIToggle.on_clicked(toggle_ai_mask)


# ======================
# 3D EXPORT & PDF REPORT
# ======================
def render_3d(event):
    """Generates an STL mesh based on the current threshold values."""
    print("Generating 3D STL model...")
    binary_vol = (data >= mask_min_val) & (data <= mask_max_val)

    if binary_vol.sum() == 0:
        print("Warning: Empty selection. Adjust sliders.")
        return

    smooth_vol = gaussian_filter(binary_vol.astype(float), sigma=1)
    try:
        global voxel_size_mm
        spacing_tuple = (voxel_size_mm, voxel_size_mm, voxel_size_mm)
        
        verts, faces, _, _ = measure.marching_cubes(smooth_vol, level=0.5, step_size=1, spacing=spacing_tuple)
        mri_mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
        for i, f in enumerate(faces): 
            mri_mesh.vectors[i] = verts[f]
            
        fname = "mri_range_model.stl"
        mri_mesh.save(fname)
        
        volume_mm3, _, _ = mri_mesh.get_mass_properties()
        volume_cm3 = volume_mm3 / 1000.0 
        
        global clinical_volume_cm3
        clinical_volume_cm3 = volume_cm3 
        
        txt_ai_info.set_text(f"3D GENERATED | VOLUME: {abs(volume_cm3):.2f} cm³")
        txt_ai_info.set_color(THEME['success'])
        fig.canvas.draw_idle()
        
        os.startfile(os.path.abspath(fname))
    except Exception as e:
        print(f"Error during 3D generation: {e}")

b3D = Button(plt.axes([0.02, 0.45, 0.09, 0.06]), "GENERATE\n3D STL")
style_button(b3D, bg_color=THEME['success'], text_color='white', bold=True)
b3D.on_clicked(render_3d)

def generate_pdf_report(event):
    """Compiles orthogonal slices and clinical data into a PDF report."""
    print("Generating Clinical PDF Report...")
    try:
        extent_A = axA.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig("temp_axial.png", bbox_inches=extent_A, facecolor='black')
        
        extent_C = axC.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig("temp_coronal.png", bbox_inches=extent_C, facecolor='black')
        
        extent_S = axS.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig("temp_sagittal.png", bbox_inches=extent_S, facecolor='black')
        
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt="MRI Clinical Analysis Report", ln=True, align='C')
        
        pdf.set_font("Arial", '', 10)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 10, txt=f"Scan Date: {current_time}", ln=True, align='C')
        
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="AI & Volumetric Analysis:", ln=True, align='L')
        
        pdf.set_font("Arial", '', 11)
        ai_text = txt_ai_info.get_text()
        if diagnosis_info != 0: 
            pdf.cell(0, 10, txt=f"AI DETECTION: TUMOR FOUND | Confidence: {diagnosis_info:.1f}%", ln=True, align='L')
            pdf.cell(0, 10, txt=f"VOLUME: {abs(clinical_volume_cm3):.2f} cm³", ln=True, align='L')
        else:
            pdf.cell(0, 10, txt=ai_text, ln=True, align='L')
        
        pdf.image("temp_axial.png", x=10, y=80, w=60)
        pdf.image("temp_coronal.png", x=75, y=80, w=60)
        pdf.image("temp_sagittal.png", x=140, y=80, w=60)
        
        report_name = "Clinical_Report.pdf"
        pdf.output(report_name)
        os.startfile(os.path.abspath(report_name))
        
    except Exception as e:
        print(f"Error generating PDF: {e}")

ax_btn_report = plt.axes([0.02, 0.92, 0.09, 0.06]) 
btn_report = Button(ax_btn_report, 'Generate \n PDF', color=THEME['accent'], hovercolor=THEME['fg'])
btn_report.on_clicked(generate_pdf_report)


# ======================
# SLIDERS & NAVIGATION
# ======================
rot_y = 0.28
slider_y = 0.16

# Axial
bRotA = Button(plt.axes([0.24, rot_y, 0.08, 0.035]), "Rot Ax")
style_button(bRotA)
bRotA.on_clicked(lambda e: (rot.update({'axial': rot['axial'] + 1}), refresh(True)))
slA = Slider(plt.axes([0.15, slider_y, 0.22, 0.02], facecolor=THEME['bg']), 'Z', 0, sz - 1, valinit=ax_idx, valstep=1)
style_slider(slA, 'Z-Slice')

# Coronal
bRotC = Button(plt.axes([0.53, rot_y, 0.08, 0.035]), "Rot Cor")
style_button(bRotC)
bRotC.on_clicked(lambda e: (rot.update({'coronal': rot['coronal'] + 1}), refresh(True)))
slC = Slider(plt.axes([0.43, slider_y, 0.22, 0.02], facecolor=THEME['bg']), 'Y', 0, sy - 1, valinit=cor_idx, valstep=1)
style_slider(slC, 'Y-Slice')

# Sagittal
bRotS = Button(plt.axes([0.82, rot_y, 0.08, 0.035]), "Rot Sag")
style_button(bRotS)
bRotS.on_clicked(lambda e: (rot.update({'sagittal': rot['sagittal'] + 1}), refresh(True)))
slS = Slider(plt.axes([0.71, slider_y, 0.22, 0.02], facecolor=THEME['bg']), 'X', 0, sx - 1, valinit=sag_idx, valstep=1)
style_slider(slS, 'X-Slice')

def update_slices(val):
    global ax_idx, cor_idx, sag_idx
    ax_idx, cor_idx, sag_idx = int(slA.val), int(slC.val), int(slS.val)
    refresh()

slA.on_changed(update_slices)
slC.on_changed(update_slices)
slS.on_changed(update_slices)

def on_click(event):
    """Synchronizes orthogonal views on mouse click."""
    if event.inaxes not in axes: return
    global ax_idx, cor_idx, sag_idx
    if event.inaxes == axA:
        ax_k, k = 'axial', rot['axial']; base_h, base_w = get_slice_dims('axial')
    elif event.inaxes == axC:
        ax_k, k = 'coronal', rot['coronal']; base_h, base_w = get_slice_dims('coronal')
    else:
        ax_k, k = 'sagittal', rot['sagittal']; base_h, base_w = get_slice_dims('sagittal')

    orig_r, orig_c = transform_coords(int(event.ydata), int(event.xdata), base_w if k % 2 != 0 else base_h,
                                      base_h if k % 2 != 0 else base_w, 4 - (k % 4))

    if ax_k == 'axial':
        sag_idx, cor_idx = orig_r, orig_c
    elif ax_k == 'coronal':
        sag_idx, ax_idx = orig_r, orig_c
    elif ax_k == 'sagittal':
        cor_idx, ax_idx = orig_r, orig_c
    slA.set_val(ax_idx)
    slC.set_val(cor_idx)
    slS.set_val(sag_idx)

fig.canvas.mpl_connect('button_press_event', on_click)


# ======================
# BOTTOM TOOLBAR
# ======================
fig.text(0.12, 0.12, "RANGE SELECTOR", color=THEME['accent'], fontsize=10, weight='bold')

bMask = Button(plt.axes([0.12, 0.05, 0.05, 0.04]), "Show\nMask")
style_button(bMask)
def t_mask(e): global mask_on; mask_on = not mask_on; refresh()
bMask.on_clicked(t_mask)

sl_min = Slider(plt.axes([0.20, 0.08, 0.20, 0.02], facecolor=THEME['bg']), '', 0.0, 1.0, valinit=mask_min_val)
style_slider(sl_min, 'Min')
sl_max = Slider(plt.axes([0.20, 0.04, 0.20, 0.02], facecolor=THEME['bg']), '', 0.0, 1.0, valinit=mask_max_val)
style_slider(sl_max, 'Max')

def update_ranges(val):
    global mask_min_val, mask_max_val
    mask_min_val = sl_min.val
    mask_max_val = sl_max.val
    if mask_on: refresh()

sl_min.on_changed(update_ranges)
sl_max.on_changed(update_ranges)

def open_dsp_window(event):
    """Initializes the external Digital Signal Processing (FFT) Tkinter module."""
    current_slice = rot90(axial_slice(ax_idx), rot['axial'])
    
    dsp_win = tk.Toplevel()
    dsp_win.title("Advanced DSP - FFT Edge Detection")
    dsp_win.geometry("800x450")
    dsp_win.configure(bg=THEME['bg'])
    
    x = (dsp_win.winfo_screenwidth() // 2) - 400
    y = (dsp_win.winfo_screenheight() // 2) - 225
    dsp_win.geometry(f"+{x}+{y}")
    
    from matplotlib.figure import Figure
    fig_dsp = Figure(figsize=(8, 4), facecolor=THEME['bg'])
    
    ax_orig = fig_dsp.add_subplot(1, 2, 1)
    ax_fft = fig_dsp.add_subplot(1, 2, 2)
    
    ax_orig.imshow(current_slice, cmap='gray')
    ax_orig.set_title("Original Slice", color=THEME['accent'])
    ax_orig.axis('off')
    
    init_radius = 0.15
    filtered_slice = DSP_Engine.apply_fft_highpass(current_slice, init_radius)
    img_fft = ax_fft.imshow(filtered_slice, cmap='gray')
    ax_fft.set_title("FFT High-Pass Filter", color=THEME['accent'])
    ax_fft.axis('off')
    
    canvas = FigureCanvasTkAgg(fig_dsp, master=dsp_win)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    ax_slider = fig_dsp.add_axes([0.2, 0.05, 0.6, 0.05], facecolor=THEME['bg'])
    dsp_slider = Slider(ax_slider, 'Cutoff Radius', 0.1, 0.2, valinit=init_radius)
    
    dsp_slider.label.set_color(THEME['fg'])
    dsp_slider.valtext.set_color(THEME['accent'])
    dsp_slider.poly.set_facecolor(THEME['accent'])
    
    def update_fft(val):
        r = dsp_slider.val
        
        # 1. מפעילים את המסנן ומקבלים רק את הקצוות (התדרים הגבוהים)
        edges = DSP_Engine.apply_fft_highpass(current_slice, r)
        
        # 2. מגדירים את "כפתור הווליום" של החידוד (אפשר לשחק עם המספר הזה)
        alpha = 1.5 
        
        # 3. פעולת החיבור: התמונה המקורית + (הקצוות כפול הווליום)
        sharpened = current_slice + (alpha * edges)
        
        # כדי שהתצוגה לא תשתגע מערכים שחורגים מהטווח של הצבעים, "חותכים" אותם
        # בהנחה שהתמונה שלך מנורמלת לטווח של 0 עד 1
        sharpened = np.clip(sharpened, 0.0, 1.0) 
        
        # 4. מציגים את התמונה המחודדת בחלון הימני
        img_fft.set_data(sharpened)
        canvas.draw_idle()
        
    dsp_slider.on_changed(update_fft)
    dsp_win.dsp_slider = dsp_slider 

ax_dsp = plt.axes([0.02, 0.25, 0.09, 0.06]) 
bDSP = Button(ax_dsp, "ADVANCED\nDSP (FFT)")
style_button(bDSP, bg_color='#FF9800', text_color='white', bold=True) 
bDSP.on_clicked(open_dsp_window)

# Sharpening Filter Controls
fig.text(0.55, 0.12, "ENHANCEMENT", color=THEME['accent'], fontsize=10, weight='bold')
bSharpen = Button(plt.axes([0.55, 0.05, 0.05, 0.04]), "Sharpen")
style_button(bSharpen)
bSharpen.on_clicked(lambda e: globals().update(sharpen_on=not sharpen_on) or refresh())

sl_rad = Slider(plt.axes([0.65, 0.08, 0.20, 0.02], facecolor=THEME['bg']), '', 0.1, 5.0, valinit=sharpen_radius)
style_slider(sl_rad, 'Radius')
sl_amt = Slider(plt.axes([0.65, 0.04, 0.20, 0.02], facecolor=THEME['bg']), '', 0.1, 5.0, valinit=sharpen_amount)
style_slider(sl_amt, 'Amount')

sl_rad.on_changed(lambda v: globals().update(sharpen_radius=v) or (refresh() if sharpen_on else None))
sl_amt.on_changed(lambda v: globals().update(sharpen_amount=v) or (refresh() if sharpen_on else None))

# Top Banner Info
txt_info = fig.text(0.50, 0.96, '', ha='center', fontsize=11, color=THEME['accent'])
txt_ai_info = fig.text(0.50, 0.91, '', ha='center', fontsize=13, weight='bold', color=THEME['accent'])

def update_info():
    i, j, k = max(0, min(sag_idx, sx - 1)), max(0, min(cor_idx, sy - 1)), max(0, min(ax_idx, sz - 1))
    msk_txt = f"[{mask_min_val:.2f}-{mask_max_val:.2f}]" if mask_on else "OFF"
    txt_info.set_text(
        f"Voxel: [{i},{j},{k}] | Val: {data[i, j, k]:.3f} | Mask: {msk_txt} | Sharpen: {'ON' if sharpen_on else 'OFF'}")

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    refresh()
    plt.show()