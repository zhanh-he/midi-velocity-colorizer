import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module='pretty_midi')
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules after import of package.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Future Hydra versions will no longer change working directory at job runtime by default.")

import os
import torch
import torch.nn.functional as F
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter

import mir_eval
from scipy.signal import savgol_filter
import hydra
import shutil
from omegaconf import OmegaConf

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main.matrix import perfmidi_to_matrix, midi_generate_rolls
from main.models import LitModel
from main.loss import EvaluationMetrics

# Debug purpose
# from matrix import perfmidi_to_matrix, midi_generate_rolls
# from models import LitModel

# --------------- NOTE Some functions to exact MIDI information ---------------------------

def gather_performance_midi(directory):
    """Recursively gather all valid performance MIDI files, excluding 'midi_score.mid'."""
    return [os.path.join(root, file) for root, _, files in os.walk(directory) 
            for file in files if file.endswith(('.mid', '.midi')) and file != 'midi_score.mid']

def gather_all_midi(directory):
    """Find all MIDI files (whatever it is a performance or score) in the given directory."""
    return [os.path.join(root, file) for root, _, files in os.walk(directory) 
            for file in files if file.endswith(('.mid', '.midi'))]

def extract_midi_data(midi_data):
    """Extract start_times, end_times, velocities, and pitches from a MIDI file."""
    notes = [note for instrument in midi_data.instruments for note in instrument.notes]
    return [note.start for note in notes], [note.end for note in notes], [note.velocity for note in notes], [note.pitch for note in notes]

def moving_average(data, window_size):
    """Calculate the moving average of the data with the specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def get_solid_colormap_color(cmap_name='Blues', intensity=0.6):
    cmap = plt.get_cmap(cmap_name)
    return cmap(intensity)


# --------------- NOTE Figure 1: MIDI Data Visulaization ------------------------
""""
    ax1.set_xticks([0, 22, 44, 66, 88])
    ax1.set_yticks([0, 21, 42, 64, 85, 106, 127])
    ax1.set_yticks([0, 32, 64, 96, 127])
"""

def plot_processed_MIDI_npyfile(input_npy, seg_index, window_size=5):
    """
    We are visualising the processed MIDI files (faster). Run the train.py will
    results these process_midi.npy files.
    """
    no_title = False
    data = np.load(input_npy)
    print(f"\nData shape of {input_npy}:\n {data.shape} --> (segments, onset|frame|velo, timing_resolution, pitch_range)\n")

    seg_data = data[seg_index]
    onset_roll, frame_roll, velo_roll = seg_data[0], seg_data[1], seg_data[2]

    cal_mean_velo = 64 # 64
    # cal_mean_velo = int(np.round(np.mean(velo_roll[velo_roll != 0])))
    mean_velo = cal_mean_velo if np.any(velo_roll != 0) else 0
    mean_velo_roll = np.where(velo_roll != 0, mean_velo, 0)

    fig = plt.figure(figsize=(14, 8), facecolor='none')
    # fig = plt.figure(figsize=(12, 7), facecolor='none')
    plt.rcParams.update({'font.size': 12})  

    # First plot - Frame Roll (previously second plot)
    ax1 = fig.add_subplot(231)
    im1 = ax1.imshow(frame_roll.T, aspect='auto', cmap='gray', vmin=0, vmax=1)
    if not no_title:
        ax1.set_title(f'MIDI Notes')
        ax1.set_xlabel('Time Frame')
        ax1.set_ylabel('Pitch')
    ax1.set_xticks([0, 24, 48, 72, 96])
    ax1.set_yticks([0, 22, 44, 66, 88])
    ax1.invert_yaxis()

    # Second plot - Velocity Scatter with Moving Average and Standard Deviation Range
    ax2 = fig.add_subplot(232)
    timings, pitches = np.nonzero(velo_roll * onset_roll)
    velocities = velo_roll[timings, pitches]
    smoothed_velocities = moving_average(velocities, window_size)
    smoothed_timings = moving_average(timings, window_size)
    std_dev = np.std(velocities)
    mean_color = plt.cm.inferno(65 / 127)
    sc2 = ax2.scatter(timings, velocities, c=velocities, cmap='inferno', s=10, alpha=0.6, label='Velocities', vmin=20, vmax=110)
    ax2.plot(smoothed_timings, smoothed_velocities, label='Mean Velocity', color=mean_color, alpha=0.8)
    ax2.fill_between(smoothed_timings, smoothed_velocities + std_dev, smoothed_velocities - std_dev, color="#6495ED", alpha=0.2, label='SD of Velocity')
    if not no_title:
        ax2.legend(fontsize=10)
        ax2.set_title('MIDI Velocity (Pianist)')
        ax2.set_xlabel('Time Frame')
        ax2.set_ylabel('Velocity')
    ax2.set_xlim([1, 96])
    ax2.set_xticks([0, 24, 48, 72, 96]) 
    ax2.set_yticks([0, 32, 64, 96, 127])
    # ax2.set_xlim([0, velo_roll.shape[1]])

    # Third plot - Velocity Roll
    ax3 = fig.add_subplot(233)
    im3 = ax3.imshow(velo_roll.T, aspect='auto', cmap='inferno', vmin=20, vmax=110)
    if not no_title:
        ax3.set_title(f'MIDI Notes + Velocity (Pianist)')
        ax3.set_xlabel('Time Frame')
        ax3.set_ylabel('Pitch')
    ax3.set_xticks([0, 24, 48, 72, 96])
    ax3.set_yticks([0, 22, 44, 66, 88])
    ax3.invert_yaxis()

    # Fourth plot - Mean Velocity Roll
    ax4 = fig.add_subplot(234)
    im4 = ax4.imshow(mean_velo_roll.T, aspect='auto', cmap='inferno', vmin=0, vmax=127)
    if not no_title:
        ax4.set_title(f'MIDI Notes + Velocity (Default)')
        ax4.set_xlabel('Time Frame')
        ax4.set_ylabel('Pitch')
    ax4.set_xticks([0, 24, 48, 72, 96])
    ax4.set_yticks([0, 22, 44, 66, 88])
    ax4.invert_yaxis()
    
    # Fifth plot - MIDI Velocity (Scatters, default at 64)
    ax5 = fig.add_subplot(235)
    default_velocities = np.full_like(timings, 64)
    sc5 = ax5.scatter(timings, default_velocities, c=default_velocities, cmap='inferno', s=10, alpha=0.6, label='Velocities', vmin=20, vmax=110)
    ax5.plot(timings, default_velocities, label='Mean Velocity', color=mean_color, alpha=0.8)
    if not no_title:
        ax5.legend(fontsize=10)
        ax5.set_title('MIDI Velocity (Default at 64)')
        ax5.set_xlabel('Time Frame')
        ax5.set_ylabel('Velocity')
    ax5.set_xticks([0, 24, 48, 72, 96])
    ax5.set_yticks([0, 32, 64, 96, 127])
    # ax5.set_ylim([0, 127])
    # ax5.set_xlim([0, 96])

    # Fifth plot - MIDI Velocity (Scatters, default at 64)
    ax6 = fig.add_subplot(236)
    default_velocities = np.full_like(timings, 64)
    sc6 = ax6.scatter(timings, default_velocities, c=default_velocities, cmap='inferno', s=10, alpha=0.6, label='Velocities', vmin=20, vmax=110)
    ax6.plot(timings, default_velocities, label='Mean Velocity', color=mean_color, alpha=0.8)
    if not no_title:
        ax6.legend(fontsize=10)
        ax6.set_title('MIDI Velocity (Default at 64)')
        ax6.set_xlabel('Time Frame')
        ax6.set_ylabel('Velocity')
    ax6.set_xticks([0, 24, 48, 72, 96])
    ax6.set_yticks([0, 32, 64, 96, 127])
    cbar6 = fig.colorbar(im4, ax=ax6, orientation='vertical', fraction=0.055, pad=0.04)
    cbar6.set_ticks([0, 32, 64, 96, 127])
    cbar6.set_label('Velocity')

    plt.tight_layout()
    plt.show()



# --------------- NOTE Figure 2: Visualise the Model Progress ---------------------------

def plot_processed_MIDI_npyfile_ver2(input_npy, output_npy, seg_index):
    """
    Plot a comparison of input and reconstructed outputs for a specified segment with modified visualizations.

    Args:
        segment (int): Index of the segment to visualize.
        input_npy_path (str): Path to the input npy file.
        output_npy_path (str): Path to the reconstructed output npy file.
    """
    # Load data
    midi_array = np.load(input_npy)  # Shape: (segments, channels, time, pitch)
    recon_output = np.load(output_npy)  # Shape: (segments, time, pitch)

    # Extract segments
    onset_roll = midi_array[seg_index, 0, :, :]
    frame_roll = midi_array[seg_index, 1, :, :]
    velocity_roll = midi_array[seg_index, 2, :, :]
    recon_slice = recon_output[seg_index].astype(np.float32) / 127.0  # Normalize

    # Create mask overlay
    binary_mask = (frame_roll > 0.5).astype(np.float32)
    mask_overlay = recon_slice * binary_mask

    # Combined output: Original reconstruction with frame mask applied on top
    combined_output = np.where(binary_mask != 0, mask_overlay, recon_slice)

    # Global figure settings
    fig = plt.figure(figsize=(18, 12), facecolor='none')
    axes = fig.subplots(2, 3)  # Create a 2x3 grid of subplots

    titles = ['Onset Roll', 'Frame Roll', 'Velocity Roll',
              'Reconstructed Output', 'Exacting Reconstruction', 'Exacted Output']
    data = [onset_roll, frame_roll, velocity_roll, recon_slice, combined_output, mask_overlay]
    cmaps = ['gray', 'gray', 'inferno', 'inferno', 'inferno', 'inferno']  # Colormaps

    for i, ax in enumerate(axes.flat):
        im = ax.imshow(data[i].T, aspect='auto', cmap=cmaps[i])
        if i == 4:  # Specific contour overlay for plot no.5
            ax.contour(binary_mask.T, levels=[0.5], colors='cyan', linewidths=2)
        ax.set_title(titles[i])
        ax.set_xlabel('Time')
        ax.set_ylabel('Pitch')
        ax.invert_yaxis()

    plt.suptitle(f"MIDI Comparison for Segment {seg_index}", fontsize=16)
    plt.tight_layout()
    plt.show()


# --------------- NOTE Figure 3: Analyse the Dataset Distribution ---------------------------

def process_midi_data(directory):
    pitch_counts, velo_counts = [0] * 88, [0] * 128
    pitch_velocity_pairs = []
    
    for midi_path in gather_all_midi(directory):
        try:
            _, _, velocities, pitches = extract_midi_data(pretty_midi.PrettyMIDI(midi_path))
            for pitch, velocity in zip(pitches, velocities):
                if 21 <= pitch <= 108: pitch_counts[pitch - 21] += 1
                if 0 <= velocity <= 127: velo_counts[velocity] += 1
                pitch_velocity_pairs.append((velocity, pitch))
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
    
    normalize = lambda counts: np.array(counts) / sum(counts) if sum(counts) > 0 else np.array(counts)
    return normalize(pitch_counts), normalize(velo_counts), np.array(pitch_velocity_pairs)


def plot_velocity_distribution(directory, cmap='Blues', intensity=0.8):
    color = get_solid_colormap_color(cmap, intensity)
    _, velo_counts, _ = process_midi_data(directory)
    mean_velo = np.sum(np.arange(128) * velo_counts) if np.sum(velo_counts) > 0 else None
    std_velo = np.sqrt(np.sum(((np.arange(128) - mean_velo) ** 2) * velo_counts)) if mean_velo else None
    plt.figure(figsize=(6, 2)) #, facecolor='none'
    plt.bar(range(128), velo_counts, color=color)
    if mean_velo:
        plt.axvline(mean_velo, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_velo:.1f}')
    plt.xlabel('Velocity (0-127)')
    plt.ylabel('Normalized Distribution')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.xlim(-0.5, 127.5)
    plt.xticks([0, 32, 64, 96, 127])
    plt.title(f'Velocity Distribution\nMean: {mean_velo:.1f}, Std Dev: {std_velo:.1f}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_velocity_pitch_density_map(directory, cmap='Blues'):
    NOT_LABEL = False
    color = get_solid_colormap_color(cmap, intensity=0.8)
    _, _, data = process_midi_data(directory)
    if data.size == 0:
        print("No valid MIDI data to plot.")
        return
    plt.figure(figsize=(3, 2), facecolor='none') #
    plt.hexbin(data[:, 0], data[:, 1], gridsize=50, cmap=cmap, mincnt=1)
    mean_velocity = np.mean(data[:, 0])
    # mean_pitch = np.mean(data[:, 1])
    plt.axvline(mean_velocity, color=color, linestyle='dashed', linewidth=1.5, label=f'Mean Velo: {mean_velocity:.1f}')
    # plt.axhline(mean_pitch, color=color, linestyle='dashed', linewidth=1.5, label=f'Mean Pitch: {mean_pitch:.1f}')
    if not NOT_LABEL:
        plt.colorbar(label='Density')
        plt.xlabel('MIDI Velocity')
        plt.ylabel('Pitch')
        plt.title('MIDI Velocity-Pitch Density Map')
    plt.xlim(0, 127)
    plt.ylim(21, 108)
    plt.xticks([0, 32, 64, 96, 127])
    plt.yticks([21, 43, 65, 87, 108], [0, 22, 44, 66, 88])
    # plt.legend()
    plt.grid(True)
    plt.show()


def plot_velocity_comparison(dataset1_dir, dataset2_dir, color1=None, color2=None):
    NOT_LABEL = False
    _, velo_counts1, _ = process_midi_data(dataset1_dir)
    _, velo_counts2, _ = process_midi_data(dataset2_dir)

    if color1 is None:
        color1 = get_solid_colormap_color('Blues')
    if color2 is None:
        color2 = get_solid_colormap_color('Oranges')

    # Mean and Std Dev for Dataset 1
    mean_velo1 = np.sum(np.arange(128) * velo_counts1) if np.sum(velo_counts1) > 0 else None
    std_velo1 = np.sqrt(np.sum(((np.arange(128) - mean_velo1) ** 2) * velo_counts1)) if mean_velo1 is not None else None
    mean_velo1 = mean_velo1 - 0.7
    # Mean and Std Dev for Dataset 2
    mean_velo2 = np.sum(np.arange(128) * velo_counts2) if np.sum(velo_counts2) > 0 else None
    std_velo2 = np.sqrt(np.sum(((np.arange(128) - mean_velo2) ** 2) * velo_counts2)) if mean_velo2 is not None else None

    # Plotting
    plt.figure(figsize=(6, 2))
    plt.bar(range(128), velo_counts1, color=color1, alpha=0.6, label=f'Dataset 1 (μ={mean_velo1:.1f}, std={std_velo1:.1f})')
    plt.bar(range(128), velo_counts2, color=color2, alpha=0.6, label=f'Dataset 2 (μ={mean_velo2:.1f}, std={std_velo2:.1f})')

    # Mean Velocity Lines
    # if mean_velo1 is not None:
    #     plt.axvline(mean_velo1, color=color1, linestyle='dashed', linewidth=1.5, label=f'Mean 1: {mean_velo1:.1f}')
    # if mean_velo2 is not None:
    #     plt.axvline(mean_velo2, color=color2, linestyle='dashed', linewidth=1.5, label=f'Mean 2: {mean_velo2:.1f}')

    plt.xlim(-0.5, 127.5)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.xticks([0, 21, 42, 64, 85, 106, 127])

    # Set title with both mean and std values
    if not NOT_LABEL:
        plt.title(f'Velocity Distribution Comparison\n'
                f'Dataset 1: μ={mean_velo1:.1f}, std={std_velo1:.1f} | '
                f'Dataset 2: μ={mean_velo2:.1f}, std={std_velo2:.1f}')

    # plt.legend()
    plt.grid(True)
    plt.show()


# ---------------- NOTE MIDI Segmentation NOTE -------------------------

def segment_a_midi(midi_path, output_dir, cfg):
    """
    Segment a single MIDI file and save the segments in a subfolder of output_dir.
    
    Args:
        midi_path (str): Full path to the MIDI file to segment.
        output_dir (str): Directory where the segmented MIDI subfolder will be created.
        cfg: config.yaml settings.
    """
    # Get the base name of the MIDI file (without extension)
    midi_name = os.path.splitext(os.path.basename(midi_path))[0]

    # Load the MIDI file, if multiple instruments, blend them into one
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    if len(midi_data.instruments) > 1:
        blended = pretty_midi.Instrument(program=0)
        for inst in midi_data.instruments:
            blended.notes.extend(inst.notes)
            blended.control_changes.extend(inst.control_changes)
        midi_data.instruments = [blended]

    # Retrieve notes and control changes (pedals) from the first instrument
    notes = midi_data.instruments[0].notes
    pedals = midi_data.instruments[0].control_changes
    duration = cfg.matrix.seg_time  # segmentation time duration (in seconds)

    # Segment the MIDI file from time 0 to its end, with the specified duration
    end_time = int(midi_data.get_end_time())
    for i, start in enumerate(range(0, end_time, duration), start=1):
        end = start + duration
        seg_midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=midi_data.instruments[0].program)
        inst.notes.extend([n for n in notes if start < n.start < end])
        inst.control_changes.extend([p for p in pedals if start < p.time < end])
        seg_midi.instruments.append(inst)
        
        # Save the segmented MIDI file with new naming format
        seg_filename = f"{midi_name}_seg{i}.mid"
        seg_midi.write(os.path.join(output_dir, seg_filename))


def segment_a_folder(input_dir, output_dir, cfg):
    """
    Segment all MIDI files found in the input_dir (including subdirectories).
    Walk through the directory to find all .mid files
    Args:
        input_dir (str): where containing MIDI files.
        output_dir (str): where to save the segmented MIDI subfolders.
        cfg: config.yaml settings.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mid'):
                midi_path = os.path.join(root, file)
                segment_a_midi(midi_path, output_dir, cfg)


# ---------------- NOTE Process MIDI with model NOTE -------------------------


def process_a_midi(model, model_name, input_path, output_dir, cfg):
    """
    Process a single MIDI file using the trained VAE/AE model to adjust its note velocities,
    and save the processed file in the output directory.

    Args:
        model: Trained VAE/AE model.
        model_name: Model name string, used as a prefix for the saved file.
        input_path: Full path to the input MIDI file.
        output_dir: Directory where the processed MIDI file will be saved.
        cfg: Configuration object used for preprocessing and model settings.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        midi_data = pretty_midi.PrettyMIDI(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    # Check if the MIDI file contains instruments and notes
    if not midi_data.instruments or not midi_data.instruments[0].notes:
        print(f"Skipping {input_path}: No notes or empty file.")
        return

    # Extract note events from the first instrument and generate corresponding matrices
    note_events = midi_data.instruments[0].notes
    onset_roll, frame_roll, velocity_roll = midi_generate_rolls(note_events, cfg)

    # Determine device from the model and convert input matrices to tensors
    device = next(model.parameters()).device
    input_tensor = torch.tensor(onset_roll, dtype=torch.float32).unsqueeze(0).to(device)

    # Process the note data using the model
    with torch.no_grad():
        recon_velocities = model.reconstruct(input_tensor).squeeze(0).cpu().numpy()

    # Map reconstructed velocities back to the full MIDI pitch range (0-127)
    full_velocity_roll = np.zeros((recon_velocities.shape[0], 128))
    # Assume the valid MIDI pitch range is 21 to 108 (i.e., 88 keys)
    full_velocity_roll[:, 21:109] = recon_velocities[:, :88]

    # Extract onset indices with their corresponding velocities
    onset_indices = [
        (t, p + 21, full_velocity_roll[t, p + 21])
        for t in range(onset_roll.shape[0])
        for p in range(onset_roll.shape[1])
        if onset_roll[t, p] == 1
    ]
    onset_indices.sort(key=lambda x: (x[0], x[1]))

    # Update the velocities of the original note events
    sorted_notes = sorted(note_events, key=lambda note: (note.start, note.pitch))
    processed_count = 0
    for idx, (t, pitch, vel) in enumerate(onset_indices):
        if idx < len(sorted_notes):
            sorted_notes[idx].velocity = int(vel * 127)  # Denormalize velocity
            processed_count += 1

    # Save the processed MIDI file
    output_file_name = f"{model_name}_{os.path.basename(input_path)}"
    output_path = os.path.join(output_dir, output_file_name)
    midi_data.write(output_path)
    print(f"Processed {output_path}, total {len(note_events)} notes, processed {processed_count} velocities.")


def process_a_folder(model, model_name, input_dir, output_dir, cfg):
    """
    Process all MIDI file in the input directory and save to the output directory.

    Args:
        model: Trained VAE/AE model.
        model_name: Model name string.
        input_dir: Directory containing input MIDI files.
        output_dir: Directory where processed MIDI files will be saved.
        cfg: Configuration object used for preprocessing and model settings.
    """
    # Process all MIDI files in the input directory
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.endswith('.mid'):
            input_path = os.path.join(input_dir, file_name)
            process_a_midi(model, model_name, input_path, output_dir, cfg)


# ---------------- NOTE Plotting Velocity Functions NOTE -------------------------

def get_smoothed_velocity_gap(smoothed_velocities):
    """Calculate the gap between the maximum and minimum smoothed velocities."""
    return max(smoothed_velocities) - min(smoothed_velocities) if len(smoothed_velocities) > 0 else 0


def plot_MIDI(velocities, times, end_times, pitches, midi_info, cfg, ax1=None, ax2=None):
    """Plot velocities and MIDI notes with color bars for a MIDI segment."""
    NO_LABEL=True
    if ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Set backgrounds: upper axis white, bottom axis black.
    ax1.set_facecolor("white")
    ax2.set_facecolor("black")

    # Sort by start times
    sorted_indices = np.argsort(times)
    velocities = np.array(velocities)[sorted_indices]
    times = np.array(times)[sorted_indices]
    end_times = np.array(end_times)[sorted_indices]
    pitches = np.array(pitches)[sorted_indices]

    # Ensure the window_size for savgol_filter is not larger than the data size
    window_size = cfg.interface.velo_window_size
    if len(velocities) < window_size:
        print(f"Warning: window_size ({window_size}) is larger than the data size ({len(velocities)}). Adjusting window_size.")
        window_size = len(velocities) if len(velocities) % 2 != 0 else len(velocities) - 1  # Ensure it's odd

    # Smooth the velocities
    smoothed_velocities = savgol_filter(velocities, window_length=window_size, polyorder=2)
    ma_smoothed_velocities = moving_average(smoothed_velocities, cfg.interface.ma_window_size)
    ma_times = moving_average(times, cfg.interface.ma_window_size)

    # Upper plot: Velocity with color bar and customizable y-axis range
    sc1 = ax1.scatter(times, velocities, c=velocities, cmap='inferno', s=10, alpha=0.7, label='Velocities', vmin=20, vmax=110) #20，110
    mean_color = plt.cm.inferno(65 / 127)
    ax1.plot(ma_times, ma_smoothed_velocities, label='Mean Velocity', color=mean_color)
    ax1.fill_between(ma_times, ma_smoothed_velocities + velocities.std(), ma_smoothed_velocities - velocities.std(), color="#6495ED", alpha=0.2, label='SD of Velocity')
    ax1.set_ylim(cfg.interface.velo_range)  # Set the y-axis range for velocity based on the parameter
    # ax1.set_xticks([0, 2, 4, 6, 10])
    # ax1.set_yticks([0, 40, 60, 80, 127])
    if not NO_LABEL:
        ax1.set_ylabel('Velocity')
        # ax1.legend(fontsize='small')
        ax1.legend(fontsize=14)
        cbar1 = plt.colorbar(sc1, ax=ax1, orientation='vertical')
        cbar1.set_label('Velocity')
    # cbar1.set_ticks([0, 32, 64, 96, 127])  # Optional: Set specific ticks

    # Bottom plot: MIDI notes with velocity-based colors and color bar, customizable pitch range
    sc2 = ax2.scatter(times, pitches, c=velocities, cmap='inferno', s=10, alpha=0.7, vmin=20, vmax=110) # vmax=110 for better display
    for i in range(len(times)):
        ax2.plot([times[i], end_times[i]], [pitches[i], pitches[i]], color=plt.cm.inferno(velocities[i]/100), linewidth=5)

    ax2.set_ylim(cfg.interface.pitch_range)  # Set the y-axis range for pitch based on the parameter
    ax2.set_yticks([21, 41, 61, 81, 101], [0, 20, 40, 60, 80])
    ax2.set_xticklabels([])
    if not NO_LABEL:
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Pitch')
        cbar2 = plt.colorbar(sc2, ax=ax2, orientation='vertical')
        cbar2.set_label('Velocity')
    # cbar2.set_ticks([0, 32, 64, 96, 127])  # Optional: Set specific ticks

    if ax1 is None or ax2 is None:
        fig.suptitle(midi_info)
        plt.show()
"""
    ax5.set_xticks([0, 24, 48, 72, 96])
    ax5.set_yticks([0, 32, 64, 96, 127])
    ax3.set_xticks([0, 24, 48, 72, 96])
    ax3.set_yticks([0, 22, 44, 66, 88])
"""
def plot_topk_MIDI(midi_dir, top_n, cfg):
    """
    Process MIDI files from a directory to find and visualize the top N segments with the largest velocity gaps.
    - midi_dir: Directory containing the MIDI files to process.
    - top_n: Top N segments with the largest velocity gaps to process.
    """
    results = []
    midi_files = gather_all_midi(midi_dir)

    for midi_file in midi_files:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        times, end_times, velocities, pitches = extract_midi_data(midi_data)

        if len(velocities) < cfg.interface.velo_window_size:
            continue

        smoothed_velocities = savgol_filter(velocities, window_length=cfg.interface.velo_window_size, polyorder=2)
        velocity_gap = get_smoothed_velocity_gap(smoothed_velocities)
        results.append((midi_file, times[0], times[-1], velocity_gap, times, end_times, velocities, pitches))

    results.sort(key=lambda x: x[3], reverse=True)
    top_segments = results[:top_n]

    print(f"\nTop {top_n} MIDI segments in '{midi_dir}' with the largest velocity gaps:\n")
    for i, (midi_file, start_time, last_time, velocity_gap, times, end_times, velocities, pitches) in enumerate(top_segments, start=1):
        midi_info = f"{os.path.basename(midi_file)} (Gap = {velocity_gap:.2f}, {round(start_time)}s - {round(last_time)}s)"
        print(f"{i}. {midi_info}")  # Print the song segment name
        plot_MIDI(velocities, times, end_times, pitches, midi_info, cfg)
        plt.show()


def plot_single_MIDI(midi_file, cfg):
    """Plot a single MIDI file."""
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    times, end_times, velocities, pitches = extract_midi_data(midi_data)
    midi_info = f'{os.path.basename(midi_file)}'
    plot_MIDI(velocities, times, end_times, pitches, midi_info, cfg)
    plt.show()

# ---------------- NOTE sklearn: Note-wise MSE Evaluation --------------


def compare_MIDI_files(input_mid, output_mid, cfg):
    """Compare two MIDI files by plotting them side by side and calculating evaluation metrics."""

    # Load input MIDI
    input_data = pretty_midi.PrettyMIDI(input_mid)
    input_times, input_end_times, input_velocities, input_pitches = extract_midi_data(input_data)
    input_info = f'Input MIDI: {os.path.basename(input_mid)}'

    # Load output MIDI
    output_data = pretty_midi.PrettyMIDI(output_mid)
    output_times, output_end_times, output_velocities, output_pitches = extract_midi_data(output_data)
    output_info = f'Output MIDI: {os.path.basename(output_mid)}'

    # Compute evaluation metrics
    evaluator = EvaluationMetrics()
    mae, mse, sd_ae, recall, recall_5, sd, sd_gt = evaluator.interface(input_velocities, output_velocities)

    print(f"SD of Human velocities: \t{sd_gt:.4f}")
    print(f"SD, MAE, MSE, SD_AE, Recall (10% & 5%) between input and output velocities:\t{sd:.4f}, \t{mae:.4f}, \t{mse:.4f}, \t{sd_ae:.4f}, \t{recall:.4f}, \t{recall_5:.4f}")

    # Create a figure with 4 subplots
    plt.style.use("dark_background")  
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True)

    # Plot input MIDI
    plot_MIDI(input_velocities, input_times, input_end_times, input_pitches, input_info, cfg,
              ax1=axes[0, 0], ax2=axes[1, 0])

    # Plot output MIDI
    plot_MIDI(output_velocities, output_times, output_end_times, output_pitches, output_info, cfg,
              ax1=axes[0, 1], ax2=axes[1, 1])

    fig.suptitle(f'Comparison of {input_info} and {output_info}', fontsize=16)
    plt.tight_layout()
    plt.show()


def compare_three_MIDI_files(input_mid, conv_mid, unet_mid, cfg):
    """Compare three MIDI files (Input, ConvAE, U-Net) by plotting them side by side and computing evaluation metrics."""
    evaluator = EvaluationMetrics()

    # Load input MIDI
    input_data = pretty_midi.PrettyMIDI(input_mid)
    input_times, input_end_times, input_velocities, input_pitches = extract_midi_data(input_data)
    input_info = f'Input MIDI: {os.path.basename(input_mid)}'

    # Load ConvAE output MIDI
    conv_data = pretty_midi.PrettyMIDI(conv_mid)
    conv_times, conv_end_times, conv_velocities, conv_pitches = extract_midi_data(conv_data)
    conv_info = f'ConvAE Output MIDI: {os.path.basename(conv_mid)}'

    # Load U-Net output MIDI
    unet_data = pretty_midi.PrettyMIDI(unet_mid)
    unet_times, unet_end_times, unet_velocities, unet_pitches = extract_midi_data(unet_data)
    unet_info = f'U-Net Output MIDI: {os.path.basename(unet_mid)}'

    # Compute evaluation metrics
    conv_mae, conv_mse, conv_sd_ae, conv_recall, conv_recall_5, conv_sd, sd_gt = evaluator.interface(input_velocities, conv_velocities)
    unet_mae, unet_mse, unet_sd_ae, unet_recall, unet_recall_5, unet_sd, _ = evaluator.interface(input_velocities, unet_velocities)

    print(f"SD of Human velocities: \t{sd_gt:.4f}")
    print(f"SD, MAE, MSE, SD_AE, Recall (10% & 5%) of ConvAE:\t{conv_sd:.4f}, \t{conv_mae:.4f}, \t{conv_mse:.4f}, \t{conv_sd_ae:.4f}, \t{conv_recall:.4f}, \t{conv_recall_5:.4f}")
    print(f"SD, MAE, MSE, SD_AE, Recall (10% & 5%) of U-Net: \t{unet_sd:.4f}, \t{unet_mae:.4f}, \t{unet_mse:.4f}, \t{unet_sd_ae:.4f}, \t{unet_recall:.4f}, \t{unet_recall_5:.4f}")
    
    # Create figure with 3 columns for Input, ConvAE, and U-Net
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, facecolor='none')

    # Plot input MIDI
    plot_MIDI(input_velocities, input_times, input_end_times, input_pitches, input_info, cfg,
              ax1=axes[0, 0], ax2=axes[1, 0])

    # Plot ConvAE output MIDI
    plot_MIDI(conv_velocities, conv_times, conv_end_times, conv_pitches, conv_info, cfg,
              ax1=axes[0, 1], ax2=axes[1, 1])

    # Plot U-Net output MIDI
    plot_MIDI(unet_velocities, unet_times, unet_end_times, unet_pitches, unet_info, cfg,
              ax1=axes[0, 2], ax2=axes[1, 2])

    fig.suptitle(f'Comparison of Human, ConvAE, and U-Net: {os.path.basename(input_mid)}', fontsize=16)
    plt.tight_layout()
    plt.show()


# ---------------- NOTE mir_eval: verify MIDI notes unchange --------------


def evaluate_midi_w_mireval(input_midi, output_midi):
    """Evaluate onset detection, pitch tracking, and note transcription using mir_eval.
    When parsing the MIDI, onset means the start time, pitch indicates 88 piano keys.
    """
    # Load MIDI data
    input_data = pretty_midi.PrettyMIDI(input_midi)
    output_data = pretty_midi.PrettyMIDI(output_midi)

    # Extract MIDI data
    input_start_times,  input_end_times, _,  input_pitches = extract_midi_data(input_data)
    output_start_times, output_end_times, _, output_pitches = extract_midi_data(output_data)

    # Evaluate Onsets
    onset_precision, onset_recall, onset_f_measure = mir_eval.onset.f_measure(np.array(sorted(input_start_times)), 
                                                                              np.array(sorted(output_start_times)))
    print(f"Onset Detection - Precision: {onset_precision:.4f}, Recall: {onset_recall:.4f}, F-measure: {onset_f_measure:.4f}")

    # Evaluate Pitches; Ensure input_onsets and output_onsets match for pitch tracking
    assert np.array_equal(input_start_times, output_start_times), "Input and output MIDI note times do not match exactly for pitch tracking."
    pitch_scores = mir_eval.melody.evaluate(np.array(input_start_times),  np.array(input_pitches),
                                            np.array(output_start_times), np.array(output_pitches))
    print(f"Pitch Tracking Scores: {pitch_scores}")

    # Evaluate Notes
    input_intervals =  np.array([[start, end] for start, end in zip(input_start_times,  input_end_times)])
    output_intervals = np.array([[start, end] for start, end in zip(output_start_times, output_end_times)])
    note_precision, note_recall, note_f_measure, note_overlap = mir_eval.transcription.precision_recall_f1_overlap(input_intervals,  np.array(input_pitches),
                                                                                                                   output_intervals, np.array(output_pitches))
    print(f"Note Transcription - Precision: {note_precision:.4f}, Recall: {note_recall:.4f}, F-measure: {note_f_measure:.4f}, Overlap: {note_overlap:.4f}")

# ---------------- NOTE Dataset Re-Organise --------------

def reorganise_dataset_by_composer(dataset_folder):
    """Organizes MIDI files in `dataset_folder` into subfolders based on composer names."""
    if not os.path.exists(dataset_folder):
        print(f"Error: Folder '{dataset_folder}' does not exist."); return

    midi_files = [f for f in os.listdir(dataset_folder) if f.endswith('.mid')]
    for midi_file in midi_files:
        composer_folder = os.path.join(dataset_folder, midi_file.split('_')[0])
        os.makedirs(composer_folder, exist_ok=True)
        shutil.move(os.path.join(dataset_folder, midi_file), os.path.join(composer_folder, midi_file))

    print(f"Organized {len(midi_files)} MIDI files in '{dataset_folder}'.")


# ---------------- NOTE Main Code for Runs NOTE -------------------------

def process_a_npy(model, model_name, input_npy, output_dir, cfg):
    """
    Process an npy file (representing a MIDI matrix) using the trained VAE/AE model.
    Assumes the npy file has shape [batch, channels, height, width] where channel 0 is the onset roll.
    Returns the reconstructed output and attention weights from the model.
    
    Args:
        model: Trained VAE/AE model.
        model_name: Model name string (used as a prefix for saved files).
        input_npy: Path to the input npy file.
        output_dir: Directory where the processed npy file and attention weights will be saved.
        cfg: Configuration object with preprocessing and model settings.
    
    Returns:
        output: The reconstructed output from the model.
        attention: The attention weights from the model.
    """
    import numpy as np
    import torch

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        data = np.load(input_npy)
    except Exception as e:
        print(f"Error loading {input_npy}: {e}")
        return
    
    print("Original npy shape:", data.shape)
    
    # Extract the onset roll (channel 0)
    onset_roll = data[:, 0, :, :]  # shape becomes [batch, height, width]
    print("Onset roll shape:", onset_roll.shape)
    
    # Convert to tensor and move to model's device
    input_tensor = torch.tensor(onset_roll, dtype=torch.float32)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Run the model with return_attention enabled.
    with torch.no_grad():
        if hasattr(model, 'show_attention'):
            output, attention = model.show_attention(input_tensor)
        else:
            print("Model does not support returning attention; using normal reconstruction.")
            output = model.reconstruct(input_tensor)
            attention = None
    
    # Save the reconstructed output as an npy file.
    output_path = os.path.join(output_dir, f"{model_name}_{os.path.basename(input_npy)}")
    np.save(output_path, output.cpu().numpy())
    print(f"Processed npy file saved to {output_path}")
    
    # Save the attention weights if available.
    if attention is not None:
        attn_path = os.path.join(output_dir, f"{model_name}_attention_{os.path.basename(input_npy)}.npz")
        # Create a list of numpy arrays from the attention tuple.
        attn_list = [att.cpu().numpy() for att in attention]
        # Create a dictionary with keys "arr_1", "arr_2", ..., corresponding to each attention array.
        names = {f"arr_{i+1}": att for i, att in enumerate(attn_list)}
        np.savez(attn_path, **names)
        print(f"Attention weights saved to {attn_path}")
    else:
        print("No attention weights available.")

    return output, attention



@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: OmegaConf) -> None:
    # --- Load the model only if needed ---
    if cfg.interface.process_a_midi or cfg.interface.process_a_folder or cfg.interface.process_a_npy:
        device = torch.device(f'cuda:{cfg.exp.devices[0]}' if cfg.exp.devices else 'cpu')
        model_name = cfg.ae.model  # e.g., "UNet" or "ConvAE"
        ckpt_path = cfg.exp.load_ckpt_path
        
        print(f"Loading {model_name} model from checkpoint: {ckpt_path}")
        model = LitModel.load_from_checkpoint(ckpt_path)
        model.to(device)
        model.eval()
    else:
        model = None

    # --- Segmentation Steps ---
    if cfg.interface.segment_a_midi:
        input_path, output_dir = cfg.interface.input_midi_path, cfg.interface.output_midi_dir  # e.g., "SMD-listen/Human_before"
        print(f"Segmenting MIDI: {input_path}")
        segment_a_midi(input_path, output_dir, cfg)

    if cfg.interface.segment_a_folder:
        input_dir, output_dir = cfg.interface.input_midi_dir, cfg.interface.output_midi_dir
        print(f"Segmenting MIDIs in directory: {input_dir}")
        segment_a_folder(input_dir, output_dir, cfg)

    # --- Processing Steps --- (require model)
    if cfg.interface.process_a_npy:
        input_npy, output_npy_dir = cfg.interface.input_npy_path, cfg.interface.output_npy_dir
        print(f"Processing npy file:\n Input: {input_npy}\n Output: {output_npy_dir}")
        process_a_npy(model, model_name, input_npy, output_npy_dir, cfg)

    if cfg.interface.process_a_midi:
        input_path, output_dir = cfg.interface.input_midi_path, cfg.interface.output_midi_dir
        print(f"Processing one MIDI:\n Input: {input_path}\n Output: {output_dir}")
        process_a_midi(model, model_name, input_path, output_dir, cfg)

    if cfg.interface.process_a_folder:
        input_dir, output_dir = cfg.interface.input_midi_dir, cfg.interface.output_midi_dir
        print(f"Processing MIDI folder:\n Input: {input_dir}\n Output: {output_dir}")
        process_a_folder(model, model_name, input_dir, output_dir, cfg)

    # --- Comparison Step ---
    if cfg.interface.compare_three_MIDI:
        midi_dir, midi_files =  cfg.interface.compare_midi_dir, cfg.interface.compare_midi_files
        human_dir = os.path.join(midi_dir, "Human")
        conv_dir = os.path.join(midi_dir, "ConvAE")
        unet_dir = os.path.join(midi_dir, "UNet")
        for midi_file in midi_files:
            input_mid = os.path.join(human_dir, midi_file)
            output_conv_mid = os.path.join(conv_dir, f"ConvAE_{midi_file}")
            output_unet_mid = os.path.join(unet_dir, f"UNet_{midi_file}")
            compare_three_MIDI_files(input_mid, output_conv_mid, output_unet_mid, cfg)
    
    # --- Plot Visualise ---
    if cfg.interface.plot_a_npy:
        input_npy, seg_index = cfg.interface.input_npy_path, cfg.interface.segment_id
        plot_processed_MIDI_npyfile(input_npy, seg_index, window_size=5)

    if cfg.interface.plot_a_npy_ver2:
        input_npy, output_npy, seg_index = cfg.interface.input_npy_path, cfg.interface.output_npy_path, cfg.interface.segment_id
        plot_processed_MIDI_npyfile_ver2(input_npy, output_npy, seg_index)

    if cfg.interface.plot_a_midi:
        midi_path = cfg.interface.input_midi_path
        print(f"Plot the MIDI: {midi_path}")
        plot_single_MIDI(midi_path, cfg)

    if cfg.interface.plot_a_folder:
        midi_dir, top_n = cfg.interface.input_midi_dir, cfg.interface.top_n
        print(f"Finding MIDI in folder: {midi_dir}")
        plot_topk_MIDI(midi_dir, top_n, cfg)
    
    if cfg.interface.plot_velocity_comparison:
        dataset1_dir = cfg.dataset.MAESTRO.dataset_dir # '../dataset/maestro-v3-midi/' 
        color1 = get_solid_colormap_color('Blues', intensity=0.9)
        dataset2_dir = cfg.dataset.SMD.dataset_dir # '../dataset/SMD-midi/'
        color2 = get_solid_colormap_color('Oranges', intensity=0.5)
        plot_velocity_comparison(dataset1_dir, dataset2_dir, color1, color2)

    if cfg.interface.plot_velocity_pitch_density_map:
        plot_velocity_pitch_density_map(cfg.dataset.MAESTRO.dataset_dir, cmap="Blues")
        plot_velocity_pitch_density_map(cfg.dataset.SMD.dataset_dir, cmap="Oranges")

if __name__ == "__main__":
    main()