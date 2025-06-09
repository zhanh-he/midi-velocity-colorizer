import torch
import numpy as np
import pretty_midi
from einops import rearrange

# import utils
from main import utils


def midi_generate_rolls(note_events, cfg, duration=None):
    """Adapt notes data within the range of piano key 21-108"""
    frames_num = cfg.matrix.resolution
    onset_roll = np.zeros((frames_num, cfg.matrix.bins))  # Now should be (frames_num, 88 or 91 if using pedals)
    velocity_roll = np.zeros((frames_num, cfg.matrix.bins))  # Now should be (frames_num, 88 or 91 if using pedals)
    frame_roll = np.zeros((frames_num, cfg.matrix.bins))  # Note roll for active notes, binary (0 or 1)

    if not note_events:
        return onset_roll, velocity_roll, frame_roll
    
    start_delta = int(min([n.start for n in note_events]))
    if not duration:
        duration = note_events[-1].end - note_events[0].start + 1
    frames_per_second = (cfg.matrix.resolution / duration)

    for note_event in note_events:
        """note_event: e.g., Note(start=1.009115, end=1.066406, pitch=40, velocity=93)"""
        if 21 <= note_event.pitch <= 108:  # Ensure the pitch is within the piano range
            pitch_index = note_event.pitch - 21  # Map the pitch to the correct index (0-87)
            bgn_frame = min(int(round((note_event.start - start_delta) * frames_per_second)), frames_num-1)
            fin_frame = min(int(round((note_event.end - start_delta) * frames_per_second)), frames_num-1)
            onset_roll[bgn_frame, pitch_index] = 1
            velocity_roll[bgn_frame : fin_frame + 1, pitch_index] = note_event.velocity
            frame_roll[bgn_frame : fin_frame + 1, pitch_index] = 1  # Mark the frames where the note is active

    return onset_roll, frame_roll, velocity_roll


def perfmidi_to_matrix(path, cfg):
    """Process performance MIDI events to roll matrices for training"""
    midi_data = pretty_midi.PrettyMIDI(path)

    # Some MIDI file is multiple tracks, containg notes of left/right hands, or different musical intruments, such as piano1 + piano2
    if len(midi_data.instruments) > 1:
        instruments = midi_data.instruments
        blended_instrument = pretty_midi.Instrument(program=0)
        for inst in instruments:
            blended_instrument.notes.extend(inst.notes)
            blended_instrument.control_changes.extend(inst.control_changes)
        midi_data.instruments.clear()
        midi_data.instruments.append(blended_instrument)
    
    note_events = midi_data.instruments[0].notes
    # pedal_events = midi_data.instruments[0].control_changes # NOTE: we are not consider the pedal

    duration = cfg.matrix.seg_time
    onset_roll, frame_roll, velocity_roll = [], [], []
    __onset_append, __frame_append, __velocity_append = onset_roll.append, frame_roll.append, velocity_roll.append # this make things faster..
    """get segment by time and produce rolls, aim for losing the cross segment events"""
    for i in range(0, int(midi_data.get_end_time()), cfg.matrix.seg_time):
        start, end = i, i + cfg.matrix.seg_time
        seg_note_events = [*filter(lambda n: (n.start > start and n.end < end), note_events)]  
        # seg_pedal_events = [*filter(lambda p: p.time > start and p.time < end, pedal_events)]
        seg_onset_roll, seg_frame_roll, seg_velocity_roll = midi_generate_rolls(seg_note_events, cfg, duration=duration)
        __onset_append(seg_onset_roll)
        __frame_append(seg_frame_roll)
        __velocity_append(seg_velocity_roll)

    # Stack the rolls into a tensor
    matrices = torch.tensor(np.array([onset_roll, frame_roll, velocity_roll]))
    matrices = rearrange(matrices, "c s f n -> s c f n") # stack them in channels: (sequence, channels=3, features, notes)
    return matrices # (s 3 h w)


def batch_to_matrix(batch, cfg, device):
    """Map the batch to input piano roll matrices
    Args:
        batch (2, b): ([path, path, ...], [label, label, ...])
    Returns: (matrix, label)
        matrix: (b, n_segs, n_channels, resolution, pitch_bins)
        label: (b, )
    """
    # if len(batch) == 2:
    files, labels = batch
    # else:
    #     files = batch
    #     labels = torch.tensor([fill_label] * len(files))
    b = len(batch[0])
    batch_matrix, batch_labels = [], []

    for idx, (path, l) in enumerate(zip(files, labels)):

        recompute = True
        if cfg.exp.load_data: # load existing data
            res = utils.load_data(path, cfg)
            if type(res) == np.ndarray: # keep computing if not exist
                seg_matrices =  res
                if cfg.matrix.n_channels == 1:
                    seg_matrices = seg_matrices[:, 1:, :, :]
                recompute = False
        
        # if 'asap-dataset/Schubert/Impromptu_op.90_D.899/4_' in path:
        #     recompute = True

        if recompute:
            seg_matrices = perfmidi_to_matrix(path, cfg)
            utils.save_data(path, seg_matrices, cfg)
        
        batch_matrix.append(seg_matrices)
        batch_labels.append(l)

    batch_matrix, batch_labels = utils.pad_batch(b, cfg,  device, batch_matrix, batch_labels)
    batch_matrix = torch.tensor(np.array(batch_matrix), device=device, dtype=torch.float32) 

    # assert(batch_matrix.shape == (b, cfg.exp.n_segs, cfg.matrix.n_channels,
    #                             int(cfg.matrix.resolution / cfg.exp.n_segs), 
    #                             cfg.matrix.bins,))
    
    return batch_matrix, batch_labels