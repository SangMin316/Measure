import os
import glob
import ntpath
import logging
import argparse
import mne
from scipy.signal import resample

import pyedflib
import numpy as np
import torch


# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
MOVE = 5
UNK = 6

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK
}

# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/Dataset/MASS/SS3/SS3_EDF/",
                        help="File path to the MASS 3 dataset.")
    parser.add_argument("--Annotation_dir", type=str, default="/Dataset/MASS/SS3/SS3_Annotations/",
                        help="File path to the MASS 3 dataset.")
    parser.add_argument("--output_dir", type=str, default="./MASS3",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG F4-LER",
                        help="Name of the channel in the dataset.")
    parser.add_argument("--log_file", type=str, default="info_ch_extract.log",
                        help="Log file.")
    args = parser.parse_args()


    # Output dir
    args.output_dir = os.path.join(args.output_dir, args.select_ch.split(' ')[-1])
    os.makedirs(args.output_dir, exist_ok=True)
    args.log_file = os.path.join(os.path.join(args.output_dir, args.log_file))

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation from EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Base.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    # psg_fnames = psg_fnames[:39]
    # ann_fnames = ann_fnames[:39]

    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))

        psg_f = pyedflib.EdfReader(psg_fnames[i])
        ann_f = pyedflib.EdfReader(ann_fnames[i])

        # assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
        start_datetime = psg_f.getStartdatetime()
        logger.info("Start datetime: {}".format(str(start_datetime)))

        file_duration = psg_f.getFileDuration()
        logger.info("File duration: {} sec".format(file_duration))
        epoch_duration = psg_f.datarecord_duration  # 2
        if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            epoch_duration = epoch_duration / 2
            logger.info("Epoch duration: {} sec (changed from 60 sec)".format(epoch_duration))
        else:
            logger.info("Epoch duration: {} sec".format(epoch_duration))

        # Extract signal from the selected channel
        ch_names = psg_f.getSignalLabels()
        ch_samples = psg_f.getNSamples()
        select_ch_idx = -1
        for s in range(psg_f.signals_in_file):
            if ch_names[s] == select_ch:
                select_ch_idx = s
                break
        if select_ch_idx == -1:
            raise Exception("Channel not found.")
        
        sampling_rate = int(psg_f.getSampleFrequencies()[select_ch_idx]/epoch_duration)
        # sampling_rate = 256

        print('sampling_rate:' ,sampling_rate)

        # n_epoch_samples = int(epoch_duration * sampling_rate)
        signals = psg_f.readSignal(select_ch_idx)
        print(signals.shape)
        logger.info("Select channel: {}".format(select_ch))
        logger.info("Select channel samples: {}".format(ch_samples[select_ch_idx]))
        logger.info("Sample rate: {}".format(sampling_rate))


        # Generate labels from onset and duration annotation
        labels = []
        ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()
        signals = signals[int(ann_onsets[0])*sampling_rate : int(ann_onsets[-1] + 30)*sampling_rate]
        print(signals.shape)
        print(len(ann_stages))
        print('ann_stages: ', len(ann_stages))
        target_freq = 100 
        # Down sampling 
        num_samples = int(len(signals) * (target_freq / sampling_rate))
        signals = resample(signals, num_samples)
        print(signals.shape)
        signals = signals.reshape(-1, target_freq*30)
        print(signals.shape)

        assert signals.shape[0] == len(ann_stages), f"signal: {signals.shape} != {len(ann_stages)}"

        for a in range(len(ann_stages)):
            ann_str = "".join(ann_stages[a])
            label = ann2label[ann_str]
            labels.append(label)
        labels = np.array(labels)

        x = signals.astype(np.float32)
        y = labels.astype(np.int32)
        print(x.shape)
        print(y.shape)


        # Remove movement and unknown
        move_idx = np.where(y == stage_dict["MOVE"])[0]
        unk_idx = np.where(y == stage_dict["UNK"])[0]
        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)
            logger.info("Remove irrelavant stages")
            logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
            logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
            logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
            logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
            select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
            x = x[select_idx]
            y = y[select_idx]
            logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))
    
        print('x: ', x.shape)
        print('y: ', y.shape)


        # after_Transition = 0
        # for label_i in range(len(y)-1):
        #     current_label = str(y[label_i])
        #     next_label = str(y[label_i + 1])
        #     if (current_label, next_label) in Object_label_mapping:
        #         Transition = Object_label_mapping[(current_label, next_label)]
        #         after_Transition = Transition + 1
        #     else:
        #         Transition = 0
            
        #     if Transition == 0 and after_Transition != 0:
        #         Transition = after_Transition
        #         after_Transition = 0
                
        #     t.append(Transition)
        # t.append(Transition)
        # t = torch.tensor(t)


        # Save
        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
        subject_N = int(filename[8:10])
        
        print('subject_N:',subject_N)
        save_dict = {
            "x": x, 
            "y": y,
            'subject_N': subject_N
        }
        
        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        logger.info("\n=======================================\n")


if __name__ == "__main__":
    main()