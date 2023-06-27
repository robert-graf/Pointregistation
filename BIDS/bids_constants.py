### Legal formats ###
formats = [
    "ct",
    "dixon",
    "t1dixon",
    "snp",
    "log",
    "msk",
    "ctd",
    "T2c",
    "T1c",
    "pd",
    "mevibe",
    "T1w",
    "T2w",
    "T2star",
    "STIR",
    "model",
]
formats_relaxed = formats +[
    "t2",
    "t1",
    "t2c",
    "t1c",
    "cta",
    "mr", "snapshot"
]
# Recommended writing style: T1c, T2c; This list is not official and can be extended.


# Actual official final folder
# func (task based and resting state functional MRI)
# dwi (diffusion weighted imaging)
# fmap (field inhomogeneity mapping data such as field maps)
# anat (structural imaging such as T1, T2, PD, and so on)
# perf (perfusion)
# meg (magnetoencephalography)
# eeg (electroencephalography)
# ieeg (intracranial electroencephalography)
# beh (behavioral)
# pet (positron emission tomography)
# micr (microscopy)


file_types = ["nii.gz", "json", "png", "jpg", "tsv", "backup", "ply", "npz", "log", "txt", "stl", "csv"]
# Description see: https://bids-specification.readthedocs.io/en/stable/99-appendices/09-entities.html
entities = {
    # Person
    "Subject": "sub",
    "Session": "ses",
    # Recording
    ## OURS
    "Sequence": "sequ",  # Not BIDS conform :( Try to replace and remove this key
    # Should be acq+chunk+ce*+trc*+rec*+mod*+run instead. where * is used only im applicable.
    # Run is used when the file collides with an other. Run gets a increasing number
    ## Official
    "Acquisition": "acq",
    # sag, ax, iso usw. and In case different sequences are used to record the same modality (for example, RARE and FLASH for T1w)
    # Examples: sag, sag-RARE, RARE
    "Task": "task",
    "Chunk/Location": "chunk",  # Location or index
    "Hemisphere": "hemi",  # [L,R]
    "Sample": "sample",  # such as tissue, primary cell or cell-free sample.
    # Sub recordings - Use when necessary.
    ## Contrast
    "Contrast enhancement phase": "ce",
    "Tracer": "trc",  # use ce before this one.
    "stain": "stain",  # like trc,ce but for stains/antibodies for contrast
    ## Reconstruction and modalities
    "Reconstruction": "rec",
    "Processed (on device)": "proc",  # TODO: possible place for the reconstruction kernels für CT? alternative rec
    "Corresponding Modality": "mod",  # Use only sublementiral material. (aka this belongs to ..._[mod].nii.gz)
    "Recording": "recording",
    "Resolution": "res",
    ### MRI parameters (Never used)
    "Reconstruction Direction": "dir",
    "Echo": "echo",  # Index!
    "Flip Angle": "flip",  # Index!
    "Inversion Time": "inv",  # Index!
    "Magnetization Transfer": "mt",  # [on, off]
    "Part": "part",  # [mag, phase, real, imag]
    "Space": "space",
    ## OURS
    "Segmentation": "seg",
    "Source": "source",
    # "snapshot": "snapshot",
    "ovl": "ovl",
    # General parameters
    # Others (often used)
    "Run/ID": "run",
    # Single class segmentation
    "Label": "label",
    # Others (never used)
    "Split": "split",
    "Density": "den",
    "Description": "desc",
}
entities_keys = {v: k for k, v in entities.items()}

parents = [
    "sourcedata",
    "rawdata",
    "derivatives",
]

sequence_splitting_keys = [
    "ses",
    "sequ",
    "acq",
    "hemi",
    "sample",
    "ce",
    "trc",
    "stain",
    # "rec",
    # "proc",
    "res",
    "run",
    # "desc",
    "split",
    "chunk",
]
sequence_naming_keys = ["seg", "label"]
# sequence_mix_format_and_naming_formats = ["ctd", "msk"]
