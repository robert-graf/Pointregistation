from pathlib import Path
from BIDS import NII, Ax_Codes
from typing import Literal
import nipy.algorithms.registration as nipy_reg
from nipy.algorithms.registration.affine import Affine


Similarity_Measures = Literal["slr", "mi", "pmi", "dpmi", "cc", "cr", "crl1"]
Affine_Transforms = Literal["affine", "affine2d", "similarity", "similarity2d", "rigid", "rigid2d"]


def registrate_nipy(
    moving: NII,
    fixed: NII,
    similarity: Similarity_Measures = "cc",
    optimizer: Affine_Transforms = "rigid",
    other_moving: list[NII] = [],
):
    hist_reg = nipy_reg.HistogramRegistration(fixed.nii, moving.nii, similarity=similarity)
    T: Affine = hist_reg.optimize(optimizer)

    aligned_img = nipy_reg.resample(moving.nii, T, fixed.nii, interp_order=0 if moving.seg else 3)
    aligned_img = fixed.set_array(aligned_img.get_data())
    aligned_img.seg = moving.seg

    out_arr = [fixed.set_array(nipy_reg.resample(i.nii, T, fixed.nii, interp_order=0 if i.seg else 3).get_data()) for i in other_moving]
    for out, other in zip(out_arr, other_moving):
        out.seg = other.seg
    return aligned_img, T, out_arr


def register_native_res(
    moving: NII,
    fixed: NII,
    similarity: Similarity_Measures = "cc",
    optimizer: Affine_Transforms = "rigid",
    other_moving: list[NII] = [],
) -> tuple[NII, NII, Affine, list[NII]]:
    """register an image to an other, with its native resolution of moving. Uses Global coordinates.

    Args:
        moving (NII): _description_
        fixed (NII): _description_
        similarity (Similarity_Measures, optional): _description_. Defaults to "cc".
        optimizer (Affine_Transforms, optional): _description_. Defaults to "rigid".

    Returns:
        (NII,NII): _description_
    """
    fixed_m_res = fixed.copy()
    fixed_m_res.resample_from_to_(moving)
    aligned_img, T, out_arr = registrate_nipy(moving, fixed_m_res, similarity, optimizer, other_moving)
    return aligned_img, fixed_m_res, T, out_arr


def crop_shared_(a: NII, b: NII):
    crop = a.compute_crop_slice()
    crop = b.compute_crop_slice(other_crop=crop)
    print(crop)
    a.apply_crop_slice_(crop)
    b.apply_crop_slice_(crop)
    return crop


if __name__ == "__main__":
    p = "/media/data/new_NAKO/NAKO/MRT/rawdata/105/sub-105013/"
    moving = NII.load(Path(p, "t1dixon", "sub-105013_acq-ax_rec-in_chunk-2_t1dixon.nii.gz"), False)
    fixed = NII.load(Path(p, "T2w", "sub-105013_acq-sag_chunk-LWS_sequ-31_T2w.nii.gz"), False)
    fixed.resample_from_to_(moving)
    # fixed.save("fixed_rep.nii.gz")
    aligned_img = registrate_nipy(moving, fixed)
