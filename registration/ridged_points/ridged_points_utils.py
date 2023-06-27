from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Literal
from BIDS import load_centroids, Ax_Codes
from BIDS.core.sitk_utils import nii_to_sitk


file = Path(__file__).resolve()
from BIDS import Logger_Interface, No_Logger

sys.path.append(str(file.parents[1]))

import numpy as np
import secrets
import nibabel as nib
import SimpleITK as sitk

from BIDS import NII, Centroids, Centroid_Reference, BIDS_FILE

from dataclasses import dataclass
import warnings


def extract_nii(d: dict[str, BIDS_FILE | list[BIDS_FILE]]) -> tuple[list[BIDS_FILE], list[str]]:
    """Extracts all BIDS_File from the family dict that contain a nii.gz

    Args:
        d (dict[str, BIDS_FILE | list[BIDS_FILE]]): a family bids dict

    Returns:
        tuple[list[BIDS_FILE],list[str]]: List of BIDS_FILS, List of unique keys
    """
    out = []
    keys = []
    for k, v in d.items():
        if isinstance(v, list):
            for i, l in enumerate(v):
                if "nii.gz" in l.file:
                    out.append(l)
                    keys.append(f"{k}_{i}")
        elif "nii.gz" in v.file:
            out.append(v)
            keys.append(k)

    return out, keys


def crop_slice(msk: np.ndarray):
    warnings.warn("ridged_points_utils.crop_slice is deprecated. Use BIDS.NII instead", DeprecationWarning)
    """Crops an Image or Mask to all areas with >0 Entries.

    Args:
        msk (np.ndarray): A numpy array that sould be cut

    Returns:
        slice, schift:
    """
    cor_msk = np.where(msk > 0)
    c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
    c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
    x0 = c_min[0]
    y0 = c_min[1]
    z0 = c_min[2]
    x1 = c_max[0]
    y1 = c_max[1]
    z1 = c_max[2]
    ex_slice = tuple([slice(z0, z1), slice(y0, y1), slice(x0, x1)])
    origin_shift = tuple([x0, y0, z0])
    return ex_slice, origin_shift


default_axcode_to = ("R", "P", "I")


def nii_to_iso_sitk_img2(in_file: BIDS_FILE | list[BIDS_FILE], tmp) -> sitk.Image:
    return nii_to_iso_sitk_img(in_file, tmp)[0]


def nii_to_iso_sitk_img(in_file: BIDS_FILE | list[BIDS_FILE], tmp=None) -> tuple[sitk.Image, NII, NII]:
    """Takes a BIDS_FILE with a nii.gz. It will be reoriented, spacing (1,1,1), saved in the tmp folder and loaded as a sitk.Image

    Raises:
        e: _description_

    Returns:
        sitk.Image | tuple[sitk.Image, NII, NII]
    """
    warnings.warn("ridged_points_utils.crop_slice is deprecated", DeprecationWarning)

    if isinstance(in_file, list):
        in_file = in_file[0]
    try:
        org = in_file.open_nii()
        ## Resample and rotate and Save Tempfiles
        iso = org.reorient(axcodes_to=default_axcode_to)
        iso.rescale_(voxel_spacing=(1, 1, 1), verbose=False)
        # The program throws an error when we have a translation in the affine.
        # We do not use the translation anyway...
        affine = iso.affine
        affine[0][-1] = 0
        affine[1][-1] = 0
        affine[2][-1] = 0
        if tmp is not None:
            file = Path(tmp, f"{secrets.token_urlsafe(20)}.nii.gz")
            iso.save(file)
            # Reload image as sitk
            out: sitk.Image = sitk.Cast(sitk.ReadImage(str(file)), sitk.sitkFloat32)
        else:
            out = nii_to_sitk(iso)
            out: sitk.Image = sitk.Cast(out, sitk.sitkFloat32)
        out.SetOrigin((0, 0, 0))
        return out, org, iso

    except Exception as e:
        print(f"[!] Failed nii_to_iso_sitk_img\n Input file: {in_file}")
        raise e


def sitk_img_to_nii(sitk_img: sitk.Image, tmp: Path | str, round: bool) -> nib.Nifti1Image:
    """Takes a sitk_img, saves it in tmp and loads it as a nifti

    Args:
        sitk_img (sitk.Image): The image
        tmp (Path): Temporarily path
        round (bool): round image, required for segmentation

    Raises:
        e

    Returns:
        nib.Nifti1Image:
    """
    from BIDS.core.sitk_utils import nib_to_sitk

    warnings.warn(
        "ridged_points_utils.sitk_img_to_nii is deprecated. Use from BIDS.sitk_utils import nib_to_sitk,nii_to_sitk instead",
        DeprecationWarning,
    )

    try:
        file = Path(tmp, f"{secrets.token_urlsafe(20)}_x.nii.gz")
        if round:
            sitk_img = sitk.Round(sitk_img)
        sitk.WriteImage(sitk_img, str(file))
        return nib.load(file)
    except Exception as e:
        print(f"[!] Failed sitk_img_to_nii\n")
        raise e


def reload_centroids(ctd_file: Centroid_Reference, img_a_org, img_a_iso) -> Centroids:
    try:
        # ctd_file is overloaded, can be a dict with BIDS, BIDS or a path
        ctd = load_centroids(ctd_file)
        if ctd.zoom == None:
            ctd.zoom = img_a_org.zoom
        ctd.rescale_((1, 1, 1))
        ctd.reorient_centroids_to_(img_a_iso)
        return ctd
    except Exception as e:
        print(f"[!] Failed reload_centroids\n\t{ctd_file}\n\t{str(e)}")
        raise e


@dataclass
class Transform_Points:
    img_moving: sitk.Image
    img_fixed: sitk.Image
    transform: sitk.VersorRigid3DTransform
    orientation: Ax_Codes

    def transform_points(self, ctd: Centroids, origins_shift2: tuple[slice, ...] | None):
        Move_L = []
        keys = []
        out = {k: value for k, value in zip(keys, Move_L)}

        for key, (x, y, z) in ctd.items():
            ctr_b = self.img_moving.TransformContinuousIndexToPhysicalPoint((x, y, z))
            ctr_b = self.transform.GetInverse().TransformPoint(ctr_b)
            ctr_b = self.img_fixed.TransformPhysicalPointToContinuousIndex(ctr_b)
            out[key] = ctr_b

        return Centroids(
            self.orientation,
            out,
            location=ctd.location,
            zoom=self.img_fixed.GetSpacing(),
            shape=self.img_fixed.GetSize(),
            sorting_list=ctd.sorting_list,
        ).shift_all_centroid_coordinates(origins_shift2)


def point_register(
    inter: list[int] | np.ndarray,
    ctd_f_iso: Centroids,
    img_fixed: sitk.Image,
    ctd_m_iso: Centroids,
    img_moving: sitk.Image,
    verbose=True,
    log: Logger_Interface = No_Logger(),
) -> tuple[sitk.ResampleImageFilter, sitk.ResampleImageFilter, Transform_Points, float, float]:
    assert len(inter) > 2
    # find shared points
    Move_L = []
    Fix_L = []
    # get real world coordinates of the corresponding vertebrae
    for key in inter:
        ctr_mass_b = ctd_m_iso[key]
        ctr_b = img_moving.TransformContinuousIndexToPhysicalPoint((ctr_mass_b[0], ctr_mass_b[1], ctr_mass_b[2]))
        Move_L.append(ctr_b)
        ctr_mass_f = ctd_f_iso[key]
        ctr_f = img_fixed.TransformContinuousIndexToPhysicalPoint((ctr_mass_f[0], ctr_mass_f[1], ctr_mass_f[2]))
        Fix_L.append(ctr_f)
    log.print("[*] used centroids:", inter, verbose=verbose)
    # Rough registration transform
    moving_image_points_flat = [c for p in Move_L for c in p if not math.isnan(c)]
    fixed_image_points_flat = [c for p in Fix_L for c in p if not math.isnan(c)]
    init_transform = sitk.VersorRigid3DTransform(
        sitk.LandmarkBasedTransformInitializer(sitk.VersorRigid3DTransform(), fixed_image_points_flat, moving_image_points_flat)
    )

    x_old = Fix_L[0]
    y_old = Move_L[0]
    error_reg = 0
    error_natural = 0
    err_count = 0
    err_count_n = 0
    log.print(
        f'{"key": <3}|{"fixed points": <23}|{"moved points after": <23}|{"moved points before": <23}|{"delta fixed/moved": <23}|{"distF": <5}|{"distM": <5}|',
        verbose=verbose,
    )
    k_old = -1000
    for k, x, y in zip(inter, np.round(Fix_L, decimals=1), np.round(Move_L, decimals=1)):
        y2 = init_transform.GetInverse().TransformPoint(y)
        y = [round(m, ndigits=1) for m in y]
        dif = [round(i - j, ndigits=1) for i, j in zip(x, y2)]

        dist = round(math.sqrt(sum([(i - j) ** 2 for i, j in zip(x, x_old)])), ndigits=1)
        dist2 = round(math.sqrt(sum([(i - j) ** 2 for i, j in zip(y, y_old)])), ndigits=1)
        x_ = f"{x[0]:7.1f},{x[1]:7.1f},{x[2]:7.1f}"
        y_ = f"{y[0]:7.1f},{y[1]:7.1f},{y[2]:7.1f}"
        y2_ = f"{y2[0]:7.1f},{y2[1]:7.1f},{y2[2]:7.1f}"
        d_ = f"{dif[0]:7.1f},{dif[1]:7.1f},{dif[2]:7.1f}"
        error_reg += math.sqrt(sum([i * i for i in dif]))
        err_count += 1

        if k - k_old < 50:
            error_natural += abs(dist - dist2)
            err_count_n += 1
        else:
            dist = ""
            dist2 = ""

        log.print(f"{str(k): <3}|{x_: <23}|{y2_: <23}|{y_: <23}|{d_: <23}|{str(dist): <5}|{str(dist2): <5}|", verbose=verbose)

        x_old = x
        y_old = y
        k_old = k

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img_fixed)

    # resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetInterpolator(sitk.sitkBSplineResampler)
    resampler.SetTransform(init_transform)
    resampler_seg = sitk.ResampleImageFilter()
    resampler_seg.SetReferenceImage(img_fixed)
    resampler_seg.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_seg.SetTransform(init_transform)
    error_reg /= max(err_count, 1)
    error_natural /= max(err_count_n, 1)
    log.print(f"Error avg registration error-vector length: {error_reg: 7.3f}", verbose=verbose)
    log.print(f"Error avg point-distances: {error_natural: 7.3f}", verbose=verbose)
    return (
        resampler,
        resampler_seg,
        Transform_Points(img_moving, img_fixed, init_transform, ctd_f_iso.orientation),
        error_reg,
        error_natural,
    )


if __name__ == "__main__":
    pass
    # a = np.zeros((100, 10, 10))
    # a = nib.Nifti1Image(a, np.eye(4))
    # a = nii_to_sitk(NII(a))
    # img = sitk.GetImageFromArray(np.random.rand(10, 10, 10))
    #
    ## ix = sitk.BSplineTransformInitializer(a, (2, 2, 2), order=3)
    ## sitk.LandmarkBasedTransformInitializer(ix, [(1, 2, 3)], [(2, 3, 4)])
    # ix = sitk.BSplineTransformInitializer(img, (2, 2, 2), order=3)
    ## sitk.LandmarkBasedTransformInitializer(ix, np.array([[1.0, 2.0, 3.0]], dtype=np.float32), np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    # LandmarkTx = sitk.LandmarkBasedTransformInitializerFilter()
    # ctr_b = [
    #    img.TransformContinuousIndexToPhysicalPoint((1, 2, 2)),
    #    img.TransformContinuousIndexToPhysicalPoint((5, 3, 3)),
    #    img.TransformContinuousIndexToPhysicalPoint((2, 3, 3)),
    #    img.TransformContinuousIndexToPhysicalPoint((9, 1, 1)),
    # ]
    # moving_image_points_flat = [c for p in ctr_b for c in p if not math.isnan(c)]
    # LandmarkTx.SetFixedLandmarks(moving_image_points_flat)
    # LandmarkTx.SetMovingLandmarks(moving_image_points_flat)
    # LandmarkTx.SetBSplineNumberOfControlPoints(4)
    # LandmarkTx.SetReferenceImage(img)
    # InitialTx = LandmarkTx.Execute(ix)
    # print(InitialTx)
