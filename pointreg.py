from __future__ import annotations
import sys
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import itertools
import nibabel as nib
from BIDS.bids_files import BIDS_FILE
from BIDS.vert_constants import Ax_Codes

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from BIDS import No_Logger
from BIDS.core.sitk_utils import nib_to_sitk, nii_to_sitk, sitk_to_nib, sitk_to_nii
from BIDS import NII, Centroids, Image_Reference, calc_centroids_from_subreg_vert, to_nii_seg, to_nii
from registration.ridged_points.ridged_points_utils import point_register, Transform_Points
from BIDS.snapshot2D.snapshot_modular import LABEL_MAX


def ridged_from_points(
    ctd_fixed: Centroids,
    ctd_movig: Centroids,
    *,
    representative_fixed: Image_Reference = None,
    representative_movig: Image_Reference = None,
    exclusion=[],
    log=No_Logger(),
    verbose=True,
) -> tuple[None, None, None] | tuple[sitk.ResampleImageFilter, sitk.ResampleImageFilter, Transform_Points]:
    # Register
    # filter points by name
    f_keys = list(filter(lambda x: x % LABEL_MAX not in exclusion, ctd_fixed.keys()))
    m_keys = list(ctd_movig.keys())
    print(f_keys)
    print(to_nii(representative_movig).orientation)

    representative_fixed = nii_to_sitk(to_nii(representative_fixed))
    representative_movig = nii_to_sitk(to_nii(representative_movig))
    print(representative_movig.GetDirection())
    # limit to only shared labels
    inter = np.intersect1d(m_keys, f_keys)
    if len(inter) <= 2:
        log.print("[!] To few points, skip registration")
        return None, None, None
    resampler, resampler_seg, transform, _, _ = point_register(
        inter, ctd_fixed, representative_fixed, ctd_movig, representative_movig, log=log, verbose=verbose
    )

    return resampler, resampler_seg, transform


def ridged_segmentation_from_seg(
    subreg_fixed: Image_Reference,
    vert_fixed: Image_Reference,
    subreg_moving: Image_Reference,
    vert_moving: Image_Reference,
    *,
    representative_fixed=None,
    representative_movig=None,
    ids=[50, 42],
    exclusion=[],
    log=No_Logger(),
    verbose=True,
):
    ref = (subreg_fixed, vert_fixed, subreg_moving, vert_moving)
    # load nii
    niis = tuple(map(to_nii_seg, ref))
    # generate sitk images
    sitk_list = tuple(map(nii_to_sitk, niis))
    # Calculate Centroids
    log.print("[*] Calc centroids ", subreg_fixed.info["sub"]) if isinstance(subreg_fixed, BIDS_FILE) else None
    cdt_fixed = calc_centroids_from_subreg_vert(niis[1], niis[0], subreg_id=ids, fixed_offset=256, verbose=verbose)
    cdt_movig = calc_centroids_from_subreg_vert(niis[3], niis[2], subreg_id=ids, fixed_offset=256, verbose=verbose)
    log.print("[*] Calc Point registration ", subreg_fixed.info["sub"]) if isinstance(subreg_fixed, BIDS_FILE) else None
    # Register
    # filter points by name

    f_keys = list(filter(lambda x: x % 256 not in exclusion, cdt_fixed.keys()))
    m_keys = list(cdt_movig.keys())
    print(f_keys)
    representative_fixed = nii_to_sitk(to_nii(representative_fixed)) if representative_fixed is not None else sitk_list[1]
    representative_movig = nii_to_sitk(to_nii(representative_movig)) if representative_movig is not None else sitk_list[3]
    # limit to only shared labels
    inter = np.intersect1d(m_keys, f_keys)
    if len(inter) <= 2:
        log.print("[!] To few points, skip registration")
        return None, None, None
    resampler, resampler_seg, transform, _, _ = point_register(
        inter,
        cdt_fixed,
        representative_fixed,
        cdt_movig,
        representative_movig,
        log=log,
        verbose=verbose,
    )

    return resampler, resampler_seg, transform


def compute_crop(fixed: Image_Reference, moving: Image_Reference, resampler: sitk.ResampleImageFilter, dist=0):
    img_nii = to_nii(moving)
    img_sitk = nii_to_sitk(img_nii)
    transformed_img = resampler.Execute(img_sitk)
    # Crop the scans to the registered regions
    ex_slice_f = to_nii(fixed).compute_crop_slice(dist=dist)
    ex_slice = sitk_to_nii(transformed_img, img_nii.seg).compute_crop_slice(dist=dist, other_crop=ex_slice_f)
    return tuple(ex_slice)


def resample(
    resampler: sitk.ResampleImageFilter,
    resampler_seg: sitk.ResampleImageFilter,
    moving_img: Image_Reference,
    slice: None | tuple[slice, ...] | nib.Nifti1Image | NII = None, #type:ignore
    is_seg=False,
):
    img_nii = to_nii(moving_img)
    img_sitk = nii_to_sitk(img_nii)
    if is_seg:
        transformed_img: sitk.Image = resampler_seg.Execute(img_sitk)
    else:
        transformed_img = resampler.Execute(img_sitk)
    if is_seg:
        transformed_img = sitk.Round(transformed_img)
    if slice is None:
        pass
    elif isinstance(slice, tuple):
        transformed_img = transformed_img[slice]
    else:
        if not isinstance(slice, NII):
            slice = to_nii(slice)
        ex_slice, _ = slice.compute_crop_slice()
        transformed_img = transformed_img[ex_slice]
    out_nii = sitk_to_nii(transformed_img, seg=is_seg)
    return out_nii


# Save registered file
def generate_file_path_reg(file: BIDS_FILE, other_file_id, folder, keys=["sub", "ses", "sequ", "reg", "acq"]):
    info = file.info.copy()
    info["reg"] = other_file_id
    file.info.clear()

    def pop(key):
        if key in info:
            file.info[key] = info.pop(key)

    for k in keys:
        pop(k)
    for k, v in info.items():
        file.info[k] = v
    out_file: Path = file.get_changed_path(
        file_type="nii.gz",
        parent="registration",
        path="{sub}/" + folder,
        info=info,
        from_info=True,
    )
    return out_file


###########################################################################################################################################
def ridged_registration_by_dict_from_seg(
    dict_fixed: dict[str, list[BIDS_FILE]],
    key_fixed: str,
    dict_moving: dict[str, list[BIDS_FILE]],
    key_moving: str,
    ids=[50, 42],
    exclusion=[],
    log=No_Logger(),
    verbose=True,
    generate_file_names=generate_file_path_reg,
    identifier_key="sequ",
    make_snapshot=True,
    save=True,
    axcodes_to=("P", "S", "R"),
    voxel_spacing=(-1, -1, -1),
):
    # rescale_and_reorient_fixed: Callable[[Image_Reference], Image_Reference] = lambda x: x,  # rescale_and_reorient

    rep_fixed = dict_fixed[key_fixed][0]
    rep_movig = dict_moving[key_moving][0]
    log.print("[*] Register ", rep_fixed.info["sub"])

    #### RESAMPLE NII ####
    rep_fixed_nii = to_nii(rep_fixed, seg=False).rescale_and_reorient_(axcodes_to=axcodes_to, voxel_spacing=voxel_spacing, verbose=verbose)
    subreg_fixed = to_nii(dict_fixed["subreg"][0], seg=True).rescale_and_reorient(axcodes_to, voxel_spacing)
    vert_fixed = to_nii(dict_fixed["vert"][0], seg=True).rescale_and_reorient(axcodes_to, voxel_spacing)

    resampler, resampler_seg, transform = ridged_segmentation_from_seg(
        subreg_fixed,
        vert_fixed,
        dict_moving["subreg"][0],
        dict_moving["vert"][0],
        representative_fixed=rep_fixed_nii,
        representative_movig=rep_movig,
        ids=ids,
        exclusion=exclusion,
        log=log,
        verbose=verbose,
    )
    fixed_id = rep_fixed.info[identifier_key]
    moving_id = rep_movig.info[identifier_key]
    folder_name = f"target-{fixed_id}_from-{moving_id}"
    log.print("[*]", folder_name, rep_fixed.info["sub"])
    if resampler is None:
        return
    if resampler_seg is None:
        return
    reg_fixed: dict[str, list[NII]] = {}
    reg_moving: dict[str, list[NII]] = {}

    rep_moving_nii: NII = resample(resampler, resampler_seg, moving_img=rep_movig)
    ex_slice_f = rep_fixed_nii.compute_crop_slice()
    ex_slice = rep_moving_nii.compute_crop_slice(other_crop=ex_slice_f)
    for key, value in dict_moving.items():
        reg_moving[key] = []
        for bids_file in value:
            if not bids_file.has_nii():
                continue

            nii_out = resample(
                resampler,
                resampler_seg,
                moving_img=bids_file,
                slice=ex_slice,
                is_seg=bids_file.get_interpolation_order() == 0,
            )
            if save:
                out_name = generate_file_names(bids_file, fixed_id, folder_name)
                log.print("[*] Save (moving)", key, bids_file.info["sub"], bids_file.info["sequ"])
                nib.save(nii_out,out_name)#type:ignore
            reg_moving[key].append(nii_out)
    for key, value in dict_fixed.items():
        reg_fixed[key] = []
        for bids_file in value:
            if not bids_file.has_nii():
                continue
            nii = to_nii(bids_file, bids_file.get_interpolation_order() == 0).rescale_and_reorient_(axcodes_to, voxel_spacing)

            if ex_slice is not None:
                nii.apply_crop_slice_(ex_slice)
            if save:
                out_name = generate_file_names(bids_file, moving_id, folder_name)
                log.print("[*] Save (fixed)", key, bids_file.info["sub"], bids_file.info["sequ"])
                nib.save(nii, out_name) #type:ignore
            reg_fixed[key].append(nii)

    if make_snapshot:
        from BIDS.snapshot2D.snapshot_modular import Snapshot_Frame, create_snapshot

        out_name = generate_file_names(rep_fixed, moving_id, folder_name)
        a = [
            Snapshot_Frame(
                image=reg_fixed[key_fixed][0],
                segmentation=reg_fixed["vert"][0],
                centroids=None,
                coronal=True,
                axial=True,
                mode="CT" if key_fixed.lower().count("ct") != 0 else "MRI",
                force_show_cdt=True,
                # crop_img=True,
            ),
            Snapshot_Frame(
                image=reg_fixed[key_fixed][0],
                segmentation=reg_fixed["subreg"][0],
                centroids=None,
                coronal=True,
                axial=True,
                mode="CT" if key_fixed.lower().count("ct") != 0 else "MRI",
                # crop_msk=True,
            ),
            Snapshot_Frame(
                image=reg_moving[key_moving][0],
                segmentation=reg_fixed["subreg"][0],
                centroids=None,
                coronal=True,
                axial=True,
                mode="CT" if key_moving.lower().count("ct") != 0 else "MRI",
            ),
            Snapshot_Frame(
                image=reg_moving[key_moving][0],
                segmentation=reg_moving["vert"][0],
                centroids=None,
                coronal=True,
                axial=True,
                mode="CT" if key_moving.lower().count("ct") != 0 else "MRI",
                force_show_cdt=True,
            ),
        ]
        folder = Path(rep_fixed.dataset, "snapshot-registration")
        folder.mkdir(exist_ok=True, parents=True)
        create_snapshot([Path(out_name.parent, "snapshot.jpg"), f"{str(folder)}/{rep_fixed.info['sub']}.jpg"], a, crop=True)
    return reg_fixed, reg_moving


###########################################################################################################################################


def ridged_registration_by_dict_from_ctd(
    list_fixed: list[BIDS_FILE],
    list_moving: list[BIDS_FILE],
    ctd_fixed: Centroids,
    ctd_movig: Centroids,
    exclusion=[],
    log=No_Logger(),
    verbose=True,
    generate_file_names=generate_file_path_reg,
    identifier_key="sequ",
    make_snapshot=True,
    save=True,
    axcodes_to=("P", "S", "R"),
    voxel_spacing=(-1, -1, -1),
    snap_shot_folder: str | Path | None = None,
    override=False,
) -> tuple[list[NII], list[NII]] | None:
    
    seg = [s.file["nii.gz"] for s in list_fixed if "nii.gz" in s.file and "seg-vert_msk" in str(s.file["nii.gz"])]
    rep_fixed = list_fixed[0]
    rep_movig = list_moving[0]
    fixed_id = rep_fixed.info[identifier_key]
    moving_id = rep_movig.info[identifier_key]
    folder_name = f"target-{fixed_id}_from-{moving_id}"
    out_name = generate_file_names(rep_movig, fixed_id, folder_name)

    log.print("[*] Register ", rep_fixed.info["sub"])
    #### RESAMPLE NII ####

    if Path(out_name.parent, "snapshot.jpg").exists() and not override:
        print("[#] skip", rep_fixed.info["sub"])
        return

    rep_fixed_nii = to_nii(rep_fixed).reorient_(axcodes_to).rescale_(voxel_spacing)
    ctd_fixed.reorient_(rep_fixed_nii.orientation).rescale_(rep_fixed_nii.zoom)

    #### Compute resampler through Points ####
    resampler, resampler_seg, transform = ridged_from_points(
        ctd_fixed,
        ctd_movig,
        representative_fixed=rep_fixed_nii,
        representative_movig=rep_movig,
        exclusion=exclusion,
        log=log,
        verbose=verbose,
    )

    log.print("[*]", folder_name, rep_fixed.info["sub"])
    if resampler is None:
        return
    if resampler_seg is None:
        return

    reg_fixed: list[NII] = []
    reg_moving: list[NII] = []
    #### Slicing
    rep_moving_nii: NII = resample(resampler, resampler_seg, moving_img=rep_movig)
    ex_slice_f = rep_fixed_nii.compute_crop_slice()
    ex_slice = rep_moving_nii.compute_crop_slice(other_crop=ex_slice_f)
    ctd_fixed.shift_all_centroid_coordinates(ex_slice)
    #### resample
    for value in list_moving:
        for bids_file in value:
            if not bids_file.has_nii():
                continue

            nii_out: NII = resample(
                resampler,
                resampler_seg,
                moving_img=bids_file,
                slice=ex_slice,
                is_seg=bids_file.get_interpolation_order() == 0,
            )
            if save:
                out_name = generate_file_names(bids_file, fixed_id, folder_name)
                log.print("[*] Save (moving)", bids_file.format, bids_file.info["sub"], bids_file.info["sequ"])
                if "seg-vert_msk" in str(out_name):
                    seg.append(out_name)
                nii_out.save(out_name)
            reg_moving.append(nii_out)
    for value in list_fixed:
        for bids_file in value:
            if not bids_file.has_nii():
                continue
            nii = to_nii(bids_file).reorient_(axcodes_to).rescale_(rep_fixed_nii.zoom)
            if ex_slice is not None:
                nii.apply_crop_slice_(ex_slice)
            if save:
                out_name = generate_file_names(bids_file, moving_id, folder_name)
                log.print("[*] Save (fixed)", bids_file.format, bids_file.info["sub"], bids_file.info["sequ"])
                nii.save(out_name)
            reg_fixed.append(nii)

    if make_snapshot:
        from BIDS.snapshot2D.snapshot_modular import Snapshot_Frame, create_snapshot

        out_name = generate_file_names(rep_fixed, moving_id, folder_name)

        if transform is not None:
            ctd = transform.transform_points(ctd_movig, ex_slice)
        else:
            ctd = ctd_fixed
        # print(seg)
        seg = []
        a = [
            Snapshot_Frame(
                image=reg_fixed[0],
                segmentation=seg[0] if len(seg) != 0 else None,
                centroids=ctd_fixed,
                coronal=True,
                axial=True,
                mode="CT" if rep_fixed.format.lower().count("ct") != 0 else "MRI",
                force_show_cdt=True,
                # crop_img=True,
            ),
            Snapshot_Frame(
                image=reg_moving[0],
                segmentation=seg[0] if len(seg) != 0 else None,
                centroids=ctd,
                coronal=True,
                axial=True,
                mode="CT" if rep_movig.format.lower().count("ct") != 0 else "MRI",
                # crop_msk=True,
            ),
        ]

        folder = Path(snap_shot_folder if snap_shot_folder is not None else rep_fixed.dataset, "snapshot-registration")
        folder.mkdir(exist_ok=True, parents=True)
        create_snapshot(
            [
                Path(out_name.parent, "snapshot.jpg"),
                f"{str(folder)}/{rep_fixed.info['sub']}_{fixed_id}_{moving_id}.jpg",
            ],
            a,
            crop=False,
        )
    return reg_fixed, reg_moving






def ridged_registration_cdt(
    ctd_fixed: Centroids,
    ctd_movig: Centroids,
    list_fixed: list[NII],
    list_moving: list[NII],
    exclusion=[],
    log=No_Logger(),
    verbose=True,
    axcodes_to:None|Ax_Codes=None,
    voxel_spacing=(-1, -1, -1),
) -> tuple[list[NII], list[NII], tuple[sitk.ResampleImageFilter, sitk.ResampleImageFilter, Transform_Points,tuple[slice,slice,slice]]]:
    
    rep_fixed = list_fixed[0]
    axcodes_to = rep_fixed.orientation if axcodes_to is None else axcodes_to
    assert axcodes_to is not None
    rep_movig = list_moving[0]
    log.print("[*] Register to", rep_fixed)
    #### RESAMPLE NII ####
    if ctd_fixed.zoom is None:
        rep_fixed= to_nii(to_nii(rep_fixed))
        ctd_fixed.zoom =rep_fixed.zoom
        ctd_fixed.orientation =rep_fixed.orientation
    rep_fixed_nii = to_nii(rep_fixed).reorient_(axcodes_to).rescale_(voxel_spacing)
    ctd_fixed.reorient_(rep_fixed_nii.orientation).rescale_(rep_fixed_nii.zoom)

    #### Compute resampler through Points ####
    resampler, resampler_seg, transform = ridged_from_points(
        ctd_fixed,
        ctd_movig,
        representative_fixed=rep_fixed_nii,
        representative_movig=rep_movig,
        exclusion=exclusion,
        log=log,
        verbose=verbose,
    )

    if resampler is None:
        raise ValueError()
    if resampler_seg is None:
        raise ValueError()
    if transform is None:
        raise ValueError()
    log.print("[*] Resample Moving to Target")
    reg_fixed: list[NII] = []
    reg_moving: list[NII] = []
    #### Slicing
    rep_moving_nii: NII = resample(resampler, resampler_seg, moving_img=rep_movig)
    ex_slice_f = rep_fixed_nii.compute_crop_slice()
    ex_slice = rep_moving_nii.compute_crop_slice(other_crop=ex_slice_f)
    #### resample
    for nii in list_moving:    
        nii_out: NII = resample(
            resampler,
            resampler_seg,
            moving_img=nii,
            slice=ex_slice,
            is_seg=nii.seg,
        )
        reg_moving.append(nii_out)
    for nii in list_fixed:
        nii = to_nii(nii).reorient_(axcodes_to).rescale_(rep_fixed_nii.zoom)
        if ex_slice is not None:
            nii.apply_crop_slice_(ex_slice)
        reg_fixed.append(nii)
    return reg_fixed, reg_moving, (resampler, resampler_seg, transform,ex_slice)
