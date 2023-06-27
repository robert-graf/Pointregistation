from __future__ import annotations
import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
from registration.reg_logger import Logger, No_Logger


import csv
import math
import os


from sitk_utils import to_str_sitk
import SimpleITK as sitk
import secrets
import traceback

# sys.path.append("..")
from BIDS.bids_files import BIDS_FILE, BIDS_Global_info, Searchquery, Subject_Container
import numpy as np
from BIDS.nii_utils import (
    Centroid_List,
    calc_centroids_from_file,
    calc_centroids_from_subreg_vert,
    is_Orientation,
    is_Orientation_tuple,
    is_Point3D,
    is_Point3D_tuple,
    to_dict,
    reorient_to,
    resample_nib,
    load_centroids,
    rescale_centroids,
    reorient_centroids_to,
    calc_centroids,
    centroids_to_dict_list,
    save_json,
    Orientation,
)
from BIDS.registration.ridged_points.ridged_points_utils import (
    extract_nii,
    crop_slice,
    nii_to_iso_sitk_img,
    nii_to_iso_sitk_img2,
    sitk_img_to_nii,
    default_axcode_to,
    reload_centroids,
    point_register,
)
import nibabel as nib

error_dict = {}


def __find_representative(u_dict: dict[str, BIDS_FILE | list[BIDS_FILE]]):
    pos_rep = ["t1", "t2", "dixon", "ct", "T1c", "T2c"]
    for key, value in u_dict.items():
        if any([key.lower() in i for i in pos_rep]):
            if isinstance(value, list):
                for v in value:
                    j = v.open_json()
                    if j is not None and "IP" in j["ImageType"]:
                        return v

                return value[0]
            return value
    print("No supported representative file")
    raise NotImplemented()


def __find_centroids(u_dict: dict[str, BIDS_FILE | list[BIDS_FILE]]) -> BIDS_FILE:
    if "ctd" in u_dict:
        # print('Fund subreg',u_dict["cdt"])
        cdt = u_dict["ctd"]
        if isinstance(cdt, list):
            return cdt[0]
        else:
            return cdt
    if "msk" in u_dict:
        msk_bids = u_dict["msk"]
        if isinstance(msk_bids, list):
            msk_bids = msk_bids[0]
        cdt_file: Path = msk_bids.get_changed_path(
            file_type="json",
            format="ctd",
            info={"seg": "subreg"},
            parent="derivatives_msk",
        )
        calc_centroids_from_file(msk_bids.file["nii.gz"], cdt_file)
        bf = BIDS_FILE(cdt_file, msk_bids.dataset, verbose=False)
        u_dict["subreg"] = bf
        print(bf)
        return bf

    print("No supported representative file")
    raise NotImplemented()


def ridged_point_registration(
    fixed_dict: dict[str, BIDS_FILE | list[BIDS_FILE]], movig_dict: dict[str, BIDS_FILE | list[BIDS_FILE]], verbose=True
):
    global error_dict
    log = No_Logger()
    tmp = os.path.join(os.getcwd(), f"temp_{secrets.token_urlsafe(22)}")
    try:

        if not Path(tmp).exists():
            Path(tmp).mkdir()
        # search for a t1,t2, dixon, ct file

        fixed_representative = __find_representative(fixed_dict)
        movig_representative = __find_representative(movig_dict)
        # print(ex_slice_f, ex_slice_m, ex_slice)
        target_sequ = fixed_representative.info["sequ"]
        from_sequ = movig_representative.info["sequ"]

        log_file: Path = fixed_representative.get_changed_path(
            file_type="png",
            parent="registration",
            path="{sub}/target-" + target_sequ + "_from-" + from_sequ,
        )
        log = Logger(Path(log_file.parent, "log.txt"))

        log.print("##################################################################")
        log.print("[#] Processing")
        log.print(
            f'[#] Fixed  image: {fixed_representative.info["sub"]: <14} sequ-{target_sequ: <6} {fixed_dict.keys()}',
            verbose=True,
        )
        log.print(
            f'[#] Moving image: {movig_representative.info["sub"]: <14} sequ-{from_sequ  : <6} {movig_dict.keys()}',
            verbose=True,
        )
        log.print("[<]fixed_representative:\n", fixed_representative, verbose=False)
        log.print("[<]moving_representative:\n", movig_representative, verbose=False)
        # Load 2 representative from A and B
        img_fixed_sitk, img_f_org, img_a_iso = nii_to_iso_sitk_img(fixed_representative, tmp)
        img_movig_sitk, img_b_org, img_b_iso = nii_to_iso_sitk_img(movig_representative, tmp)
        log.print("[<] representative fixed :", to_str_sitk(img_fixed_sitk), verbose=verbose)
        log.print("[<] representative moving:", to_str_sitk(img_movig_sitk), verbose=verbose)
        # get centroid correspondence of A and B
        # f_img is the related file

        ctd_f_iso, axis_f_code = reload_centroids(__find_centroids(fixed_dict), img_f_org, img_a_iso)
        ctd_m_iso, axis_m_code = reload_centroids(__find_centroids(movig_dict), img_b_org, img_b_iso)
        assert axis_f_code == axis_m_code
        log.print("[<] centroid fixed:", list(ctd_f_iso.keys()), axis_f_code, verbose=verbose)
        log.print("[<] centroid movig:", list(ctd_m_iso.keys()), axis_m_code, verbose=verbose)

        ########################## Registration Step 1 ###########################
        # filter points by name
        f_keys = list(ctd_f_iso.keys())
        m_keys = list(ctd_m_iso.keys())
        # limit to only shared labels
        inter = np.intersect1d(m_keys, f_keys)
        if len(inter) < 3:
            log.print("[ ] not enough centroids:", inter)
            return
        log.print("[*] used centroids:", inter)

        resampler, resampler_seg, init_transform, error_reg, error_natural = point_register(
            inter, ctd_f_iso, img_fixed_sitk, ctd_m_iso, img_movig_sitk, log=log
        )
        # assert False, "Implement resamler_seg"
        error_dict[f'{fixed_representative.info["sub"]}_{target_sequ}_{from_sequ}'] = [error_reg, error_natural]
        ##################### Get Extended Centroids from Temporary new space #############
        m_subreg = nii_to_iso_sitk_img2(movig_dict["subreg"], tmp)
        m_vert = nii_to_iso_sitk_img2(movig_dict["vert"], tmp)
        transformed_subreg_sitk: sitk.Image = resampler_seg.Execute(m_subreg)
        transformed_vert = resampler_seg.Execute(m_vert)
        transformed_subreg = sitk_img_to_nii(transformed_subreg_sitk, tmp, round=True)
        transformed_vert = sitk_img_to_nii(transformed_vert, tmp, round=True)
        ctd_m_iso_extended = calc_centroids_from_file_and_processus_spinosus(transformed_vert, transformed_subreg)
        assert ctd_m_iso_extended is not None
        for key, value in ctd_m_iso_extended.items():
            # New Space
            value = transformed_subreg_sitk.TransformContinuousIndexToPhysicalPoint(value)
            value = init_transform.TransformPoint(value)
            # Old Space
            ctd_m_iso_extended[key] = img_movig_sitk.TransformPhysicalPointToContinuousIndex(value)
        # print(dict_transformed_movig_ctd)
        #
        ##################### Get Extended Centroids from Dixon ###################################
        assert "msk" in fixed_dict, "No centroid file"
        assert not isinstance(fixed_dict["msk"], list), f"{fixed_dict['msk']} contains more than one file"
        msk_bids: BIDS_FILE = fixed_dict["msk"]
        # assert "nii.gz" in msk_bids.file
        do_second_reg = True
        if "procspin" in fixed_dict:
            manual_ps_file: Path = fixed_dict["procspin"][0].file["nii.gz"]
            ctd_f_iso_extended = calc_centroids_from_file_and_manual_processus_spinosus(msk_bids.file["nii.gz"], manual_ps_file)
        elif "spinalcord" in fixed_dict:
            cord_file: Path = fixed_dict["spinalcord"][0].file["nii.gz"]
            ctd_f_iso_extended = calc_centroids_from_file_and_cord(msk_bids.file["nii.gz"], cord_file)
        else:
            log.print("[?] Could not find a procspin of spinalcord file. Skip second registration step.")
            ctd_f_iso_extended = None
            do_second_reg = False
        if do_second_reg:
            assert ctd_f_iso_extended is not None
            # filter points by name
            f_keys = list(ctd_f_iso_extended.keys())
            m_keys = list(ctd_m_iso_extended.keys())
            # limit to only shared labels
            inter = np.intersect1d(m_keys, f_keys)

            resampler, resampler_seg, init_transform2, error_reg, error_natural = point_register(
                inter, ctd_f_iso_extended, img_fixed_sitk, ctd_m_iso_extended, img_movig_sitk, log=log
            )
            error_dict[f'{fixed_representative.info["sub"]}_{target_sequ}_{from_sequ}'] += [error_reg, error_natural]
        ###################################################

        transformed_img = resampler.Execute(img_movig_sitk)
        # Crop the scans to the registered regions
        ex_slice_f, _ = crop_slice(sitk.GetArrayFromImage(img_fixed_sitk))
        ex_slice_m, _ = crop_slice(sitk.GetArrayFromImage(transformed_img))

        ex_slice = [slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(ex_slice_f, ex_slice_m)]

        ####### MAKE INTER SNAPSHOT
        out_file: Path = fixed_representative.get_changed_path(
            file_type="png",
            parent="registration",
            path="{sub}/target-" + target_sequ + "_from-" + from_sequ,
            info={"ovl": "intermediate"},
        )
        sc = None
        if "spinalcord" in fixed_dict:
            _, _, sc = nii_to_iso_sitk_img(fixed_dict["spinalcord"], tmp)
        # create_snapshot_mr(
        #    img_a_iso,
        #    out_file,
        #    transformed_vert,
        #    transformed_subreg,
        #    spinalcord_bids=sc,
        #    # ctd_bids=ctd_m_iso_extended,
        #    crop=False,
        #    check=False,
        #    fast=False,
        # )

        #######

        # Save registered file
        def register_and_save_file(img: sitk.Image, file: BIDS_FILE, target_space: bool):
            # print(f"[*] register {file.format}")
            if not target_space:
                if file.get_interpolation_order() != 0:
                    img = resampler.Execute(img)
                else:
                    img = resampler_seg.Execute(img)
            img = img[ex_slice]
            if "e" in file.info:
                file.remove("e")

            info = file.info.copy()
            info["reg"] = from_sequ if target_space else target_sequ
            if file.format == "dixon":
                j = file.open_json()
                if "IP" in j["ImageType"]:
                    info["acq"] = "real"
                elif "W" in j["ImageType"]:
                    info["acq"] = "water"
                elif "F" in j["ImageType"]:
                    info["acq"] = "fat"
            file.info.clear()

            def pop(key):
                if key in info:
                    file.info[key] = info.pop(key)

            pop("sub")
            pop("ses")
            pop("sequ")
            pop("reg")
            pop("acq")
            for k, v in info.items():
                file.info[k] = v

            # sub-spinegan0042_ses-20220517_sequ-301_reg-None_acq-water_dixon.nii.gz

            out_file: Path = file.get_changed_path(
                file_type="nii.gz",
                parent="registration",
                path="{sub}/target-" + target_sequ + "_from-" + from_sequ,
                info=info,
                from_info=True,
            )
            if not out_file.exists():
                out_file.parent.mkdir(exist_ok=True, parents=True)
            if file.get_interpolation_order() == 0:
                img = sitk.Round(img)
            log.print(f"[#] saving {file.format}:\t{out_file.name}")
            sitk.WriteImage(img, str(out_file))

        # Single file from B
        # register_and_save_file(img_movig_sitk, movig_representative, False)
        for bids in extract_nii(movig_dict)[0]:
            try:
                if "nii.gz" in bids.file:
                    img = nii_to_iso_sitk_img2(bids, tmp)
                    register_and_save_file(img, bids, False)
            except Exception as e:
                log.print(f"[!] Fail to register a sub_file, others will be registered \n\t{bids}\n\t {str(traceback.format_exc())}")

        # Single file from A
        # register_and_save_file(img_fixed_sitk, fixed_representative, True)
        for bids in extract_nii(fixed_dict)[0]:
            try:
                if "nii.gz" in bids.file:
                    img = nii_to_iso_sitk_img2(bids, tmp)
                    register_and_save_file(img, bids, True)
            except Exception as e:
                log.print(f"[!] Fail to register a sub_file, others will be registered \n\t{bids}\n\t {str(traceback.format_exc())}")

    except Exception as e:
        log.print("[!] Failed")
        log.print("fixed")
        log.print_dict(fixed_dict)
        log.print("moving")
        log.print_dict(movig_dict)
        log.print(str(traceback.format_exc()))
    finally:
        log.close()
        del log
        import shutil

        shutil.rmtree(tmp)


def calc_centroids_from_file_and_cord(path: Path, path_cord: Path, out_path: Path | None = None, verbose=True):
    print("[*] Generate ctd json from _msk.nii.gz and label-spinalcord")

    msk_nii = nib.load(str(path))
    msk_nii = reorient_to(msk_nii, axcodes_to=default_axcode_to)
    msk_nii = resample_nib(msk_nii, voxel_spacing=(1, 1, 1), order=0, verbose=False)
    msk_np = msk_nii.get_fdata().copy()

    cord_nii = nib.load(str(path_cord))
    cord_nii = reorient_to(cord_nii, axcodes_to=default_axcode_to)
    cord_nii = resample_nib(cord_nii, voxel_spacing=(1, 1, 1), order=0, verbose=False)
    cord_np = cord_nii.get_fdata().copy()
    assert default_axcode_to[2] == "I", "Function is using fixed dimensions implementation, 'I' must be last dimension"
    for i in range(cord_nii.shape[2]):
        s = msk_np[:, :, i]
        # Remove spinalcord, were there is no centroid
        if s.sum() == 0:
            cord_np[:, :, i][cord_np[:, :, i] != 0] = 0
        # Set spinalcord an same ID as vertebra. (vert_id)
        else:
            arr_unique = np.unique(msk_np[:, :, i])
            assert len(arr_unique) == 2, arr_unique
            cord_np[:, :, i][cord_np[:, :, i] != 0] = arr_unique[1]
    # nib.save(reorient_to(nib.Nifti1Image(cord_np, msk_nii.affine, msk_nii.header), axcodes_to=old_affine), str(out_path) + ".nii.gz")
    ctd_list = calc_centroids(msk_nii, decimals=2)
    cord_list = calc_centroids(nib.Nifti1Image(cord_np, cord_nii.affine, cord_nii.header), decimals=2)

    ctd_list = append_points_and_normalize_distance(ctd_list, cord_list)
    ctd_dict = centroids_to_dict_list(ctd_list)
    if verbose:
        print("[*] Found the following label:", [v["label"] for v in ctd_dict[1:] if is_Point3D(v)])
    if out_path is None:
        return to_dict(ctd_dict)
    save_json(ctd_dict, str(out_path))

    # sub-spinegan0042_ses-20220517_sequ-406_seg-subreg_ctd.json


def calc_centroids_from_file_and_manual_processus_spinosus(path: Path, path_ps: Path, out_path: Path | None = None, verbose=True):
    print("[*] Generate ctd json from label-procspin_msk.nii.gz and processus spinosus")
    msk_nii = nib.load(str(path))
    msk_nii = reorient_to(msk_nii, axcodes_to=default_axcode_to)
    msk_nii = resample_nib(msk_nii, voxel_spacing=(1, 1, 1), order=0, verbose=False)

    ps_nii = nib.load(str(path_ps))
    ps_nii = reorient_to(ps_nii, axcodes_to=default_axcode_to)
    ps_nii = resample_nib(ps_nii, voxel_spacing=(1, 1, 1), order=0, verbose=False)

    ctd_list = calc_centroids(msk_nii, decimals=2)
    ctd_ps_list = calc_centroids(ps_nii, decimals=2)

    cdt_v_list = append_points_and_normalize_distance(ctd_list, ctd_ps_list)
    ctd_dict = centroids_to_dict_list(cdt_v_list)
    if verbose:
        print("[*] Found the following label:", [v["label"] for v in ctd_dict[1:] if is_Point3D(v)])
    if out_path is None:
        return to_dict(ctd_dict)
    save_json(ctd_dict, str(out_path))


def calc_centroids_from_file_and_processus_spinosus(vert_file, subreg_file, out_path=None, verbose=True):
    print("[*] Generate ctd json from vertebra-body and processus spinosus")
    # Vertebra
    cdt_v_list = calc_centroids_from_subreg_vert(
        vert_file,
        subreg_file,
        decimals=2,
        axcodes_to=default_axcode_to,
    )

    # processus spinosus
    cdt_ps_list: Centroid_List = calc_centroids_from_subreg_vert(
        vert_file,
        subreg_file,
        subreg_id=42,
        decimals=2,
        axcodes_to=default_axcode_to,
    )
    # arcus vertebrae/foramen vertebrae
    cdt_av_list: Centroid_List = calc_centroids_from_subreg_vert(
        vert_file,
        subreg_file,
        subreg_id=41,
        decimals=2,
        axcodes_to=default_axcode_to,
    )
    found_key_ps = [k[0] for k in cdt_ps_list[1:]]  # [key,x,y,z]
    cdt_av_list = [v for i, v in enumerate(cdt_av_list) if i == 0 or v[0] not in found_key_ps]
    found_key_av = [k[0] for k in cdt_av_list[1:]]
    cdt_v_list = append_points_and_normalize_distance(cdt_v_list, cdt_ps_list)
    cdt_v_list = append_points_and_normalize_distance(cdt_v_list, cdt_av_list)
    ctd_dict = centroids_to_dict_list(cdt_v_list)
    if verbose:
        print("[*] Found the following label:", [v["label"] for v in ctd_dict[1:] if is_Point3D(v)])
        if len(found_key_av) != 0:
            print("[*] No processus spinosus for these keys, fall back to arcus vertebrae:", found_key_av)
    if out_path is None:
        return to_dict(ctd_dict)
    save_json(ctd_dict, str(out_path))


def append_points_and_normalize_distance(
    cdt_v_list: Centroid_List,
    cdt_ps_list: Centroid_List,
    fixed_distance: float = 50,
    point_index_offset: int = 100,
    fixed_dimensions: list[int | str] = ["I"],
):
    cdt_v_list = cdt_v_list.copy()
    fixed_dimensions2: list[int] = fixed_dimensions.copy()  # type: ignore

    assert cdt_v_list[0] == cdt_ps_list[0], f"different rotation {cdt_v_list[0]} != {cdt_ps_list[0]}"

    # Select dimensions
    for i, v in enumerate(fixed_dimensions):
        if isinstance(v, str):
            assert is_Orientation_tuple(cdt_v_list[0]), cdt_v_list[0]
            x: Orientation = cdt_v_list[0]
            idx = list(x).index(v)
            fixed_dimensions2[i] = idx + 1
        else:
            assert isinstance(v, int), "only int work as index"
            assert v <= len(cdt_v_list[0]), "out of bounds"
            assert v > 0, "No negative numbers as index; No 0 because index starts at 1. (Internal offset of list)"
    # set height component of the ps-centroid to the same hight as vertebra centroid
    for key in cdt_v_list[1:].copy():
        assert is_Point3D_tuple(key)
        for key2 in cdt_ps_list[1:].copy():
            assert is_Point3D_tuple(key2)
            key2 = list(key2)
            if key[0] == key2[0]:
                # key = [id, p,i,r] where i is the hight, when axcodes == (PIR)
                for idx in fixed_dimensions2:
                    key2[idx] = key[idx]
                # Make a point in the same direction as sp with a fixed distance of 10 px
                length = math.sqrt(sum([(b - a) * (b - a) for a, b in zip(key[1:], key2[1:])]))

                inter = [round(a + fixed_distance * (b - a) / length, ndigits=2) for a, b in zip(key[1:], key2[1:])]
                # print("key", key, key2, inter)
                cdt_v_list.append((int(key2[0] + point_index_offset), inter[0], inter[1], inter[2]))
    return cdt_v_list


def _parallelized_preprocess_scan(subj_name, subject: Subject_Container, force_override_A=True):
    def limit(query: Searchquery, ct=False):

        query.filter("sub", "spinegan0079")
        if not ct:
            query.filter("sequ", "301")
        pass
        # query.filter("sub", "spinegan0003")

    query1: Searchquery = subject.new_query()
    limit(query1)
    # query1.filter("sub", "spinegan0015")
    # It must exist a dixon and a msk
    query1.filter("format", "dixon")
    # A nii.gz must exist
    query1.filter("Filetype", "nii.gz")
    query1.filter("format", "msk")

    #######
    # Sometimes CT and dixon are on the same sequ-number
    # Removing all CT stuff from the dixon filter
    # Other file clashes will be interpreted as dixon (fixed image) is owner.
    #######
    if len(query1.candidates) == 0:
        return
    query1.flatten()
    query1.filter("format", lambda x: isinstance(x, str) and x.lower() != "ct")
    query1.filter("format", lambda x: isinstance(x, str) and x.lower() != "snp")
    query1.filter("seg", lambda x: isinstance(x, str) and x.lower() != "subreg", required=False)
    query1.filter("seg", lambda x: isinstance(x, str) and x.lower() != "vert", required=False)
    query1.unflatten()
    # query1.filter("seg", lambda x: isinstance(x, str) and x.lower() != "subreg")

    def key_transform(x: BIDS_FILE):
        if "seg" not in x.info:
            return None
        # seg-subreg_ctd.nii.gz
        if "subreg" == x.info["seg"] and x.format == "ctd":
            return "ctd"
        return None

    query2 = subject.new_query()
    limit(query2, ct=True)

    # Only files with a seg-subreg + ctd file.
    query2.filter("format", "ctd")
    query2.filter("seg", "subreg", required=True)
    # It must exist a ct
    query2.filter("format", "ct")

    #######
    # Sometimes CT and dixon are on the same sequ-number
    # Removing all dixon stuff from the ct filter
    # Other file clashes will be interpreted as dixon (fixed image) is owner.
    #######
    query2.flatten()
    query2.filter("format", lambda x: x.lower() != "dixon")  # type: ignore
    query2.filter("format", lambda x: x.lower() != "snp")  # type: ignore
    query2.filter("label", lambda x: x.lower() != "spinalcord", required=False)  # type: ignore
    query2.filter("parent", lambda x: x.lower() != "derivatives_msk")  # type: ignore
    query2.unflatten()
    query2.filter("seg", lambda x: isinstance(x, str) and x.lower() != "subreg")

    for dict_A in query1.loop_dict(key_transform=key_transform):
        for dict_B in query2.loop_dict(key_transform=key_transform):
            if not "subreg" in dict_B:

                continue
            ridged_point_registration(dict_A, dict_B)
    global error_dict

    dataset = next(iter(subject.sequences.values()))[0].dataset
    filepath = Path(dataset, f"registration/{subj_name}/reg_error.csv")
    if not filepath.exists():
        return
    with open(str(filepath), "w") as output:
        writer = csv.writer(output)
        for key, value in error_dict.items():
            value = [round(v, ndigits=3) for v in value]
            writer.writerow([key, *value])
    error_dict = {}


def parallel_execution(n_jobs, path="/media/data/dataset-spinegan/register_test", force_override_A=False):
    from joblib import Parallel, delayed

    global_info = BIDS_Global_info(
        [path],
        [
            "rawdata",
            "rawdata_ct",
            "rawdata_dixon",
            "derivatives",
            "derivatives_msk",
            "derivatives_spinalcord",
        ],  # "sourcedata"
        additional_key=["sequ", "seg", "e", "ovl"],
        verbose=False,
    )
    print(f"Found {len(global_info.subjects)} subjects in {global_info.datasets}")

    if n_jobs > 1:
        print("[*] Running {} parallel jobs. Note that stdout will not be sequential".format(n_jobs))

    Parallel(n_jobs=n_jobs)(
        delayed(_parallelized_preprocess_scan)(subj_name, subject, force_override_A)
        for subj_name, subject in global_info.enumerate_subjects()
    )
    with open(str(Path(path, f"registration/reg_error.csv")), "w") as output:
        writer = csv.writer(output)
        for subj_name, subject in global_info.enumerate_subjects(sort=True):
            path2 = Path(path, f"registration/{subj_name}/reg_error.csv")
            if not path2.exists():
                continue
            with open(str(path2), "r") as input:
                reader = csv.reader(input)
                for row in reader:
                    writer.writerow(row)
            path2.unlink()

    return None


def profile(force_override_A=True):
    from joblib import Parallel, delayed

    global global_info

    global_info = BIDS_Global_info(
        ["/media/data/dataset-spinegan/register_test"],
        [
            "rawdata",
            "rawdata_ct",
            "rawdata_dixon",
            "derivatives",
            "derivatives_msk",
            "derivatives_spinalcord",
        ],  # "sourcedata"
        additional_key=["sequ", "seg", "e", "ovl"],
        verbose=False,
    )
    print(f"Found {len(global_info.subjects)} subjects in {global_info.datasets}")

    import cProfile

    cProfile.run(
        f"[_parallelized_preprocess_scan(subj_name, subject, {force_override_A}) for subj_name, subject in global_info.enumerate_subjects()]",
        sort="cumulative",
    )
    return None


if __name__ == "__main__":
    a = ""
    # ridged_point_registration()
    parallel_execution(8, path="/media/data/robert/datasets/dataset_spinegan/")
    # profile()
