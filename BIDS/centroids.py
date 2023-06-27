from dataclasses import dataclass, field

from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio
from scipy.ndimage import center_of_mass
import json

from typing import Type, TypedDict, Union, Tuple, List, TypeVar, TypeGuard
from typing_extensions import Self
import BIDS.bids_files
from BIDS.nii_wrapper import to_nii, to_nii_optional, NII, Image_Reference
import warnings
from BIDS.vert_constants import *

from functools import partial

_Point3D = TypedDict("Point3D", X=float, Y=float, Z=float, label=int)
_Orientation = TypedDict("Orientation", direction=Tuple[str, str, str])
_Centroid_DictList = list[Union[_Orientation, _Point3D]]
from typing_extensions import Self

C = TypeVar("C", bound="Centroids")


Centroid_Reference = Union[
    BIDS.bids_files.BIDS_FILE,
    Path,
    str,
    Tuple[Image_Reference, Image_Reference, list[int]],
    C,
]


ctd_info_blacklist = ["zoom", "location", "shape", "direction", "format"]


@dataclass
class Centroids:
    """
    orientation is a tuple of three string values that represents the orientation of the image.
    centroids is a dictionary of centroid points, where the keys are the labels for the centroid
    points and values are a tuple of three float values that represent the x, y, and z coordinates
    of the centroid. location is an enum value that indicates the location of the centroid points
    in the image. zoom is a tuple of three float values that represents the zoom level of the image.
    shape is a tuple of three integer values that represent the shape of the image.
    sorting_list is a list of integer values that represents the order of the centroid points.
    """

    orientation: Ax_Codes = ("R", "A", "S")
    centroids: Centroid_Dict = field(default_factory=dict)
    location: Location = Location.Unknown
    zoom: None | Zooms = field(init=True, default=None)  # type: ignore
    shape: tuple[int, int, int] | None = field(default=None, repr=False, compare=False)
    sorting_list: List[int] | None = field(default=None, repr=False, compare=False)
    format: int | None = field(default=None, repr=False, compare=False)
    info: dict = field(
        default_factory=dict, compare=False
    )  # additional info (key,value pairs)
    # internal
    _zoom: None | Zooms = field(init=False, default=None, repr=False, compare=False)

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        if isinstance(value, property):
            pass
        elif value is None:
            self._zoom = None
        else:
            self._zoom = tuple(float(v) for v in value)

    @property
    def location_str(self):
        if isinstance(self.location, str):
            return self.location
        return self.location.name

    def clone(self, **qargs):
        return self.copy(**qargs)

    def copy(
        self,
        centroids: Centroid_Dict | None = None,
        orientation: Ax_Codes | None = None,
        zoom: Zooms | None = None,
        shape: tuple[int, int, int] | None = None,
    ) -> Self:
        return Centroids(
            orientation=orientation if orientation is not None else self.orientation,
            centroids=centroids if centroids is not None else self.centroids,
            location=self.location,
            zoom=zoom if zoom is not None else self.zoom,
            shape=shape if shape is not None else self.shape,
            sorting_list=self.sorting_list,
            info=self.info,
            format=self.format,
        )

    def crop_centroids(self, o_shift: tuple[slice, slice, slice], inplace=False):
        """When you crop an image, you have to also crop the centroids.
        There are actually no boundary to be moved, but the origin must be moved to the new 0,0,0
        Points outside the frame are NOT removed. See NII.compute_crop_slice()

        Args:
            o_shift (tuple[slice, slice, slice]): translation of the origin, cause by the crop
            inplace (bool, optional): inplace. Defaults to True.

        Returns:
            Self
        """

        try:
            centroids = {
                k: (
                    float(x - o_shift[0].start),
                    float(y - o_shift[1].start),
                    float(z - o_shift[2].start),
                )
                for k, (x, y, z) in self.centroids.items()
            }

            def map_v(sli: slice, i):
                end = sli.stop
                if end is None:
                    return self.shape[i]
                if end >= 0:
                    return end
                else:
                    return end + self.shape[i]

            shape = tuple(
                int(map_v(o_shift[i], i) - o_shift[i].start) for i in range(3)
            )

        except AttributeError:
            o: tuple[float, float, float] = o_shift  # type: ignore

            centroids = {
                k: (x - o[0], y - o[1], z - o[2])
                for k, (x, y, z) in self.centroids.items()
            }
            shape = None

        if inplace:
            self.centroids = centroids
            self.shape = shape  # TODO recompute ctd
            return self
        else:
            out = self.copy(centroids=centroids)
            out.shape = shape
            return out

    def crop_centroids_(self, o_shift: tuple[slice, slice, slice]):
        return self.crop_centroids(o_shift, inplace=True)

    def shift_all_centroid_coordinates(
        self,
        translation_vector: tuple[slice, slice, slice] | None,
        inplace=True,
        **kwargs,
    ):
        if translation_vector is None:
            return self
        return self.crop_centroids(translation_vector, inplace=inplace, **kwargs)

    def reorient(
        self,
        axcodes_to: Ax_Codes = ("P", "I", "R"),
        decimals=3,
        verbose: logging = False,
        inplace=False,
        _shape=None,
    ):
        """
        This method reorients the centroids of an image from the current orientation to the specified orientation.
        It updates the position of the centroids, zoom level, and shape of the image accordingly.

        Args:
            axcodes_to (Ax_Codes): An Ax_Codes object representing the desired orientation of the centroids.
            decimals (int, optional): Number of decimal places to round the coordinates of the centroids after reorientation. Defaults to 3.
            verbose (bool, optional): If True, print a message indicating the current and new orientation of the centroids. Defaults to False.
            inplace (bool, optional): If True, update the current centroid object with the reoriented values. If False, return a new centroid object with reoriented values. Defaults to False.
            _shape (tuple[int], optional): The shape of the image. Required if the shape is not already present in the centroid object.
        Returns:
            If inplace is True, returns the updated centroid object. If inplace is False, returns a new centroid object with reoriented values.
        """
        ctd_arr = np.transpose(np.asarray(list(self.centroids.values())))
        v_list = list(self.centroids.keys())
        if ctd_arr.shape[0] == 0:
            log.print(
                "No centroids present",
                verbose=verbose if not isinstance(verbose, bool) else True,
                type=log_file.Log_Type.WARNING,
            )
            return self

        ornt_fr = nio.axcodes2ornt(self.orientation)  # original centroid orientation
        ornt_to = nio.axcodes2ornt(axcodes_to)
        if (ornt_fr == ornt_to).all():
            log.print(
                "ctd is already rotated to image with ", axcodes_to, verbose=verbose
            )
            return self
        trans = nio.ornt_transform(ornt_fr, ornt_to).astype(int)
        perm: list[int] = trans[:, 0].tolist()

        if self.shape is not None:
            shape = tuple([self.shape[perm.index(i)] for i in range(len(perm))])

            if _shape != shape and _shape is not None:
                raise ValueError(
                    f"Different shapes {shape} <-> {_shape}, types {type(shape)} <-> {type(_shape)}"
                )
        else:
            shape = _shape
        assert (
            shape is not None
        ), "Require shape information for flipping dimensions. Set self.shape or use reorient_centroids_to"
        shp = np.asarray(shape)
        ctd_arr[perm] = ctd_arr.copy()
        for ax in trans:
            if ax[1] == -1:
                size = shp[ax[0]]
                ctd_arr[ax[0]] = np.around(size - ctd_arr[ax[0]], decimals) - 1
        points = {}
        ctd_arr = np.transpose(ctd_arr).tolist()
        for v, point in zip(v_list, ctd_arr):
            points[v] = tuple(point)

        log.print(
            "[*] Centroids reoriented from",
            nio.ornt2axcodes(ornt_fr),
            "to",
            axcodes_to,
            verbose=verbose,
        )
        if self.zoom is not None:
            zoom_i = np.array(self.zoom)
            zoom_i[perm] = zoom_i.copy()
            zoom: Zooms | None = tuple(zoom_i)
        else:
            zoom = None
        if inplace:
            self.orientation = axcodes_to
            self.centroids = points
            self.zoom = zoom
            self.shape = shape
            return self
        return self.copy(
            orientation=axcodes_to, centroids=points, zoom=zoom, shape=shape
        )

    def reorient_(
        self,
        axcodes_to: Ax_Codes = ("P", "I", "R"),
        decimals=3,
        verbose: logging = False,
        _shape=None,
    ):
        return self.reorient(
            axcodes_to, decimals=decimals, verbose=verbose, inplace=True, _shape=_shape
        )

    def reorient_centroids_to(
        self,
        img: NII | nib.Nifti1Image,
        decimals=3,
        verbose: logging = False,
        inplace=False,
    ) -> Self:
        # reorient centroids to image orientation
        if not isinstance(img, NII):
            img = NII(img)
        axcodes_to = nio.aff2axcodes(img.affine)
        return self.reorient(
            axcodes_to,
            decimals=decimals,
            verbose=verbose,
            inplace=inplace,
            _shape=img.shape,
        )

    def reorient_centroids_to_(
        self, img: NII | nib.Nifti1Image, decimals=3, verbose: logging = False
    ) -> Self:
        return self.reorient_centroids_to(
            img, decimals=decimals, verbose=verbose, inplace=True
        )

    def rescale(
        self,
        voxel_spacing=(1, 1, 1),
        decimals=3,
        verbose: logging = True,
        inplace=False,
    ) -> Self:
        """Rescale the centroid coordinates to a new voxel spacing in the current x-y-z-orientation.

        Args:
            voxel_spacing (tuple[float, float, float], optional): New voxel spacing in millimeters. Defaults to (1, 1, 1).
            decimals (int, optional): Number of decimal places to round the rescaled coordinates to. Defaults to 3.
            verbose (bool, optional): Whether to print a message indicating that the centroid coordinates have been rescaled. Defaults to True.
            inplace (bool, optional): Whether to modify the current instance or return a new instance. Defaults to False.

        Returns:
            Centroid: If inplace=True, returns the modified Centroids instance. Otherwise, returns a new Centroids instance with rescaled centroid coordinates.
        """
        assert (
            self.zoom is not None
        ), "this Centroids doesn't have a zoom set. Use centroid.zoom = nii.zoom"
        zms = self.zoom
        shp = list(self.shape) if self.shape is not None else None
        ctd_arr = np.transpose(np.asarray(list(self.centroids.values())))
        v_list = list(self.centroids.keys())
        for i in range(3):
            fkt = zms[i] / voxel_spacing[i]
            if len(v_list) != 0:
                ctd_arr[i] = np.around(ctd_arr[i] * fkt, decimals=decimals)
            if shp is not None:
                shp[i] *= fkt
        points = {}
        ctd_arr = np.transpose(ctd_arr).tolist()
        for v, point in zip(v_list, ctd_arr):
            points[v] = tuple(point)
        log.print(
            "[*] Rescaled centroid coordinates to spacing (x, y, z) =",
            voxel_spacing,
            "mm",
            verbose=verbose,
        )
        if shp is not None:
            shp = tuple(shp)
        if inplace:
            self.centroids = points
            self.zoom = voxel_spacing
            self.shape = shp
            return self
        return self.copy(centroids=points, zoom=voxel_spacing, shape=shp)

    def rescale_(
        self, voxel_spacing=(1, 1, 1), decimals=3, verbose: logging = False
    ) -> Self:
        return self.rescale(
            voxel_spacing=voxel_spacing,
            decimals=decimals,
            verbose=verbose,
            inplace=True,
        )

    def map_labels(
        self,
        label_map: dict[int | str, int | str],
        verbose: logging = False,
        inplace=False,
    ):
        """Maps labels to new values based on a label map dictionary.

        Args:
            label_map (dict): A dictionary that maps labels to new values.
                The keys and values can be either integers or strings.
            verbose (bool, optional): If True, print the label map dictionary.
                Defaults to False.
            inplace (bool, optional): If True, modify the centroids in place.
                Defaults to False.

        Returns:
            Centroids: A new Centroids object with the mapped labels.

        Examples:
            >>> centroid_obj = Centroids(...)
            >>> label_map = {'1': '3', 4: '2', '5': 6}
            >>> new_centroids = centroid_obj.map_labels(label_map)

        """
        label_map_ = {
            v_name2idx[k]
            if k in v_name2idx
            else int(k): v_name2idx[v]
            if v in v_name2idx
            else int(v)
            for k, v in label_map.items()
        }
        log.print_dict(label_map_, "label_map", verbose=verbose)
        cent = {}
        for label, value in self.items():
            if label in label_map_:
                new_label = label_map_[label]
                if new_label != 0:
                    cent[new_label] = value
            else:
                cent[label] = value
        if inplace:
            self.centroids = cent
            return self
        else:
            return self.copy(cent)

    def map_labels_(
        self, label_map: dict[int | str, int | str], verbose: logging = False
    ):
        return self.map_labels(label_map, verbose=verbose, inplace=True)

    def save(
        self,
        out_path: Path | str,
        make_parents=False,
        additional_info: dict | None = None,
        verbose: logging = True,
        save_hint=0,
    ) -> None:
        """
        Saves the centroids to a JSON file.

        Args:
            out_path (Path | str): The path where the JSON file will be saved.
            make_parents (bool, optional): If True, create any necessary parent directories for the output file.
                Defaults to False.
            verbose (bool, optional): If True, print status messages to the console. Defaults to True.
            save_hint: 0 Default, 1 Gruber, 2 POI (readable), 10 ISO-POI (outdated)
        Returns:
            None

        Raises:
            TypeError: If any of the centroids have an invalid type.

        Example:
            >>> centroids = Centroids(...)
            >>> centroids.save("output/centroids.json")
        """
        if make_parents:
            Path(out_path).parent.mkdir(exist_ok=True, parents=True)

        self.sort()
        # TODO add function here that automatically orders the out_path keys in correct order?
        out_path = str(out_path)
        if len(self.centroids) == 0:
            log.print(
                "Centroids empty, not saved:",
                out_path,
                type=log_file.Log_Type.FAIL,
                verbose=verbose,
            )
            return
        json_object, print_add = _centroids_to_dict_list(
            self, additional_info, save_hint, verbose
        )

        # Problem with python 3 and int64 serialization.
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            raise TypeError(type(o))

        with open(out_path, "w") as f:
            json.dump(json_object, f, default=convert, indent=4)
        log.print(
            "Centroids saved:",
            out_path,
            print_add,
            type=log_file.Log_Type.SAVE,
            verbose=verbose,
        )

    def sort(self, inplace=True, natural_numbers=False):
        """Sort vertebra dictionary by sorting_list"""
        if natural_numbers:
            centroids = dict(sorted(self.centroids.items()))
        else:
            if self.sorting_list is None:
                return self
            centroids = dict(
                sorted(
                    self.centroids.items(),
                    key=partial(order_vert_name, v_idx_order=self.sorting_list),
                )
            )
        if inplace:
            self.centroids = centroids
            return self
        else:
            return self.copy(centroids=centroids)

    def make_point_cloud_nii(self, affine, s=8):
        assert self.shape is not None, "need shape information"
        assert self.zoom is not None, "need shape information"
        arr = np.zeros(self.shape)
        s1 = s // 2
        s2 = s - s1
        for key, (x, y, z) in self.items():
            arr[
                max(int(x - s1 / self.zoom[0]), 0) : min(
                    int(x + s2 / self.zoom[0]), self.shape[0]
                ),
                max(int(y - s1 / self.zoom[1]), 0) : min(
                    int(y + s2 / self.zoom[1]), self.shape[1]
                ),
                max(int(z - s1 / self.zoom[2]), 0) : min(
                    int(z + s2 / self.zoom[2]), self.shape[2]
                ),
            ] = key
        nii = nib.Nifti1Image(arr, affine=affine)
        return NII(nii, seg=True)

    def __iter__(self):
        return iter(self.centroids.keys())

    def __contains__(self, key: int) -> bool:
        return key in self.centroids

    def __getitem__(self, key: int | slice) -> Tuple[float, float, float]:
        if isinstance(key, int) or isinstance(key, np.integer):
            return self.centroids[int(key)]
        else:
            return self.centroids[
                LABEL_MAX * key.start + key.stop
            ]  # (subreg_id,vertebra_id)

    def __setitem__(self, key: int | slice, value: Tuple[float, float, float]):
        if isinstance(key, int) or isinstance(key, np.integer):
            self.centroids[int(key)] = value
        else:
            print(type(key))
            self.centroids[
                LABEL_MAX * key.start + key.stop
            ] = value  # (subreg_id,vertebra_id)

    def __len__(self) -> int:
        return self.centroids.__len__()

    def items(self):
        self.sort()
        return self.centroids.items()

    def items2(self):
        self.sort()
        for x, y in self.centroids.items():
            yield x // LABEL_MAX, x % LABEL_MAX, y

    def keys(self):
        return self.centroids.keys()

    def values(self):
        return self.centroids.values()

    def remove_centroid_(self, *label: int | Location):
        return self.remove_centroid(*label, inplace=True)

    def remove_centroid(self, *label: int | Location, inplace=False):
        obj: Centroids = self.copy() if inplace == False else self
        for l in label:
            if isinstance(l, Location):
                l = l.value
            obj.centroids.pop(l, None)
        return obj

    def round(self, ndigits, inplace=False):
        out = {}
        for label, (x, y, z) in self.items():
            out[label] = (
                round(x, ndigits=ndigits),
                round(y, ndigits=ndigits),
                round(z, ndigits=ndigits),
            )
        if inplace:
            self.centroids = out
            return self
        return self.copy(out)

    def round_(self, ndigits):
        return self.round(ndigits=ndigits, inplace=True)

    def get_subset_by_lower_key(self, vert_key):
        out: Centroid_Dict = {}
        for sub, vert, v in self.items2():
            if vert == vert_key:
                out[sub * LABEL_MAX + vert] = v
        return self.copy(out)

    @classmethod
    def load(cls, poi: Centroid_Reference):
        return load_centroids(poi)


###### SORTING #####
def order_vert_name(
    elem: tuple[int, tuple[float, float, float]], v_idx_order: list[int], mod=LABEL_MAX
):
    idx = elem[0] % mod
    off = elem[0] // mod
    assert (
        idx in v_idx_order
    ), f"{elem[0]} {idx} - {off} not in v_name2idx, only got {v_idx_order},{idx == v_idx_order[0]}"
    return v_idx_order.index(idx) + off


class VertebraCentroids(Centroids):
    def __init__(
        self,
        _orientation,
        _centroids,
        _shape: tuple[int, int, int] | None,
        zoom=None,
        _location=Location.Vertebra_Corpus,
    ):
        super().__init__(
            orientation=_orientation,
            centroids=_centroids,
            shape=_shape,
            location=_location,
            zoom=zoom,
            sorting_list=v_idx_order.copy(),
        )
        self.sort()

    @classmethod
    def from_centroids(cls, ctd: Centroids):
        a = ctd.centroids.copy()
        a = {k: v for k, v in a.items() if k <= 30}  # filter all non Vertebra_Centroids
        return cls(ctd.orientation, a, _shape=ctd.shape, zoom=ctd.zoom)

    def get_xth_vertebra(self, index: int):
        """

        Args:
            index:

        Returns:
            the <index>-th vertebra in this Centroid list (if index == -1, returns the last one)
        """
        self.sort()
        centroid_keys = list(self.keys())
        if index == -1:
            return self.centroids[centroid_keys[len(self.centroids) - 1]]
        return self.centroids[centroid_keys[index]]

    def get_last_l_centroid(self):
        """

        Returns:
            The last L centroid coordinates. None if not found
        """
        # Can only be L6, L5, or L4
        idx = 25 if 25 in self else 24 if 24 in self else 23 if 23 in self else None
        if idx is not None:
            return idx, self.centroids[idx]
        return None, None


######## Saving #######
def _is_Point3D(obj) -> TypeGuard[_Point3D]:
    return "label" in obj and "X" in obj and "Y" in obj and "Z" in obj


FORMAT_DOCKER = 0
FORMAT_GRUBER = 1
FORMAT_POI = 2
FORMAT_OLD_POI = 10
format_key = {FORMAT_DOCKER: "docker", FORMAT_GRUBER: "guber", FORMAT_POI: "POI"}
format_key2value = {value: key for key, value in format_key.items()}


def _centroids_to_dict_list(
    ctd: Centroids, additional_info: dict | None, save_hint=0, verbose: logging = False
):
    ori: _Orientation = {"direction": ctd.orientation}
    print_out = ""
    if hasattr(ctd, "location") and ctd.location != Location.Unknown:
        ori["location"] = ctd.location_str  # type: ignore
    if ctd.zoom != None:
        ori["zoom"] = ctd.zoom  # type: ignore
    if ctd.shape != None:
        ori["shape"] = ctd.shape  # type: ignore
    if save_hint in format_key:
        ori["format"] = format_key[save_hint]  # type: ignore
        print_out = "in format " + format_key[save_hint]

    if additional_info is not None:
        for k, v in additional_info.items():
            if k not in ori:
                ori[k] = v

    for (
        k,
        v,
    ) in ctd.info.items():
        if k not in ori:
            ori[k] = v

    dict_list: list[Union[_Orientation, _Point3D | dict]] = [ori]

    if save_hint == FORMAT_OLD_POI:
        ctd = ctd.rescale((1, 1, 1), verbose=verbose).reorient_(
            ("R", "P", "I"), verbose=verbose
        )
        dict_list = []

    temp_dict = {}
    ctd.sort(natural_numbers=True) if ctd.sorting_list != None else ctd.sort()
    for v, (x, y, z) in ctd.items():
        subreg_id = v // LABEL_MAX
        vert_id = v % LABEL_MAX

        if save_hint == FORMAT_DOCKER:
            dict_list.append({"label": v, "X": x, "Y": y, "Z": z})
        elif save_hint == FORMAT_GRUBER:
            v = (
                v_idx2name[vert_id].replace("T", "TH")
                + "_"
                + conversion_poi2text[subreg_id]
            )
            dict_list.append({"label": v, "X": x, "Y": y, "Z": z})
        elif save_hint == FORMAT_POI:
            v_name = v_idx2name[vert_id] if vert_id in v_idx2name else str(vert_id)
            # sub_name = v_idx2name[subreg_id]
            if v_name not in temp_dict:
                temp_dict[v_name] = {}
            temp_dict[v_name][subreg_id] = (x, y, z)
        elif save_hint == FORMAT_OLD_POI:
            if vert_id not in temp_dict:
                temp_dict[vert_id] = {}
            temp_dict[vert_id][str(subreg_id)] = str((float(x), float(y), float(z)))
        else:
            raise NotImplementedError(save_hint)
    if len(temp_dict) != 0:
        if save_hint == FORMAT_OLD_POI:
            for k, v in temp_dict.items():
                out_dict = {"vert_label": str(k), **v}
                dict_list.append(out_dict)
        else:
            dict_list.append(temp_dict)
    return dict_list, print_out


######### Load #############
# Handling centroids #


def load_centroids(ctd_path: Centroid_Reference, verbose=True) -> Centroids:
    """
    Load centroids from a file or a BIDS file object.

    Args:
        ctd_path (Centroid_Reference): Path to a file or BIDS file object from which to load centroids.
            Alternatively, it can be a tuple containing the following items:
            - vert: str, the name of the vertebra.
            - subreg: str, the name of the subregion.
            - ids: list[int | Location], a list of integers and/or Location objects used to filter the centroids.

    Returns:
        A Centroids object containing the loaded centroids.

    Raises:
        AssertionError: If `ctd_path` is not a recognized type.

    """
    if isinstance(ctd_path, Centroids):
        return ctd_path
    elif isinstance(ctd_path, BIDS.bids_files.BIDS_FILE):
        dict_list: _Centroid_DictList = ctd_path.open_json()  # type: ignore
    elif isinstance(ctd_path, str) or isinstance(ctd_path, Path):
        with open(ctd_path) as json_data:
            dict_list: _Centroid_DictList = json.load(json_data)
            json_data.close()
    elif isinstance(ctd_path, tuple):
        vert = ctd_path[0]
        subreg = ctd_path[1]
        ids: list[int | Location] = ctd_path[2]  # type: ignore
        return calc_centroids_from_subreg_vert(vert, subreg, subreg_id=ids)
    else:
        assert False, f"{type(ctd_path)}\n{ctd_path}"
    ### format_POI_old has no META header
    if "direction" not in dict_list[0] and "vert_label" in dict_list[0]:
        return _load_format_POI_old(dict_list)

    assert (
        "direction" in dict_list[0]
    ), f'File format error: first index must be a "Direction" but got {dict_list[0]}'
    axcode: Ax_Codes = tuple(dict_list[0]["direction"])  # type: ignore
    location: Location = dict_list[0].get("location", Location.Unknown)  # type: ignore
    zoom: Zooms = dict_list[0].get("zoom", None)  # type: ignore
    shape = dict_list[0].get("shape", None)  # type: ignore
    format = dict_list[0].get("format", None)

    info = {}
    for k, v in dict_list[0].items():
        if k not in ctd_info_blacklist:
            info[k] = v

    format = format_key2value[format] if format is not None else None
    centroids: Centroid_Dict = {}
    if format is None or format == FORMAT_DOCKER or format == FORMAT_GRUBER:
        _load_docker_centroids(dict_list, centroids, format)
    elif format == FORMAT_POI:
        _load_POI_centroids(dict_list, centroids)
    else:
        raise NotImplementedError(format)
    return Centroids(
        axcode, centroids, location, zoom=zoom, shape=shape, format=format, info=info
    )


def _load_docker_centroids(dict_list, centroids, format):
    for d in dict_list[1:]:
        assert (
            "direction" not in d
        ), f'File format error: only first index can be a "direction" but got {dict_list[0]}'
        if "nan" in str(d):  # skipping NaN centroids
            continue
        elif _is_Point3D(d):
            try:
                centroids[int(d["label"])] = (d["X"], d["Y"], d["Z"])
            except Exception:
                try:
                    number, subreg = str(d["label"]).split("_", maxsplit=1)
                    number = number.replace("TH", "T").replace("SA", "S1")
                    # print(number, subreg)
                    vert_id = v_name2idx[number]
                    subreg_id = conversion_poi[subreg]
                    # print(vert_id, subreg_id)

                    centroids[subreg_id * LABEL_MAX + vert_id] = (
                        d["X"],
                        d["Y"],
                        d["Z"],
                    )

                    # print(f'Label {d["label"]} to {number},{subreg}; {vert_id}-{subreg_id}')
                except:
                    print(
                        f'Label {d["label"]} is not an integer and cannot be converted to an int'
                    )
                    #    # warnings.warn(f'Label {d["label"]} is not an integer')
                    centroids[d["label"]] = (d["X"], d["Y"], d["Z"])
        else:
            raise ValueError(d)


def _load_format_POI_old(dict_list):
    # [
    # {
    #    "vert_label": "8",
    #    "85": "(281, 185, 274)",
    #    ...
    # }{...}
    # ...
    # ]
    centroids: Centroid_Dict = {}
    for d in dict_list:
        d: dict[str, str]
        vert_id = int(d["vert_label"])
        for k, v in d.items():
            if k == "vert_label":
                continue
            sub_id = int(k)
            t = v.replace("(", "").replace(")", "").replace(" ", "").split(",")
            t = tuple(float(x) for x in t)
            centroids[sub_id * LABEL_MAX + vert_id] = t
    return Centroids(
        ("R", "P", "I"),
        centroids,
        Location.Multi,
        zoom=(1, 1, 1),
        shape=None,
        format=FORMAT_OLD_POI,
    )


def _to_int(vert_id):
    try:
        return int(vert_id)
    except Exception:
        return v_name2idx[vert_id]


def _load_POI_centroids(dict_list, centroids: Centroid_Dict):
    assert len(dict_list) == 2
    d: dict[int | str, dict[int | str, tuple[float, float, float]]] = dict_list[1]
    for vert_id, v in d.items():
        vert_id = _to_int(vert_id)
        for sub_id, t in v.items():
            sub_id = _to_int(sub_id)
            centroids[sub_id * LABEL_MAX + vert_id] = tuple(t)


def loc2int(i: int | Location):
    if isinstance(i, int):
        return i
    return i.value


def int2loc(i: int | Location | list[int | Location]):
    if isinstance(i, List):
        return [int2loc(j) for j in i]
    elif isinstance(i, int):
        try:
            return Location(i)
        except Exception:
            return i
    return i


def calc_centroids_labeled_buffered(
    msk_reference: Image_Reference,
    subreg_reference: Image_Reference | None = None,
    out_path: Path | None = None,
    subreg_id: int | Location | list[int | Location] = 50,
    verbose=True,
    override=False,
    decimals=3,
    world=False,
    additional_folder=False,
) -> Centroids:
    """
    Computes the centroids of the given mask `msk_reference` with respect to the given subregion `subreg_reference`,
    and saves them to a file at `out_path` (if `override=False` and the file already exists, the function loads and returns
    the existing centroids from the file).

    If `out_path` is None and `msk_reference` is a `BIDS.bids_files.BIDS_FILE` object, the function generates a path to
    save the centroids file based on the `label` attribute of the file and the given `subreg_id`.

    If `subreg_reference` is None, the function computes the centroids using only `msk_reference`.

    If `subreg_reference` is not None, the function computes the centroids with respect to the given `subreg_id` in the
    subregion defined by `subreg_reference`.

    Args:
        msk_reference (Image_Reference): The mask to compute the centroids from, as a `Path`, `BIDS.bids_files.BIDS_FILE`,
        or `nii` file.

        subreg_reference (Image_Reference | None, optional): The subregion mask to compute the centroids relative to,
        as a `Path`, `BIDS.bids_files.BIDS_FILE`, or `nii` file. Defaults to None.

        out_path (Path | None, optional): The path to save the computed centroids to, as a `.json` file with format `ctd`.
        If None, the function generates a path to save the file based on `msk_reference`, `subreg_id`, and the label of
        `msk_reference` (if it is a `BIDS.bids_files.BIDS_FILE`). Defaults to None.

        subreg_id (int | Location | list[int | Location], optional): The ID of the subregion to compute centroids in,
        or a list of IDs to compute multiple centroids. Defaults to 50.

        verbose (bool, optional): Whether to print verbose output during the computation. Defaults to True.

        override (bool, optional): Whether to overwrite any existing centroids file at `out_path`. Defaults to False.

        decimals (int, optional): The number of decimal places to round the computed centroid coordinates to. Defaults to 3.

        world (bool, optional): Whether to return the centroids in world (mm) coordinates instead of voxel coordinates.
        Defaults to False.

        additional_folder (bool, optional): Whether to add a `/ctd/` folder to the path generated for the output file.
        Defaults to False.

    Returns:
        Centroids: The computed centroids, as a `Centroids` object.
    """
    assert out_path is not None or isinstance(
        msk_reference, BIDS.bids_files.BIDS_FILE
    ), "Automatic path generation is only possible with a BIDS_FILE"
    if out_path is None and isinstance(msk_reference, BIDS.bids_files.BIDS_FILE):
        if not isinstance(subreg_id, list) and subreg_id != -1:
            name = subreg_idx2name[loc2int(subreg_id)]
        elif subreg_reference is None:
            name = msk_reference.get("label", default="full")
        else:
            name = "multi"
        assert name is not None
        out_path = msk_reference.get_changed_path(
            file_type="json",
            format="ctd",
            info={"label": name.replace("_", "-")},
            parent="derivatives"
            if msk_reference.get_parent() == "rawdata"
            else msk_reference.get_parent(),
            additional_folder="ctd" if additional_folder else None,
        )
    assert out_path is not None
    if not override and out_path.exists():
        print(f"[*] Load ctd json from {out_path}") if verbose else None
        return load_centroids(out_path)
    print(f"[*] Generate ctd json towards {out_path}") if verbose else None
    msk_nii = to_nii(msk_reference, True)
    sub_nii = to_nii_optional(subreg_reference, True)
    if sub_nii is not None:
        ctd = calc_centroids_from_subreg_vert(
            msk_nii,
            sub_nii,
            decimals=decimals,
            world=world,
            subreg_id=subreg_id,
            verbose=verbose,
        )
    else:
        ctd = calc_centroids(msk_nii, Location(0), decimals=decimals, world=world)

    ctd.save(out_path, verbose=verbose)
    return ctd


def calc_centroids_from_subreg_vert(
    vert_msk: Image_Reference,
    subreg: Image_Reference,
    decimals=1,
    world=False,
    subreg_id: int | Location | list[int | Location] = 50,
    axcodes_to: Ax_Codes | None = None,
    verbose=False,
    fixed_offset=0,
    extend_to: Centroids | None = None,
) -> Centroids:
    """Calculates the centroids of a subregion within a vertebral mask.

    Args:
        vert_msk (Image_Reference): A vertebral mask image reference.
        subreg (Image_Reference): An image reference for the subregion of interest.
        decimals (int, optional): Number of decimal places to round the output coordinates to. Defaults to 1.
        world (bool, optional): Whether to return centroids in world coordinates. Defaults to False.
        subreg_id (int | Location | list[int | Location], optional): The ID(s) of the subregion(s) to calculate centroids for. Defaults to 50.
        axcodes_to (Ax_Codes | None, optional): A tuple of axis codes indicating the target orientation of the images. Defaults to None.
        verbose (bool, optional): Whether to print progress messages. Defaults to False.
        fixed_offset (int, optional): A fixed offset value to add to the calculated centroid coordinates. Defaults to 0.
        extend_to (Centroids | None, optional): An existing Centroids object to extend with the new centroid values. Defaults to None.

    Returns:
        Centroids: A Centroids object containing the calculated centroid coordinates.
    """
    vert_msk = to_nii(vert_msk, seg=True)
    subreg_msk = to_nii(subreg, seg=True)
    if verbose:
        print(
            "[*] Calc centroids from subregion id", int2loc(subreg_id), vert_msk.shape
        )

    if axcodes_to is not None:
        # Like: ("P","I","R")
        vert_msk = vert_msk.reorient(
            verbose=verbose, axcodes_to=axcodes_to, inplace=False
        )
        subreg_msk = subreg_msk.reorient(
            verbose=verbose, axcodes_to=axcodes_to, inplace=False
        )
    # Recursive call for multiple subregion ids
    if isinstance(subreg_id, list):
        centroids = Centroids(
            vert_msk.orientation,
            {},
            Location.Multi,
            zoom=vert_msk.zoom,
            shape=vert_msk.shape,
            format=FORMAT_POI,
        )

        offset = fixed_offset if fixed_offset != 0 else LABEL_MAX
        print("list") if verbose else None
        for i, id in enumerate(subreg_id):
            centroids = calc_centroids_from_subreg_vert(
                vert_msk,
                subreg_msk,
                subreg_id=loc2int(id),
                fixed_offset=loc2int(id) * offset,
                verbose=verbose,
                extend_to=centroids,
                decimals=decimals,
                world=world,
            )
        return centroids
    # Prepare mask to binary mask
    vert = vert_msk.get_seg_array()
    subreg_msk = subreg_msk.get_seg_array()
    assert subreg_msk.shape == vert.shape, (
        "Shape miss-match" + str(subreg_msk.shape) + str(vert.shape)
    )
    subreg_id = loc2int(subreg_id)
    if subreg_id == 50:
        subreg_msk[subreg_msk == 49] = 50
    vert[subreg_msk != subreg_id] = 0
    msk_data = vert
    nii = nip.Nifti1Image(msk_data, vert_msk.affine, vert_msk.header)
    ctd_list = calc_centroids(
        nii,
        decimals=decimals,
        world=world,
        fixed_offset=fixed_offset,
        extend_to=extend_to,
    )
    return ctd_list


def calc_centroids(
    msk: Image_Reference,
    location: Location = Location.Unknown,
    decimals=3,
    world=False,
    fixed_offset=0,
    extend_to: Centroids | None = None,
) -> Centroids:
    """
    Calculates the centroid coordinates of each region in the mask.

    Args:
        msk (Image_Reference): Input mask.
        location (Location, optional): The anatomical location of the centroid. Defaults to Location.Unknown.
        decimals (int, optional): Number of decimal places to round the centroid coordinates to. Defaults to 3.
        world (bool, optional): If True, the centroids are returned in world coordinates. Defaults to False.
        fixed_offset (int, optional): Fixed offset added to the region label to differentiate multiple regions. Defaults to 0.
        extend_to (Centroids, optional): An existing Centroids object to add the calculated centroids to.
                                         If provided, the object will be updated in-place and returned. Defaults to None.

    Returns:
        Centroids: A Centroids object containing the calculated centroids.

    Raises:
        AssertionError: If the extend_to object has a different orientation, location, or zoom than the input mask.

    Notes:
        - Centroids are in voxel coordinates unless world=True.
        - If extend_to is provided, the calculated centroids will be added to the existing object and the updated object will be returned.
        - The region label is assumed to be an integer.
        - NaN values in the binary mask are ignored.
    """
    msk = to_nii(msk, seg=True)
    msk_data = msk.get_seg_array()
    axc = nio.aff2axcodes(msk.affine)
    ctd_list: dict[int, tuple[float, float, float]] = {}
    verts = np.unique(msk_data)
    verts = verts[verts != 0]
    verts = verts[~np.isnan(verts)]  # remove NaN values
    for i in verts:
        msk_temp = np.zeros(msk_data.shape, dtype=bool)
        msk_temp[msk_data == i] = True
        ctr_mass: list[float] = center_of_mass(msk_temp)  # type: ignore
        if world:
            ctr_mass = msk.affine[:3, :3].dot(ctr_mass) + msk.affine[:3, 3]
            ctr_mass = ctr_mass.tolist()  # type: ignore
        ctd_list[int(i + fixed_offset)] = tuple([round(x, decimals) for x in ctr_mass])
    if extend_to is None:
        return Centroids(axc, ctd_list, location, zoom=msk.zoom, shape=msk.shape)
    else:
        if extend_to.zoom == None:
            extend_to.zoom = msk.zoom
        assert (
            axc == extend_to.orientation
        ), f"orientation Location must be the same.  extend_to {extend_to.orientation}, new {axc}"
        assert (
            location == extend_to.location or location == Location.Unknown
        ), f"Mismatched Location.  extend_to {extend_to.location}, new {location.name}"
        assert (
            msk.zoom == extend_to.zoom
        ), f"Mismatched zoom.  extend_to {extend_to.zoom}, new {msk.zoom}"

        for key, value in ctd_list.items():
            extend_to[key] = value
        return extend_to
