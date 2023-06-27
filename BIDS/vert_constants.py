from typing import Tuple, Dict, Literal
import BIDS.logger.log_file as log_file

log = log_file.Reflection_Logger()
logging = bool | log_file.Logger_Interface
# R: Right, L: Left; S: Superio (up), I: Inverior (down); A: Anterior (front), P: Posterior (back)
Directions = Literal["R", "L", "S", "I", "A", "P"]
Ax_Codes = tuple[Directions, Directions, Directions]
LABEL_MAX = 256
Zooms = Tuple[float, float, float]

Centroid_Dict = Dict[int, Tuple[float, float, float]]

Label_Map = dict[int | str, int | str] | dict[str, str] | dict[int, int]

from enum import Enum


class Location(Enum):
    Unknown = 0
    # Vertebral subregions
    Vertebra_Full = 40  # TODO Requires special processing
    Arcus_Vertebrae = 41
    Spinosus_Process = 42
    Costal_Process_Left = 43
    Costal_Process_Right = 44
    Superior_Articular_Left = 45
    Superior_Articular_Right = 46
    Inferior_Articular_Left = 47
    Inferior_Articular_Right = 48
    Vertebra_Corpus_border = 49
    Vertebra_Corpus = 50
    Dens_axis = 51  # TODO Was does it mean
    Vertebral_Body_Endplate_Superior = 52
    Vertebral_Body_Endplate_Inferior = 53
    Superior_Articulate_Process_Facet_Joint_Surface_Left = 54
    Superior_Articulate_Process_Facet_Joint_Surface_Right = 55
    Inferior_Articulate_Process_Facet_Joint_Surface_Left = 56
    Inferior_Articulate_Process_Facet_Joint_Surface_Right = 57
    Vertebra_Disc_Superior = 58
    Vertebra_Disc_Inferior = 59
    Vertebra_Disc = 100
    Spinal_Cord = 60
    # 60-80 Free
    # Muscle inserts
    # 81-91
    Muscle_Inserts_Spinosus_Process = 81
    Muscle_Inserts_Transverse_Process_left = 83
    Muscle_Inserts_Transverse_Process_right = 82
    Muscle_Inserts_Vertebral_Body_left = 84
    Muscle_Inserts_Vertebral_Body_right = 85
    Muscle_Inserts_Articulate_Process_Inferior_left = 86
    Muscle_Inserts_Articulate_Process_Inferior_right = 87
    Muscle_Inserts_Articulate_Process_Superior_left = 88
    Muscle_Inserts_Articulate_Process_Superior_right = 89
    Muscle_Inserts_Rib_left = 90
    Muscle_Inserts_Rib_right = 91
    # Ligament attachment points
    # 101-151
    Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left = 109
    Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median = 101
    Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right = 117
    Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Left = 111
    Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Median = 103
    Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Right = 119

    Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Left = 110
    Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Median = 102
    Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Right = 118
    Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Left = 112
    Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Median = 104
    Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Right = 120

    Ligament_Attachment_Point_Flava_Superior_Left = 149
    Ligament_Attachment_Point_Flava_Superior_Median = 125
    Ligament_Attachment_Point_Flava_Superior_Right = 141
    Ligament_Attachment_Point_Flava_Inferior_Left = 151
    Ligament_Attachment_Point_Flava_Inferior_Median = 127
    Ligament_Attachment_Point_Flava_Inferior_Right = 143

    Ligament_Attachment_Point_Interspinosa_Superior_Left = 133
    Ligament_Attachment_Point_Interspinosa_Superior_Right = 134
    Ligament_Attachment_Point_Interspinosa_Inferior_Left = 135
    Ligament_Attachment_Point_Interspinosa_Inferior_Right = 136
    # Additional body points
    # 108-143
    Additional_Vertebral_Body_Anterior_Central_left = 116
    Additional_Vertebral_Body_Anterior_Central_median = 108
    Additional_Vertebral_Body_Anterior_Central_right = 124
    Additional_Vertebral_Body_Posterior_Central_left = 114
    Additional_Vertebral_Body_Posterior_Central_median = 106
    Additional_Vertebral_Body_Posterior_Central_right = 122
    Additional_Vertebral_Body_Middle_Superior_left = 113
    Additional_Vertebral_Body_Middle_Superior_median = 105
    Additional_Vertebral_Body_Middle_Superior_right = 121
    Additional_Vertebral_Body_Middle_Inferior_left = 115
    Additional_Vertebral_Body_Middle_Inferior_median = 107
    Additional_Vertebral_Body_Middle_Inferior_right = 123

    Multi = 256


def vert_subreg_labels(with_border: bool = True) -> list[Location]:
    labels = [
        Location.Arcus_Vertebrae,
        Location.Spinosus_Process,
        Location.Costal_Process_Left,
        Location.Costal_Process_Right,
        Location.Superior_Articular_Left,
        Location.Superior_Articular_Right,
        Location.Inferior_Articular_Left,
        Location.Inferior_Articular_Right,
        Location.Vertebra_Corpus,
    ]
    if with_border:
        labels.append(Location.Vertebra_Corpus_border)
    return labels


# fmt: off

subreg_idx2name = {}
for k in range(255):
    try:
        subreg_idx2name[k] = Location(k).name
    except Exception:
        pass
v_idx2name = {
     1: "C1",     2: "C2",     3: "C3",     4: "C4",     5: "C5",     6: "C6",     7: "C7", 
     8: "T1",     9: "T2",    10: "T3",    11: "T4",    12: "T5",    13: "T6",    14: "T7",    15: "T8",    16: "T9",    17: "T10",   18: "T11",   19: "T12", 28: "T13",
    20: "L1",    21: "L2",    22: "L3",    23: "L4",    24: "L5",    25: "L6",    
    26: "S1",    29: "S2",    30: "S3",    31: "S4",    32: "S5",    33: "S6",
    27: "Cocc", **subreg_idx2name
}
v_name2idx = {value: key for key,value in v_idx2name.items()}
v_idx_order = list(v_idx2name.keys())

# fmt: on

conversion_poi = {
    "SSL": 81,  # this POI is not included in our POI list
    "ALL_CR_S": 109,
    "ALL_CR": 101,
    "ALL_CR_D": 117,
    "ALL_CA_S": 111,
    "ALL_CA": 103,
    "ALL_CA_D": 119,
    "PLL_CR_S": 110,
    "PLL_CR": 102,
    "PLL_CR_D": 118,
    "PLL_CA_S": 112,
    "PLL_CA": 104,
    "PLL_CA_D": 120,
    "FL_CR_S": 149,
    "FL_CR": 125,
    "FL_CR_D": 141,
    "FL_CA_S": 151,
    "FL_CA": 127,
    "FL_CA_D": 143,
    "ISL_CR": 134,
    "ISL_CA": 136,
    "ITL_S": 142,
    "ITL_D": 144,
}
conversion_poi2text = {k: v for v, k in conversion_poi.items()}
