from BIDS import BIDS_Global_info

if __name__ == "__main__":
    global_bids = BIDS_Global_info(['/media/data/robert/datasets/spinegan_T2w_all_reg_iso/sourcedata/translated/'],["rawdata","derivatives"])
    for name