
CSF_Map_hr = niftiread('magnitude_resamToCsi_bet_pve_0.nii');
GM_Map_hr  = niftiread('magnitude_resamToCsi_bet_pve_1.nii');
WM_Map_hr  = niftiread('magnitude_resamToCsi_bet_pve_2.nii');

nii_header = niftiinfo('mask.nii');
out_path = pwd;
dimensions_lr = [22 22 21];

[GM_lr, WM_lr, CSF_lr] = Generate_Segmentations_DeutRes(out_path, GM_Map_hr, WM_Map_hr, CSF_Map_hr, dimensions_lr, nii_header);

