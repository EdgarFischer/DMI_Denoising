function [GM_Map_lr, WM_Map_lr, CSF_Map_lr] = Generate_Segmentations_DeutRes(out_path, GM_Map_hr, WM_Map_hr, CSF_Map_hr, dimensions_lr, nii_header)
    
    GM_Map_hr(GM_Map_hr < 0) = 0;
    WM_Map_hr(WM_Map_hr < 0) = 0;
    CSF_Map_hr(CSF_Map_hr < 0) = 0;

    summed_Map_hr = GM_Map_hr + WM_Map_hr + CSF_Map_hr;
    summed_Map_lr = imresize3(single(summed_Map_hr), [22,22,21], 'nearest');
    Brain_Mask_lr = logical(summed_Map_lr);

    GM_Map_hr = (GM_Map_hr./summed_Map_hr); % If properly normalised then just dividing each voxel by 1
    WM_Map_hr = (WM_Map_hr./summed_Map_hr); % If sum is over 1 then reducing content in each tissue
    CSF_Map_hr = (CSF_Map_hr./summed_Map_hr); % Everything outside mapped region is divided by zero so will be NaN

    GM_Map_hr(summed_Map_hr == 0) = 0;
    WM_Map_hr(summed_Map_hr == 0) = 0;
    CSF_Map_hr(summed_Map_hr == 0) = 0;

    % Make sum map and calculate in HR what the 'air' component is
    Sum_Map_hr = WM_Map_hr + GM_Map_hr + CSF_Map_hr;
    Non_MR_hr = 1-Sum_Map_hr;

    % 3D Fourier transform to k-space
    GM_Map_lr = fftshift(fft(ifftshift(GM_Map_hr,1), [], 1), 1);
    GM_Map_lr = fftshift(fft(ifftshift(GM_Map_lr,2), [], 2), 2);
    GM_Map_lr = fftshift(fft(ifftshift(GM_Map_lr,3), [], 3), 3);

    WM_Map_lr = fftshift(fft(ifftshift(WM_Map_hr,1), [], 1), 1);
    WM_Map_lr = fftshift(fft(ifftshift(WM_Map_lr,2), [], 2), 2);
    WM_Map_lr = fftshift(fft(ifftshift(WM_Map_lr,3), [], 3), 3);
    
    CSF_Map_lr = fftshift(fft(ifftshift(CSF_Map_hr,1), [], 1), 1);
    CSF_Map_lr = fftshift(fft(ifftshift(CSF_Map_lr,2), [], 2), 2);
    CSF_Map_lr = fftshift(fft(ifftshift(CSF_Map_lr,3), [], 3), 3);
    
    Non_MR_lr = fftshift(fft(ifftshift(Non_MR_hr,1), [], 1), 1);
    Non_MR_lr = fftshift(fft(ifftshift(Non_MR_lr,2), [], 2), 2);
    Non_MR_lr = fftshift(fft(ifftshift(Non_MR_lr,3), [], 3), 3);
    
    original_size = size(GM_Map_hr);
    start_index = floor((original_size - dimensions_lr) / 2) + 1;
    end_index = start_index + dimensions_lr - 1;

    % Cutting out desired resolution
    WM_Map_lr = WM_Map_lr(start_index(1):end_index(1),start_index(2):end_index(2),start_index(3):end_index(3)); % Check these numbers are right
    GM_Map_lr = GM_Map_lr(start_index(1):end_index(1),start_index(2):end_index(2),start_index(3):end_index(3)); % Check these numbers are right
    CSF_Map_lr = CSF_Map_lr(start_index(1):end_index(1),start_index(2):end_index(2),start_index(3):end_index(3)); % Check these numbers are right
    Non_MR_lr = Non_MR_lr(start_index(1):end_index(1),start_index(2):end_index(2),start_index(3):end_index(3)); % Check these numbers are right
    
    % Approx k space weighting of concentric ring trajectory
    GM_Map_lr=HammingFilter(GM_Map_lr,[1 2 3],100,'OuterProduct',1);
    WM_Map_lr=HammingFilter(WM_Map_lr,[1 2 3],100,'OuterProduct',1);
    CSF_Map_lr=HammingFilter(CSF_Map_lr,[1 2 3],100,'OuterProduct',1);
    Non_MR_lr=HammingFilter(Non_MR_lr,[1 2 3],100,'OuterProduct',1);

    % FoV shift correction
    for x=1:size(GM_Map_lr,1)
        GM_Map_lr(x,:,:)=GM_Map_lr(x,:,:)*exp(1i*pi*x/size(GM_Map_lr,1));
        WM_Map_lr(x,:,:)=WM_Map_lr(x,:,:)*exp(1i*pi*x/size(WM_Map_lr,1));
        CSF_Map_lr(x,:,:)=CSF_Map_lr(x,:,:)*exp(1i*pi*x/size(CSF_Map_lr,1));
        Non_MR_lr(x,:,:)=Non_MR_lr(x,:,:)*exp(1i*pi*x/size(Non_MR_lr,1));
    end

    for y=1:size(GM_Map_lr,2)
        GM_Map_lr(:,y,:)=GM_Map_lr(:,y,:)*exp(1i*pi*y/size(GM_Map_lr,2));
        WM_Map_lr(:,y,:)=WM_Map_lr(:,y,:)*exp(1i*pi*y/size(WM_Map_lr,2));
        CSF_Map_lr(:,y,:)=CSF_Map_lr(:,y,:)*exp(1i*pi*y/size(CSF_Map_lr,2));
        Non_MR_lr(:,y,:)=Non_MR_lr(:,y,:)*exp(1i*pi*y/size(Non_MR_lr,2));
    end

    % Fourier transform back to image space

    GM_Map_lr = fftshift(ifft(ifftshift(GM_Map_lr,1), [], 1), 1);
    GM_Map_lr = fftshift(ifft(ifftshift(GM_Map_lr,2), [], 2), 2);
    GM_Map_lr = abs(fftshift(ifft(ifftshift(GM_Map_lr,3), [], 3), 3));

    WM_Map_lr = fftshift(ifft(ifftshift(WM_Map_lr,1), [], 1), 1);
    WM_Map_lr = fftshift(ifft(ifftshift(WM_Map_lr,2), [], 2), 2);
    WM_Map_lr = abs(fftshift(ifft(ifftshift(WM_Map_lr,3), [], 3), 3));

    CSF_Map_lr = fftshift(ifft(ifftshift(CSF_Map_lr,1), [], 1), 1);
    CSF_Map_lr = fftshift(ifft(ifftshift(CSF_Map_lr,2), [], 2), 2);
    CSF_Map_lr = abs(fftshift(ifft(ifftshift(CSF_Map_lr,3), [], 3), 3));
    
    Non_MR_lr = fftshift(ifft(ifftshift(Non_MR_lr,1), [], 1), 1);
    Non_MR_lr = fftshift(ifft(ifftshift(Non_MR_lr,2), [], 2), 2);
    Non_MR_lr = abs(fftshift(ifft(ifftshift(Non_MR_lr,3), [], 3), 3));
    
    GM_Map_lr(Brain_Mask_lr==0) = 0;
    WM_Map_lr(Brain_Mask_lr==0) = 0;
    CSF_Map_lr(Brain_Mask_lr==0) = 0;
    Non_MR_lr(Brain_Mask_lr==0) = 0;

    summed_Map_lr = GM_Map_lr + WM_Map_lr + CSF_Map_lr + Non_MR_lr;

    % Normalising low rank masks
    GM_Map_lr = (GM_Map_lr ./ summed_Map_lr); GM_Map_lr(Brain_Mask_lr==0) = 0;
    WM_Map_lr = (WM_Map_lr ./ summed_Map_lr); WM_Map_lr(Brain_Mask_lr==0) = 0;
    CSF_Map_lr = (CSF_Map_lr ./ summed_Map_lr); CSF_Map_lr(Brain_Mask_lr==0) = 0;

    % Saving low resolution maps and brain mask as nifti files
    %GM_Map_lr = single(rot90(GM_Map_lr,2));
    %WM_Map_lr = single(rot90(WM_Map_lr,2));
    %CSF_Map_lr = single(rot90(CSF_Map_lr,2));
    niftiwrite(GM_Map_lr, strcat(out_path, '/GM_Map_lr.nii'),nii_header);
    niftiwrite(WM_Map_lr, strcat(out_path, '/WM_Map_lr.nii'),nii_header);
    niftiwrite(CSF_Map_lr, strcat(out_path, '/CSF_Map_lr.nii'),nii_header);

end