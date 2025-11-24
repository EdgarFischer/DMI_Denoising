%% tMPPCA 5D batch denoising script (gemeinsam über alle 5 Dimensionen)

clear; clc;

% --- Basisverzeichnis relativ zur aktuellen Datei setzen ---
base_path = fileparts(mfilename('fullpath'));

% --- Liste der Eingabeordner (relativ zu base_path) ---
folders = { ...
    '../../datasets/Tumor_2_normalized', ...
    % weitere Ordner hier ergänzen
    };

% --- Parameter ---
window = [5 5 5];   % räumliches Fenster

for i = 1:numel(folders)
    input_folder = fullfile(base_path, folders{i});
    input_file   = fullfile(input_folder, 'data.mat');

    fprintf('Lade %s ...\n', input_file);
    S = load(input_file);      % erwartet Variable 'Data'
    data = S.Data;

    % --- Ausgabepfad ---
    output_folder = fullfile(base_path, [folders{i} '_tMPPCA_5D_On_All_Reps']);
    if ~exist(output_folder, 'dir'); mkdir(output_folder); end
    output_file = fullfile(output_folder, 'data.mat');

    % --- Prüfen, dass Daten 5D sind ---
    if ndims(data) ~= 5
        error('Data in %s ist nicht 5-dimensional.', folders{i});
    end

    % --- 5D-denoising in einem Rutsch ---
    % Indizes: {1:3} = Voxel (räumlich, fensterbasiert), 4 & 5 = Mess-/Modi-Indizes
    fprintf('Starte 5D tMPPCA ...\n');
    [denoised_data, Sigma2, P, SNR_gain] = denoise_recursive_tensor( ...
        data, window, 'indices', {4, 5, 1:2, 3});  %4 5 1:3

    % SNR_gain ist voxelweise (3D). Gebe Mittelwert aus:
    snr_gain_mean = mean(SNR_gain(:), 'omitnan');
    fprintf('  Mittlerer SNR gain (voxel-mean): %.2f\n', snr_gain_mean);

    % --- Speichern ---
    fprintf('Speichere Ergebnis nach %s ...\n', output_file);
    Data = denoised_data;
    save(output_file, 'Data', '-v7.3');
end

fprintf('Fertig!\n');
