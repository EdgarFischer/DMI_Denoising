%% tMPPCA batch denoising script

clear; clc;

% --- Basisverzeichnis relativ zur aktuellen Datei setzen ---
base_path = fileparts(mfilename('fullpath'));

% --- Liste der Eingabeordner (relativ zu base_path) ---
folders = { ...
    '../../datasets/Tumor_2_normalized', ...
    % weitere Ordner hier ergänzen
    };

% --- Parameter ---
window = [5 5 5];   % Beispiel-Fenstergröße, ggf. anpassen

% --- Schleife über alle Ordner ---
for i = 1:numel(folders)
    input_folder = fullfile(base_path, folders{i});
    input_file   = fullfile(input_folder, 'data.mat');

    fprintf('Lade %s ...\n', input_file);
    S = load(input_file);  % lädt Variable 'Data'
    data = S.Data;

    % --- Ausgabepfad vorbereiten ---
    output_folder = fullfile(base_path, [folders{i} '_tMPPCA_4D']);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    output_file = fullfile(output_folder, 'data.mat');

    % --- Prüfen, dass Daten 5D sind ---
    if ndims(data) ~= 5
        error('Data in %s ist nicht 5-dimensional.', folders{i});
    end

    % --- Neue Variable für das Ergebnis ---
    denoised_data = zeros(size(data), 'like', data);
    SNR_gain_all = zeros(1, size(data,5));

    % --- Über letzte Dimension loopen ---
    for t = 1:size(data,5)
        fprintf('  Verarbeite Index %d/%d ...\n', t, size(data,5));
        current_data = squeeze(data(:,:,:,:,t));

        [denoised_block, ~, SNR_gain] = denoise_recursive_tensor(current_data, window, 'indices',{4 1:3});
        
        % Mittelwert über alle Voxel für diesen Index
        SNR_gain_all(t) = mean(SNR_gain(:), 'omitnan');

        fprintf('    Mittlerer SNR gain für Index %d: %.2f\n', t, SNR_gain_all(t));

        denoised_data(:,:,:,:,t) = denoised_block;
    end

    % --- Speichern ---
    fprintf('Speichere Ergebnis nach %s ...\n', output_file);
    Data = denoised_data;
    save(output_file, 'Data', '-v7.3');
end

fprintf('Fertig!\n');
