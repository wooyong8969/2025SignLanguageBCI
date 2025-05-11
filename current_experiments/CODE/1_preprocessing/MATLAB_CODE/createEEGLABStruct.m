function EEG = createEEGLABStruct(epochedEEG, labels, fs, loc_path, selected_labels, save_filename)
    % 차원 변환
    num_epochs = length(epochedEEG);
    [num_samples, num_channels] = size(epochedEEG{1});
    EEG_data = zeros(num_channels, num_samples, num_epochs);
    for i = 1:num_epochs
        EEG_data(:, :, i) = epochedEEG{i}';
    end

    % 빈 EEG 구조체 생성
    EEG = eeg_emptyset();
    EEG.data = EEG_data;
    EEG.srate = fs;
    EEG.nbchan = num_channels;
    EEG.pnts = num_samples;
    EEG.trials = num_epochs;
    EEG.xmin = 0;
    EEG.xmax = (num_samples - 1) / fs;
    EEG.labels = labels;

    % 채널 위치 설정
    chanlocs = readlocs(loc_path, 'filetype', 'chanedit');
    [~, idx] = ismember(selected_labels, {chanlocs.labels});
    EEG.chanlocs = chanlocs(idx(idx > 0));

    % CAR
    EEG.data = EEG.data - mean(EEG.data, 2);

    % μV 변환 (Gain 24 기준)
    EEG.data = EEG.data * 0.02235;
end
