function [epochedEEG, labels] = epochEEGfromCSV(csv_path, epoch_table_path, fs, cut_time)

    % CSV와 엑셀 불러오기
    raw_data = readmatrix(csv_path);
    epochs_table = readtable(epoch_table_path);
    
    % 1열은 sample index, 2~17열은 EEG
    EEG_data = raw_data(cut_time*fs+1:end, 2:17);
    
    % epoch 정보
    start_col = epochs_table.Properties.VariableNames{1};
    end_col = epochs_table.Properties.VariableNames{2};
    stimulus_col = epochs_table.Properties.VariableNames{3};

    % 에포킹
    epochedEEG = {};
    labels = {};
    for i = 1:height(epochs_table)
        start_sample = round(epochs_table{i, start_col} * fs) + 1;
        end_sample = round(epochs_table{i, end_col} * fs);
        epochedEEG{end+1} = EEG_data(start_sample:end_sample, :);
        labels{end+1} = string(epochs_table{i, stimulus_col});
    end

end