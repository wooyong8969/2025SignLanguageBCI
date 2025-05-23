classdef EEGPreprocessor
    properties
        EEG         % EEGLAB EEG 구조체
        fs          % 샘플링 주파수
        labels      % 자극 라벨
        chanlocs    % 전극 위치 정보
    end

    methods
        %% 생성자
        function obj = EEGPreprocessor(eeg_struct, labels, chanlocs)
            obj.EEG = eeg_struct;
            obj.fs = eeg_struct.srate;
            obj.labels = labels;
            obj.chanlocs = chanlocs;
        end

        %% Bandpass 필터링
        function obj = applyBandpass(obj, low_cut, high_cut)
            [b, a] = butter(4, [low_cut high_cut] / (obj.fs / 2), 'bandpass');
            for ch = 1:size(obj.EEG.data, 1)
                for ep = 1:size(obj.EEG.data, 3)
                    obj.EEG.data(ch,:,ep) = filtfilt(b, a, double(obj.EEG.data(ch,:,ep)));
                end
            end
            disp('Bandpass filtering complete.');
        end

        %% Notch 필터링
        function obj = applyNotch(obj, notch_freq)
            wo = notch_freq / (obj.fs / 2);
            bw = wo / 35;
            [b, a] = iirnotch(wo, bw);
            for ch = 1:size(obj.EEG.data, 1)
                for ep = 1:size(obj.EEG.data, 3)
                    obj.EEG.data(ch,:,ep) = filtfilt(b, a, obj.EEG.data(ch,:,ep));
                end
            end
            disp('Notch filtering complete.');
        end

        %% ICA 수행
        function obj = runICA(obj)
            obj.EEG = eeg_checkset(obj.EEG);
            obj.EEG = pop_runica(obj.EEG, 'extended', 1);
            disp('ICA complete.');
        end

        %% ADJUST 실행 (자동 제거 X)
        function obj = runADJUST(obj)
            report_name = fullfile(pwd, ['ADJUST_Report_' datestr(now,'yyyymmdd_HHMMSS') '.txt']);
        
            backupICA = struct();
            if isfield(obj.EEG, 'icaweights'),  backupICA.icaweights  = obj.EEG.icaweights; end
            if isfield(obj.EEG, 'icasphere'),   backupICA.icasphere   = obj.EEG.icasphere; end
            if isfield(obj.EEG, 'icachansind'), backupICA.icachansind = obj.EEG.icachansind; end
        
            try
                % reshape compatibility 조정
                data2D = reshape(obj.EEG.data, size(obj.EEG.data, 1), []);
                ica_result = obj.EEG.icaweights * obj.EEG.icasphere * data2D;
        
                pca_dim = size(ica_result, 1);
                obj.EEG.icaact = reshape(ica_result, pca_dim, size(obj.EEG.data, 2), size(obj.EEG.data, 3));
        
                EEG_adjusted = ADJUST(obj.EEG, report_name);
                if iscell(EEG_adjusted), EEG_adjusted = EEG_adjusted{1}; end
        
                if ~isfield(EEG_adjusted, 'icaweights') && isfield(backupICA, 'icaweights')
                    EEG_adjusted.icaweights  = backupICA.icaweights;
                    EEG_adjusted.icasphere   = backupICA.icasphere;
                    EEG_adjusted.icachansind = backupICA.icachansind;
                end
        
                obj.EEG = EEG_adjusted;
                disp('ADJUST complete.');
            catch ME
                warning('ADJUST 실행 중 오류 발생: %s', ME.message);
            end
        end


        %% 평균 재참조
        function obj = rereference(obj)
            obj.EEG = pop_reref(obj.EEG, []);
            disp('Re-referencing complete.');
        end

        %% EEG 구조 반환
        function eeg_out = getEEG(obj)
            eeg_out = obj.EEG;
        end

        %% ICA 컴포넌트 수동 제거
        function obj = removeComponents(obj)
            pop_selectcomps(obj.EEG);  % 컴포넌트 시각화
            comps = input('제거할 ICA 컴포넌트 번호를 콤마로 구분하여 입력하세요 (예: 1,3,5): ', 's');
            comps_to_remove = str2num(comps);
            obj.EEG = pop_subcomp(obj.EEG, comps_to_remove, 0);
            disp('Remove Components complete.');
        end
    end
end
