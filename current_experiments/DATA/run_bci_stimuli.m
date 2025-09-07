function run_bci_stimuli(stim_type)
    % stim_type: 'text', 'sign', 'multi' 중 하나

    % 자극 클래스 정의
    classes = {'hello', 'helpme', 'sorry', 'thanku'};
    keys = {'K', 'L', 'M', 'N'};  % class별 키보드 입력
    n_repeat = 5;  % 각 class당 반복 횟수

    % Java keyboard 객체
    import java.awt.Robot;
    import java.awt.event.*;
    kb = Robot();

    % Psychtoolbox 초기화
    Screen('Preference', 'SkipSyncTests', 1);  % 테스트용
    screens = Screen('Screens');
    screenNumber = max(screens);
    [w, ~] = Screen('OpenWindow', screenNumber, [255 255 255]);  % 흰 배경
    Screen('TextFont', w, 'NanumGothic');  % 한글 텍스트용 (필요 시)

    % 이미지 사전 로딩
    for i = 1:length(classes)
        fname = fullfile('StimuliImage', sprintf('%s_%s.PNG', stim_type, classes{i}));
        imgs.(classes{i}) = imread(fname);
    end
    fixation_img = imread(fullfile('StimuliImage', 'fixation.png'));
    blank_img = uint8(255 * ones(size(fixation_img)));  % 흰 배경

    % trial 리스트 생성 및 셔플
    trial_list = repelem(1:4, n_repeat);  % class index
    trial_list = trial_list(randperm(length(trial_list)));

    % 자극 루프 시작
    for t = 1:length(trial_list)
        class_idx = trial_list(t);
        class_name = classes{class_idx};
        key = keys{class_idx};

        fprintf('Trial %d: %s\n', t, class_name);

        % === 1. Fixation 3초 ===
        tex = Screen('MakeTexture', w, fixation_img);
        Screen('DrawTexture', w, tex); Screen('Flip', w); Screen('Close', tex);
        WaitSecs(3);

        % === 2. 자극 제시 3초 ===
        tex = Screen('MakeTexture', w, imgs.(class_name));
        Screen('DrawTexture', w, tex); Screen('Flip', w); Screen('Close', tex);
        
        % → 마커 전송 (키보드 입력)
        keycode = eval(sprintf('KeyEvent.VK_%s', key));
        kb.keyPress(keycode); pause(0.05); kb.keyRelease(keycode);

        WaitSecs(3);

        % === 3. Fixation 1초 ===
        tex = Screen('MakeTexture', w, fixation_img);
        Screen('DrawTexture', w, tex); Screen('Flip', w); Screen('Close', tex);
        WaitSecs(1);

        % === 4. 상상 2초 (빈 화면) ===
        tex = Screen('MakeTexture', w, blank_img);
        Screen('DrawTexture', w, tex); Screen('Flip', w); Screen('Close', tex);
        WaitSecs(2);
    end

    % 종료 메시지
    Screen('TextSize', w, 32);
    DrawFormattedText(w, '실험이 종료되었습니다.', 'center', 'center', [0 0 0]);
    Screen('Flip', w); WaitSecs(3);

    Screen('CloseAll'); ShowCursor;
end
