function test_bci_trigger()
    import java.awt.Robot;
    import java.awt.event.*;
    kb = Robot();

    Screen('Preference', 'SkipSyncTests', 1);
    screens = Screen('Screens');
    screenNumber = max(screens);
    [w, ~] = Screen('OpenWindow', screenNumber, [255 255 255]);

    fixation = imread('StimuliImage/fixation.png');
    blank = uint8(255 * ones(size(fixation)));

    % 1. 고정 십자 2초
    tex = Screen('MakeTexture', w, fixation);
    Screen('DrawTexture', w, tex); Screen('Flip', w); Screen('Close', tex);
    WaitSecs(2);

    % 2. 자극 (예: HELLO) 2초 + 키보드 입력 (K)
    img = imread('StimuliImage/multi_hello.PNG');
    tex = Screen('MakeTexture', w, img);
    Screen('DrawTexture', w, tex); Screen('Flip', w); Screen('Close', tex);

    kb.keyPress(KeyEvent.VK_K); pause(0.05); kb.keyRelease(KeyEvent.VK_K);  % D11 마커

    WaitSecs(2);

    % 3. 빈 화면 2초
    tex = Screen('MakeTexture', w, blank);
    Screen('DrawTexture', w, tex); Screen('Flip', w); Screen('Close', tex);
    WaitSecs(2);

    Screen('CloseAll');
    ShowCursor;
end
