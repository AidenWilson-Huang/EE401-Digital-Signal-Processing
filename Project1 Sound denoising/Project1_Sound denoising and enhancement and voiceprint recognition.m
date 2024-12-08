%% ��ȡ��Ƶ����Դ�ļ�
[male_audio, fs_male] = audioread('HJrecord.wav');
fs = fs_male;  % ȷ��������һ��

%% Ϊ��Ƶ��Ӱ�����
% ���������
SNR = 10;  % 10 dB �������
noisy_audio = addWhiteNoise(male_audio, SNR);

%% ������λ����Ƶ������������ 10 ������ӳ٣�
delay_samples = round(0.01 * fs);  % �ӳ� 10 ����
male_audio_shifted = [zeros(delay_samples, 1); male_audio(1:end-delay_samples)];

%% ʹ��LMS�㷨������������
% ���� male_audio Ϊ����������noisy_audio Ϊ���������źţ�male_audio_shifted Ϊ�����ο�
mu = 0.01;  % LMS�㷨�Ĳ���
filter_order = 32;  % ����Ӧ�˲����Ľ���
[denoised_audio, ~] = lms_noise_cancellation(noisy_audio, male_audio, male_audio_shifted, mu, filter_order);  % ���ݴ������źţ������������������ο��ź�
% ����������źźͽ�����źŵ�ƽ��������
energy_ratio = sum(noisy_audio.^2) / sum(denoised_audio.^2);
% �Խ�������Ƶ���й�һ������
denoised_audio = denoised_audio / max(abs(male_audio));  % ���������Ƚ��й�һ��

%% ʹ�ô�ͨ�˲�����ǿ����
low_cutoff = 1000;   % ��Ƶ��ֹƵ�ʣ���λ��Hz��
high_cutoff = 5000;  % ��Ƶ��ֹƵ�ʣ���λ��Hz��

% ȷ����ֹƵ���� (0, 1) ��Χ�ڣ����й�һ������
nyquist_frequency = fs / 2;  % Nyquist Ƶ��
low_cutoff_normalized = low_cutoff / nyquist_frequency;  % ��һ����Ƶ��ֹ
high_cutoff_normalized = high_cutoff / nyquist_frequency;  % ��һ����Ƶ��ֹ

% ʹ�ù�һ����Ľ�ֹƵ�ʽ��д�ͨ�˲������
[b, a] = butter(4, [low_cutoff_normalized, high_cutoff_normalized], 'bandpass');
enhanced_audio = filter(b, a, male_audio_shifted);  % ʹ����λ�����Ƶ�����˲�


%% ����ת��
% ������ȡ
extractVoiceprint('LJYrecord.wav');


%% ��ʾ�Ͳ��Ŵ���Ч��
figure;
subplot(3,1,1);
plot(male_audio);
title('Original');
subplot(3,1,2);
plot(noisy_audio);
title('Noisy Audio');
subplot(3,1,3);
plot(enhanced_audio);
title('After Noise Cancellation');

% ����ת��ǰ����Ƶ
disp('Source Audio');
sound(male_audio, fs);  % ����ԭʼ��Ƶ
pause(length(male_audio) / fs + 1);  % �ȴ����Ž���

% ���Ŵ��������ĵ���Ƶ
disp('Added noise');
sound(noisy_audio, fs);  % ����ԭʼ��Ƶ
pause(length(male_audio) / fs + 1);  % �ȴ����Ž���

% ����ת�������Ƶ
disp('After Filtered');
sound(enhanced_audio, fs);  % ����ת�������Ƶ

% �������մ�������Ƶ
audiowrite('converted_female_voice.wav', enhanced_audio, fs);  % ���洦������Ƶ

%% ����ģ��
% ʹ��������ֵ��������ת�� Edit data:2024/11/14 12:26 By Huangjie
function shifted_audio = pitchShift(audio, factor, fs)
    N = length(audio);
    t = (0:N-1) / fs;  % ԭʼʱ����
    new_t = t * factor;  % �޸ĺ��ʱ���ᣬ��������
    
    % ʹ��������ֵ���в�ֵ
    shifted_audio = interp1(t, audio, new_t, 'spline', 'extrap');  % ʹ��������ֵ
end

%ʹ��MFCC��������ȡ���� 
function [mfcc_features, fs] = extractVoiceprint(audio_file)
    % ���룺
    % audio_file - Ҫ��ȡ���Ƶ���Ƶ�ļ�·��
    % �����
    % mfcc_features - ��ȡ��MFCC��������
    % fs - ��Ƶ�Ĳ�����
    
    % ��ȡ��Ƶ�ļ�
    [audio, fs] = audioread(audio_file);  % fsΪ�����ʣ�audioΪ��Ƶ�ź�
    
    % �����ƵΪ��������ת��Ϊ������
    if size(audio, 2) == 2
        audio = mean(audio, 2);  % ת��Ϊ������
    end
    
    % ����MFCC����
    % ����˵����
    % audio - �������Ƶ�ź�
    % fs - ��Ƶ�Ĳ�����
    % 'NumCoeffs' - ��ȡ��MFCCϵ��������һ��ѡȡ13����
    % 'WindowLength' - ���ڳ���
    % 'OverlapLength' - �ص�����
    mfcc_features = mfcc(audio, fs, 'NumCoeffs', 13, 'WindowLength', 256, 'OverlapLength', 128);
    
    % mfcc_featuresΪһ��������������MFCC��ϵ������13����������Ϊÿһ֡��MFCC����
    % ���Զ���������һ���������ֵ������׼���ȣ�
    
    % ��ʾMFCC�����������Ҫ���ӻ���
    figure;
    imagesc(mfcc_features');
    colormap jet;
    title('MFCC Features');
    xlabel('Frame Index');
    ylabel('MFCC Coefficients');
    colorbar;
    
    % ����MFCC����
end

function [output, w] = lms_noise_cancellation(noisy_signal, desired_signal, noise_reference, mu, filter_order)
    % ���룺
    % noisy_signal - ���������źţ�ԭʼ��Ƶ��
    % desired_signal - ����������Ŀ���źţ�
    % noise_reference - �����ο��źţ�������λ�������������
    % mu - LMS�㷨�Ĳ������������������ٶȣ�
    % filter_order - ����Ӧ�˲����Ľ���
    %
    % �����
    % output - ������������ź�
    % w - ���յ��˲���ϵ��

    N = length(noisy_signal);  % �źų���
    output = zeros(N, 1);  % ��ʼ������ź�
    w = zeros(filter_order, 1);  % ��ʼ���˲���ϵ��
    x_buffer = zeros(filter_order, 1);  % �˲��������źŻ�����
    
    % LMS�㷨����
    for n = filter_order:N
        % ��ȡ��ǰ�ο��źŵ�һС��
        x_buffer = noise_reference(n-filter_order+1:n);
        
        % �˲������������Ԥ��������ź�
        y_hat = w' * x_buffer;
        
        % ��������źţ��������ź���Ԥ��������ź�֮�
        e = desired_signal(n) - y_hat;  % ʹ������������Ϊ�������
        
        % �����˲���ϵ��
        w = w + mu * e * x_buffer;
        
        % �洢ȥ�����ź�
        output(n) = e;
    end
end

function noisy_audio = addWhiteNoise(audio, SNR)
    % ���룺
    % audio  - ԭʼ��Ƶ�źţ���������
    % SNR    - ����ȣ���λ��dB��Ĭ��ֵΪ10 dB��
    %
    % �����
    % noisy_audio  - ��Ӱ����������Ƶ�ź�

    if nargin < 2
        SNR = 10;  % Ĭ�������Ϊ 10 dB
    end

    % ���ɰ�����
    noise = randn(size(audio));  % ���ɱ�׼��̬�ֲ����������
    
    % �����źź������Ĺ���
    signal_power = sum(audio.^2) / length(audio);  % ԭʼ�źŵĹ���
    noise_power = sum(noise.^2) / length(noise);  % �����Ĺ���
    
    % �����������������ӣ�ʹ��ָ��������ȵ���ʵ��
    scaling_factor = sqrt(signal_power / (noise_power * 10^(SNR / 10)));
    
    % ��������ӵ���Ƶ�ź���
    noisy_audio = audio + scaling_factor * noise;  % ��������ӵ���Ƶ�ź���
end
