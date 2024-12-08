%% 读取音频处理源文件
[male_audio, fs_male] = audioread('HJrecord.wav');
fs = fs_male;  % 确保采样率一致

%% 为音频添加白噪声
% 设置信噪比
SNR = 10;  % 10 dB 的信噪比
noisy_audio = addWhiteNoise(male_audio, SNR);

%% 定义移位的音频（假设我们用 10 毫秒的延迟）
delay_samples = round(0.01 * fs);  % 延迟 10 毫秒
male_audio_shifted = [zeros(delay_samples, 1); male_audio(1:end-delay_samples)];

%% 使用LMS算法进行噪声消除
% 假设 male_audio 为清晰人声，noisy_audio 为带噪音的信号，male_audio_shifted 为噪声参考
mu = 0.01;  % LMS算法的步长
filter_order = 32;  % 自适应滤波器的阶数
[denoised_audio, ~] = lms_noise_cancellation(noisy_audio, male_audio, male_audio_shifted, mu, filter_order);  % 传递带噪音信号，清晰人声，和噪声参考信号
% 计算带噪音信号和降噪后信号的平均能量比
energy_ratio = sum(noisy_audio.^2) / sum(denoised_audio.^2);
% 对降噪后的音频进行归一化处理
denoised_audio = denoised_audio / max(abs(male_audio));  % 除以最大幅度进行归一化

%% 使用带通滤波器增强语音
low_cutoff = 1000;   % 低频截止频率（单位：Hz）
high_cutoff = 5000;  % 高频截止频率（单位：Hz）

% 确保截止频率在 (0, 1) 范围内，进行归一化处理
nyquist_frequency = fs / 2;  % Nyquist 频率
low_cutoff_normalized = low_cutoff / nyquist_frequency;  % 归一化低频截止
high_cutoff_normalized = high_cutoff / nyquist_frequency;  % 归一化高频截止

% 使用归一化后的截止频率进行带通滤波器设计
[b, a] = butter(4, [low_cutoff_normalized, high_cutoff_normalized], 'bandpass');
enhanced_audio = filter(b, a, male_audio_shifted);  % 使用移位后的音频进行滤波


%% 声纹转换
% 声纹提取
extractVoiceprint('LJYrecord.wav');


%% 显示和播放处理效果
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

% 播放转换前的音频
disp('Source Audio');
sound(male_audio, fs);  % 播放原始音频
pause(length(male_audio) / fs + 1);  % 等待播放结束

% 播放存在噪音的的音频
disp('Added noise');
sound(noisy_audio, fs);  % 播放原始音频
pause(length(male_audio) / fs + 1);  % 等待播放结束

% 播放转换后的音频
disp('After Filtered');
sound(enhanced_audio, fs);  % 播放转换后的音频

% 保存最终处理后的音频
audiowrite('converted_female_voice.wav', enhanced_audio, fs);  % 保存处理后的音频

%% 函数模块
% 使用样条插值进行音高转换 Edit data:2024/11/14 12:26 By Huangjie
function shifted_audio = pitchShift(audio, factor, fs)
    N = length(audio);
    t = (0:N-1) / fs;  % 原始时间轴
    new_t = t * factor;  % 修改后的时间轴，缩放音高
    
    % 使用样条插值进行插值
    shifted_audio = interp1(t, audio, new_t, 'spline', 'extrap');  % 使用样条插值
end

%使用MFCC的声纹提取函数 
function [mfcc_features, fs] = extractVoiceprint(audio_file)
    % 输入：
    % audio_file - 要提取声纹的音频文件路径
    % 输出：
    % mfcc_features - 提取的MFCC特征矩阵
    % fs - 音频的采样率
    
    % 读取音频文件
    [audio, fs] = audioread(audio_file);  % fs为采样率，audio为音频信号
    
    % 如果音频为立体声，转换为单声道
    if size(audio, 2) == 2
        audio = mean(audio, 2);  % 转换为单声道
    end
    
    % 计算MFCC特征
    % 参数说明：
    % audio - 输入的音频信号
    % fs - 音频的采样率
    % 'NumCoeffs' - 提取的MFCC系数数量（一般选取13个）
    % 'WindowLength' - 窗口长度
    % 'OverlapLength' - 重叠长度
    mfcc_features = mfcc(audio, fs, 'NumCoeffs', 13, 'WindowLength', 256, 'OverlapLength', 128);
    
    % mfcc_features为一个矩阵，列数代表MFCC的系数数（13个），行数为每一帧的MFCC特征
    % 可以对特征做进一步处理（如均值化，标准化等）
    
    % 显示MFCC特征（如果需要可视化）
    figure;
    imagesc(mfcc_features');
    colormap jet;
    title('MFCC Features');
    xlabel('Frame Index');
    ylabel('MFCC Coefficients');
    colorbar;
    
    % 返回MFCC特征
end

function [output, w] = lms_noise_cancellation(noisy_signal, desired_signal, noise_reference, mu, filter_order)
    % 输入：
    % noisy_signal - 带噪音的信号（原始音频）
    % desired_signal - 清晰人声（目标信号）
    % noise_reference - 噪声参考信号（例如移位后的清晰人声）
    % mu - LMS算法的步长参数（控制收敛速度）
    % filter_order - 自适应滤波器的阶数
    %
    % 输出：
    % output - 噪声消除后的信号
    % w - 最终的滤波器系数

    N = length(noisy_signal);  % 信号长度
    output = zeros(N, 1);  % 初始化输出信号
    w = zeros(filter_order, 1);  % 初始化滤波器系数
    x_buffer = zeros(filter_order, 1);  % 滤波器输入信号缓冲区
    
    % LMS算法迭代
    for n = filter_order:N
        % 获取当前参考信号的一小段
        x_buffer = noise_reference(n-filter_order+1:n);
        
        % 滤波器的输出，即预测的噪声信号
        y_hat = w' * x_buffer;
        
        % 计算误差信号（带噪音信号与预测的噪声信号之差）
        e = desired_signal(n) - y_hat;  % 使用清晰语音作为期望输出
        
        % 更新滤波器系数
        w = w + mu * e * x_buffer;
        
        % 存储去噪后的信号
        output(n) = e;
    end
end

function noisy_audio = addWhiteNoise(audio, SNR)
    % 输入：
    % audio  - 原始音频信号（单声道）
    % SNR    - 信噪比，单位：dB（默认值为10 dB）
    %
    % 输出：
    % noisy_audio  - 添加白噪声后的音频信号

    if nargin < 2
        SNR = 10;  % 默认信噪比为 10 dB
    end

    % 生成白噪声
    noise = randn(size(audio));  % 生成标准正态分布的随机噪声
    
    % 计算信号和噪声的功率
    signal_power = sum(audio.^2) / length(audio);  % 原始信号的功率
    noise_power = sum(noise.^2) / length(noise);  % 噪声的功率
    
    % 计算噪声的缩放因子，使得指定的信噪比得以实现
    scaling_factor = sqrt(signal_power / (noise_power * 10^(SNR / 10)));
    
    % 将噪声添加到音频信号中
    noisy_audio = audio + scaling_factor * noise;  % 将噪声添加到音频信号中
end
