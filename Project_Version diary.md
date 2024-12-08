## Version 1.0

### 1. 项目进度概览

- **频谱分析**：比较了男声和女声的频率特性，分析它们的频谱差异。
- **音高转换**：通过改变音频的音高（通过调整采样率或时间伸缩），将男声转换为女声。
- **降噪与增强**：利用滤波器对音频信号进行降噪与增强，改善音频的清晰度和可懂度。
- **比较不同滤波器效果**：通过设计和应用不同类型的滤波器（如带通、低通和高通滤波器），对比它们对音频的影响。

### 2. **数据与方法**

#### 2.1 音频数据

- **男声数据**：使用 `male_voice.wav` 作为男声音频数据。
- **女声数据**：使用 `female_voice.wav` 作为女声音频数据。
- 假设两种音频的采样率一致，确保可以进行直接比较和处理。

#### 2.2 频谱分析

通过对男声和女声信号进行快速傅里叶变换（FFT），分析其频谱特性。具体分析目标是：

- **频率范围**：通过FFT获得频谱，并计算频率分布情况。
- **频谱差异**：比较男声与女声的频谱幅度，尤其是低频与高频成分的区别。

#### 2.3 音高转换

通过改变音高将男声转换为女声。实现方法是通过 **重采样** 或 **时间伸缩** 技术来改变音高。此步骤的目标是：

- **音高调整**：调整男声的音高，使其接近女声的音高范围。
- **声音保真度**：保证音高转换后音频的质量，避免音质损失或失真。

#### 2.4 降噪与语音增强

使用带通滤波器对音频信号进行降噪处理，特别是通过滤波增强语音频段（通常为 300 Hz 到 3400 Hz）。此步骤的目标是：

- **噪声抑制**：去除或减少低频和高频噪声，提高信号质量。
- **语音增强**：强化人类语音的频率成分，使语音更清晰易懂。

#### 2.5 滤波器设计与应用

设计并应用不同类型的滤波器（带通、低通和高通滤波器），对音频进行处理，并比较其效果。具体步骤包括：

- **带通滤波器**：用于增强语音频段，通常在 300 Hz 到 3400 Hz 之间。
- **低通滤波器**：用于去除高频噪声。
- **高通滤波器**：用于去除低频噪声或环境噪声。

### 3. **具体实验步骤**

#### 3.1 音频数据读取与采样率检查

- 使用 `audioread` 函数加载男声和女声音频数据。
- 检查两个音频文件的采样率是否一致，如果不一致则报错或重新采样。

#### 3.2 频谱分析

- 对男声和女声音频信号进行快速傅里叶变换（FFT）。
- 计算并绘制音频的频谱图，比较其频率分布，观察男声和女声在低频与高频上的差异。

#### 3.3 音高转换

- 设计音高转换函数 `pitchShift`，通过重采样或时间伸缩方法改变音高，使男声音频接近女声音频的音高。
- 播放转换后的音频，听觉上评估转换效果。

#### 3.4 降噪与增强处理

- 设计一个带通滤波器（300 Hz 到 3400 Hz），对音频进行语音增强处理。
- 使用 `filter` 函数应用滤波器，并绘制滤波前后的波形对比。

#### 3.5 滤波器对比实验

- 设计并应用三种不同的滤波器：带通滤波器、低通滤波器和高通滤波器。
- 对每种滤波器应用后的音频信号进行可视化，比较滤波效果，观察不同滤波器对音频的影响。

#### 3.6 音频播放与效果展示

- 使用 `sound` 或 `audioplayer` 播放处理前后的音频，直观对比原始音频和处理后音频的变化。
- 对比男声和转换后的女声的音质、音高以及清晰度。

### 4. **实验结果与分析**

#### 4.1 频谱分析结果

- 通过频谱图展示男声和女声的频率差异。
- 从频谱图中可以看出，男声的频率分布通常会集中在较低的频率区间（约 85 Hz 到 180 Hz），而女声则主要集中在较高的频率区间（约 165 Hz 到 255 Hz）。

#### 4.2 音高转换结果

- 通过音高转换，男声的频率范围应当调整到更接近女声的频率区间。
- 语音质量（清晰度、音质）可能会受到影响，因此可以进一步优化音高转换算法（例如通过 `resample` 或 `phase vocoder`）。

#### 4.3 降噪与增强效果

- 使用带通滤波器后，语音的清晰度得到了增强，同时低频噪声和高频噪声被有效去除。
- 降噪后，音频信号的语音部分更加突出，背景噪声被显著减少。

#### 4.4 滤波器比较

- 比较不同滤波器（带通、低通、高通）的效果，带通滤波器更适合增强语音频段，而低通和高通滤波器更适用于去除噪声。
- 每种滤波器的效果通过波形和频谱图可视化，进一步验证其对信号的影响。

### 5. **结论**

- 通过频谱分析，我们可以清楚地看到男声和女声在频率特性上的显著差异。
- 通过音高转换，能够有效地将男声转化为女声，但音质可能会受到一定的影响，进一步优化算法可以提高转换质量。
- 降噪与语音增强方法对提升语音清晰度起到了积极作用，带通滤波器在此过程中发挥了重要作用。
- 通过滤波器对比实验，明确了带通滤波器、低通滤波器和高通滤波器在不同场景下的应用。

### 6. 改进目标

- 可以进一步探讨基于 **小波变换** 或 **谱减法** 等先进降噪算法来提高语音质量。
- 寻找更长的音频样本，扩大**训练样本**，而不是采用过于简短且重复的音频段落。
- 使用 **音频合成** 技术，进一步改善音高转换效果，减少音质损失。
- 研究更多音频处理方法，如 **动态范围压缩** 或 **语音分离**，提升音频处理的整体效果



```matlab
%% Project 1 analyse the difference between male and female voice.
% 读取音频文件
[male_audio, fs_male] = audioread('male_voice.wav');
[female_audio, fs_female] = audioread('female_voice.wav');

% 确保两个音频文件采样率一致
if fs_male ~= fs_female
    error('采样率不一致');
end
fs = fs_male;  % 假设采样率相同

% 分析男声与女声的频率特征
male_spectrum = abs(fft(male_audio));
female_spectrum = abs(fft(female_audio));

% 可视化频谱差异
figure;
subplot(2,1,1);
plot(linspace(0, fs, length(male_spectrum)), male_spectrum);
title('Man Voice');
xlabel('frequency(Hz)');
ylabel('Amplitude');

subplot(2,1,2);
plot(linspace(0, fs, length(female_spectrum)), female_spectrum);
title('Women Voice');
xlabel('frequency(Hz)');
ylabel('Amplitude');

%% Project2 VOic

% 声音转换：通过调整音高来转换男声为女声
% 通过改变音高来进行转换
pitch_shift_factor = 1.2; % 1.2倍音高转换
male_audio_shifted = pitchShift(male_audio, pitch_shift_factor, fs);

% 降噪和增强语音：通过滤波器进行简单的降噪处理
% 使用一个带通滤波器来增强语音频段
low_cutoff = 300;  % 低频截止
high_cutoff = 3400; % 高频截止
[b, a] = butter(4, [low_cutoff, high_cutoff] / (fs / 2), 'bandpass');
enhanced_audio = filter(b, a, male_audio_shifted);

% 显示降噪后的音频效果
figure;
subplot(3,1,1);
plot(male_audio);
title('Original');
subplot(3,1,2);
plot(male_audio_shifted);
title('After Converter');
subplot(3,1,3);
plot(enhanced_audio);
title('Noise cancellation');

% 播放转换前的男声
disp('Before Converter');
sound(male_audio, fs);  % 播放原始男声
pause(length(male_audio) / fs + 1);  % 等待播放结束

% 播放转换后的音频
disp('After Converter');
sound(enhanced_audio, fs);  % 播放转换后的女声

% 设计和应用不同的滤波器：比较不同滤波器的效果
% 设计不同的滤波器进行对比
[b1, a1] = butter(4, [500, 3000] / (fs / 2), 'bandpass');  % 带通滤波器
[b2, a2] = butter(4, 2000 / (fs / 2), 'low');  % 低通滤波器
[b3, a3] = butter(4, 1000 / (fs / 2), 'high');  % 高通滤波器

% 应用滤波器
audio_bandpass = filter(b1, a1, male_audio);
audio_lowpass = filter(b2, a2, male_audio);
audio_highpass = filter(b3, a3, male_audio);

% 可视化不同滤波器效果
figure;
subplot(4,1,1);
plot(male_audio);
title('ORIGINAL');
subplot(4,1,2);
plot(audio_bandpass);
title('BANDPASS');
subplot(4,1,3);
plot(audio_lowpass);
title('LOWPASS');
subplot(4,1,4);
plot(audio_highpass);
title('HIGHPASS');

% 保存最终处理后的音频
audiowrite('converted_female_voice.wav', enhanced_audio, fs);

% 音高转换的函数
function shifted_audio = pitchShift(audio, factor, fs)
    N = length(audio);
    t = (0:N-1) / fs;
    shifted_audio = zeros(size(audio));
    
    % 通过插值法改变音高
    for i = 1:N
        new_index = round(i * factor);
        if new_index <= N
            shifted_audio(i) = audio(new_index);
        end
    end
end

```



## Version 2

### 1. 预期目标

- 找到更加有效的音频训练样例 【没找到，自己随便录的】
- 添加声纹采集等函数方便转换多种音色 【已完成声纹采集/声纹映射未完成】
- 添加新的滤波器实现更有效的音效增强【已添加自适应滤波器以实现音效增强】
- 更改音高方法（线性插值-》样条插值） 【已完成】



音频训练样例文本

| Case                                                         |
| ------------------------------------------------------------ |
| This recording was used as the audio training text for the DSP project. |



### 计划添加：

#### 1.**自适应滤波器（Adaptive Filter）**【已完成】

自适应滤波器根据输入信号的变化动态调整滤波器的参数，这对于去除背景噪声特别有用，尤其是当噪声的频谱不固定时。它可以有效地去除背景噪声，而不会损害语音内容。

#### 应用：

- 用于去除背景噪声，例如噪声环境中的电话信号或录音中的风噪声。
- 例如，使用自适应滤波器进行噪声估计和噪声消除（比如 `LMS` 或 `RLS` 算法）。

自适应滤波器的实现比较复杂，通常需要结合外部噪声信号来进行噪声的估计。如果你有噪声模型，可以使用自适应滤波器。

```
matlab复制代码% 示例：使用LMS（最小均方误差）算法进行噪声消除
mu = 0.01;  % 步长
N = length(male_audio);  % 信号长度
d = male_audio;  % 假设 male_audio 中有噪声
n = randn(size(d));  % 生成噪声信号
y = zeros(N, 1);  % 输出信号初始化
e = zeros(N, 1);  % 误差初始化

% 使用LMS算法
for k = 1:N
    y(k) = mu * n(k);  % 假设噪声信号估计为输入
    e(k) = d(k) - y(k);  % 计算误差
end
```

#### 2. 白噪声【已完成】

#### 3.声纹转换

### 2.代码

```matlab
% Last Edit Date: 2024/11/14 17：43
% Group: G7


%% 读取音频处理源文件
[male_audio, fs_male] = audioread('HJrecord.m4a');
fs = fs_male;  % 确保采样率一致

%% 为音频添加白噪声
% 设置信噪比
SNR = 10;  % 10 dB 的信噪比
noisy_audio = addWhiteNoise(male_audio, SNR);

%% 定义移位的音频（假设我们用 10 毫秒的延迟）
delay_samples = round(0.01 * fs);  % 延迟 10 毫秒
male_audio_shifted = [zeros(delay_samples, 1); male_audio(1:end-delay_samples)];

%% 使用LMS算法进行噪声消除
% 假设 male_audio 为噪声参考
mu = 0.01;  % LMS算法的步长
filter_order = 32;  % 自适应滤波器的阶数
[denoised_audio, ~] = lms_noise_cancellation(male_audio_shifted, male_audio, mu, filter_order);  % 假设 male_audio 为噪声参考

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

% 声纹映射


%% 声音转换：通过调整音高来转换男声为女声
pitch_shift_factor = 1.2;  % 1.2倍音高转换
male_audio_shifted = pitchShift(male_audio, pitch_shift_factor, fs);

%% 显示和播放处理效果
% figure;
% subplot(3,1,1);
% plot(male_audio);
% title('Original');
% subplot(3,1,2);
% plot(male_audio_shifted);
% title('After Pitch Shift');
% subplot(3,1,3);
% plot(enhanced_audio);
% title('After Noise Cancellation');

% 播放转换前的男声
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

function [output, w] = lms_noise_cancellation(noisy_signal, noise_reference, mu, filter_order)
    % 输入：
    % noisy_signal - 带噪音的信号（原始音频）
    % noise_reference - 噪声参考信号（例如环境噪声）
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
        e = noisy_signal(n) - y_hat;
        
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


```



## Version 3

修正了训练音频的输入问题

```matlab
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
denoised_audio = denoised_audio / max(abs(denoised_audio));  % 除以最大幅度进行归一化

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

% 声纹映射


%% 声音转换：通过调整音高来转换男声为女声
pitch_shift_factor = 1.2;  % 1.2倍音高转换
male_audio_shifted = pitchShift(male_audio, pitch_shift_factor, fs);

%% 显示和播放处理效果
% figure;
% subplot(3,1,1);
% plot(male_audio);
% title('Original');
% subplot(3,1,2);
% plot(male_audio_shifted);
% title('After Pitch Shift');
% subplot(3,1,3);
% plot(enhanced_audio);
% title('After Noise Cancellation');

% 播放转换前的男声
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

```



## 稿子

Next, I will introduce our group's Project 1 and Project 2, which are the Noise Cancellation and Audio Enhancement Project and the Audio Pitch and Audio Variable Speed No Pitch Project.

首先我们来看项目一，项目1的目的如其名，进行的是关于音频的噪声消除以及音效增强工作，用户能通过在清晰环境录制一段音频以用于训练，使其获得其特定的自适应降噪参数以实现在嘈杂环境下的清晰通话。

Let's first look at project 1, which as the name suggests, works on noise cancellation and sound enhancement of audio, where the user is able to record a piece of audio in a clear environment to be used for training, so that it acquires its specific adaptive noise cancellation parameters to achieve a clear call in noisy environments.

接下来我们来看具体的信号处理流程：

Next we look at the specific signal processing flow:

**读取音频文件**：

- 通过 `audioread` 读取原始的男性语音文件，并获取其采样率。

**添加白噪声**：

- 使用 `addWhiteNoise` 函数在原始语音中加入白噪声，并通过设定信噪比（SNR）来模拟带噪的语音环境。

**移位音频（作为噪声参考）**：

- 假设有10毫秒的音频延迟，将原始男性语音信号进行时间移位，得到一个延迟版本，作为噪声参考信号。

**LMS算法进行噪声消除**：

- 利用LMS（最小均方）自适应滤波算法，将带噪音的音频和清晰的语音（目标信号）与噪声参考信号（延迟版本的语音）一起输入，进行噪声消除，得到去噪后的语音信号。

**语音信号的带通滤波**：

- 使用带通滤波器保留语音信号中1000 Hz到5000 Hz之间的频率范围，进一步提高语音的清晰度。

**声纹提取（MFCC）**：

- 提取语音的MFCC特征，这些特征用于分析语音的个性化特征。MFCC是语音处理中的标准特征，通常用于语音识别或声纹识别。

**播放与保存处理后的音频**：

- 最后，播放原始的男性语音、加入噪声后的语音、降噪后的语音以及处理后的最终语音，并保存最终处理后的音频文件。

接着我们来看项目二，音频的变速不变调处理。

