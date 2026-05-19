#!/bin/bash

server_url="http://10.251.11.55:10018"
# voice_id="ff609584-a1c3-4f29-837a-615871f9ecd7"
# input_text="根据您描述的反复胃痛、饭后加重、偶尔反酸，建议预约消化内科就诊。医生可能会建议幽门螺杆菌检测或胃镜检查。就诊前请清淡饮食，避免饮酒和辛辣食物。若出现呕血、黑便或剧烈腹痛，请立即前往急诊。"
for i in {1..10}; do 
    # 流式 wav（浏览器/播放器可边下边播）
    curl ${server_url}/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input":"根据您描述的反复胃痛、饭后加重、偶尔反酸，建议预约消化内科就诊。医生可能会建议幽门螺杆菌检测或胃镜检查。就诊前请清淡饮食，避免饮酒和辛辣食物。若出现呕血、黑便或剧烈腹痛，请立即前往急诊。","voice":"ff609584-a1c3-4f29-837a-615871f9ecd7","response_format":"wav"}' \
    -o speech_${i}.wav   # 保存为 speech_${i}.wav
done

# 合并所有音频文件
ffmpeg -i speech_1.wav -i speech_2.wav -i speech_3.wav -i speech_4.wav -i speech_5.wav -i speech_6.wav -i speech_7.wav -i speech_8.wav -i speech_9.wav -i speech_10.wav -c copy merged_speech.wav