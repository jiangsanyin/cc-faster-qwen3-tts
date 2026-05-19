#!/usr/bin/env python3
"""
流式TTS测试客户端 - 带图形界面

功能：
1. 自定义输入文本进行流式语音合成
2. 实时播放返回的流式音频
3. 显示关键性能指标（TTFA、块间间隔、总耗时等）

使用方法：
    python tts_streaming_client.py

依赖安装：
    pip install PyQt5 pygame requests numpy
"""

import sys
import time
import threading
import queue
import io
import wave
import struct
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

import numpy as np
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QGroupBox, QGridLayout,
    QComboBox, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

# 尝试导入 pygame 用于音频播放
try:
    import pygame
    pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=512)
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("警告: pygame 未安装，将无法播放音频。请运行: pip install pygame")


@dataclass
class RequestMetrics:
    """请求指标数据"""
    req_id: int
    text: str
    text_len: int
    
    # 时间戳
    t_start: float = 0.0  # 请求发起时间
    t_first_byte: float = 0.0  # 首字节时间（HTTP响应头返回）
    t_first_audio: float = 0.0  # 首块可播放音频时间
    t_end: float = 0.0  # 请求结束时间
    
    # 指标
    ttfb_ms: float = 0.0  # 首字节延迟 (Time To First Byte)
    ttfa_ms: float = 0.0  # 首音频延迟 (Time To First Audio)
    total_ms: float = 0.0  # 总耗时 (ms)
    audio_s: float = 0.0  # 音频时长 (秒)
    n_chunks: int = 0  # 音频块数
    
    # 块间间隔
    chunk_times: List[float] = field(default_factory=list)
    inter_chunk_max_ms: float = 0.0
    inter_chunk_p95_ms: float = 0.0
    
    # 状态
    status: str = "pending"  # pending, running, success, error
    error_msg: str = ""


class TTSClientThread(QThread):
    """TTS客户端线程 - 处理流式请求和音频播放"""
    
    # 信号定义
    metrics_updated = pyqtSignal(object)  # 指标更新
    chunk_received = pyqtSignal(int, int)  # (req_id, chunk_index)
    log_message = pyqtSignal(str)  # 日志消息
    playback_started = pyqtSignal(int)  # req_id
    playback_finished = pyqtSignal(int)  # req_id
    
    def __init__(self, server_url: str, voice: str, text: str, req_id: int):
        super().__init__()
        self.server_url = server_url
        self.voice = voice
        self.text = text
        self.req_id = req_id
        self.metrics = RequestMetrics(
            req_id=req_id,
            text=text,
            text_len=len(text)
        )
        self._stop_flag = False
        self.audio_queue = queue.Queue()
        
    def run(self):
        """执行流式TTS请求"""
        self.metrics.t_start = time.perf_counter()
        self.metrics.status = "running"
        self.metrics_updated.emit(self.metrics)
        
        url = f"{self.server_url}/v1/audio/speech"
        payload = {
            "model": "tts-1",
            "input": self.text,
            "voice": self.voice,
            "response_format": "wav"
        }
        
        try:
            self.log_message.emit(f"[请求 {self.req_id}] 开始请求: {self.text[:30]}...")
            
            # 发起流式请求
            response = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            # 记录首字节时间（HTTP 响应头返回时间）
            self.metrics.t_first_byte = time.perf_counter()
            self.metrics.ttfb_ms = (self.metrics.t_first_byte - self.metrics.t_start) * 1000
            
            # WAV 头 (44 bytes)
            wav_header = None
            pcm_data = b""
            chunk_index = 0
            sample_rate = 24000
            first_audio_logged = False
            
            # 音频播放线程
            playback_thread = None
            if PYGAME_AVAILABLE:
                playback_thread = threading.Thread(target=self._playback_worker)
                playback_thread.daemon = True
                playback_thread.start()
            
            for chunk in response.iter_content(chunk_size=4096):
                if self._stop_flag:
                    break
                    
                if not chunk:
                    continue
                
                chunk_recv_time = time.perf_counter()
                
                # 首块包含 WAV 头
                if wav_header is None:
                    if len(chunk) >= 44:
                        wav_header = chunk[:44]
                        pcm_data = chunk[44:]
                        self.log_message.emit(
                            f"[请求 {self.req_id}] 收到首块: 总长度={len(chunk)}, "
                            f"WAV头=44字节, PCM数据={len(pcm_data)}字节, "
                            f"耗时={(chunk_recv_time - self.metrics.t_start)*1000:.1f}ms"
                        )
                        # 如果首块除了 WAV 头还有 PCM 数据，记录 TTFA
                        if pcm_data and not first_audio_logged:
                            self.metrics.t_first_audio = chunk_recv_time
                            self.metrics.ttfa_ms = (self.metrics.t_first_audio - self.metrics.t_start) * 1000
                            first_audio_logged = True
                            self.log_message.emit(
                                f"[请求 {self.req_id}] TTFB: {self.metrics.ttfb_ms:.1f}ms, "
                                f"TTFA: {self.metrics.ttfa_ms:.1f}ms (首块含PCM)"
                            )
                            self.playback_started.emit(self.req_id)
                    else:
                        # 首块太小，可能只有部分 WAV 头
                        self.log_message.emit(
                            f"[请求 {self.req_id}] 收到不完整首块: {len(chunk)}字节"
                        )
                        continu
                else:
                    pcm_data = chunk
                    # 收到第一个真正的 PCM 数据块时记录 TTFA
                    if not first_audio_logged and pcm_data:
                        self.metrics.t_first_audio = chunk_recv_time
                        self.metrics.ttfa_ms = (self.metrics.t_first_audio - self.metrics.t_start) * 1000
                        first_audio_logged = True
                        self.log_message.emit(
                            f"[请求 {self.req_id}] TTFB: {self.metrics.ttfb_ms:.1f}ms, "
                            f"TTFA: {self.metrics.ttfa_ms:.1f}ms (第{chunk_index+1}块)"
                        )
                        self.playback_started.emit(self.req_id)
                
                if pcm_data:
                    # 记录块时间
                    now = time.perf_counter()
                    self.metrics.chunk_times.append(now)
                    self.metrics.n_chunks += 1
                    
                    # 计算音频时长 (16-bit mono, 24000Hz)
                    audio_samples = len(pcm_data) // 2
                    chunk_audio_s = audio_samples / sample_rate
                    self.metrics.audio_s += chunk_audio_s
                    
                    # 放入播放队列
                    if PYGAME_AVAILABLE:
                        self.audio_queue.put(pcm_data)
                    
                    self.chunk_received.emit(self.req_id, chunk_index)
                    chunk_index += 1
                    
                    self.metrics_updated.emit(self.metrics)
            
            # 等待播放完成
            if PYGAME_AVAILABLE:
                self.audio_queue.put(None)  # 结束标记
                if playback_thread:
                    playback_thread.join(timeout=5)
            
            self.metrics.t_end = time.perf_counter()
            self.metrics.total_ms = (self.metrics.t_end - self.metrics.t_start) * 1000
            
            # 计算块间间隔
            if len(self.metrics.chunk_times) > 1:
                gaps = [
                    (self.metrics.chunk_times[i] - self.metrics.chunk_times[i-1]) * 1000
                    for i in range(1, len(self.metrics.chunk_times))
                ]
                self.metrics.inter_chunk_max_ms = max(gaps) if gaps else 0
                self.metrics.inter_chunk_p95_ms = float(np.percentile(gaps, 95)) if gaps else 0
            
            self.metrics.status = "success"
            self.log_message.emit(
                f"[请求 {self.req_id}] 完成: TTFB={self.metrics.ttfb_ms:.1f}ms, "
                f"TTFA={self.metrics.ttfa_ms:.1f}ms, "
                f"总耗时={self.metrics.total_ms:.1f}ms, 音频={self.metrics.audio_s:.2f}s, "
                f"块数={self.metrics.n_chunks}, inter_chunk_p95={self.metrics.inter_chunk_p95_ms:.1f}ms"
            )
            
        except Exception as e:
            self.metrics.status = "error"
            self.metrics.error_msg = str(e)
            self.log_message.emit(f"[请求 {self.req_id}] 错误: {e}")
            
        finally:
            self.metrics_updated.emit(self.metrics)
            self.playback_finished.emit(self.req_id)
    
    def _playback_worker(self):
        """音频播放工作线程"""
        while True:
            try:
                pcm_data = self.audio_queue.get(timeout=1)
                if pcm_data is None:
                    break
                
                # 将 PCM 数据转换为 pygame 可播放的格式
                audio_array = np.frombuffer(pcm_data, dtype=np.int16)
                
                # 转换为立体声格式 (复制单声道到两个声道)
                # pygame.sndarray.make_sound 需要 (samples, 2) 的形状用于立体声
                stereo_array = np.column_stack((audio_array, audio_array))
                
                sound = pygame.sndarray.make_sound(stereo_array)
                sound.play()
                
                # 等待播放完成 (非阻塞方式)
                while pygame.mixer.get_busy():
                    time.sleep(0.01)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.log_message.emit(f"[播放] 错误: {e}")
                break
    
    def stop(self):
        """停止请求"""
        self._stop_flag = True


class MetricsTableWidget(QTableWidget):
    """指标表格控件"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        headers = [
            "请求ID", "文本", "状态", "TTFB(ms)", "TTFA(ms)", "总耗时(ms)",
            "音频时长(s)", "块数", "inter_chunk_max(ms)", "inter_chunk_p95(ms)"
        ]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setAlternatingRowColors(True)
        
    def add_metrics(self, metrics: RequestMetrics):
        """添加指标行"""
        row = self.rowCount()
        self.insertRow(row)
        
        self.setItem(row, 0, QTableWidgetItem(str(metrics.req_id)))
        self.setItem(row, 1, QTableWidgetItem(metrics.text[:20] + "..." if len(metrics.text) > 20 else metrics.text))
        
        status_item = QTableWidgetItem(metrics.status)
        if metrics.status == "success":
            status_item.setBackground(QColor(200, 255, 200))
        elif metrics.status == "error":
            status_item.setBackground(QColor(255, 200, 200))
        self.setItem(row, 2, status_item)
        
        self.setItem(row, 3, QTableWidgetItem(f"{metrics.ttfb_ms:.1f}"))
        self.setItem(row, 4, QTableWidgetItem(f"{metrics.ttfa_ms:.1f}"))
        self.setItem(row, 5, QTableWidgetItem(f"{metrics.total_ms:.1f}"))
        self.setItem(row, 6, QTableWidgetItem(f"{metrics.audio_s:.2f}"))
        self.setItem(row, 7, QTableWidgetItem(str(metrics.n_chunks)))
        self.setItem(row, 8, QTableWidgetItem(f"{metrics.inter_chunk_max_ms:.1f}"))
        self.setItem(row, 9, QTableWidgetItem(f"{metrics.inter_chunk_p95_ms:.1f}"))
        
        # 滚动到最新行
        self.scrollToBottom()
        
    def clear_all(self):
        """清空表格"""
        self.setRowCount(0)


class StatisticsWidget(QWidget):
    """统计信息控件"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QGridLayout(self)
        
        # 标签
        self.labels = {}
        stats = [
            ("total_requests", "总请求数:"),
            ("success_count", "成功数:"),
            ("error_count", "失败数:"),
            ("ttfb_avg", "TTFB 平均(ms):"),
            ("ttfb_p95", "TTFB P95(ms):"),
            ("ttfa_avg", "TTFA 平均(ms):"),
            ("ttfa_p95", "TTFA P95(ms):"),
            ("total_avg", "总耗时平均(ms):"),
            ("total_p95", "总耗时 P95(ms):"),
            ("audio_total", "音频总时长(s):"),
            ("inter_chunk_p95_avg", "块间P95平均(ms):"),
        ]
        
        for i, (key, label) in enumerate(stats):
            row, col = i // 2, (i % 2) * 2
            layout.addWidget(QLabel(label), row, col)
            value_label = QLabel("-")
            value_label.setFont(QFont("Consolas", 10))
            layout.addWidget(value_label, row, col + 1)
            self.labels[key] = value_label
    
    def update_stats(self, metrics_list: List[RequestMetrics]):
        """更新统计信息"""
        if not metrics_list:
            return
            
        ttfb_values = [m.ttfb_ms for m in metrics_list if m.ttfb_ms > 0]
        ttfa_values = [m.ttfa_ms for m in metrics_list if m.ttfa_ms > 0]
        total_values = [m.total_ms for m in metrics_list if m.total_ms > 0]
        audio_total = sum(m.audio_s for m in metrics_list)
        inter_chunk_values = [m.inter_chunk_p95_ms for m in metrics_list if m.inter_chunk_p95_ms > 0]
        
        self.labels["total_requests"].setText(str(len(metrics_list)))
        self.labels["success_count"].setText(str(len([m for m in metrics_list if m.status == "success"])))
        self.labels["error_count"].setText(str(len([m for m in metrics_list if m.status == "error"])))
        
        if ttfb_values:
            self.labels["ttfb_avg"].setText(f"{np.mean(ttfb_values):.1f}")
            self.labels["ttfb_p95"].setText(f"{np.percentile(ttfb_values, 95):.1f}")
        
        if ttfa_values:
            self.labels["ttfa_avg"].setText(f"{np.mean(ttfa_values):.1f}")
            self.labels["ttfa_p95"].setText(f"{np.percentile(ttfa_values, 95):.1f}")
        
        if total_values:
            self.labels["total_avg"].setText(f"{np.mean(total_values):.1f}")
            self.labels["total_p95"].setText(f"{np.percentile(total_values, 95):.1f}")
        
        self.labels["audio_total"].setText(f"{audio_total:.2f}")
        
        if inter_chunk_values:
            self.labels["inter_chunk_p95_avg"].setText(f"{np.mean(inter_chunk_values):.1f}")


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.client_threads = []
        self.req_counter = 0
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("流式TTS测试客户端")
        self.setMinimumSize(1000, 700)
        
        # 主控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 服务器配置
        config_group = QGroupBox("服务器配置")
        config_layout = QGridLayout(config_group)
        
        config_layout.addWidget(QLabel("服务器地址:"), 0, 0)
        self.server_url_edit = QTextEdit()
        self.server_url_edit.setPlainText("http://10.251.11.55:10018")
        self.server_url_edit.setMaximumHeight(30)
        config_layout.addWidget(self.server_url_edit, 0, 1)
        
        config_layout.addWidget(QLabel("音色ID:"), 0, 2)
        self.voice_edit = QTextEdit()
        self.voice_edit.setPlainText("ff609584-a1c3-4f29-837a-615871f9ecd7")
        self.voice_edit.setMaximumHeight(30)
        config_layout.addWidget(self.voice_edit, 0, 3)
        
        main_layout.addWidget(config_group)
        
        # 输入区域
        input_group = QGroupBox("输入文本")
        input_layout = QVBoxLayout(input_group)
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("请输入要合成的文本...")
        self.input_text.setPlainText("根据您描述的反复胃痛、饭后加重、偶尔反酸，建议预约消化内科就诊。医生可能会建议幽门螺杆菌检测或胃镜检查。就诊前请清淡饮食，避免饮酒和辛辣食物。若出现呕血、黑便或剧烈腹痛，请立即前往急诊。")
        input_layout.addWidget(self.input_text)
        
        btn_layout = QHBoxLayout()
        self.send_btn = QPushButton("发送请求")
        self.send_btn.clicked.connect(self.send_single_request)
        self.send_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_all_requests)
        self.stop_btn.setEnabled(False)
        
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_results)
        
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()
        input_layout.addLayout(btn_layout)
        
        main_layout.addWidget(input_group)
        
        # 指标表格
        metrics_group = QGroupBox("请求指标")
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_table = MetricsTableWidget()
        metrics_layout.addWidget(self.metrics_table)
        main_layout.addWidget(metrics_group)
        
        # 统计信息
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout(stats_group)
        self.stats = StatisticsWidget()
        stats_layout.addWidget(self.stats)
        main_layout.addWidget(stats_group)
        
        # 日志区域
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)
        
    
    def log(self, message: str):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_text.append(f"[{timestamp}] {message}")
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def send_single_request(self):
        """发送单次请求"""
        server_url = self.server_url_edit.toPlainText().strip()
        voice = self.voice_edit.toPlainText().strip()
        text = self.input_text.toPlainText().strip()
        
        if not text:
            QMessageBox.warning(self, "警告", "请输入要合成的文本")
            return
        
        self.req_counter += 1
        client = TTSClientThread(server_url, voice, text, self.req_counter)
        client.metrics_updated.connect(self.on_metrics_updated)
        client.log_message.connect(self.log)
        client.chunk_received.connect(self.on_chunk_received)
        
        self.client_threads.append(client)
        client.start()
        
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    
    def on_metrics_updated(self, metrics: RequestMetrics):
        """指标更新回调"""
        # 更新表格
        self.metrics_table.add_metrics(metrics)
        
        # 更新统计
        all_metrics = [
            t.metrics for t in self.client_threads 
            if t.metrics.status in ["success", "error"]
        ]
        self.stats.update_stats(all_metrics)
        
        # 检查是否所有请求完成
        if all(t.isFinished() for t in self.client_threads):
            self.send_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def on_chunk_received(self, req_id: int, chunk_index: int):
        """块接收回调"""
        pass  # 可以在这里更新实时状态
    
    def stop_all_requests(self):
        """停止所有请求"""
        for t in self.client_threads:
            if t.isRunning():
                t.stop()
        self.log("已停止所有请求")
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def clear_results(self):
        """清空测试结果"""
        self.client_threads.clear()
        self.metrics_table.clear_all()
        self.log_text.clear()



def main():
    app = QApplication(sys.argv)
    
    # 设置样式
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
