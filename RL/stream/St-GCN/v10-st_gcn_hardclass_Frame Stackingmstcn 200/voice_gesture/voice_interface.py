import sherpa_onnx
import numpy as np
import os

class SherpaKWSInterface:
    def __init__(self, num_classes=13, model_dir=None):
        self.num_classes = num_classes
        
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kws_models")
        
        # 配置 Sherpa-onnx
        # 请确保 kws_models 目录下有相应的 onnx 模型文件
        config = sherpa_onnx.KeywordSpotterConfig(
            model=sherpa_onnx.OnlineModelConfig(
                transducer=sherpa_onnx.OnlineTransducerModelConfig(
                    encoder=f"{model_dir}/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
                    decoder=f"{model_dir}/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
                    joiner=f"{model_dir}/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
                ),
                tokens=f"{model_dir}/tokens.txt",
                num_threads=1,
            ),
            keywords_file=f"{model_dir}/keywords.txt",
        )
        self.spotter = sherpa_onnx.KeywordSpotter(config)
        self.stream = self.spotter.create_stream()
        
        # 状态向量：[13个类别的置信度, 1个全局静音标志]
        # 维度 = 14
        self.current_state = np.zeros(num_classes + 1)
        self.decay_factor = 0.95 # 信号衰减因子，让语音指令在状态中“存活”一段时间

        # 关键词 ID 映射表 (必须与 keywords.txt 保持一致)
        self.keyword_map = {
            "向前": 0,
            "向上": 1,
            "放开": 2,
            "捏住": 3,
            "顺时针": 4,
            "逆时针": 5,
            "向后": 6,
            "向右": 7,
            "确认": 8,
            "锁定": 9,
            "向左": 10,
            "点赞": 11,
            "向下": 12
        }

    def process_audio_chunk(self, samples):
        """接收麦克风数据（float32 array）"""
        self.stream.accept_waveform(sample_rate=16000, waveform=samples)
        
        if self.spotter.is_ready(self.stream):
            self.spotter.decode(self.stream)
            result = self.spotter.get_result(self.stream)
            
            if result.keyword:
                # result.keyword 返回的是关键词文本
                # 比如 "向前", "锁定" 等
                detected_id = self._parse_keyword_id(result.keyword)
                
                # 激活对应状态，置信度设为 1.0
                if 0 <= detected_id < self.num_classes:
                    self.current_state[detected_id] = 1.0
                    print(f"🎤 语音检测到: {result.keyword} (ID: {detected_id})")

    def get_state(self):
        """被 RL 环境调用，获取当前语音状态"""
        # 返回当前状态的副本
        state = self.current_state.copy()
        
        # 每一帧调用后，让信号自然衰减
        # 这样 RL 就能知道：数值是 1.0 代表刚说完，0.5 代表说完了一会儿
        self.current_state *= self.decay_factor 
        
        # 如果所有信号都很弱，认为处于静音/噪声状态
        if np.max(self.current_state[:-1]) < 0.1:
            self.current_state[-1] = 1.0 # 最后一个维度表示“无语音”
        else:
            self.current_state[-1] = 0.0
            
        return state

    def _parse_keyword_id(self, keyword_str):
        # 如果 keyword_str 包含 "@"，取前面部分 (虽然 sherpa 通常直接返回文本)
        if "@" in keyword_str:
            keyword_str = keyword_str.split("@")[1].split("/")[0] # 例如 "0@锁定/1.0" -> "锁定"
        
        # 有时候 sherpa 返回的 keyword 带有空格，需要 strip
        clean_kw = keyword_str.strip()
        
        return self.keyword_map.get(clean_kw, -1)
