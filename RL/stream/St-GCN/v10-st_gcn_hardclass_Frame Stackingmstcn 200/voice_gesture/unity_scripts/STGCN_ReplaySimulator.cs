/*
 * ST-GCN Sentis CSV 模拟器 (v13.9.4 Voice Simulation Support)
 * =================================================
 * [更新] 支持手动模拟语音指令触发，用于测试多模态 Agent。
 */

using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;

[RequireComponent(typeof(STGCN_GesturePredictor))]
public class STGCN_ReplaySimulator : MonoBehaviour
{
    [Header("测试控制")]
    public bool playOnStart = false;
    
    [Header("批量测试设置")]
    public string testFolderPath = "D:/Path/To/Test/Data"; 
    public string expectedGestureName = "air_quotes";
    
    [Header("模拟设置")]
    public float simulationFPS = 60.0f;

    [Header("引用")]
    public STGCN_GesturePredictor predictor;

    private List<float[]> m_CsvData;
    private int m_TotalFiles = 0;
    private int m_CorrectPredictions = 0;
    private List<int> m_FramesList = new List<int>();
    private List<float> m_TimeList = new List<float>();
    private List<float> m_ConfList = new List<float>(); 

    private bool m_IsWaitingForResult;
    private string m_LastPredictionResult;
    private int m_LastPredictionFrames;
    private float m_LastStgcnConf;
    private float m_LastAgentConf;

    void Awake() { if (predictor == null) predictor = GetComponent<STGCN_GesturePredictor>(); }

    void Start()
    {
        if (predictor != null) predictor.OnGestureRecognized += OnPredictionReceived;
        if (playOnStart) StartBatchTest();
    }

    void OnDestroy() { if (predictor != null) predictor.OnGestureRecognized -= OnPredictionReceived; }

    private void OnPredictionReceived(string gestureName, int gestureIndex, int frames)
    {
        m_LastPredictionResult = gestureName;
        m_LastPredictionFrames = frames;
        m_LastStgcnConf = predictor.LastTriggerConfidence;
        m_LastAgentConf = predictor.LastAgentConfidence;
        m_IsWaitingForResult = false;
    }

    [ContextMenu("Start Batch Test")]
    public void StartBatchTest() { StopAllCoroutines(); StartCoroutine(BatchTestCoroutine()); }

    // [新增] 模拟语音指令 - 在 Unity 编辑器组件菜单中点击调用
    [ContextMenu("Simulate Random Voice")]
    public void SimulateRandomVoice()
    {
        if (predictor != null) 
        {
            int rndCmd = UnityEngine.Random.Range(0, predictor.audioDim);
            Debug.Log($"[Simulator] 模拟语音指令索引: {rndCmd}");
            predictor.SimulateVoiceCommand(rndCmd, 1.0f);
        }
    }

    // [新增] 模拟特定指令 (例如索引 0 - 向前)
    [ContextMenu("Simulate Voice 'Forward' (0)")]
    public void SimulateVoiceForward() { predictor?.SimulateVoiceCommand(0, 1.0f); }

    private IEnumerator BatchTestCoroutine()
    {
        yield return new WaitForSeconds(0.5f);
        m_CorrectPredictions = 0; m_FramesList.Clear(); m_TimeList.Clear(); m_ConfList.Clear();

        if (!Directory.Exists(testFolderPath)) { Debug.LogError($"文件夹不存在: {testFolderPath}"); yield break; }
        string[] csvFiles = Directory.GetFiles(testFolderPath, "*.csv");
        m_TotalFiles = csvFiles.Length;

        for (int i = 0; i < m_TotalFiles; i++)
        {
            string file = csvFiles[i];
            if (!LoadCsvData(file)) continue;

            m_IsWaitingForResult = true;
            m_LastPredictionResult = "NO_PREDICTION";
            m_LastPredictionFrames = 0;
            m_LastStgcnConf = 0f;
            m_LastAgentConf = 0f;

            yield return StartCoroutine(ReplayCoroutine());

            float simulatedTime = m_LastPredictionFrames * (1.0f / predictor.targetFramesPerSecond);
            bool isCorrect = (m_LastPredictionResult == expectedGestureName);
            
            if (m_LastPredictionResult != "NO_PREDICTION") {
                if (isCorrect) m_CorrectPredictions++;
                m_FramesList.Add(m_LastPredictionFrames);
                m_TimeList.Add(simulatedTime);
                m_ConfList.Add(m_LastStgcnConf);
            }

            string log = "";
            if (isCorrect)
            {
                log = $"<color=green>PASS</color> | 文件: {Path.GetFileName(file)} | 结果: {m_LastPredictionResult} | Conf: {m_LastStgcnConf:F2} | Agent: {m_LastAgentConf:F2}";
            }
            else
            {
                string actualGuess = (m_LastPredictionResult == "NO_PREDICTION") ? predictor.CurrentBestGesture : m_LastPredictionResult;
                float actualConf = (m_LastPredictionResult == "NO_PREDICTION") ? predictor.CurrentStgcnProb : m_LastStgcnConf;
                float actualAgent = (m_LastPredictionResult == "NO_PREDICTION") ? predictor.CurrentAgentProb : m_LastAgentConf;
                
                string failType = (m_LastPredictionResult == "NO_PREDICTION") ? "MISSED (未触发)" : "WRONG (识别错误)";
                
                log = $"<color=red>FAIL</color> | 文件: {Path.GetFileName(file)}\n" +
                      $"  🔴 状态: {failType}\n" +
                      $"  🟢 期望: {expectedGestureName}\n" +
                      $"  ⚠️ 实际: {actualGuess} (ST-GCN: {actualConf:F2}, Agent: {actualAgent:F2})";
            }
            Debug.Log(log);
            yield return null;
        }

        predictor.m_IsInSimulationMode = false;
        predictor.StopTracking("测试结束");
        PrintFinalReport();
    }

    private void PrintFinalReport()
    {
        float accuracy = (m_TotalFiles > 0) ? (float)m_CorrectPredictions / m_TotalFiles * 100f : 0;
        float avgFrames = (m_FramesList.Count > 0) ? (float)m_FramesList.Average() : 0;
        float avgTime = (m_TimeList.Count > 0) ? (float)m_TimeList.Average() : 0;
        float avgConf = (m_ConfList.Count > 0) ? (float)m_ConfList.Average() : 0;

        string report = $"=== 报告 ===\n总数: {m_TotalFiles}\n准确率: <color=yellow>{accuracy:F2}%</color>\n平均ST-GCN置信度: {avgConf:F4}\n平均耗时: {avgTime:F3}s";
        Debug.Log(report);
    }

    private IEnumerator ReplayCoroutine()
    {
        if (m_CsvData == null || m_CsvData.Count == 0) yield break;
        predictor.m_IsInSimulationMode = true;
        predictor.StartTracking();
        predictor.SetMaxSteps(m_CsvData.Count);
        float delay = (simulationFPS > 0) ? (1.0f / simulationFPS) : 0;

        // [新增] 确定语音触发时机 (模拟 Python 训练逻辑: 70% 概率触发)
        int voiceTriggerFrame = -1;
        int targetVoiceClass = -1;

        if (enableVoiceSimulation && Random.value < 0.7f)
        {
            // 尝试查找期望手势对应的索引
            targetVoiceClass = System.Array.IndexOf(predictor.gestureClassNames, expectedGestureName);
            if (targetVoiceClass >= 0)
            {
                // 随机选择一个触发帧 (例如在 10% ~ 40% 的进度处)
                int start = Mathf.Max(5, (int)(m_CsvData.Count * 0.1f));
                int end = Mathf.Min(m_CsvData.Count - 5, (int)(m_CsvData.Count * 0.5f));
                if (end > start) voiceTriggerFrame = Random.Range(start, end);
            }
        }

        for (int i = 0; i < m_CsvData.Count; i++) {
            if (!m_IsWaitingForResult) yield break;

            // [新增] 注入语音信号
            if (i == voiceTriggerFrame && targetVoiceClass >= 0)
            {
                Debug.Log($"[Simulator] Frame {i}: Injecting Voice Command '{expectedGestureName}' (ID: {targetVoiceClass})");
                predictor.SimulateVoiceCommand(targetVoiceClass, 1.0f);
            }

            try { predictor.ProcessFrame(m_CsvData[i]); }
            catch (System.Exception e) { Debug.LogError($"推理错误: {e.Message}"); yield break; }
            if (delay > 0) yield return new WaitForSeconds(delay); else yield return null;
        }
    }

    private bool LoadCsvData(string path)
    {
        m_CsvData = new List<float[]>();
        try {
            string[] lines = File.ReadAllLines(path);
            if (lines.Length <= 1) return false;
            string[] headers = lines[0].Split(',');
            int featCount = predictor.numFeatures;
            for (int i = 1; i < lines.Length; i++) {
                string[] vals = lines[i].Split(',');
                float[] feats = new float[featCount];
                int idx = 0;
                for (int j = 0; j < vals.Length; j++) {
                    if (headers[j] == "hand_type") continue;
                    if (idx < featCount && float.TryParse(vals[j], out float v)) feats[idx++] = v;
                }
                if (idx == featCount) m_CsvData.Add(feats);
            }
            return m_CsvData.Count > 0;
        } catch { return false; }
    }
}
