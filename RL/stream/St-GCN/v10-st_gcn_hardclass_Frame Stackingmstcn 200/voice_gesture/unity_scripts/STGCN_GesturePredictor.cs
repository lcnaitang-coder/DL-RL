/*
 * ST-GCN 实时手势识别 (v13.9.4 Voice Integrated)
 * =================================================
 * [更新] 集成语音状态输入 (AudioDim=14)，支持多模态 PPO Agent。
 * [更新] 实时暴露当前帧的预测状态 (CurrentBestGesture)，用于失败分析。
 */

using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;
using Leap;
using System.Linq;

public class STGCN_GesturePredictor : MonoBehaviour
{
    [Header("1. ONNX 模型文件")]
    public ModelAsset feModelAsset;    
    public ModelAsset chModelAsset;    
    public ModelAsset agentModelAsset; 

    [Header("2. 模型维度参数")]
    public int numFeatures = 218;
    public int numClasses = 13;
    public int stackSize = 5;
    public int audioDim = 14; // [新增] 语音状态维度
    // StateDim = (GestureProbs * Stack) + Step + Stability + AudioState
    public int StateDim => (numClasses * stackSize) + 2 + audioDim; 

    [Header("3. 阈值参数")]
    public float stopThreshold = 0.5f;
    public float voiceDecay = 0.95f; // [新增] 语音信号衰减系数 (建议与 Python 训练代码保持一致)

    // --- [新增] 实时状态暴露 (即使不触发，外部也能看到它在想什么) ---
    public string CurrentBestGesture { get; private set; } = "None";
    public float CurrentStgcnProb { get; private set; } = 0f;
    public float CurrentAgentProb { get; private set; } = 0f;

    // --- [原有] 触发时的快照 ---
    public float LastTriggerConfidence { get; private set; } = 0f;
    public float LastAgentConfidence { get; private set; } = 0f;

    private readonly int[] BUFFER_SIZES = { 2, 4, 16, 64, 96 }; 
    private const int BUFFER_FEAT = 64;
    private const int NUM_LAYERS = 2;
    private const float UNIT_SCALE = 1.0f;

    [Header("4. 手势设置")]
    public string[] gestureClassNames; 

    [Header("5. 实时检测调优")]
    public float targetFramesPerSecond = 120f;
    public int realtimeFrameLimit = 200;
    public float trackingStartVelocity = 0.6f;

    public delegate void GestureRecognizedAction(string gestureName, int gestureIndex, int frames);
    public event GestureRecognizedAction OnGestureRecognized;

    private Worker m_FE_Worker;
    private Worker m_CH_Worker;
    private Worker m_Agent_Worker;

    private Dictionary<string, Tensor<float>> m_InputBuffers = new Dictionary<string, Tensor<float>>();
    private Dictionary<string, string> m_OutputToInputMap = new Dictionary<string, string>();

    private const string IN_FRAME = "input_frame";
    private string m_Name_Out_Embedding; 
    private string m_Name_Out_CH_Prob; 
    private string m_Name_Out_Agent_Stop;

    private ScalerParams m_Scaler;
    private float[] m_CurrentFrameFeatures;
    private float[] m_NormalizedFeatures;
    private float[] m_StateBuffer;

    // [新增] 语音状态缓存
    private float[] m_CurrentVoiceState;

    private int m_CurrentStep = 0;
    private int m_MaxSteps = 200;
    public bool m_IsTracking = false;

    [HideInInspector] public bool m_IsInSimulationMode = false;

    private Queue<float[]> m_ProbStack = new Queue<float[]>();
    private Queue<float> m_MaxProbHistory = new Queue<float>();
    private const int STABILITY_LEN = 10;

    private Controller leapController;
    private float collectionInterval;
    private float collectionTimer = 0f;

    [System.Serializable]
    private class ScalerParams { public List<float> mean; public List<float> scale; }

    private static readonly Finger.FingerType[] fingerTypes = {
        Finger.FingerType.THUMB, Finger.FingerType.INDEX, Finger.FingerType.MIDDLE,
        Finger.FingerType.RING, Finger.FingerType.PINKY
    };
    private static readonly Bone.BoneType[] boneTypes = {
        Bone.BoneType.METACARPAL, Bone.BoneType.PROXIMAL,
        Bone.BoneType.INTERMEDIATE, Bone.BoneType.DISTAL
    };

    struct BufferInfo { public string name; public int size; public int originalIndex; }

    void Start()
    {
        leapController = new Controller();
        m_IsInSimulationMode = false;
        LoadScalerParams();
        if (!enabled) return;

        m_CurrentFrameFeatures = new float[numFeatures];
        m_NormalizedFeatures = new float[numFeatures];
        m_StateBuffer = new float[StateDim];
        m_CurrentVoiceState = new float[audioDim]; // [新增] 初始化语音状态

        if (feModelAsset) {
            Model model = ModelLoader.Load(feModelAsset);
            m_FE_Worker = new Worker(model, BackendType.GPUCompute);
            if (!DetectAndMapOutputs(model)) { enabled = false; return; }
        } else { Debug.LogError("Feature Extractor 未分配！"); enabled = false; return; }

        if (chModelAsset) {
            Model model = ModelLoader.Load(chModelAsset);
            m_CH_Worker = new Worker(model, BackendType.GPUCompute);
            m_Name_Out_CH_Prob = model.outputs[0].name;
        }

        if (agentModelAsset) {
            Model model = ModelLoader.Load(agentModelAsset);
            m_Agent_Worker = new Worker(model, BackendType.GPUCompute);
            m_Name_Out_Agent_Stop = model.outputs[0].name;
        }

        InitBuffers();
        if (targetFramesPerSecond <= 0) targetFramesPerSecond = 120f;
        collectionInterval = 1.0f / targetFramesPerSecond;
        Debug.Log("✅ ST-GCN System Ready (Voice Integrated).");
    }

    bool DetectAndMapOutputs(Model model)
    {
        m_OutputToInputMap.Clear();
        var inputs = new Dictionary<string, Tensor>();
        inputs[IN_FRAME] = new Tensor<float>(new TensorShape(1, 1, numFeatures), new float[numFeatures]);
        var sortedExpectedInputs = new List<string>();
        for (int b_idx = 0; b_idx < BUFFER_SIZES.Length; b_idx++) {
            for (int l = 1; l <= NUM_LAYERS; l++) {
                string name = $"l{l}_b{b_idx + 1}_in";
                sortedExpectedInputs.Add(name);
                int size = BUFFER_SIZES[b_idx];
                inputs[name] = new Tensor<float>(new TensorShape(1, size, BUFFER_FEAT), new float[size * BUFFER_FEAT]);
            }
        }
        foreach (var kv in inputs) m_FE_Worker.SetInput(kv.Key, kv.Value);
        m_FE_Worker.Schedule();
        string foundEmbeddingName = "";
        var bufferInfos = new List<BufferInfo>();
        for(int i=0; i<model.outputs.Count; i++) {
            var outputDef = model.outputs[i];
            var t = m_FE_Worker.PeekOutput(outputDef.name);
            if (t == null) continue;
            if (t.shape.rank == 2 && t.shape[1] == 32) foundEmbeddingName = outputDef.name;
            else if (t.shape.rank == 3 && t.shape[2] == BUFFER_FEAT) bufferInfos.Add(new BufferInfo { name = outputDef.name, size = t.shape[1], originalIndex = i });
        }
        foreach(var kv in inputs) kv.Value.Dispose(); 
        if (string.IsNullOrEmpty(foundEmbeddingName)) { Debug.LogError("❌ 未找到 Embedding 输出！"); return false; }
        m_Name_Out_Embedding = foundEmbeddingName;
        bufferInfos.Sort((a, b) => {
            int sizeComp = a.size.CompareTo(b.size);
            if (sizeComp != 0) return sizeComp;
            return a.originalIndex.CompareTo(b.originalIndex);
        });
        if (bufferInfos.Count != sortedExpectedInputs.Count) { Debug.LogError("❌ Buffer 数量不匹配!"); return false; }
        for (int i = 0; i < bufferInfos.Count; i++) m_OutputToInputMap[bufferInfos[i].name] = sortedExpectedInputs[i];
        return true;
    }

    void OnDestroy() { m_FE_Worker?.Dispose(); m_CH_Worker?.Dispose(); m_Agent_Worker?.Dispose(); DisposeBuffers(); }

    void InitBuffers()
    {
        DisposeBuffers();
        m_InputBuffers.Clear();
        m_ProbStack.Clear();
        m_MaxProbHistory.Clear();
        
        // [新增] 重置语音状态
        if (m_CurrentVoiceState != null) System.Array.Clear(m_CurrentVoiceState, 0, m_CurrentVoiceState.Length);

        for (int l = 1; l <= NUM_LAYERS; l++) {
            for (int b = 1; b <= BUFFER_SIZES.Length; b++) {
                string name = $"l{l}_b{b}_in";
                int size = BUFFER_SIZES[b - 1];
                m_InputBuffers[name] = new Tensor<float>(new TensorShape(1, size, BUFFER_FEAT), new float[size * BUFFER_FEAT]);
            }
        }
        for(int i=0; i<stackSize; i++) m_ProbStack.Enqueue(new float[numClasses]);
    }

    void DisposeBuffers() { if (m_InputBuffers != null) { foreach (var kv in m_InputBuffers) kv.Value?.Dispose(); m_InputBuffers.Clear(); } }

    void LoadScalerParams()
    {
        TextAsset jsonFile = Resources.Load<TextAsset>("scaler_params_v13_6");
        if (jsonFile == null) jsonFile = Resources.Load<TextAsset>("scaler_params3");
        if (jsonFile == null) { Debug.LogError("缺少 scaler_params Json 文件"); enabled = false; return; }
        m_Scaler = JsonUtility.FromJson<ScalerParams>(jsonFile.text);
    }

    void Update()
    {
        if (m_IsInSimulationMode) return;
        collectionTimer += Time.deltaTime;
        if (collectionTimer < collectionInterval) return;
        collectionTimer -= collectionInterval;
        if (leapController == null || !leapController.IsConnected) return;

        Frame frame = leapController.Frame();
        if (frame.Hands.Count > 0)
        {
            Hand hand = frame.Hands[0];
            if (!m_IsTracking)
            {
                float velocity = hand.PalmVelocity.magnitude;
                if (velocity > trackingStartVelocity || Input.GetKeyDown(KeyCode.Space)) StartTracking();
            }
            if (m_IsTracking) { ExtractFeatures(hand, m_CurrentFrameFeatures); ProcessFrame(m_CurrentFrameFeatures); }
        }
        else if (m_IsTracking) StopTracking("手离开");
    }

    public void StartTracking() { m_IsTracking = true; m_CurrentStep = 0; if (!m_IsInSimulationMode) m_MaxSteps = (realtimeFrameLimit > 0) ? realtimeFrameLimit : 200; InitBuffers(); }
    public void StopTracking(string reason) { m_IsTracking = false; m_CurrentStep = 0; InitBuffers(); if (!m_IsInSimulationMode) Debug.Log($"停止: {reason}"); }
    public void SetMaxSteps(int steps) { m_MaxSteps = steps; m_CurrentStep = 0; }

    // [新增] 外部调用此方法更新语音输入
    // probs: 包含所有关键词置信度的数组 (应为 audioDim 长度)
    public void UpdateVoiceInput(float[] probs)
    {
        if (probs == null || probs.Length != audioDim) return;
        // 取最大值保留 (Max Pooling over time until decay)
        for (int i = 0; i < audioDim; i++)
        {
            m_CurrentVoiceState[i] = Mathf.Max(m_CurrentVoiceState[i], probs[i]);
        }
    }
    
    // [新增] 用于调试或模拟触发特定语音指令
    public void SimulateVoiceCommand(int commandIdx, float confidence = 1.0f)
    {
        if (commandIdx >= 0 && commandIdx < audioDim)
        {
            m_CurrentVoiceState[commandIdx] = Mathf.Max(m_CurrentVoiceState[commandIdx], confidence);
        }
    }

    public void ProcessFrame(float[] rawFeatures)
    {
        if (!m_IsTracking && m_IsInSimulationMode) StartTracking();
        NormalizeFeatures(rawFeatures);

        using (var inputFrame = new Tensor<float>(new TensorShape(1, 1, numFeatures), m_NormalizedFeatures))
        {
            m_FE_Worker.SetInput(IN_FRAME, inputFrame);
            foreach (var kv in m_InputBuffers) m_FE_Worker.SetInput(kv.Key, kv.Value);
            m_FE_Worker.Schedule();
            var embedding = m_FE_Worker.PeekOutput(m_Name_Out_Embedding) as Tensor<float>;
            
            foreach (var outputName in m_OutputToInputMap.Keys) {
                var outTensor = m_FE_Worker.PeekOutput(outputName) as Tensor<float>;
                if (outTensor == null) continue;
                var nextBuffer = new Tensor<float>(outTensor.shape, outTensor.DownloadToArray());
                string inputName = m_OutputToInputMap[outputName];
                if (m_InputBuffers.ContainsKey(inputName)) { m_InputBuffers[inputName].Dispose(); m_InputBuffers[inputName] = nextBuffer; }
            }

            m_CH_Worker.SetInput("input_feat", embedding);
            m_CH_Worker.Schedule();
            var probsTensor = m_CH_Worker.PeekOutput(m_Name_Out_CH_Prob) as Tensor<float>;
            RunPPOAgent(probsTensor.DownloadToArray());
        }
    }

    void RunPPOAgent(float[] currentProbs)
    {
        m_ProbStack.Enqueue(currentProbs); if (m_ProbStack.Count > stackSize) m_ProbStack.Dequeue();
        
        // [更新] 实时更新当前状态
        float maxP = currentProbs.Max();
        int maxIdx = System.Array.IndexOf(currentProbs, maxP);
        CurrentBestGesture = (maxIdx < gestureClassNames.Length) ? gestureClassNames[maxIdx] : $"Class_{maxIdx}";
        CurrentStgcnProb = maxP;

        // [新增] 衰减语音状态 (模拟 Python env step)
        // 确保在构建状态前衰减，或者构建后衰减？ Python中是 step() -> _get_obs() -> decay -> return state.
        // 所以当前帧使用衰减后的值。
        for (int i = 0; i < audioDim; i++) m_CurrentVoiceState[i] *= voiceDecay;

        m_MaxProbHistory.Enqueue(maxP); if (m_MaxProbHistory.Count > STABILITY_LEN) m_MaxProbHistory.Dequeue();
        float stability = CalculateStability(); float normStep = (m_MaxSteps <= 1) ? 0 : (float)m_CurrentStep / (m_MaxSteps - 1);
        
        BuildAgentState(normStep, stability);

        using (var stateTensor = new Tensor<float>(new TensorShape(1, StateDim), m_StateBuffer)) {
            m_Agent_Worker.SetInput("input_state", stateTensor); m_Agent_Worker.Schedule();
            var stopOut = m_Agent_Worker.PeekOutput(m_Name_Out_Agent_Stop) as Tensor<float>;
            float stopProb = stopOut.DownloadToArray()[0];
            
            CurrentAgentProb = stopProb;

            CheckTrigger(currentProbs, maxP, stopProb);
        }
    }

    void CheckTrigger(float[] probs, float maxProb, float stopProb)
    {
        if (stopProb > stopThreshold || (m_CurrentStep >= m_MaxSteps - 1 && m_MaxSteps > 0))
        {
            int index = System.Array.IndexOf(probs, maxProb);
            string gesture = (index < gestureClassNames.Length) ? gestureClassNames[index] : $"Class_{index}";
            
            LastTriggerConfidence = maxProb;
            LastAgentConfidence = stopProb;

            if (!m_IsInSimulationMode) Debug.Log($"✅ 触发: {gesture} (ST-GCN: {maxProb:F2}, Agent: {stopProb:F2})");
            OnGestureRecognized?.Invoke(gesture, index, m_CurrentStep + 1);
            StopTracking("Triggered");
        }
        else m_CurrentStep++;
    }

    void BuildAgentState(float step, float stab) 
    { 
        int k = 0; 
        // 1. Stacked Gesture Probs
        foreach (var pArr in m_ProbStack) { 
            int len = Mathf.Min(pArr.Length, numClasses); 
            for (int i = 0; i < len; i++) m_StateBuffer[k++] = pArr[i]; 
            for (int i = len; i < numClasses; i++) m_StateBuffer[k++] = 0f; 
        } 
        // 2. Step & Stability
        m_StateBuffer[k++] = step; 
        m_StateBuffer[k++] = stab; 

        // 3. [新增] Audio State
        for (int i = 0; i < audioDim; i++)
        {
            m_StateBuffer[k++] = m_CurrentVoiceState[i];
        }
    }

    float CalculateStability() { if (m_MaxProbHistory.Count < 2) return 0f; float avg = m_MaxProbHistory.Average(); float sum = m_MaxProbHistory.Sum(d => Mathf.Pow(d - avg, 2)); return Mathf.Sqrt(sum / (m_MaxProbHistory.Count - 1)); }
    void NormalizeFeatures(float[] raw) { if (m_Scaler == null) return; for (int i = 0; i < numFeatures; i++) { if (Mathf.Abs(m_Scaler.scale[i]) < 1e-6f) m_NormalizedFeatures[i] = 0; else m_NormalizedFeatures[i] = (raw[i] - m_Scaler.mean[i]) / m_Scaler.scale[i]; } }

    void ExtractFeatures(Hand hand, float[] featureArray)
    {
        try {
            int idx = 0;
            AppendVector3(featureArray, ref idx, hand.PalmPosition);
            AppendVector3(featureArray, ref idx, hand.PalmVelocity);
            AppendVector3(featureArray, ref idx, hand.PalmNormal);
            AppendVector3(featureArray, ref idx, hand.Direction);
            AppendQuaternion(featureArray, ref idx, hand.Rotation);
            featureArray[idx++] = hand.GrabStrength;
            featureArray[idx++] = hand.PinchStrength;
            foreach (var ft in fingerTypes) {
                Finger f = hand.GetFinger(ft);
                if (f != null) { foreach (var bt in boneTypes) { Bone b = f.GetBone(bt); if (b != null) { AppendVector3(featureArray, ref idx, b.PrevJoint); AppendVector3(featureArray, ref idx, b.NextJoint); AppendQuaternion(featureArray, ref idx, b.Rotation); } else FillWithZeros(featureArray, ref idx, 10); } }
                else FillWithZeros(featureArray, ref idx, 40);
            }
        } catch (System.Exception e) { Debug.LogError($"Leap Error: {e.Message}"); StopTracking("Error"); }
    }

    private void AppendVector3(float[] arr, ref int i, Vector3 v) { arr[i++] = v.x; arr[i++] = v.y; arr[i++] = v.z; }
    private void AppendQuaternion(float[] arr, ref int i, Quaternion q) { arr[i++] = q.x; arr[i++] = q.y; arr[i++] = q.z; arr[i++] = q.w; }
    private void FillWithZeros(float[] arr, ref int i, int count) { for (int k = 0; k < count; k++) if (i < arr.Length) arr[i++] = 0f; }
}
