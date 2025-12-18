
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def draw_architecture_diagram(save_path="system_architecture_v13_7.png"):
    # 设置风格
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # 颜色定义
    colors = {
        'input': '#E3F2FD',      # 浅蓝
        'model': '#FFF3E0',      # 浅橙
        'rl': '#E8F5E9',         # 浅绿
        'deploy': '#F3E5F5',     # 浅紫
        'arrow': '#546E7A',      # 灰蓝
        'box_edge': '#455A64',   # 深灰
        'text': '#263238'        # 深黑
    }

    # 辅助函数：绘制圆角矩形
    def draw_box(x, y, w, h, label, subtext="", color='#FFFFFF', edge_color='#000000'):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.2",
            ec=edge_color,
            fc=color,
            alpha=0.9,
            zorder=2
        )
        ax.add_patch(box)
        
        # 主标题
        ax.text(x + w/2, y + h/2 + (0.3 if subtext else 0), label, 
                ha='center', va='center', fontsize=11, fontweight='bold', color=colors['text'], zorder=3)
        # 副标题
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.3, subtext, 
                    ha='center', va='center', fontsize=9, color='#546E7A', zorder=3)
        return box

    # 辅助函数：绘制箭头
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=colors['arrow'], lw=2, shrinkA=5, shrinkB=5),
            zorder=1
        )
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.1, label, ha='center', va='bottom', fontsize=9, color=colors['box_edge'],
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # ==============================================================================
    # 1. 区域划分 (背景块)
    # ==============================================================================
    
    # Input Layer
    ax.add_patch(patches.Rectangle((0.5, 0.5), 4, 15, color=colors['input'], alpha=0.3, zorder=0))
    ax.text(2.5, 15.2, "Phase 1: Input & Sensing", ha='center', fontsize=14, fontweight='bold', color='#1565C0')
    
    # Perception Layer (ST-GCN)
    ax.add_patch(patches.Rectangle((5, 0.5), 6, 15, color=colors['model'], alpha=0.3, zorder=0))
    ax.text(8, 15.2, "Phase 2: Perception (ST-GCN v13.7)", ha='center', fontsize=14, fontweight='bold', color='#EF6C00')
    
    # Fusion & RL Layer
    ax.add_patch(patches.Rectangle((11.5, 0.5), 7, 15, color=colors['rl'], alpha=0.3, zorder=0))
    ax.text(15, 15.2, "Phase 3: Decision (PPO + Voice)", ha='center', fontsize=14, fontweight='bold', color='#2E7D32')
    
    # Deployment Layer
    ax.add_patch(patches.Rectangle((19, 0.5), 4.5, 15, color=colors['deploy'], alpha=0.3, zorder=0))
    ax.text(21.25, 15.2, "Phase 4: Unity Deployment", ha='center', fontsize=14, fontweight='bold', color='#6A1B9A')

    # ==============================================================================
    # 2. 绘制组件
    # ==============================================================================

    # --- Input Zone ---
    draw_box(1, 12, 3, 1.5, "Leap Motion", "Hand Tracking Data", color='#FFFFFF')
    draw_box(1, 9, 3, 1.5, "CSV Data / Realtime", "218 Features (Frame)", color='#FFFFFF')
    
    draw_box(1, 5, 3, 1.5, "Microphone", "Audio Stream", color='#FFFFFF')
    draw_box(1, 2, 3, 1.5, "Sherpa-onnx", "Keyword Spotting", color='#FFFFFF')

    # --- Perception Zone (ST-GCN) ---
    # Feature Extractor Flow
    draw_box(5.5, 12, 5, 2, "ST-GCN Feature Extractor", "v13.7 Enhanced Attention", color='#FFE0B2')
    
    # Internal Details of ST-GCN (Small boxes inside)
    # Just schematic representation
    draw_box(6, 10, 1.2, 0.8, "GCN", color='#FFCC80')
    draw_box(7.4, 10, 1.2, 0.8, "MS-TCN", "5-Branch", color='#FFCC80')
    draw_box(8.8, 10, 1.2, 0.8, "Attention", "Channel", color='#FFCC80')
    
    draw_box(5.5, 7.5, 5, 1.2, "Feature Embedding", "32-dim Vector", color='#FFCC80')
    draw_box(5.5, 5, 5, 1.5, "Classifier Head", "Dense -> Softmax", color='#FFE0B2')
    draw_box(5.5, 2.5, 5, 1.2, "Gesture Probabilities", "13 Classes", color='#FFCC80')

    # --- Decision Zone (RL) ---
    # Voice Processing
    draw_box(12, 3, 2.5, 1.5, "Voice State", "14-dim + Decay", color='#C8E6C9')
    
    # State Construction
    draw_box(12, 6, 6, 4, "RL State Construction", "Total: 81-dim Input", color='#A5D6A7')
    # Details inside State Construction
    draw_box(12.5, 8.5, 5, 0.8, "Gesture Stack (13 x 5)", "65 dims", color='#FFFFFF')
    draw_box(12.5, 7.5, 5, 0.8, "Step & Stability", "2 dims", color='#FFFFFF')
    draw_box(12.5, 6.5, 5, 0.8, "Voice State Vector", "14 dims", color='#FFFFFF')

    # PPO Agent
    draw_box(12, 11, 6, 2, "PPO Agent (Actor)", "MLP: 81 -> 128 -> 128 -> 1", color='#81C784')
    
    # Output Action
    draw_box(14, 13.5, 2, 1, "Action", "Trigger / Wait", color='#FFFFFF', edge_color='#2E7D32')

    # --- Deployment Zone ---
    draw_box(19.5, 11, 3.5, 1.5, "ppo_actor.onnx", "Sentis Inference", color='#E1BEE7')
    draw_box(19.5, 8, 3.5, 1.5, "fe_v13_7.onnx", "Feature Extractor", color='#E1BEE7')
    draw_box(19.5, 5, 3.5, 1.5, "head_v13_7.onnx", "Classifier", color='#E1BEE7')
    
    draw_box(19.5, 2, 3.5, 1.5, "Unity C# Scripts", "Predictor & Simulator", color='#F8BBD0')

    # ==============================================================================
    # 3. 连线 (Arrows)
    # ==============================================================================

    # Input -> Preprocessing
    draw_arrow(2.5, 12, 2.5, 10.5) # Leap -> CSV
    draw_arrow(2.5, 5, 2.5, 3.5)   # Mic -> Sherpa

    # Preprocessing -> Model
    draw_arrow(4, 9.75, 5.5, 13, "Frame (218)") # CSV -> ST-GCN FE
    draw_arrow(4, 2.75, 12, 3.75, "Keywords")   # Sherpa -> Voice State (Long jump)

    # ST-GCN Internal
    draw_arrow(8, 12, 8, 10.8) # FE -> Internal
    draw_arrow(6.6, 10.4, 7.4, 10.4) # GCN -> MS-TCN
    draw_arrow(8.6, 10.4, 8.8, 10.4) # MS-TCN -> Att
    draw_arrow(8, 10, 8, 8.7) # Internal -> Embedding
    
    draw_arrow(8, 7.5, 8, 6.5) # Embedding -> Head
    draw_arrow(8, 5, 8, 3.7)   # Head -> Probs

    # Model -> RL Fusion
    draw_arrow(10.5, 3.1, 12.5, 8.9, "Prob Queue") # Probs -> RL State (Stack)
    draw_arrow(13.25, 4.5, 13.25, 6.5) # Voice State -> RL State (Voice)

    # RL Internal
    draw_arrow(15, 10, 15, 11) # State -> Actor

    # RL -> Output
    draw_arrow(15, 13, 15, 13.5) # Actor -> Action

    # ONNX Export Mapping (Dotted lines)
    # FE -> ONNX
    ax.annotate("", xy=(19.5, 8.75), xytext=(10.5, 13),
                arrowprops=dict(arrowstyle="->", color='#9C27B0', lw=1.5, linestyle="--", connectionstyle="arc3,rad=-0.2"))
    # Head -> ONNX
    ax.annotate("", xy=(19.5, 5.75), xytext=(10.5, 5.75),
                arrowprops=dict(arrowstyle="->", color='#9C27B0', lw=1.5, linestyle="--"))
    # Actor -> ONNX
    ax.annotate("", xy=(19.5, 11.75), xytext=(18, 12),
                arrowprops=dict(arrowstyle="->", color='#9C27B0', lw=1.5, linestyle="--"))

    # Unity Integration
    draw_arrow(21.25, 5, 21.25, 3.5) # Head -> Script
    draw_arrow(21.25, 8, 21.25, 3.5) # FE -> Script
    draw_arrow(21.25, 11, 21.25, 3.5) # Actor -> Script

    # Title
    ax.text(12, 0.2, "Multimodal Gesture & Voice Control System Architecture (v13.7)", 
            ha='center', fontsize=18, fontweight='bold', color='#37474F')

    # Save
    plt.tight_layout()
    
    # 1. Save as PNG (Preview)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Architecture diagram saved to {save_path}")

    # 2. Save as SVG (Editable in Visio)
    # 设置字体类型为 'none' 以保留文本的可编辑性 (否则会被转为路径/形状)
    plt.rcParams['svg.fonttype'] = 'none'
    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Editable architecture diagram saved to {svg_path} (Import into Visio to edit)")


if __name__ == "__main__":
    draw_architecture_diagram()
