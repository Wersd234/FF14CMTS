import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time

app = FastAPI()

# 开启 CUDNN 优化并限制显存（给大模型留空间）
torch.backends.cudnn.benchmark = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE == 'cuda':
    # 限制仅使用 15% 显存 (约 3.6GB)，其余留给 LLM
    torch.cuda.set_per_process_memory_fraction(0.15, 0)

# 技能映射表 (用于返回给前端)
# 0:制作, 1:加工, 2:精修, 3:观察
ACTION_MAP = {0: 100001, 1: 100002, 2: 100003, 3: 100004}


class CraftingState(BaseModel):
    cp: int
    durability: int
    progress: int
    quality: int
    condition: int  # 0白, 1红, 2彩, 3黑
    max_progress: int
    base_progress: float  # 基础制作效率 100% 的数值
    base_quality: float  # 基础加工效率 100% 的数值


class GPUCraftingEnv:
    def __init__(self, batch_size=100000):
        self.batch_size = batch_size
        self.device = DEVICE
        # 球色倍率：白(1.0), 红(1.5), 彩(4.0), 黑(0.5)
        self.mults = torch.tensor([1.0, 1.5, 4.0, 0.5], device=DEVICE)

    def batch_step(self, states, actions, base_prog, base_qual):
        """GPU 上的单步矩阵推演"""
        cp = states[:, 0]
        durability = states[:, 1]
        progress = states[:, 2]
        quality = states[:, 3]
        condition = states[:, 4].long()
        max_progress = states[:, 5]

        # ================= 1. 安全锁 (Action Masking) =================
        # CP 不足时，强行替换为 0 (制作) 以防报错自杀
        actions = torch.where((actions == 1) & (cp < 18), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 2) & (cp < 88), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 3) & (cp < 7), torch.tensor(0, device=DEVICE), actions)

        is_synth = (actions == 0)  # 制作 0 CP, 10 耐久, 120%
        is_touch = (actions == 1)  # 加工 18 CP, 10 耐久, 100%
        is_mend = (actions == 2)  # 精修 88 CP, 恢复 30 耐久
        is_obs = (actions == 3)  # 观察 7 CP

        # ================= 2. 扣除属性 =================
        cp = cp - torch.where(is_touch, 18, 0) - torch.where(is_mend, 88, 0) - torch.where(is_obs, 7, 0)

        dur_cost = torch.where(is_synth | is_touch, 10, 0)
        durability = durability - dur_cost + torch.where(is_mend, 30, 0)
        durability = torch.clamp(durability, max=80)  # 假设最大耐久80

        # ================= 3. 增加进展与品质 (结合基础值) =================
        progress = progress + torch.where(is_synth, base_prog * 1.2, 0.0)

        quality_gain = torch.where(is_touch, base_qual * 1.0, 0.0)
        # 乘以发牌员颜色倍率
        quality = quality + (quality_gain * self.mults[condition])

        # ================= 4. 发牌员逻辑 (下一个球) =================
        rands = torch.rand(self.batch_size, device=self.device)
        next_cond = torch.where(rands < 0.65, torch.tensor(0, device=DEVICE),
                                torch.where(rands < 0.90, torch.tensor(1, device=DEVICE),
                                            torch.tensor(2, device=DEVICE)))

        # 彩球后必黑球
        is_exc = (condition == 2)
        next_cond = torch.where(is_exc, torch.tensor(3, device=DEVICE), next_cond)

        # 黑球后无黑/彩，大概率白，小概率红
        is_poor = (condition == 3)
        next_cond = torch.where(is_poor, torch.where(rands < 0.8, torch.tensor(0, device=DEVICE),
                                                     torch.tensor(1, device=DEVICE)), next_cond)

        new_states = torch.stack([cp, durability, progress, quality, next_cond, max_progress], dim=1)

        # 存活判定：耐久没碎 且 还没搓满
        is_active = (durability > 0) & (progress < max_progress)
        return new_states, is_active


def gpu_mcts(state_data: CraftingState):
    start_time = time.time()
    env = GPUCraftingEnv(batch_size=100000)

    initial_tensor = torch.tensor([
        state_data.cp, state_data.durability, state_data.progress,
        state_data.quality, state_data.condition, state_data.max_progress
    ], dtype=torch.float32, device=DEVICE)

    states = initial_tensor.repeat(env.batch_size, 1)
    num_actions = 4

    # 随机分配第一步
    first_actions = torch.randint(0, num_actions, (env.batch_size,), device=DEVICE)
    actions = first_actions.clone()
    active_mask = torch.ones(env.batch_size, dtype=torch.bool, device=DEVICE)

    # 深度推演 15 步
    search_depth = 15
    for _ in range(search_depth):
        states, is_active = env.batch_step(states, actions, state_data.base_progress, state_data.base_quality)
        active_mask = active_mask & is_active
        # 如果活着，随机选下一个技能；死了给 0
        actions = torch.where(active_mask, torch.randint(0, num_actions, (env.batch_size,), device=DEVICE),
                              torch.tensor(0, device=DEVICE))

    # 评分系统: 进展没满得 0 分，进展满了按品质算分
    final_progress = states[:, 2]
    final_quality = states[:, 3]
    success_mask = final_progress >= state_data.max_progress
    scores = final_quality * success_mask.float()

    best_action = 0
    best_score = -1.0

    print(f"--- MCTS (耗时: {(time.time() - start_time) * 1000:.1f}ms) ---")
    for a in range(num_actions):
        avg_score = scores[first_actions == a].mean().item()
        print(f"技能 {a} 期望收益: {avg_score:.2f}")
        if avg_score > best_score:
            best_score = avg_score
            best_action = a

    return ACTION_MAP[best_action]


@app.post("/solve_step")
async def solve_step(state: CraftingState):
    best_action_id = gpu_mcts(state)
    return {"action_id": best_action_id}


@app.get("/warmup")
async def warmup():
    """唤醒/预热 CUDA 张量"""
    dummy = CraftingState(cp=500, durability=80, progress=0, quality=0, condition=0, max_progress=1000,
                          base_progress=600, base_quality=300)
    gpu_mcts(dummy)
    return {"status": "warmed up"}