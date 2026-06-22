import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time

app = FastAPI()

# 开启 CUDNN 优化
torch.backends.cudnn.benchmark = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 映射 FF14 的真实技能 ID (你需要根据真实 ID 调整)
# 这里假设：0=制作(100001), 1=加工(100002), 2=精修(100003), 3=观察(100004)
ACTION_MAP = {0: 100001, 1: 100002, 2: 100003, 3: 100004}


class CraftingState(BaseModel):
    cp: int
    durability: int
    progress: int
    quality: int
    condition: int  # 0:白, 1:红, 2:彩, 3:黑
    max_progress: int  # 配方需要的满进展


class GPUCraftingEnv:
    def __init__(self, batch_size=100000):
        self.batch_size = batch_size
        self.device = DEVICE
        self.mults = torch.tensor([1.0, 1.5, 4.0, 0.5], device=DEVICE)

    def batch_step(self, states, actions):
        """在 GPU 上瞬间推演十万个状态"""
        cp = states[:, 0]
        durability = states[:, 1]
        progress = states[:, 2]
        quality = states[:, 3]
        condition = states[:, 4].long()
        max_progress = states[:, 5]

        # 分离动作掩码 (Action Masks)
        is_synth = (actions == 0)  # 制作 (进+10, 耐-10)
        is_touch = (actions == 1)  # 加工 (品+10*倍率, 耐-10, CP-18)
        is_mend = (actions == 2)  # 精修 (耐+30, CP-88)
        is_obs = (actions == 3)  # 观察 (CP-7)

        # 1. 更新 CP (如果 CP 不够强行放技能，这里简单设为负数，后续判定死亡)
        cp_cost = torch.zeros_like(cp)
        cp_cost += torch.where(is_touch, 18, 0)
        cp_cost += torch.where(is_mend, 88, 0)
        cp_cost += torch.where(is_obs, 7, 0)
        cp = cp - cp_cost

        # 2. 更新耐久
        dur_cost = torch.zeros_like(durability)
        dur_cost += torch.where(is_synth | is_touch, 10, 0)
        dur_cost -= torch.where(is_mend, 30, 0)  # 回复耐久
        durability = durability - dur_cost
        durability = torch.clamp(durability, max=80)  # 假设最大耐久80

        # 3. 更新进展
        progress = progress + torch.where(is_synth, 10, 0)

        # 4. 更新品质 (带入球色倍率)
        quality_gain = 10 * self.mults[condition]
        quality = quality + torch.where(is_touch, quality_gain, 0)

        # 5. 生成下一个球色 (核心发牌逻辑)
        rands = torch.rand(self.batch_size, device=self.device)
        next_cond = torch.where(rands < 0.65, torch.tensor(0, device=self.device),
                                torch.where(rands < 0.90, torch.tensor(1, device=self.device),
                                            torch.tensor(2, device=self.device)))

        # 彩球后必黑球规则
        is_excellent = (condition == 2)
        next_cond = torch.where(is_excellent, torch.tensor(3, device=self.device), next_cond)

        new_states = torch.stack([cp, durability, progress, quality, next_cond, max_progress], dim=1)

        # 判定状态存活 (耐久>0 且 CP>=0 且 进展未满)
        # 注意：如果进展满了，说明成功了，也属于“结束”，但在 MCTS 算分时会有高额奖励
        is_active = (durability > 0) & (cp >= 0) & (progress < max_progress)

        return new_states, is_active


def gpu_mcts(state_data: CraftingState):
    start_time = time.time()
    env = GPUCraftingEnv(batch_size=100000)  # 3090 跑十万次毫无压力

    initial_tensor = torch.tensor([
        state_data.cp, state_data.durability, state_data.progress,
        state_data.quality, state_data.condition, state_data.max_progress
    ], dtype=torch.float32, device=DEVICE)

    states = initial_tensor.repeat(env.batch_size, 1)

    # 第一步：均分 4 个可用技能给十万个分支
    num_actions = 4
    first_actions = torch.randint(0, num_actions, (env.batch_size,), device=DEVICE)
    actions = first_actions.clone()

    active_mask = torch.ones(env.batch_size, dtype=torch.bool, device=DEVICE)

    # 推演深度 (例如往后看 15 步)
    search_depth = 15
    for _ in range(search_depth):
        states, is_active = env.batch_step(states, actions)
        active_mask = active_mask & is_active
        # 如果分支还活着，随机分配下一个技能；死了的就随便给个 0
        actions = torch.where(active_mask, torch.randint(0, num_actions, (env.batch_size,), device=DEVICE),
                              torch.tensor(0, device=DEVICE))

    # --- 评分系统 (Heuristic) ---
    final_progress = states[:, 2]
    final_quality = states[:, 3]
    max_progress = states[:, 5]

    # 1. 如果进展没推满，直接给 0 分 (失败品)
    success_mask = final_progress >= max_progress

    # 2. 最终得分 = 品质
    scores = final_quality * success_mask.float()

    # 统计第一步该选哪个技能最好
    best_action = 0
    best_score = -1.0

    print(f"--- MCTS 推演结果 (耗时: {(time.time() - start_time) * 1000:.1f} ms) ---")
    for a in range(num_actions):
        avg_score = scores[first_actions == a].mean().item()
        print(f"技能 {a} 期望品质: {avg_score:.2f}")
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
    """用于 C# 启动时预热 CUDA"""
    dummy_state = CraftingState(cp=500, durability=80, progress=0, quality=0, condition=0, max_progress=100)
    gpu_mcts(dummy_state)
    return {"status": "warmed up"}