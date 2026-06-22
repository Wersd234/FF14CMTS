import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time

app = FastAPI()
torch.backends.cudnn.benchmark = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.15, 0)

# ================= 扩展技能库 =================
# 0:制作, 1:加工, 2:精修, 3:观察, 4:改革, 5:崇敬, 6:俭约, 7:阔步, 8:比尔格的祝福
# 扩展到 24 个技能！包含了赌狗、连击、发牌判定、7.0绝技
ACTION_MAP = {
    0: 100001, 1: 100002, 2: 100003, 3: 100004,
    4: 100005, 5: 100006, 6: 100007, 7: 100008, 8: 100009,
    9: 100010, 10: 100011, 11: 100012, 12: 100013, 13: 100014,
    14: 100015, 15: 100016,
    # === 新增：赌狗、连击、发牌、绝技 ===
    16: 100017, # 仓促 (Hasty Touch): 0CP, 100%品质, 60%胜率
    17: 100018, # 高速制作 (Rapid Synth): 0CP, 500%进展, 50%胜率
    18: 100019, # 中级加工 (Standard Touch): 连击18CP/否则32CP, 125%品质
    19: 100020, # 上级加工 (Advanced Touch): 连击18CP/否则46CP, 150%品质
    20: 100021, # 秘诀 (Tricks of Trade): 回复20CP, 仅红彩球可用
    21: 100022, # 集中加工 (Precise Touch): 18CP, 150%品质, IQ+2, 仅红彩球
    22: 100023, # 集中制作 (Intensive Synth): 6CP, 400%进展, 仅红彩球
    23: 100024  # 工匠的绝技 (Trained Perfection): 0CP, 下一步耐久0, 限用1次
}


class CraftingState(BaseModel):
    cp: int
    durability: int
    progress: int
    quality: int
    condition: int
    max_progress: int
    base_progress: float
    base_quality: float


# 扩展到 24 个技能！包含了赌狗、连击、发牌判定、7.0绝技
ACTION_MAP = {
    0: 100001, 1: 100002, 2: 100003, 3: 100004,
    4: 100005, 5: 100006, 6: 100007, 7: 100008, 8: 100009,
    9: 100010, 10: 100011, 11: 100012, 12: 100013, 13: 100014,
    14: 100015, 15: 100016,
    # === 新增：赌狗、连击、发牌、绝技 ===
    16: 100017,  # 仓促 (Hasty Touch): 0CP, 100%品质, 60%胜率
    17: 100018,  # 高速制作 (Rapid Synth): 0CP, 500%进展, 50%胜率
    18: 100019,  # 中级加工 (Standard Touch): 连击18CP/否则32CP, 125%品质
    19: 100020,  # 上级加工 (Advanced Touch): 连击18CP/否则46CP, 150%品质
    20: 100021,  # 秘诀 (Tricks of Trade): 回复20CP, 仅红彩球可用
    21: 100022,  # 集中加工 (Precise Touch): 18CP, 150%品质, IQ+2, 仅红彩球
    22: 100023,  # 集中制作 (Intensive Synth): 6CP, 400%进展, 仅红彩球
    23: 100024  # 工匠的绝技 (Trained Perfection): 0CP, 下一步耐久0, 限用1次
}


class GPUCraftingEnv:
    def __init__(self, batch_size=150000):
        self.batch_size = batch_size
        self.device = DEVICE
        self.mults = torch.tensor([1.0, 1.5, 4.0, 0.5], device=DEVICE)

    def batch_step(self, states, actions, base_prog, base_qual):
        # 17 维恐怖张量
        cp = states[:, 0]
        dur = states[:, 1]
        prog = states[:, 2]
        qual = states[:, 3]
        cond = states[:, 4].long()
        max_p = states[:, 5]
        iq = states[:, 6]
        innov = states[:, 7]
        vener = states[:, 8]
        wn = states[:, 9]
        gs = states[:, 10]
        manip = states[:, 11]
        muscle = states[:, 12]
        step = states[:, 13]
        combo = states[:, 14]  # 连击状态: 0无, 1加工后, 2中级后
        p_avail = states[:, 15]  # 绝技可用: 1可用, 0不可用
        p_active = states[:, 16]  # 绝技激活: 1激活, 0未激活

        # ================= 1. 动态 CP 计算与安全锁 =================
        # 处理连击降低 CP 消耗
        cost_18 = torch.where(combo == 1, 18, 32)
        cost_19 = torch.where(combo == 2, 18, 46)

        cost_map = {
            1: 18, 2: 88, 3: 7, 4: 18, 5: 18, 6: 98, 7: 32, 8: 24, 9: 96,
            10: 6, 11: 24, 12: 7, 13: 18, 14: 40, 15: 25, 16: 0, 17: 0,
            20: 0, 21: 18, 22: 6, 23: 0
        }

        # 强制替换违规动作 (统一替换为 0号 基础制作兜底)
        for act_idx, cost in cost_map.items():
            actions = torch.where((actions == act_idx) & (cp < cost), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 18) & (cp < cost_18), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 19) & (cp < cost_19), torch.tensor(0, device=DEVICE), actions)

        # 红/彩球专属技能锁
        is_good_exc = (cond == 1) | (cond == 2)
        actions = torch.where(((actions == 20) | (actions == 21) | (actions == 22)) & ~is_good_exc,
                              torch.tensor(0, device=DEVICE), actions)

        # 绝技与起手锁
        actions = torch.where((actions == 23) & (p_avail == 0), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where(((actions == 10) | (actions == 11)) & (step > 0), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 8) & (iq == 0), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 15) & (wn > 0), torch.tensor(0, device=DEVICE), actions)

        acts = {i: (actions == i) for i in range(24)}

        # ================= 2. RNG (赌狗) 概率结算 =================
        # 掷骰子
        rng_rolls = torch.rand(self.batch_size, device=self.device)
        is_success = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        # 仓促 60%, 高速 50%
        is_success = torch.where(acts[16], rng_rolls < 0.60, is_success)
        is_success = torch.where(acts[17], rng_rolls < 0.50, is_success)

        is_any_touch = acts[1] | acts[8] | acts[11] | acts[14] | acts[15] | acts[16] | acts[18] | acts[19] | acts[21]
        is_any_synth = acts[0] | acts[10] | acts[12] | acts[13] | acts[17] | acts[22]

        # ================= 3. 结算 CP 与 耐久 =================
        cp_cost = torch.zeros_like(cp)
        for act_idx, cost in cost_map.items():
            cp_cost += torch.where(acts[act_idx], cost, 0)
        cp_cost += torch.where(acts[18], cost_18, 0)
        cp_cost += torch.where(acts[19], cost_19, 0)

        cp = cp - cp_cost + torch.where(acts[20], 20, 0)  # 秘诀回CP
        cp = torch.clamp(cp, max=999)  # 假设不能溢出太多

        dur_base_cost = torch.zeros_like(dur)
        dur_base_cost += torch.where(
            acts[0] | acts[1] | acts[10] | acts[11] | acts[12] | acts[16] | acts[17] | acts[18] | acts[19] | acts[21] |
            acts[22], 10, 0)
        dur_base_cost += torch.where(acts[13] | acts[14], 20, 0)
        dur_base_cost += torch.where(acts[15], 5, 0)

        # 俭约减半
        dur_cost = torch.where(wn > 0, dur_base_cost / 2.0, dur_base_cost.float()).int()

        # 7.0绝技免耐久：如果绝技激活，本次耐久消耗归 0
        dur_cost = torch.where(p_active == 1, 0, dur_cost)

        dur = dur - dur_cost + torch.where(acts[2], 30, 0)
        dur = torch.clamp(dur, max=80)

        # ================= 4. 结算进展与品质 =================
        # 进展效率 (如果是赌狗且失败，效率归0)
        synth_eff = torch.where(acts[0], 1.2, 0.0) + torch.where(acts[10], 3.0, 0.0) + \
                    torch.where(acts[12], 1.5, 0.0) + torch.where(acts[13], 3.6, 0.0) + \
                    torch.where(acts[22], 4.0, 0.0) + \
                    torch.where(acts[17] & is_success, 5.0, 0.0)  # 高速成功给500%

        muscle_bonus = torch.where((muscle > 0) & is_any_synth, 2.5, 1.0)
        prog_mult = synth_eff * muscle_bonus * (1.0 + torch.where(vener > 0, 0.5, 0.0))
        prog = prog + torch.where(is_any_synth, base_prog * prog_mult, 0.0)

        # 品质效率
        touch_eff = torch.where(acts[1] | acts[11] | acts[15], 1.0, 0.0) + \
                    torch.where(acts[18], 1.25, 0.0) + torch.where(acts[19], 1.5, 0.0) + \
                    torch.where(acts[14], 2.0, 0.0) + torch.where(acts[21], 1.5, 0.0) + \
                    torch.where(acts[8], 1.0 + 0.2 * iq, 0.0) + \
                    torch.where(acts[16] & is_success, 1.0, 0.0)  # 仓促成功给100%

        qual_mult = (touch_eff + torch.where(innov > 0, 0.5, 0.0) + torch.where(gs > 0, 1.0, 0.0)) * (1.0 + 0.1 * iq)
        # 品质增加量乘以球色
        qual = qual + torch.where(is_any_touch, base_qual * qual_mult * self.mults[cond], 0.0)

        # ================= 5. 巨型 Buff 轮转 =================
        muscle = torch.where(is_any_synth, 0, torch.clamp(muscle - 1, min=0))
        muscle = torch.where(acts[10], 5, muscle)

        innov = torch.where(acts[4], 4, torch.clamp(innov - 1, min=0))
        vener = torch.where(acts[5], 4, torch.clamp(vener - 1, min=0))
        wn = torch.where(acts[6], 8, torch.clamp(wn - 1, min=0))
        gs = torch.where(acts[7], 3, torch.clamp(gs - 1, min=0))
        gs = torch.where(is_any_touch, 0, gs)

        # 绝技状态更新
        # 如果刚才这一步消耗了耐久，并且绝技激活中，那么绝技就消耗掉了
        p_active = torch.where((dur_base_cost > 0) & (p_active == 1), 0, p_active)
        # 本回合按下绝技
        p_active = torch.where(acts[23], 1, p_active)
        p_avail = torch.where(acts[23], 0, p_avail)

        # 连击状态更新 (Buff 不会打断连击，只有其他触摸或制作会打断)
        is_buff_action = acts[2] | acts[3] | acts[4] | acts[5] | acts[6] | acts[7] | acts[9] | acts[20] | acts[23]
        combo = torch.where(acts[1], 1, torch.where(acts[18], 2, torch.where(is_buff_action, combo, 0)))

        # 内静 (成功时才加)
        iq_gain = torch.where(acts[11] | acts[14] | acts[21], 2, 0) + \
                  torch.where(acts[1] | acts[15] | acts[18] | acts[19] | (acts[16] & is_success), 1, 0)
        iq = torch.clamp(iq + iq_gain, max=10)
        iq = torch.where(acts[8], 0, iq)

        manip = torch.where(acts[9], 8, torch.clamp(manip - 1, min=0))
        dur = torch.where((manip > 0) & (dur > 0), dur + 5, dur)
        dur = torch.clamp(dur, max=80)

        step = step + 1

        # 发牌员
        rands2 = torch.rand(self.batch_size, device=self.device)
        next_cond = torch.where(rands2 < 0.65, torch.tensor(0, device=DEVICE),
                                torch.where(rands2 < 0.90, torch.tensor(1, device=DEVICE),
                                            torch.tensor(2, device=DEVICE)))
        next_cond = torch.where(cond == 2, torch.tensor(3, device=DEVICE), next_cond)
        next_cond = torch.where(cond == 3, torch.where(rands2 < 0.8, torch.tensor(0, device=DEVICE),
                                                       torch.tensor(1, device=DEVICE)), next_cond)

        new_states = torch.stack(
            [cp, dur, prog, qual, next_cond, max_p, iq, innov, vener, wn, gs, manip, muscle, step, combo, p_avail,
             p_active], dim=1)
        is_active = (dur > 0) & (prog < max_p)
        return new_states, is_active


# 加上这个装饰器，告诉 PyTorch 我们不练大模型，不计算梯度，显存占用直降 90%！
@torch.no_grad()
def gpu_mcts(state_data: CraftingState):
    start_time = time.time()
    env = GPUCraftingEnv(batch_size=150000)

    # 初始 17 维张量
    initial_tensor = torch.tensor([
        state_data.cp, state_data.durability, state_data.progress,
        state_data.quality, state_data.condition, state_data.max_progress,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
    ], dtype=torch.float32, device=DEVICE)

    states = initial_tensor.repeat(env.batch_size, 1)
    num_actions = len(ACTION_MAP)

    first_actions = torch.randint(0, num_actions, (env.batch_size,), device=DEVICE)
    actions = first_actions.clone()
    active_mask = torch.ones(env.batch_size, dtype=torch.bool, device=DEVICE)

    # 🚀 这里改成了 30 步！完美覆盖一个高难配方的完整生命周期
    search_depth = 30

    for _ in range(search_depth):
        states, is_active = env.batch_step(states, actions, state_data.base_progress, state_data.base_quality)
        active_mask = active_mask & is_active
        actions = torch.where(active_mask, torch.randint(0, num_actions, (env.batch_size,), device=DEVICE),
                              torch.tensor(0, device=DEVICE))

    # 评分逻辑
    final_prog = states[:, 2]
    final_qual = states[:, 3]
    success = final_prog >= state_data.max_progress

    scores = final_qual * success.float()

    best_action = 0
    best_score = -1.0

    print(f"--- 15万次 x 30步 宇宙推演结束 (耗时: {(time.time() - start_time) * 1000:.1f}ms) ---")
    for a in range(num_actions):
        avg_score = scores[first_actions == a].mean().item()
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
    dummy = CraftingState(cp=500, durability=80, progress=0, quality=0, condition=0, max_progress=1000,
                          base_progress=600, base_quality=300)
    gpu_mcts(dummy)
    return {"status": "warmed up"}