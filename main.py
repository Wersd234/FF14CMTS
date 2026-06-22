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

class CraftingState(BaseModel):
    cp: int
    durability: int
    progress: int
    quality: int
    condition: int
    max_quality: int
    max_progress: int
    base_progress: float
    base_quality: float


# ================= 扩展到 25 个技能 (拆分俭约与长期俭约) =================
ACTION_MAP = {
    0: 100001, 1: 100002, 2: 100003, 3: 100004,
    4: 100005, 5: 100006,
    6: 100007, # 俭约 (Waste Not) - 56 CP, 4回合
    7: 100008, # 阔步
    8: 100009, # 比尔格的祝福
    9: 100010, # 掌握
    10: 100011, # 坚信
    11: 100012, # 闲静
    12: 100013, # 模范制作
    13: 100014, # 坯料制作
    14: 100015, # 坯料加工
    15: 100016, # 专心加工
    16: 100017, # 仓促
    17: 100018, # 高速制作
    18: 100019, # 中级加工
    19: 100020, # 上级加工
    20: 100021, # 秘诀
    21: 100022, # 集中加工
    22: 100023, # 集中制作
    23: 100024, # 工匠的绝技
    24: 281     # 长期俭约 (Waste Not II) - 98 CP, 8回合 (注：真实技能ID是281，如果报错请根据游戏ID修改)
}


class GPUCraftingEnv:
    def __init__(self, batch_size=150000):
        self.batch_size = batch_size
        self.device = DEVICE
        self.mults = torch.tensor([1.0, 1.5, 4.0, 0.5], device=DEVICE)

    def batch_step(self, states, actions, base_prog, base_qual):
        # 17 维张量 (wn 维度现在表示两种俭约的合并倒计时)
        cp = states[:, 0];
        dur = states[:, 1];
        prog = states[:, 2];
        qual = states[:, 3]
        cond = states[:, 4].long();
        max_p = states[:, 5];
        iq = states[:, 6]
        innov = states[:, 7];
        vener = states[:, 8]
        wn = states[:, 9]  # 俭约状态统合记录 (不管是大的还是小的，只看剩余步数)
        gs = states[:, 10];
        manip = states[:, 11];
        muscle = states[:, 12]
        step = states[:, 13];
        combo = states[:, 14];
        p_avail = states[:, 15];
        p_active = states[:, 16]

        # ================= 1. 安全锁与 CP 计算 =================
        cost_18 = torch.where(combo == 1, 18, 32)
        cost_19 = torch.where(combo == 2, 18, 46)

        cost_map = {
            1: 18, 2: 88, 3: 7, 4: 18, 5: 18,
            6: 56,  # 俭约
            7: 32, 8: 24, 9: 96,
            10: 6, 11: 24, 12: 7, 13: 18, 14: 40, 15: 25, 16: 0, 17: 0,
            20: 0, 21: 18, 22: 6, 23: 0,
            24: 98  # 长期俭约
        }

        for act_idx, cost in cost_map.items():
            actions = torch.where((actions == act_idx) & (cp < cost), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 18) & (cp < cost_18), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 19) & (cp < cost_19), torch.tensor(0, device=DEVICE), actions)

        # 绝技与前置锁
        is_good_exc = (cond == 1) | (cond == 2)
        actions = torch.where(((actions == 20) | (actions == 21) | (actions == 22)) & ~is_good_exc,
                              torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 23) & (p_avail == 0), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where(((actions == 10) | (actions == 11)) & (step > 0), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 8) & (iq == 0), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 15) & (wn > 0), torch.tensor(0, device=DEVICE), actions)  # 专心加工严禁俭约

        acts = {i: (actions == i) for i in range(25)}

        # RNG (仓促60%, 高速50%)
        rng_rolls = torch.rand(self.batch_size, device=self.device)
        is_success = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        is_success = torch.where(acts[16], rng_rolls < 0.60, is_success)
        is_success = torch.where(acts[17], rng_rolls < 0.50, is_success)

        is_any_touch = acts[1] | acts[8] | acts[11] | acts[14] | acts[15] | acts[16] | acts[18] | acts[19] | acts[21]
        is_any_synth = acts[0] | acts[10] | acts[12] | acts[13] | acts[17] | acts[22]

        # ================= 2. 扣除 CP =================
        cp_cost = torch.zeros_like(cp)
        for act_idx, cost in cost_map.items():
            cp_cost += torch.where(acts[act_idx], cost, 0)
        cp_cost += torch.where(acts[18], cost_18, 0)
        cp_cost += torch.where(acts[19], cost_19, 0)
        cp = torch.clamp(cp - cp_cost + torch.where(acts[20], 20, 0), max=999)

        # ================= 3. 扣除耐久 =================
        dur_base_cost = torch.zeros_like(dur)
        dur_base_cost += torch.where(
            acts[0] | acts[1] | acts[10] | acts[11] | acts[12] | acts[16] | acts[17] | acts[18] | acts[19] | acts[21] |
            acts[22], 10, 0)
        dur_base_cost += torch.where(acts[13] | acts[14], 20, 0)
        dur_base_cost += torch.where(acts[15], 5, 0)

        # 俭约减半 (当前 wn > 0 即生效)
        dur_cost = torch.where(wn > 0, dur_base_cost / 2.0, dur_base_cost.float()).int()
        dur_cost = torch.where(p_active == 1, 0, dur_cost)  # 7.0 绝技免耐久

        dur = torch.clamp(dur - dur_cost + torch.where(acts[2], 30, 0), max=80)

        # ================= 4. 进展与品质 =================
        synth_eff = torch.where(acts[0], 1.2, 0.0) + torch.where(acts[10], 3.0, 0.0) + \
                    torch.where(acts[12], 1.5, 0.0) + torch.where(acts[13], 3.6, 0.0) + \
                    torch.where(acts[22], 4.0, 0.0) + torch.where(acts[17] & is_success, 5.0, 0.0)
        prog_mult = synth_eff * torch.where((muscle > 0) & is_any_synth, 2.5, 1.0) * (
                    1.0 + torch.where(vener > 0, 0.5, 0.0))
        prog = prog + torch.where(is_any_synth, base_prog * prog_mult, 0.0)

        touch_eff = torch.where(acts[1] | acts[11] | acts[15], 1.0, 0.0) + \
                    torch.where(acts[18], 1.25, 0.0) + torch.where(acts[19], 1.5, 0.0) + \
                    torch.where(acts[14], 2.0, 0.0) + torch.where(acts[21], 1.5, 0.0) + \
                    torch.where(acts[8], 1.0 + 0.2 * iq, 0.0) + torch.where(acts[16] & is_success, 1.0, 0.0)

        qual_mult = (touch_eff + torch.where(innov > 0, 0.5, 0.0) + torch.where(gs > 0, 1.0, 0.0)) * (1.0 + 0.1 * iq)
        qual = qual + torch.where(is_any_touch, base_qual * qual_mult * self.mults[cond], 0.0)

        # ================= 5. Buff 更新 =================
        muscle = torch.where(is_any_synth, 0, torch.clamp(muscle - 1, min=0))
        muscle = torch.where(acts[10], 5, muscle)

        innov = torch.where(acts[4], 4, torch.clamp(innov - 1, min=0))
        vener = torch.where(acts[5], 4, torch.clamp(vener - 1, min=0))

        # ⚠️ 区分两种俭约的覆盖逻辑：
        wn = torch.clamp(wn - 1, min=0)  # 先全员减1
        wn = torch.where(acts[6], 4, wn)  # 小俭约强行设为4
        wn = torch.where(acts[24], 8, wn)  # 大俭约强行设为8

        gs = torch.where(acts[7], 3, torch.clamp(gs - 1, min=0))
        gs = torch.where(is_any_touch, 0, gs)

        p_active = torch.where((dur_base_cost > 0) & (p_active == 1), 0, p_active)
        p_active = torch.where(acts[23], 1, p_active)
        p_avail = torch.where(acts[23], 0, p_avail)

        is_buff = acts[2] | acts[3] | acts[4] | acts[5] | acts[6] | acts[7] | acts[9] | acts[20] | acts[23] | acts[24]
        combo = torch.where(acts[1], 1, torch.where(acts[18], 2, torch.where(is_buff, combo, 0)))

        iq_gain = torch.where(acts[11] | acts[14] | acts[21], 2, 0) + torch.where(
            acts[1] | acts[15] | acts[18] | acts[19] | (acts[16] & is_success), 1, 0)
        iq = torch.where(acts[8], 0, torch.clamp(iq + iq_gain, max=10))

        manip = torch.where(acts[9], 8, torch.clamp(manip - 1, min=0))
        dur = torch.clamp(torch.where((manip > 0) & (dur > 0), dur + 5, dur), max=80)

        step = step + 1

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

    # 评分逻辑: 进展满后看品质，品质封顶不给额外分
    final_prog = states[:, 2]
    # 把品质“截断”在满品质数值，防止溢出浪费动作
    final_qual = torch.clamp(states[:, 3], max=state_data.max_quality)

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