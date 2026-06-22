import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time

app = FastAPI()
torch.backends.cudnn.benchmark = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE == 'cuda':
    # 限制 20% 显存，150万次推演绰绰有余，完美避开大模型抢占
    torch.cuda.set_per_process_memory_fraction(0.20, 0)

# ================= 27 技能大满贯映射表 =================
ACTION_MAP = {
    0: 100001, 1: 100002, 2: 100003, 3: 100004, 4: 100005, 5: 100006, 6: 100007, 7: 100008, 8: 100009,
    9: 100010, 10: 100011, 11: 100012, 12: 100013, 13: 100014, 14: 100015, 15: 100016, 16: 100017, 17: 100018,
    18: 100019, 19: 100020, 20: 100021, 21: 100022, 22: 100023, 23: 100024, 24: 281,
    25: 100025,  # 巧夺天工 (Trained Finesse)
    26: 100026  # 精密制作 (Delicate Synth)
}


class CraftingState(BaseModel):
    cp: int;
    durability: int;
    progress: int;
    quality: int;
    condition: int
    max_progress: int;
    max_quality: int;
    base_progress: float;
    base_quality: float
    step: int;
    iq: int;
    innov: int;
    vener: int;
    wn: int;
    gs: int;
    manip: int
    muscle: int;
    combo: int;
    p_avail: int;
    p_active: int


class GPUCraftingEnv:
    def __init__(self, batch_size=1500000):  # 💥 150 万只猴子并发！
        self.batch_size = batch_size
        self.device = DEVICE
        self.mults = torch.tensor([1.0, 1.5, 4.0, 0.5], device=DEVICE)

    def batch_step(self, states, actions, base_prog, base_qual):
        cp = states[:, 0];
        dur = states[:, 1];
        prog = states[:, 2];
        qual = states[:, 3]
        cond = states[:, 4].long();
        max_p = states[:, 5];
        iq = states[:, 6]
        innov = states[:, 7];
        vener = states[:, 8];
        wn = states[:, 9];
        gs = states[:, 10]
        manip = states[:, 11];
        muscle = states[:, 12];
        step = states[:, 13]
        combo = states[:, 14];
        p_avail = states[:, 15];
        p_active = states[:, 16]

        cost_18 = torch.where(combo == 1, 18, 32)
        cost_19 = torch.where(combo == 2, 18, 46)

        cost_map = {
            1: 18, 2: 88, 3: 7, 4: 18, 5: 18, 6: 56, 7: 32, 8: 24, 9: 96,
            10: 6, 11: 24, 12: 7, 13: 18, 14: 40, 15: 25, 16: 0, 17: 0,
            20: 0, 21: 18, 22: 6, 23: 0, 24: 98, 25: 32, 26: 32
        }

        # 强制替换违规动作为 0(制作)兜底，防止崩溃
        for act_idx, cost in cost_map.items():
            actions = torch.where((actions == act_idx) & (cp < cost), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 18) & (cp < cost_18), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 19) & (cp < cost_19), torch.tensor(0, device=DEVICE), actions)

        # 技能释放前置条件锁
        is_good_exc = (cond == 1) | (cond == 2)
        actions = torch.where(((actions == 20) | (actions == 21) | (actions == 22)) & ~is_good_exc,
                              torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 23) & (p_avail == 0), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where(((actions == 10) | (actions == 11)) & (step > 0), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 8) & (iq == 0), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 15) & (wn > 0), torch.tensor(0, device=DEVICE), actions)
        actions = torch.where((actions == 25) & (iq < 10), torch.tensor(0, device=DEVICE), actions)  # 巧夺必须内静10

        acts = {i: (actions == i) for i in range(27)}

        # 赌狗判定
        rng_rolls = torch.rand(self.batch_size, device=self.device)
        is_success = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        is_success = torch.where(acts[16], rng_rolls < 0.60, is_success)
        is_success = torch.where(acts[17], rng_rolls < 0.50, is_success)

        is_any_touch = acts[1] | acts[8] | acts[11] | acts[14] | acts[15] | acts[16] | acts[18] | acts[19] | acts[21] | \
                       acts[25] | acts[26]
        is_any_synth = acts[0] | acts[10] | acts[12] | acts[13] | acts[17] | acts[22] | acts[25] | acts[26]

        # 结算 CP
        cp_cost = torch.zeros_like(cp)
        for act_idx, cost in cost_map.items():
            cp_cost += torch.where(acts[act_idx], cost, 0)
        cp_cost += torch.where(acts[18], cost_18, 0)
        cp_cost += torch.where(acts[19], cost_19, 0)
        cp = torch.clamp(cp - cp_cost + torch.where(acts[20], 20, 0), max=999)

        # 结算耐久
        dur_base_cost = torch.zeros_like(dur)
        dur_base_cost += torch.where(
            acts[0] | acts[1] | acts[10] | acts[11] | acts[12] | acts[16] | acts[17] | acts[18] | acts[19] | acts[21] |
            acts[22] | acts[26], 10, 0)
        dur_base_cost += torch.where(acts[13] | acts[14], 20, 0)
        dur_base_cost += torch.where(acts[15], 5, 0)

        dur_cost = torch.where(wn > 0, dur_base_cost / 2.0, dur_base_cost.float()).int()
        dur_cost = torch.where(p_active == 1, 0, dur_cost)  # 绝技免消耗
        dur = torch.clamp(dur - dur_cost + torch.where(acts[2], 30, 0), max=80)

        # 结算进展 (包含巧夺/精密)
        synth_eff = torch.where(acts[0] | acts[25], 1.0, 0.0) + torch.where(acts[10], 3.0, 0.0) + \
                    torch.where(acts[12] | acts[26], 1.5, 0.0) + torch.where(acts[13], 3.6, 0.0) + \
                    torch.where(acts[22], 4.0, 0.0) + torch.where(acts[17] & is_success, 5.0, 0.0)
        prog_mult = synth_eff * torch.where((muscle > 0) & is_any_synth, 2.5, 1.0) * (
                    1.0 + torch.where(vener > 0, 0.5, 0.0))
        prog = prog + torch.where(is_any_synth, base_prog * prog_mult, 0.0)

        # 结算品质 (包含巧夺/精密)
        touch_eff = torch.where(acts[1] | acts[11] | acts[15] | acts[25] | acts[26], 1.0, 0.0) + \
                    torch.where(acts[18], 1.25, 0.0) + torch.where(acts[19], 1.5, 0.0) + \
                    torch.where(acts[14], 2.0, 0.0) + torch.where(acts[21], 1.5, 0.0) + \
                    torch.where(acts[8], 1.0 + 0.2 * iq, 0.0) + torch.where(acts[16] & is_success, 1.0, 0.0)

        qual_mult = (touch_eff + torch.where(innov > 0, 0.5, 0.0) + torch.where(gs > 0, 1.0, 0.0)) * (1.0 + 0.1 * iq)
        qual = qual + torch.where(is_any_touch, base_qual * qual_mult * self.mults[cond], 0.0)

        # Buff 轮转更新
        muscle = torch.where(is_any_synth, 0, torch.clamp(muscle - 1, min=0))
        muscle = torch.where(acts[10], 5, muscle)

        innov = torch.where(acts[4], 4, torch.clamp(innov - 1, min=0))
        vener = torch.where(acts[5], 4, torch.clamp(vener - 1, min=0))

        wn = torch.clamp(wn - 1, min=0)
        wn = torch.where(acts[6], 4, wn)
        wn = torch.where(acts[24], 8, wn)

        gs = torch.where(acts[7], 3, torch.clamp(gs - 1, min=0))
        gs = torch.where(is_any_touch, 0, gs)

        p_active = torch.where((dur_base_cost > 0) & (p_active == 1), 0, p_active)
        p_active = torch.where(acts[23], 1, p_active)
        p_avail = torch.where(acts[23], 0, p_avail)

        is_buff = acts[2] | acts[3] | acts[4] | acts[5] | acts[6] | acts[7] | acts[9] | acts[20] | acts[23] | acts[24]
        combo = torch.where(acts[1], 1, torch.where(acts[18], 2, torch.where(is_buff, combo, 0)))

        # 增加精密制作(26)的层数
        iq_gain = torch.where(acts[11] | acts[14] | acts[21], 2, 0) + torch.where(
            acts[1] | acts[15] | acts[18] | acts[19] | acts[26] | (acts[16] & is_success), 1, 0)
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


@torch.no_grad()
def gpu_mcts(state_data: CraftingState):
    start_time = time.time()
    env = GPUCraftingEnv(batch_size=1500000)  # 150 万宇宙

    initial_tensor = torch.tensor([
        state_data.cp, state_data.durability, state_data.progress,
        state_data.quality, state_data.condition, state_data.max_progress,
        state_data.iq, state_data.innov, state_data.vener, state_data.wn,
        state_data.gs, state_data.manip, state_data.muscle,
        state_data.step, state_data.combo, state_data.p_avail, state_data.p_active
    ], dtype=torch.float32, device=DEVICE)

    states = initial_tensor.repeat(env.batch_size, 1)
    num_actions = len(ACTION_MAP)

    first_actions = torch.randint(0, num_actions, (env.batch_size,), device=DEVICE)
    actions = first_actions.clone()
    active_mask = torch.ones(env.batch_size, dtype=torch.bool, device=DEVICE)

    search_depth = 40

    for _ in range(search_depth):
        new_states, step_active = env.batch_step(states, actions, state_data.base_progress, state_data.base_quality)
        # 冻结死亡宇宙
        states = torch.where(active_mask.unsqueeze(1), new_states, states)
        active_mask = active_mask & step_active
        actions = torch.where(active_mask, torch.randint(0, num_actions, (env.batch_size,), device=DEVICE),
                              torch.tensor(0, device=DEVICE))

    # ================= 🌟 零容忍计分系统 🌟 =================
    final_cp = states[:, 0]
    final_dur = states[:, 1]
    final_prog = states[:, 2]
    final_qual = torch.clamp(states[:, 3], max=state_data.max_quality)

    success = final_prog >= state_data.max_progress
    progress_ratio = final_prog / state_data.max_progress

    # 失败宇宙只有进展分，无品质分
    dense_score = progress_ratio * 10000.0 + final_dur * 10.0 + final_cp * 1.0
    # 成功宇宙解锁品质分与千万大奖
    success_score = 10000000.0 + final_qual * 100.0 + final_dur * 10.0 + final_cp * 5.0
    scores = torch.where(success, success_score, dense_score)

    # ================= 🛡️ 根节点合法过滤器 =================
    legal_actions = []
    cp = state_data.cp;
    cond = state_data.condition;
    step = state_data.step
    iq = state_data.iq;
    innov = state_data.innov;
    vener = state_data.vener
    wn = state_data.wn;
    gs = state_data.gs;
    manip = state_data.manip
    combo = state_data.combo;
    p_avail = state_data.p_avail

    cost_18_act = 18 if combo == 1 else 32
    cost_19_act = 18 if combo == 2 else 46
    cost_map = {
        1: 18, 2: 88, 3: 7, 4: 18, 5: 18, 6: 56, 7: 32, 8: 24, 9: 96,
        10: 6, 11: 24, 12: 7, 13: 18, 14: 40, 15: 25, 16: 0, 17: 0,
        20: 0, 21: 18, 22: 6, 23: 0, 24: 98, 25: 32, 26: 32
    }

    for a in range(num_actions):
        is_legal = True
        cost = cost_map.get(a, 0)
        if a == 18: cost = cost_18_act
        if a == 19: cost = cost_19_act

        if cp < cost: is_legal = False
        if a in [20, 21, 22] and cond not in [1, 2]: is_legal = False
        if a == 23 and p_avail == 0: is_legal = False
        if a in [10, 11] and step > 0: is_legal = False
        if a == 8 and iq == 0: is_legal = False
        if a == 15 and wn > 0: is_legal = False
        if a == 25 and iq < 10: is_legal = False

        # 严禁重复刷 Buff
        if a == 9 and manip > 0: is_legal = False
        if a in [6, 24] and wn > 0: is_legal = False
        if a == 4 and innov > 0: is_legal = False
        if a == 5 and vener > 0: is_legal = False
        if a == 7 and gs > 0: is_legal = False

        if is_legal: legal_actions.append(a)

    if not legal_actions: legal_actions = [0]

    best_action = legal_actions[0]
    best_score = -1.0

    print(f"--- 150万次 x {search_depth}步 推演结束 (耗时: {(time.time() - start_time) * 1000:.1f}ms) ---")

    for a in legal_actions:
        avg_score = scores[first_actions == a].mean().item()

        # 风险厌恶：大幅度削弱仓促(16)和高速制作(17)的得分
        if a in [16, 17]:
            avg_score = avg_score * 0.02

        if avg_score > 0:
            print(f"合法技能 [{a}] 期望分: {avg_score:.0f}")

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
    dummy = CraftingState(
        cp=600, durability=80, progress=0, quality=0, condition=0,
        max_progress=1000, max_quality=1000, base_progress=600, base_quality=300,
        step=0, iq=0, innov=0, vener=0, wn=0, gs=0, manip=0, muscle=0, combo=0, p_avail=1, p_active=0
    )
    gpu_mcts(dummy)
    return {"status": "warmed up"}