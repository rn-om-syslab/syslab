import os, re, json, time, signal, shutil, math, argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
 
 
SAVE_DIR = r""
CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
LOG_DIR  = os.path.join(SAVE_DIR, "logs")
STATE_FILE   = os.path.join(SAVE_DIR, "training_state.json")
METRICS_FILE = os.path.join(LOG_DIR, "metrics.json")
 
ADV_ROUNDS = 15
ADV_STEPS_PER_ROUND = 12_000
CHECKPOINT_FREQ = 6_000
 
PPO_KWARGS = dict(learning_rate=3e-4, n_steps=1024, batch_size=256,n_epochs=8,
                  gamma=0.995, gae_lambda=0.95, clip_range=0.2,
                  ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)
NET_ARCH = [256,256]
 
# rnd -> (use_obstacle, ev_speed_mult)
# CURRICULUM = {0:(False,1.0), 5:(True,1.0), 10:(True,1.1)}  # earlier version
CURRICULUM = {0:(False,1.0), 3:(False,1.05), 6:(True,1.05), 10:(True,1.10)}
 
MASS = 0.087
DRAG_COEFF = 0.02
VMAX_TRACKER = 4.0
# vmax_evader  =4.5
VMAX_V = 3.5
VEL_TAU= 0.15
DT = 1.0/20
 
BOUND_XY=7.0
BOUND_ZLO = 0.4
BOUND_ZHI= 4.0
 
# cyl obstacle in middle
OBS_POS    = np.array([0.0,0.0,0.0])
OBS_RADIUS = 0.8
OBS_HEIGHT= 3.5
 
MAX_STEPS = 400
CAPTURE_DIST=0.7
 
# fov
FOV_HALF_DEG=60.0
FOV_RANGE = 6.0   # was 5
fov_half_rad= math.radians(FOV_HALF_DEG)
 
 
class Drone:
    def __init__(self, pos, vmax_h, vmax_v=VMAX_V):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.vmax_h=vmax_h
        self.vmax_v=vmax_v
 
    def step(self, cmd, dt):
        # first-order lag + drag
        target = np.array([cmd[0]*self.vmax_h, cmd[1]*self.vmax_h, cmd[2]*self.vmax_v])
        a = dt / (VEL_TAU + dt)
        self.vel = (1-a)*self.vel +a*target
        self.vel += -DRAG_COEFF *self.vel * np.abs(self.vel) / MASS * dt
        self.pos += self.vel * dt
 
    def clamp(self):
        if   self.pos[0] >=  BOUND_XY: self.pos[0]= BOUND_XY; self.vel[0]=min(self.vel[0],0.0)
        elif self.pos[0] <= -BOUND_XY: self.pos[0]=-BOUND_XY; self.vel[0]=max(self.vel[0],0.0)
        if   self.pos[1] >=  BOUND_XY: self.pos[1]= BOUND_XY; self.vel[1]=min(self.vel[1],0.0)
        elif self.pos[1] <= -BOUND_XY: self.pos[1]=-BOUND_XY; self.vel[1]=max(self.vel[1],0.0)
        if   self.pos[2] >= BOUND_ZHI: self.pos[2]= BOUND_ZHI; self.vel[2]=min(self.vel[2],0.0)
        elif self.pos[2] <= BOUND_ZLO: self.pos[2]= BOUND_ZLO; self.vel[2]=max(self.vel[2],0.0)
 
    def collide_obstacle(self):
        dx,dy = self.pos[0]-OBS_POS[0],self.pos[1]-OBS_POS[1]
        d = np.sqrt(dx*dx + dy*dy)
        if d < OBS_RADIUS and self.pos[2] < OBS_HEIGHT:
            if d<1e-4: dx,dy,d = 1.0,0.0,1.0
            s = OBS_RADIUS/d
            self.pos[0]= OBS_POS[0] + dx*s
            self.pos[1]= OBS_POS[1]+ dy*s
            radial = np.array([dx,dy])/d
            vr = self.vel[0]*radial[0] + self.vel[1]*radial[1]
            if vr<0:
                self.vel[0] -=vr*radial[0]
                self.vel[1] -= vr*radial[1]
 
    def oob(self):
        return (abs(self.pos[0])>BOUND_XY or abs(self.pos[1])>BOUND_XY
                or self.pos[2]<BOUND_ZLO or self.pos[2]>BOUND_ZHI)
 
 
class PursuitBase(gym.Env):
    def __init__(self, render=False, use_fov=True):
        super().__init__()
        self.render_mode=render
        self.use_fov=use_fov
 
        self.tracker = Drone([0,0,1.5], VMAX_TRACKER)
        self.evader  = Drone([2,0,1.5], vmax_evader)
 
        self.step_count=0
        self.episode_count=0
        self.use_obstacle=False
        self.evader_speed_mult=1.0
        self.tracker_wins=0
        self.evader_wins =0
        self.total_eps =0
        self.prev_evader_pos = np.zeros(3)
        self.total_dist_t=0.0
        self.total_dist_e=0.0
 
        self.tracker_heading = np.array([1.0,0.0,0.0])
        self.last_seen_rel = np.zeros(3)
        self.steps_since_seen=0
        self.ever_seen=True
 
        # pybullet ids
        self.tracker_id=None
        self.evader_id =None
        self.obstacle_id=None
        self.fov_lines = []
        self.lkp_marker= None
 
        if self.render_mode: self.init_pb()
 
    def init_pb(self):
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.resetDebugVisualizerCamera(cameraDistance=12,cameraYaw=45,cameraPitch=-30,
                                     cameraTargetPosition=[0,0,1.5])
        p.setGravity(0,0,0)
        # arena box
        b=BOUND_XY


        c = np.array([[-b,-b,BOUND_ZLO],[b,-b,BOUND_ZLO],[b,b,BOUND_ZLO],[-b,b,BOUND_ZLO],
                      [-b,-b,BOUND_ZHI],[b,-b,BOUND_ZHI],[b,b,BOUND_ZHI],[-b,b,BOUND_ZHI]])
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i,j in edges:
            p.addUserDebugLine(c[i].tolist(), c[j].tolist(), [0.4,0.4,0.4], 1.5)
        # drones (red tracker, blue evader)
        tvs = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[1.0,0.2,0.2,1.0])
        evs = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[0.2,0.5,1.0,1.0])
        self.tracker_id =p.createMultiBody(0, baseVisualShapeIndex=tvs, basePosition=[0,0,1.5])
        self.evader_id  = p.createMultiBody(0, baseVisualShapeIndex=evs, basePosition=[2,0,1.5])
        # last-known marker (yellow), hidden offscreen until needed


        lvs = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[1.0,1.0,0.0,0.7])
        self.lkp_marker = p.createMultiBody(0, baseVisualShapeIndex=lvs, basePosition=[99,99,99])
        for _ in range(20):
            self.fov_lines.append(p.addUserDebugLine([0,0,0],[0,0,1],[0,1,0],1.0))
 
    def maybe_draw_obstacle(self):
        if not self.render_mode or self.obstacle_id is not None or not self.use_obstacle: return
        cyl = p.createVisualShape(p.GEOM_CYLINDER, radius=OBS_RADIUS, length=OBS_HEIGHT,
                                  rgbaColor=[0.5,0.5,0.5,0.6])
        self.obstacle_id = p.createMultiBody(0,baseVisualShapeIndex=cyl,
                                             basePosition=[OBS_POS[0],OBS_POS[1],OBS_HEIGHT/2])
 
    def draw_fov_cone(self):
        if not self.render_mode or not self.fov_lines: return
        h = self.tracker_heading
        up = np.array([0,0,1.0]) if abs(h[2])<0.9 else np.array([1.0,0,0])
        p1 = np.cross(h, up); p1 /= (np.linalg.norm(p1)+1e-8)
        p2 = np.cross(h, p1); p2/= (np.linalg.norm(p2)+1e-8)
        r = math.tan(fov_half_rad) * FOV_RANGE
        ctr = self.tracker.pos + h*FOV_RANGE
        color = [0,1,0] if self.evader_in_fov() else [1,0.7,0]
        n = len(self.fov_lines)
        for lid,ang in zip(self.fov_lines, np.linspace(0, 2*math.pi, n, endpoint=False)):
            tip = ctr + (p1*math.cos(ang) + p2*math.sin(ang)) * r
            p.addUserDebugLine(self.tracker.pos.tolist(), tip.tolist(), color,1.0,
                                replaceItemUniqueId=lid)
 
    def refresh_visuals(self):
        if not self.render_mode: return
        I=[0,0,0,1]
        p.resetBasePositionAndOrientation(self.tracker_id, self.tracker.pos.tolist(), I)
        p.resetBasePositionAndOrientation(self.evader_id,  self.evader.pos.tolist(),  I)
        if self.lkp_marker is not None:
            if self.steps_since_seen > 0 and self.use_fov:
                lkp = (self.tracker.pos + self.last_seen_rel).tolist()
            else:
                lkp = [99,99,99]
            p.resetBasePositionAndOrientation(self.lkp_marker, lkp, I)
        self.maybe_draw_obstacle()
        self.draw_fov_cone()
 
    def set_curriculum(self, use_obstacle=False,  evader_speed_mult=1.0):
        self.use_obstacle=use_obstacle
        self.evader_speed_mult = evader_speed_mult
        self.evader.vmax_h = vmax_evader * evader_speed_mult
 
    def spawn(self):
        self.step_count=0
        self.total_dist_t=0.0
        self.total_dist_e=0.0
        # evader spawn (avoid obstacle)
        for _ in range(20):
            ex = np.random.uniform(-BOUND_XY*0.7, BOUND_XY*0.7)
            ey = np.random.uniform(-BOUND_XY*0.7,BOUND_XY*0.7)
            if not self.use_obstacle or np.hypot(ex-OBS_POS[0], ey-OBS_POS[1]) > OBS_RADIUS+0.5: break
        ez = np.random.uniform(1.0, 2.8)
        # tracker 3-5m from evader
        for _ in range(20):
            off = np.random.uniform(3.0,5.0)
            ang = np.random.uniform(0, 2*np.pi)
            tx = np.clip(ex+off*np.cos(ang), -BOUND_XY*0.85, BOUND_XY*0.85)
            ty = np.clip(ey+off*np.sin(ang),-BOUND_XY*0.85, BOUND_XY*0.85)
            if not self.use_obstacle or np.hypot(tx-OBS_POS[0], ty-OBS_POS[1]) > OBS_RADIUS+0.5: break
        tz = np.random.uniform(1.0,2.8)
        # tz = ez  # tried matching tracker altitude to evader, made it too easy
        self.evader.pos[:]  = [ex,ey,ez]; self.evader.vel[:] = 0
        self.tracker.pos[:]= [tx,ty,tz]; self.tracker.vel[:] = 0
        # random heading, mostly horizontal
        h = np.random.randn(3).astype(np.float32)
        h[2] *= 0.3
        h /= (np.linalg.norm(h)+1e-8)
        self.tracker_heading = h
        self.prev_evader_pos =self.evader.pos.copy()
        self.last_seen_rel   = self.evader.pos - self.tracker.pos
        self.steps_since_seen=0
        self.ever_seen=True
 
    def update_visibility(self):
        # called after both drones have moved- updates last_seen_rel +counter
        if self.use_fov:
            if self.evader_in_fov():
                self.last_seen_rel = self.evader.pos - self.tracker.pos
                self.steps_since_seen=0
            else:
                self.steps_since_seen += 1
        else:
            self.last_seen_rel = self.evader.pos - self.tracker.pos
            self.steps_since_seen=0
 
    @staticmethod
    def predict_or_fallback(model, vn, obs_raw, fallback):
        # run frozen opponent policy if loaded, else use fallback action
        if model is None: return fallback()
        if vn is not None:
            obs = vn.normalize_obs(obs_raw.reshape(1,-1)).reshape(-1).astype(np.float32)
        else:
            obs = obs_raw
        try:
            act,_ = model.predict(obs, deterministic=False)
            return np.clip(act,-1.0,1.0)
        except Exception:
            return fallback()
 
    def evader_in_fov(self):
        diff = self.evader.pos - self.tracker.pos
        d = np.linalg.norm(diff)
        if d<1e-6 or d>FOV_RANGE: return False
        return float(np.dot(diff/d, self.tracker_heading))>= math.cos(fov_half_rad)
 
    def update_heading(self):
        v=self.tracker.vel
        s=np.linalg.norm(v)
        if s >0.15: self.tracker_heading = (v/s).astype(np.float32)
 
    def evader_velocity(self):
        return (self.evader.pos - self.prev_evader_pos) / DT
 
    def obs_tracker(self):
        rel  = self.evader.pos - self.tracker.pos
        evel = np.clip(self.evader_velocity(), -vmax_evader*1.5, vmax_evader*1.5)
        if self.use_fov and not self.evader_in_fov():
            # out of view -- zero the velocity, use last-seen pos
            evel = np.zeros(3, dtype=np.float32)
            rel  = self.last_seen_rel
        else:
            self.last_seen_rel = rel.copy()
            self.steps_since_seen = 0
        return np.array([rel[0],rel[1],rel[2],
                         self.tracker.vel[0],self.tracker.vel[1],self.tracker.vel[2],
                         evel[0],evel[1],evel[2]], dtype=np.float32)
 
    def obs_evader(self):
        rel = self.tracker.pos - self.evader.pos
        dist = np.linalg.norm(rel)
        warned = dist < 5.0
        threat = rel/(dist+1e-6) * min(dist,5.0)/5.0 if warned else np.zeros(3)
        wx = min(BOUND_XY - self.evader.pos[0], BOUND_XY + self.evader.pos[0]) / BOUND_XY
        wy = min(BOUND_XY - self.evader.pos[1], BOUND_XY + self.evader.pos[1]) / BOUND_XY
        return np.array([threat[0],threat[1],threat[2],
                         1.0 if warned else 0.0,
                         self.evader.vel[0],self.evader.vel[1],self.evader.vel[2],
                         wx,wy], dtype=np.float32), dist
 
    def get_win_rates(self):
        if self.total_eps==0: return 0.0, 0.0
        return self.tracker_wins/self.total_eps, self.evader_wins/self.total_eps
 
    def close(self):
        if self.render_mode:
            try: p.disconnect()
            except Exception: pass
 
 
class TrackerEnv(PursuitBase):
    def __init__(self, evader_model=None, evader_vecnorm=None, render=False, use_fov=True):
        super().__init__(render=render, use_fov=use_fov)
        self.observation_space = spaces.Box(-np.inf, np.inf, (9,), np.float32)
        self.action_space = spaces.Box(-1.0,1.0, (3,), np.float32)
        self.evader_model = evader_model
        self.evader_vecnorm = evader_vecnorm
 
    def set_evader_model(self, m, vecnorm=None):
        self.evader_model=m
        self.evader_vecnorm=vecnorm
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.spawn()
        self.episode_count += 1
        obs = self.obs_tracker()
        d = float(np.linalg.norm(self.evader.pos - self.tracker.pos))
        print(f"\n[tracker ep {self.episode_count}] dist={d:.2f}m  "
              f"obstacle={self.use_obstacle}  T:{self.tracker_wins} E:{self.evader_wins}")
        self.refresh_visuals()
        return obs, {}
 
    def step(self, action):
        self.step_count += 1
        action = np.clip(action,-1.0,1.0)
        prev_t = self.tracker.pos.copy()
        prev_e_for_vel = self.evader.pos.copy()
 
        self.tracker.step(action, DT)
        self.tracker.clamp()
        if self.use_obstacle: self.tracker.collide_obstacle()
        self.update_heading()
 
        # evader move (frozen opponent or fallback)
        e_obs_raw,_ = self.obs_evader()
        e_obs_raw = e_obs_raw.astype(np.float32)
        def ev_fallback():
            away = self.evader.pos - self.tracker.pos
            away /= (np.linalg.norm(away)+1e-6)
            return np.clip(away + np.random.uniform(-0.3,0.3,3), -1, 1)
        e_act = self.predict_or_fallback(self.evader_model, self.evader_vecnorm, e_obs_raw, ev_fallback)
        self.evader.step(e_act, DT)
        self.evader.clamp()
        if self.use_obstacle: self.evader.collide_obstacle()
 
        # visibility tracking
        self.update_visibility()
 
        self.prev_evader_pos = prev_e_for_vel
        self.total_dist_t += np.linalg.norm(self.tracker.pos - prev_t)
        self.total_dist_e += np.linalg.norm(self.evader.pos - prev_e_for_vel)
 
        obs = self.obs_tracker()
        dist = float(np.linalg.norm(self.evader.pos - self.tracker.pos))
        r, info = self._reward(dist, prev_t)
        done = self._done(dist, info)
        self.refresh_visuals()
        return obs, r, done, False, info
 
    def _reward(self, dist, prev_t):
        r = 0.0
        rel = self.evader.pos - self.tracker.pos
        nt = np.linalg.norm(rel)
        if nt > 0.1:
            # closing-rate shaping: dot of velocity onto unit vector toward evader
            r += float(np.dot(self.tracker.pos-prev_t, rel/nt)) * 20.0
        r += 0.5 * np.exp(-dist/3.0)
        # r += 1.0 / (1.0 + dist)   # old shaping, too greedy
        if self.use_fov and self.evader_in_fov(): r += 0.05
        if dist <= CAPTURE_DIST:
            r += 60.0 + 50.0*(1.0 - self.step_count/MAX_STEPS)
        if self.tracker.oob(): r -= 20.0
        if self.use_obstacle:
            d_obs = np.hypot(self.tracker.pos[0]-OBS_POS[0], self.tracker.pos[1]-OBS_POS[1])
            if d_obs < OBS_RADIUS+0.3 and self.tracker.pos[2] < OBS_HEIGHT:
                r -= 0.3 * (1.0 - (d_obs-OBS_RADIUS)/0.3)
        r -= 0.02   # time penalty
        return r, {'distance':dist}
 
    def _done(self, dist, info):
        if dist <= CAPTURE_DIST:
            print(f"[tracker wins] dist={dist:.2f}m in {self.step_count} steps")
            self.tracker_wins+=1; self.total_eps+=1
            return True
        if self.tracker.oob():
            self.total_eps+=1; return True
        if self.step_count >= MAX_STEPS:
            self.evader_wins+=1; self.total_eps+=1
            return True
        return False
 
 
class EvaderEnv(PursuitBase):
    def __init__(self, tracker_model=None, tracker_vecnorm=None, render=False, use_fov=True):
        super().__init__(render=render, use_fov=use_fov)
        self.observation_space = spaces.Box(-np.inf, np.inf, (9,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.tracker_model = tracker_model
        self.tracker_vecnorm=tracker_vecnorm
        self.prev_action = np.zeros(3, dtype=np.float32)
 
    def set_tracker_model(self, m, vecnorm=None):
        self.tracker_model=m
        self.tracker_vecnorm=vecnorm
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.spawn()
        self.episode_count += 1
        self.prev_action = np.zeros(3, dtype=np.float32)
        obs, dist = self.obs_evader()
        print(f"\n[evader ep {self.episode_count}] dist={dist:.2f}m  "
              f"obstacle={self.use_obstacle}  E:{self.evader_wins} T:{self.tracker_wins}")
        self.refresh_visuals()
        return obs, {}
 
    def step(self, evader_action):
        self.step_count += 1
        evader_action = np.clip(evader_action,-1.0,1.0)
        prev_t = self.tracker.pos.copy()
        prev_e = self.evader.pos.copy()
 
        self.evader.step(evader_action, DT)
        self.evader.clamp()
        if self.use_obstacle: self.evader.collide_obstacle()
 
        # tracker's velocity estimate needs the pre-step evader pos
        self.prev_evader_pos = prev_e
        t_obs_raw = self.obs_tracker().astype(np.float32)
        def tr_fallback():
            rel = self.evader.pos - self.tracker.pos
            rel /= (np.linalg.norm(rel)+1e-6)
            return np.clip(rel,-1,1)
        t_act = self.predict_or_fallback(self.tracker_model, self.tracker_vecnorm, t_obs_raw, tr_fallback)
        self.tracker.step(t_act, DT)
        self.tracker.clamp()
        if self.use_obstacle: self.tracker.collide_obstacle()
        self.update_heading()
 
        self.update_visibility()
 
        self.total_dist_t += np.linalg.norm(self.tracker.pos - prev_t)
        self.total_dist_e += np.linalg.norm(self.evader.pos - prev_e)
 
        obs, dist = self.obs_evader()
        r, info = self._reward(dist, evader_action)
        done = self._done(dist, info)
        self.prev_action = evader_action.copy()
        self.refresh_visuals()
        return obs, r, done, False, info
 
    def _reward(self, dist, action):
        r = 0.05*min(dist,6.0)
        spd = np.linalg.norm(self.evader.vel)
        r += 0.15 * min(1.0, spd/2.5)
        # dodge bonus -- sharp direction change when close
        if dist < 3.0 and self.step_count > 1:
            pn = np.linalg.norm(self.prev_action)
            cn = np.linalg.norm(action)
            if pn>0.15 and cn>0.15:
                cos_turn = float(np.dot(self.prev_action/pn, action/cn))
                turn = max(0.0, 0.5 - cos_turn)
                prox = 1.0 - dist/3.0
                r += 5.0 * turn * prox
        if dist<3.5: r += 1.5 * abs(self.evader.vel[2])/VMAX_V
        if self.use_obstacle:
            d_obs = np.hypot(self.evader.pos[0]-OBS_POS[0], self.evader.pos[1]-OBS_POS[1])
            if 0.9 < d_obs < 2.2 and dist < 4.0: r += 0.5
        if self.use_fov and not self.evader_in_fov(): r += 0.1
        # if self.steps_since_seen > 10: r += 0.2  # too easy to game
        if dist <= CAPTURE_DIST: r -= 30.0
        if self.evader.oob(): r -= 1.0
        if self.use_obstacle:
            d_obs = np.hypot(self.evader.pos[0]-OBS_POS[0], self.evader.pos[1]-OBS_POS[1])
            if d_obs<OBS_RADIUS+0.2 and self.evader.pos[2]<OBS_HEIGHT: r -= 0.3
        return r, {'distance':dist}
 
    def _done(self, dist, info):
        if dist <= CAPTURE_DIST:
            print(f"[tracker catches evader] dist={dist:.2f}m at step {self.step_count}")
            self.tracker_wins+=1; self.total_eps+=1
            return True
        if self.step_count >= MAX_STEPS:
            print(f"[evader survives {MAX_STEPS} steps]")
            self.evader_wins+=1; self.total_eps+=1
            return True
        return False
 
 
# training infra below
 
_STOP=False
_FORCE=False
 
def sigint_handler(sig, frame):
    global _STOP,_FORCE
    if not _STOP:
        print("\n[interrupt] saving after this rollout"); _STOP=True
    elif not _FORCE:
        _FORCE=True; raise KeyboardInterrupt
signal.signal(signal.SIGINT, sigint_handler)
 
 
class EarlyStop(BaseCallback):
    def _on_step(self): return not _STOP
 
 
class MetricsCB(BaseCallback):
    def __init__(self, role, rnd, verbose=0):
        super().__init__(verbose)
        self.role=role; self.rnd=rnd
        self.r_buf=[]; self.l_buf=[]; self.upd=0
 
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if info and info.get("episode"):
                self.r_buf.append(float(info["episode"]["r"]))
                self.l_buf.append(float(info["episode"]["l"]))
        return True
 
    def _on_rollout_end(self):
        self.upd += 1
        if not self.r_buf: return
        self.write_metrics()
        if self.upd % 3 == 0:
            print(f"  [{self.role} r{self.rnd}] upd={self.upd} steps={self.num_timesteps:,} "
                  f"eps={len(self.r_buf)} r10={np.mean(self.r_buf[-10:]):.1f} "
                  f"len10={np.mean(self.l_buf[-10:]):.0f}")
 
    def write_metrics(self):
        entry = {"role":self.role, "round":self.rnd,
                 "timesteps":self.num_timesteps, "update":self.upd,
                 "mean_r":float(np.mean(self.r_buf[-10:])),
                 "mean_l":float(np.mean(self.l_buf[-10:])),
                 "eps":len(self.r_buf)}
        os.makedirs(LOG_DIR, exist_ok=True)
        existing = []
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE) as f: existing = json.load(f)
            except Exception: pass
        existing.append(entry)
        with open(METRICS_FILE, "w") as f: json.dump(existing, f, indent=2)
 
    def flush(self): self.write_metrics()
 
 
class RoundCheckpointCB(BaseCallback):
    # save snapshot at end of each round so visualizer can replay
    def __init__(self, role, rnd, verbose=0):
        super().__init__(verbose)
        self.role=role
        self.rnd=rnd
    def _on_step(self): return True
    def _on_training_end(self):
        path = os.path.join(CHECKPOINT_DIR, f"{self.role}_round_{self.rnd:02d}.zip")
        self.model.save(path)
        print(f"  [{self.role}] round snapshot -> {os.path.basename(path)}")
 
 
def make_callbacks(prefix, role, rnd):
    ckpt = CheckpointCallback(CHECKPOINT_FREQ, CHECKPOINT_DIR, prefix)
    met  = MetricsCB(role, rnd)
    snap = RoundCheckpointCB(role, rnd)
    return [ckpt, met, snap, EarlyStop()], met
 
 
def latest_ckpt(prefix):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best = (0, None)
    for f in os.listdir(CHECKPOINT_DIR):
        m = re.match(rf"^{prefix}_(\d+)_steps\.zip$", f)
        if m:
            n = int(m.group(1))
            if n > best[0]: best = (n, os.path.join(CHECKPOINT_DIR, f))
    return best
 
 
def load_or_new(prefix, env):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for d in (SAVE_DIR, CHECKPOINT_DIR, LOG_DIR): os.makedirs(d, exist_ok=True)
    main = os.path.join(SAVE_DIR, f"{prefix}.zip")
    if os.path.exists(main) and os.path.getsize(main) > 1000:
        print(f"[{prefix}] loading {main}")
        return PPO.load(main, env=env, device=device)
    steps, ckpt = latest_ckpt(prefix)
    if ckpt:
        print(f"[{prefix}] loading checkpoint ({steps:,} steps)")
        return PPO.load(ckpt, env=env, device=device)
    print(f"[{prefix}] new model")
    return PPO("MlpPolicy", env, **PPO_KWARGS, tensorboard_log=LOG_DIR,
               policy_kwargs=dict(net_arch=NET_ARCH), verbose=1, device=device)
 
 
def phase_params(rnd):
    cur = CURRICULUM[0]
    for r,params in sorted(CURRICULUM.items()):
        if rnd >= r: cur = params
    return cur
 
 
def save_state(s):
    s["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(STATE_FILE, "w") as f: json.dump(s, f, indent=2)
 
def load_state():
    if not os.path.exists(STATE_FILE): return None
    with open(STATE_FILE) as f: s = json.load(f)
    s.setdefault("start_round",0)
    s.setdefault("tracker_steps",0)
    s.setdefault("evader_steps",0)
    print(f"[resume] round={s['start_round']+1} T={s['tracker_steps']:,} E={s['evader_steps']:,}")
    return s
 
def clear_state():
    if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
 
 
def make_vec(env_fn):
    v = DummyVecEnv([env_fn])
    return VecNormalize(v, norm_obs=True, norm_reward=False, clip_obs=10.0)
 
def save_vn(v, path): v.save(path)
def load_vn(v, path):
    return VecNormalize.load(path, v) if os.path.exists(path) else v
 
 
def train():
    global _STOP
    for d in (SAVE_DIR, CHECKPOINT_DIR, LOG_DIR): os.makedirs(d, exist_ok=True)
    print("="*64)
    print(f"  PURSUIT-EVASION  (asymmetric obs + obstacle + FOV)")
    print(f"  {ADV_ROUNDS} rounds x {ADV_STEPS_PER_ROUND:,} steps/agent")
    print(f"  device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*64)
 
    st = load_state()
    start_round   = st["start_round"]   if st else 0
    tracker_steps = st["tracker_steps"] if st else 0
    evader_steps  = st["evader_steps"]  if st else 0
 
    t_vn_path = os.path.join(SAVE_DIR, "tracker_vecnorm.pkl")
    e_vn_path = os.path.join(SAVE_DIR, "evader_vecnorm.pkl")
    t_model_path = os.path.join(SAVE_DIR, "tracker.zip")
    e_model_path = os.path.join(SAVE_DIR, "evader.zip")
 
    use_obs, espeed = phase_params(start_round)
 
    def t_bootstrap():
        e = TrackerEnv(evader_model=None); e.set_curriculum(use_obs, espeed)
        return Monitor(e)
    t_venv = make_vec(t_bootstrap); t_venv = load_vn(t_venv, t_vn_path)
    tracker_model = load_or_new("tracker", t_venv); t_venv.close()
 
    tracker_for_evader = PPO.load(t_model_path) if os.path.exists(t_model_path) else None
    def e_bootstrap():
        e = EvaderEnv(tracker_model=tracker_for_evader); e.set_curriculum(use_obs, espeed)
        return Monitor(e)
    e_venv = make_vec(e_bootstrap); e_venv = load_vn(e_venv, e_vn_path)
    evader_model = load_or_new("evader", e_venv); e_venv.close()
 
    rnd = start_round
    t_cb=None; e_cb=None
    try:
        for rnd in range(start_round, ADV_ROUNDS):
            if _STOP: break
            use_obs, espeed = phase_params(rnd)
            print(f"\n{'='*64}")
            print(f"  ROUND {rnd+1}/{ADV_ROUNDS}  obstacle={use_obs}  "
                  f"evader_speed={espeed:.2f}x  T={tracker_steps:,}  E={evader_steps:,}")
            print(f"{'='*64}")
 
            # ---- tracker phase ----
            print(f"\n  [tracker] {ADV_STEPS_PER_ROUND:,} steps")
            ev_vn = None
            if os.path.exists(e_vn_path):
                ev_vn = VecNormalize.load(e_vn_path,
                    DummyVecEnv([lambda: EvaderEnv(tracker_model=None)]))
                ev_vn.training=False; ev_vn.norm_reward=False
            def t_fn():
                e = TrackerEnv(evader_model=evader_model, evader_vecnorm=ev_vn)
                e.set_curriculum(use_obs, espeed); return Monitor(e)
            t_venv = make_vec(t_fn); t_venv = load_vn(t_venv, t_vn_path)
            tracker_model.set_env(t_venv)
            cbs, t_cb = make_callbacks("tracker", "tracker", rnd+1)
            try:
                tracker_model.learn(ADV_STEPS_PER_ROUND, callback=cbs,
                                    reset_num_timesteps=True, progress_bar=True)
            finally:
                save_vn(t_venv, t_vn_path); t_venv.close()
            tracker_steps += ADV_STEPS_PER_ROUND
            tracker_model.save(t_model_path)
            shutil.copy(t_vn_path,
                os.path.join(CHECKPOINT_DIR, f"tracker_vecnorm_round_{rnd+1:02d}.pkl"))
            print(f"  [tracker] saved ({tracker_steps:,} steps)")
            if _STOP: break
 
            # ---- evader phase ----
            print(f"\n  [evader] {ADV_STEPS_PER_ROUND:,} steps")
            tr_vn = None
            if os.path.exists(t_vn_path):
                tr_vn = VecNormalize.load(t_vn_path,
                    DummyVecEnv([lambda: TrackerEnv(evader_model=None)]))
                tr_vn.training=False; tr_vn.norm_reward=False
            def e_fn():
                e = EvaderEnv(tracker_model=tracker_model, tracker_vecnorm=tr_vn)
                e.set_curriculum(use_obs, espeed); return Monitor(e)
            e_venv = make_vec(e_fn); e_venv = load_vn(e_venv, e_vn_path)
            evader_model.set_env(e_venv)
            cbs, e_cb = make_callbacks("evader", "evader", rnd+1)
            try:
                evader_model.learn(ADV_STEPS_PER_ROUND, callback=cbs,
                                   reset_num_timesteps=True, progress_bar=True)
            finally:
                save_vn(e_venv, e_vn_path); e_venv.close()
            evader_steps += ADV_STEPS_PER_ROUND
            evader_model.save(e_model_path)
            shutil.copy(e_vn_path,
                os.path.join(CHECKPOINT_DIR, f"evader_vecnorm_round_{rnd+1:02d}.pkl"))
            print(f"  [evader] saved ({evader_steps:,} steps)")
 
        if not _STOP:
            print("\ndone -- all rounds complete")
            tracker_model.save(t_model_path); evader_model.save(e_model_path)
            clear_state()
 
    finally:
        if _STOP:
            if t_cb: t_cb.flush()
            if e_cb: e_cb.flush()
            tracker_model.save(t_model_path); evader_model.save(e_model_path)
            save_state({"start_round":rnd, "tracker_steps":tracker_steps,
                        "evader_steps":evader_steps})
 
 
def demo(episodes=5):
    tp  = os.path.join(SAVE_DIR, "tracker.zip")
    ep_ = os.path.join(SAVE_DIR, "evader.zip")
    tvn = os.path.join(SAVE_DIR, "tracker_vecnorm.pkl")
    evn = os.path.join(SAVE_DIR, "evader_vecnorm.pkl")
    if not os.path.exists(tp) or not os.path.exists(ep_):
        print(f"[demo] no trained models in {SAVE_DIR}, train first"); return
 
    tracker = PPO.load(tp, device='cpu')
    evader  = PPO.load(ep_, device='cpu')
 
    t_norm=None; e_norm=None
    if os.path.exists(tvn):
        t_norm = VecNormalize.load(tvn,
            DummyVecEnv([lambda: TrackerEnv(evader_model=None)]))
        t_norm.training=False; t_norm.norm_reward=False
    if os.path.exists(evn):
        e_norm = VecNormalize.load(evn,
            DummyVecEnv([lambda: EvaderEnv(tracker_model=None)]))
        e_norm.training=False; e_norm.norm_reward=False
 
    env = TrackerEnv(evader_model=evader, evader_vecnorm=e_norm,
                     render=True, use_fov=True)
    env.set_curriculum(use_obstacle=True, evader_speed_mult=1.10)
 
    for ep_i in range(episodes):
        obs,_ = env.reset()
        done=False; steps=0; info={}
        while not done:
            o = obs
            if t_norm is not None:
                o = t_norm.normalize_obs(obs.reshape(1,-1)).reshape(-1).astype(np.float32)
            act,_ = tracker.predict(o, deterministic=True)
            obs, r, done, _, info = env.step(act)
            steps += 1
            time.sleep(DT)
        result = "TAGGED" if info.get('distance',99) <= CAPTURE_DIST else "ESCAPED"
        print(f"  demo ep {ep_i+1}: {result}  steps={steps}  dist={info.get('distance',0):.2f}")
        time.sleep(0.6)
    env.close()
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()
    if args.demo: demo(args.episodes)
    else: train()
