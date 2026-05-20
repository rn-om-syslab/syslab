
import os, sys, time, argparse, threading, queue
import numpy as np
import cv2, av
import gymnasium as gym
from gymnasium import spaces

try:
    import msvcrt
    HAVE_MSVCRT=True
except ImportError:
    HAVE_MSVCRT=False

# pyav changed exception names between versions
try: AVErr = av.FFmpegError
except AttributeError:
    try: AVErr = av.AVError
    except AttributeError: AVErr = Exception

try:
    from djitellopy import Tello
    HAVE_TELLO = True
except ImportError: HAVE_TELLO=False
try:
    from pupil_apriltags import Detector
    HAVE_APRILTAG=True
except ImportError: HAVE_APRILTAG = False

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# grab SAVE_DIR + DT from sim module, else fallback
SAVE_DIR_FALLBACK = os.path.join(os.path.expanduser("~"), "OneDrive","Documents",
    "School","Senior","Syslab","airsim_rl","Scripts","save_pybullet_v3")
DT_FALLBACK = 0.05
SAVE_DIR, DT = None, None
for modname in ('diagnose_connection','pursuit_evasion_pybullet'):
    try:
        m = __import__(modname)
        SAVE_DIR = getattr(m,'SAVE_DIR', SAVE_DIR_FALLBACK)
        DT = getattr(m,'DT', DT_FALLBACK)
        print(f"[init] sim config from {modname}"); break
    except ImportError: continue
if SAVE_DIR is None:
    SAVE_DIR, DT = SAVE_DIR_FALLBACK, DT_FALLBACK
    print("[warn] sim module not found, using fallback")

# defaults / consts
DEFAULT_MODEL_FILE = "tracker.zip"
DEFAULT_VN_FILE = "tracker_vecnorm.pkl"
CMD_SCALE=50

# tello camera (measured)
TELLO_FX,TELLO_FY = 921.0, 921.0
TELLO_CX,TELLO_CY = 480.0, 360.0
# webcam fallback for dry-run
WEBCAM_FX,WEBCAM_FY = 600.0,600.0
WEBCAM_CX,WEBCAM_CY = 320.0,240.0

TAG_SIZE_M = 0.10
TAG_FAMILY = 'tag36h11'

HOVER_DIST = 0.3
HOVER_DIST_EXIT = 0.5
# HOVER_DIST=0.5  # tried bigger zone, drone overshoots
LOST_TAG_HOVER_FRAMES=12
MIN_FLIGHT_BATTERY=30
LOW_BATTERY_LAND = 20

YAW_GAIN_PER_RADIAN=90.0
YAW_DEADBAND_RAD = 0.04
MAX_YAW_CMD=80

DETECT_SCALE        = 0.5    # 0.5 = half-res for speed
ACTION_SMOOTH_ALPHA = 0.15   # low=jitter, high=lag, hand-tuned
USE_TARGET_PREDICTION=True
PREDICT_MAX_DT=0.5

TELLO_VIDEO_URL='udp://0.0.0.0:11111'
DECODE_THREADS = 4

RECORD_FPS = 20
RECORD_QUEUE_MAX=32
# RECORD_FPS = 30   # webcam handles this, tello caps lower

# dummy env so VecNormalize.load has something to wrap
class MockEnv(gym.Env):
    def __init__(self, obs_dim):
        super().__init__()
        self.observation_space=spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space=spaces.Box(-1.0, 1.0, (3,), np.float32)
    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}
    def step(self, a):
        return np.zeros(self.observation_space.shape, np.float32), 0.0, False, False, {}
    def render(self): pass
    def close(self): pass

# low-latency pyav reader. always exposes the newest frame
class TelloVideo:
    def __init__(self, url=TELLO_VIDEO_URL, store_frame=True):
        self.url = url
        self.store_frame=store_frame
        self._frame=None; self._frame_id=0
        self._lock=threading.Lock()
        self._stop=threading.Event()
        self._thread=None
        self._fps_t0=time.time(); self._fps_n=0; self._fps=0.0
        self.dropped=0

    def start(self, timeout=10.0):
        # low-latency opts
        opts = {'fflags':'nobuffer','flags':'low_delay','probesize':'32',
                'analyzeduration':'0','sync':'ext','rtbufsize':'50M',
                'reorder_queue_size':'0'}
        try:
            self._container = av.open(self.url, mode='r', options=opts, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"open stream failed: {e}")
        # multi-thread decode
        try:
            s = self._container.streams.video[0]
            s.thread_type='AUTO'
            s.codec_context.thread_type='AUTO'
            s.codec_context.thread_count = DECODE_THREADS
        except Exception as e:
            print(f"[warn] thread_type=AUTO: {e}")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # wait for first frame
        t0=time.time()
        while time.time()-t0 < timeout:
            with self._lock:
                if self._frame_id > 0: return
            time.sleep(0.05)
        raise RuntimeError("no video frame within timeout")

    def stop(self):
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout=2.0)
        try: self._container.close()
        except Exception: pass

    @property
    def frame(self):
        with self._lock: return self._frame
    @property
    def frame_id(self):
        with self._lock: return self._frame_id
    @property
    def fps(self): return self._fps

    def _loop(self):
        try:
            for packet in self._container.demux(video=0):
                if self._stop.is_set(): break
                try:
                    for f in packet.decode():
                        img = f.to_ndarray(format='bgr24')
                        with self._lock:
                            if self.store_frame: self._frame = img
                            self._frame_id += 1
                        self._fps_n += 1
                        if time.time()-self._fps_t0 > 1.0:
                            self._fps = self._fps_n/(time.time()-self._fps_t0)
                            self._fps_n=0; self._fps_t0=time.time()
                except AVErr:
                    self.dropped += 1; continue
                except Exception as e:
                    self.dropped += 1
                    if self.dropped <= 3:
                        print(f"[video] decode err: {type(e).__name__}: {e}")
                    continue
        except Exception as e:
            print(f"[video] loop ended: {e}")

# apriltag detect w/ optional downscaling
class TagTracker:
    def __init__(self, fx, fy, cx, cy):
        if not HAVE_APRILTAG:
            raise RuntimeError("pupil-apriltags not installed")
        self.detector = Detector(families=TAG_FAMILY, nthreads=2)
        # scale intrinsics to match downscale
        self.fx,self.fy = fx*DETECT_SCALE, fy*DETECT_SCALE
        self.cx,self.cy = cx*DETECT_SCALE, cy*DETECT_SCALE

    def detect(self, frame_bgr):
        if DETECT_SCALE != 1.0:
            small = cv2.resize(frame_bgr, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray, estimate_tag_pose=True,
            camera_params=[self.fx,self.fy,self.cx,self.cy], tag_size=TAG_SIZE_M)
        if not tags: return None, False, None
        # closest tag wins if multiple
        tag = min(tags, key=lambda t: float(np.linalg.norm(t.pose_t)))
        tc = tag.pose_t.flatten()
        # cam frame -> body frame
        rel_body = np.array([tc[2], -tc[0], -tc[1]], dtype=np.float32)
        if DETECT_SCALE != 1.0:
            tag.corners = tag.corners / DETECT_SCALE
        return rel_body, True, tag

# detection on its own thread  keeps video latency out of control loop
class VisionThread(threading.Thread):
    def __init__(self, video_reader, tag_tracker, keep_frame=False):
        super().__init__(daemon=True)
        self.video=video_reader
        self.tags=tag_tracker
        self.keep_frame=keep_frame
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._latest = {'frame':None,'rel':None,'visible':False,'tag':None,'ts':0.0}
        self._last_fid = -1
        self._fps_n=0; self._fps_t0=time.time(); self._fps=0.0

    def run(self):
        while not self._stop_evt.is_set():
            fid = self.video.frame_id
            if fid == self._last_fid:
                time.sleep(0.005); continue
            self._last_fid = fid
            frame = self.video.frame
            if frame is None:
                time.sleep(0.005); continue
            try:
                rel, vis, tag = self.tags.detect(frame)
            except Exception:
                rel, vis, tag = None, False, None
            with self._lock:
                self._latest = {
                    'frame':   frame if self.keep_frame else None,
                    'rel':     rel,
                    'visible': vis,
                    'tag':     tag if self.keep_frame else None,
                    'ts':      time.time(),
                }
            self._fps_n += 1
            if time.time() - self._fps_t0 > 1.0:
                self._fps = self._fps_n / (time.time() - self._fps_t0)
                self._fps_n=0; self._fps_t0=time.time()

    def stop(self): self._stop_evt.set()
    def get_latest(self):
        with self._lock: return dict(self._latest)
    @property
    def fps(self): return self._fps

# builds the obs vec. supports 3-dim or 9-dim mode (difference in sim vers.)
# between vision frames, predicts forward using smoothed evader vel.
class ObsBuilder:
    VEL_SMOOTH_ALPHA = 0.6
    def __init__(self, obs_dim):
        if obs_dim not in (3, 9):
            raise ValueError(f"obs_dim {obs_dim}?")
        self.obs_dim = obs_dim
        self.last_tag_pos = None
        self.last_tag_time = None
        self.smoothed_ev_vel = np.zeros(3, dtype=np.float32)
        self.lost_frames = 0

    def update_visibility(self, visible):
        self.lost_frames = 0 if visible else self.lost_frames + 1

    def build_obs(self, tag_rel_body, tag_visible, own_vel,
                  use_prediction=True, detection_age=0.0):
        now = time.time()
        if tag_visible and tag_rel_body is not None:
            new_rel = np.array(tag_rel_body, dtype=np.float32)
            # velocity from finite diff
            if self.last_tag_pos is not None and self.last_tag_time is not None:
                dt = max(now-self.last_tag_time, 1e-3)
                raw_vel = (new_rel-self.last_tag_pos)/dt + own_vel
                raw_vel = np.clip(raw_vel, -6.0, 6.0)
                self.smoothed_ev_vel = (self.VEL_SMOOTH_ALPHA*self.smoothed_ev_vel
                                      + (1-self.VEL_SMOOTH_ALPHA)*raw_vel)
            self.last_tag_pos = new_rel.copy()
            self.last_tag_time = now
            rel_used = new_rel
        else:
            # no detection -- predict forward from last
            if self.last_tag_pos is None or self.lost_frames >= LOST_TAG_HOVER_FRAMES:
                rel_used = np.zeros(3, dtype=np.float32)
            elif use_prediction and detection_age < PREDICT_MAX_DT:
                rel_vel = self.smoothed_ev_vel - own_vel
                rel_used = self.last_tag_pos + rel_vel*detection_age
            else:
                rel_used = self.last_tag_pos.copy()

        dist = float(np.linalg.norm(rel_used)) if self.last_tag_pos is not None else 99.0
        if self.obs_dim == 3:
            obs = np.array(rel_used, dtype=np.float32)
        else:
            obs = np.concatenate([rel_used, own_vel, self.smoothed_ev_vel]).astype(np.float32)
        return obs, dist, self.lost_frames >= LOST_TAG_HOVER_FRAMES, rel_used

def compute_yaw_cmd(rel_body):
    fwd, left = rel_body[0], rel_body[1]
    if abs(fwd) < 0.05 and abs(left) < 0.05: return 0
    bearing = np.arctan2(left, fwd)
    if abs(bearing) < YAW_DEADBAND_RAD: return 0
    return int(np.clip(-YAW_GAIN_PER_RADIAN*bearing, -MAX_YAW_CMD, MAX_YAW_CMD))

def vn_stats_shape(vn):
    try: return vn.obs_rms.mean.shape
    except AttributeError: return None

def load_model(model_path, vn_path):
    if not os.path.exists(model_path):
        raise RuntimeError(f"no model at {model_path}")
    model = PPO.load(model_path, device='cpu')
    expected = model.observation_space.shape
    if len(expected) != 1:
        raise RuntimeError(f"weird obs space {expected}")
    obs_dim = expected[0]
    if obs_dim not in (3,9):
        raise RuntimeError(f"obs_dim={obs_dim}? need 3 or 9")
    print(f"[init] model wants obs ({obs_dim},)")

    vn = None
    if os.path.exists(vn_path):
        try:
            mock_venv = DummyVecEnv([lambda: MockEnv(obs_dim)])
            cand = VecNormalize.load(vn_path, mock_venv)
            shp = vn_stats_shape(cand)
            if shp == (obs_dim,):
                cand.training=False; cand.norm_reward=False
                vn = cand
                print(f"[init] loaded vecnorm {shp}")
            else:
                print(f"[warn] vecnorm shape {shp} != model ({obs_dim},), skipping")
        except Exception as e:
            print(f"[warn] vecnorm load: {type(e).__name__}: {e}")
    else:
        print(f"[init] no vecnorm at {vn_path}")
    return model, vn, obs_dim

def normalize_obs_safe(obs, vn):
    if vn is None: return obs
    try:
        if vn_stats_shape(vn) != obs.shape: return obs
        return vn.normalize_obs(obs.reshape(1,-1)).reshape(-1).astype(np.float32)
    except Exception as e:
        print(f"[warn] normalize_obs error: {e}")
        return obs

def safe_velocity(tello):
    try:
        return np.array([tello.get_speed_x()/100.0, tello.get_speed_y()/100.0,
                         tello.get_speed_z()/100.0], dtype=np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)

def safe_battery(tello):
    try: return tello.get_battery()
    except Exception: return None

def poll_key_headless():
    if not HAVE_MSVCRT: return None
    if msvcrt.kbhit():
        try: return msvcrt.getch().decode('utf-8', errors='ignore').lower()
        except Exception: return None
    return None

# HUD
def draw_hud(frame, state):
    out = frame.copy()
    h, w = out.shape[:2]

    # tag outline + center dot
    tag = state.get('tag')
    if tag is not None:
        corners = tag.corners.astype(int)
        for i in range(4):
            cv2.line(out, tuple(corners[i]), tuple(corners[(i+1)%4]), (0,255,0), 2)
        c = tuple(np.mean(corners, axis=0).astype(int))
        cv2.circle(out, c, 4, (0,255,0), -1)

    # screen center crosshair
    cx, cy = w//2, h//2
    cv2.line(out, (cx-10,cy), (cx+10,cy), (180,180,180), 1)
    cv2.line(out, (cx,cy-10),(cx,cy+10), (180,180,180), 1)

    # estimated evader velocity arr
    if state.get('visible'):
        ev = state.get('ev_vel', np.zeros(3))
        if np.linalg.norm(ev[:2]) > 0.1 and tag is not None:
            c = tuple(np.mean(tag.corners.astype(int), axis=0))
            end = (int(c[0] - ev[1]*40), int(c[1] - ev[0]*40))
            cv2.arrowedLine(out, c, end, (0,200,255), 2, tipLength=0.25)

    # commanded velocity arrow from center
    cmd = state.get('cmd', (0,0,0,0))
    fb, lr, ud, yaw = cmd
    end = (cx + int(lr*1.2), cy - int(fb*1.2))
    cv2.arrowedLine(out, (cx,cy), end, (255,120,0), 2, tipLength=0.25)
    visible = state.get('visible', False)
    dist = state.get('dist', None)
    rel = state.get('rel', None)
    age_ms = state.get('age_ms', 0)
    lines=[]
    if visible:
        lines.append(f"TAG  dist {dist:.2f} m")
        if rel is not None:
            lines.append(f"rel  [{rel[0]:+.2f} {rel[1]:+.2f} {rel[2]:+.2f}]")
        lines.append(f"age  {age_ms:.0f} ms")
    else:
        lines.append("NO TAG")
        lines.append(f"lost {state.get('lost', 0)} frames")
    lines.append(f"cmd  fb {fb:+d}  lr {lr:+d}  ud {ud:+d}  yaw {yaw:+d}")
    bat = state.get('battery')
    if bat is not None: lines.append(f"bat  {bat}%")
    ctrl_hz = state.get('ctrl_hz', 0.0)
    vid_fps = state.get('vid_fps', 0.0)
    lines.append(f"ctrl {ctrl_hz:.0f} Hz  vid {vid_fps:.0f} fps")

    y = 24
    for line in lines:
        cv2.putText(out, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        y += 22
    mode = state.get('mode')
    if mode:
        cv2.putText(out, mode, (w-180,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, mode, (w-180,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,255), 2, cv2.LINE_AA)
    return out

class HUDRecorder:
    def __init__(self, path, fps=RECORD_FPS):
        root, ext = os.path.splitext(path)
        if ext.lower() not in ('.avi','.mp4','.mov'):
            path = root + '.avi'
        self.path = path
        self.fps = fps
        self._q = queue.Queue(maxsize=RECORD_QUEUE_MAX)
        self._stop = threading.Event()
        self._thread = None
        self._writer = None
        self._size = None
        self._first_ts=None; self._last_ts=None
        self.dropped=0; self.written=0

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[rec] -> {os.path.abspath(self.path)} @ {self.fps} fps")

    def submit(self, frame, state):
        if frame is None: return
        try:
            self._q.put_nowait((time.time(), frame, state))
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait((time.time(), frame, state))
                self.dropped += 1
            except queue.Empty:
                pass

    def stop(self):
        t_drain = time.time()
        while not self._q.empty() and time.time()-t_drain < 4.0:
            time.sleep(0.05)
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout=6.0)
        if self._writer is not None:
            self._writer.release(); self._writer=None
        if not os.path.exists(self.path):
            print(f"[rec] file missing at {self.path}"); return
        mb = os.path.getsize(self.path) / 1024 /1024
        print(f"[rec] done. {self.written} frames, dropped {self.dropped}")
        print(f"[rec] file: {os.path.abspath(self.path)}  ({mb:.1f} MB)")
        if self.written >= 2 and self._first_ts and self._last_ts:
            real_s = self._last_ts - self._first_ts
            real_fps = self.written/real_s if real_s > 0 else self.fps
            playback_s = self.written / float(self.fps)
            print(f"[rec] real time: {real_s:.1f}s  playback: {playback_s:.1f}s (writer fps={self.fps})")
            print(f"[rec] measured: {real_fps:.1f} fps")
            if abs(real_fps - self.fps) > 1.0:
                root, ext = os.path.splitext(self.path)
                fixed = f"{root}_realtime{ext}"
                print(f"[rec] playback speed is off, run:")
                print(f"[rec]   ffmpeg -y -r {real_fps:.2f} -i \"{self.path}\" -c copy \"{fixed}\"")

    def _open_writer(self, frame):
        h, w = frame.shape[:2]
        self._size = (w, h)
        ext = os.path.splitext(self.path)[1].lower()
        if   ext == '.avi':           codecs = ['MJPG','XVID']
        elif ext in ('.mp4','.mov'):  codecs = ['mp4v','avc1']
        else:                         codecs = ['MJPG']
        for c in codecs:
            fourcc = cv2.VideoWriter_fourcc(*c)
            w_obj = cv2.VideoWriter(self.path, fourcc, self.fps, self._size)
            if w_obj.isOpened():
                self._writer = w_obj
                print(f"[rec] writer: codec={c} size={self._size} fps={self.fps}")
                return
            w_obj.release()
        print(f"[rec] no codec opened for {self.path}")

    def _loop(self):
        last_status = time.time()
        while not self._stop.is_set() or not self._q.empty():
            try:
                ts, frame, state = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                hud = draw_hud(frame, state)
                if self._writer is None: self._open_writer(hud)
                if self._writer is None:
                    self.dropped += 1; continue
                if (hud.shape[1], hud.shape[0]) != self._size:
                    hud = cv2.resize(hud, self._size)
                self._writer.write(hud)
                self.written += 1
                if self._first_ts is None: self._first_ts = ts
                self._last_ts = ts
                # live status so you can see writes happening
                if time.time() - last_status > 1.0:
                    sz_mb = (os.path.getsize(self.path)/1024/1024
                             if os.path.exists(self.path) else 0)
                    elapsed = (ts - self._first_ts) if self._first_ts else 0
                    inst = self.written/elapsed if elapsed > 0 else 0
                    print(f"[rec] {self.written:5d} frames | {sz_mb:5.1f} MB | "
                          f"{elapsed:5.1f}s | {inst:4.1f} fps | "
                          f"q={self._q.qsize()}/{RECORD_QUEUE_MAX} drop={self.dropped}")
                    last_status = time.time()
            except Exception as e:
                self.dropped += 1
                if self.dropped <= 5:
                    print(f"[rec] frame err #{self.dropped}: {type(e).__name__}: {e}")

# webcam dry-run (no tello)
def dry_run(args):
    print("="*60); print("DRY RUN  webcam"); print("="*60)
    model_path = os.path.join(SAVE_DIR, args.model)
    vn_path = os.path.join(SAVE_DIR, args.vecnorm)
    model, vn, obs_dim = load_model(model_path, vn_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[err] no webcam"); return

    tags = TagTracker(WEBCAM_FX, WEBCAM_FY, WEBCAM_CX, WEBCAM_CY)
    ob = ObsBuilder(obs_dim)
    rec = HUDRecorder(args.record) if args.record else None
    if rec: rec.start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            rel, vis, tag = tags.detect(frame)
            ob.update_visibility(vis)
            own_vel = np.zeros(3, dtype=np.float32)
            obs, dist, hover, rel_used = ob.build_obs(rel, vis, own_vel,
                use_prediction=False, detection_age=0.0)
            obs_n = normalize_obs_safe(obs, vn)
            action,_ = model.predict(obs_n, deterministic=True)
            action = np.clip(action, -1.0, 1.0)
            fb  = int(action[0]*args.cmd_scale)
            lr  = int(-action[1]*args.cmd_scale)
            ud  = int(action[2]*args.cmd_scale)
            yaw = compute_yaw_cmd(rel) if (vis and not args.no_yaw) else 0
            state = {'visible':vis, 'tag':tag, 'rel':rel_used,
                     'dist':dist if vis else None,
                     'ev_vel':ob.smoothed_ev_vel,
                     'cmd':(fb,lr,ud,yaw),
                     'age_ms':0, 'lost':ob.lost_frames, 'mode':'DRY RUN'}
            disp = draw_hud(frame, state)
            cv2.imshow("Dry Run", disp)
            if rec: rec.submit(frame, state)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        if rec: rec.stop()
        cap.release()
        cv2.destroyAllWindows()

def fly(args):
    if not HAVE_TELLO:
        print("[err] djitellopy not installed"); return

    motors_off = args.motors_off
    show_video = args.display
    recording = bool(args.record)

    print("="*60)
    print("MOTORS-OFF" if motors_off else "FLIGHT MODE")
    print(f"hover={args.hover_dist}  yaw={'OFF' if args.no_yaw else 'ON'}  "
          f"scale={args.cmd_scale}  display={'ON' if show_video else 'OFF'}  "
          f"record={'ON' if recording else 'OFF'}")
    if not show_video:
        print("[hint] q=land  space=hover  esc=emergency")
    print("="*60)

    model_path = os.path.join(SAVE_DIR, args.model)
    vn_path = os.path.join(SAVE_DIR, args.vecnorm)
    model, vn, obs_dim = load_model(model_path, vn_path)

    print("[init] connecting...")
    tello = Tello()
    try:
        tello.connect()
    except Exception as e:
        # try once more w/o auto-wait
        print(f"[warn] {e}, retry")
        try: tello.connect(False)
        except Exception as e2:
            print(f"[err] {e2}"); return

    bat = safe_battery(tello)
    if bat is not None:
        print(f"[init] battery {bat}%")
        if not motors_off and bat < MIN_FLIGHT_BATTERY:
            print("[abort] battery low"); return

    print("[init] streamon")
    try: tello.streamon()
    except Exception as e: print(f"[warn] streamon: {e}")
    time.sleep(1.5)

    print("[init] opening video")
    video = TelloVideo()
    try:
        video.start(timeout=12.0)
        print("[init] video up")
    except Exception as e:
        print(f"[err] video init: {e}")
        try: tello.streamoff()
        except Exception: pass
        return

    tags = TagTracker(TELLO_FX, TELLO_FY, TELLO_CX, TELLO_CY)
    keep_frame = show_video or recording
    vision = VisionThread(video, tags, keep_frame=keep_frame)
    vision.start()
    time.sleep(0.5)

    ob = ObsBuilder(obs_dim)
    rec = HUDRecorder(args.record) if recording else None
    if rec: rec.start()

    if not motors_off:
        print("[init] takeoff")
        try: tello.takeoff()
        except Exception as e:
            print(f"[err] takeoff: {e}")
            vision.stop(); video.stop()
            if rec: rec.stop()
            try: tello.streamoff()
            except Exception: pass
            return
        time.sleep(2.5)
        try:
            tello.move_up(40); time.sleep(1.5)
        except Exception as e:
            print(f"[warn] move_up: {e}")

    print("[ready]  q=land  space=hover  esc=emergency")

    paused=False
    last_bat_check = time.time()
    last_log=0; last_disp=0
    DISP_INTERVAL = 0.1
    smooth_act = np.zeros(3, dtype=np.float32)
    loop_n=0; perf_n=0; perf_t0=time.time(); actual_hz=0.0
    in_hover_zone=False
    cur_bat = bat

    try:
        while True:
            t0 = time.time()
            loop_n += 1
            perf_n += 1
            if time.time() - perf_t0 > 1.0:
                actual_hz = perf_n /(time.time() - perf_t0)
                perf_n=0; perf_t0=time.time()
            det = vision.get_latest()
            visible = det['visible']
            rel_body = det['rel'] if visible else None
            det_age = time.time() - det['ts'] if det['ts'] > 0 else 99.0

            ob.update_visibility(visible)
            own_vel = safe_velocity(tello)
            obs, dist, force_hover, rel_used = ob.build_obs(rel_body, visible, own_vel,
                use_prediction=USE_TARGET_PREDICTION, detection_age=det_age)

            obs_n = normalize_obs_safe(obs, vn)
            action,_ = model.predict(obs_n, deterministic=True)
            action = np.clip(action, -1.0, 1.0)
            smooth_act = ACTION_SMOOTH_ALPHA*smooth_act + (1-ACTION_SMOOTH_ALPHA)*action

            # hover hysteresis once you're close, stay close until target moves away
            have_pos = ob.last_tag_pos is not None
            if have_pos:
                if in_hover_zone:
                    if dist > HOVER_DIST_EXIT: in_hover_zone = False
                else:
                    if dist < args.hover_dist: in_hover_zone = True

            why = None
            if paused:                         why = "paused"
            elif force_hover:       why = "tag lost"
            elif det_age > 0.5:                why = f"stale {det_age:.1f}s"
            elif have_pos and in_hover_zone:   why = f"close ({dist:.2f}m)"

            yaw_input = rel_used if have_pos else None

            if why and "close" not in why:
                fb=lr=ud=yaw=0
                # search: rotate toward last bearing, vertical sweep after a bit to find aprilg"
                if why == "tag lost" and ob.last_tag_pos is not None:
                    last_y = float(ob.last_tag_pos[1])
                    yaw = 40 if last_y < 0 else -40
                    if ob.lost_frames > 25:
                        ud = 15 if (ob.lost_frames // 30) % 2 == 0 else -15
            elif why and "close" in why:
                fb=lr=ud=0
                yaw = compute_yaw_cmd(yaw_input) if (have_pos and not args.no_yaw) else 0
            else:
                fb  = int(smooth_act[0]*args.cmd_scale)
                lr  = int(-smooth_act[1]*args.cmd_scale)
                ud  = int(smooth_act[2]*args.cmd_scale)
                yaw = compute_yaw_cmd(yaw_input) if (have_pos and not args.no_yaw) else 0

            # send rc
            if not motors_off and not paused:
                try: tello.send_rc_control(lr, fb, ud, yaw)
                except Exception as e: print(f"[warn] rc: {e}")

            mode = ("PAUSED" if paused else "MOTORS OFF" if motors_off
                    else why.upper() if why else "TRACKING")
            state = {
                'visible': visible,
                'tag':   det['tag'] if (show_video or recording) else None,
                'rel':     rel_used if have_pos else None,
                'dist':  dist if have_pos else None,
                'ev_vel':  ob.smoothed_ev_vel,
                'cmd':   (fb, lr, ud, yaw),
                'age_ms':  det_age*1000.0,
                'lost':    ob.lost_frames,
                'battery': cur_bat,
                'ctrl_hz': actual_hz,
                'vid_fps': video.fps,
                'mode':    mode,
            }

            if rec: rec.submit(det['frame'], state)

            # per-second log
            if time.time() - last_log > 1.0:
                if have_pos:
                    ru = rel_used if rel_used is not None else np.zeros(3)
                    print(f"  [{loop_n}] dist={dist:.2f} "
                          f"rel=[{ru[0]:+.2f},{ru[1]:+.2f},{ru[2]:+.2f}] "
                          f"ev=[{ob.smoothed_ev_vel[0]:+.1f},{ob.smoothed_ev_vel[1]:+.1f},"
                          f"{ob.smoothed_ev_vel[2]:+.1f}] "
                          f"cmd=[{fb:+3d},{lr:+3d},{ud:+3d},y={yaw:+3d}] "
                          f"vid={video.fps:.0f} vis={vision.fps:.0f} "
                          f"ctrl={actual_hz:.0f}Hz hover={in_hover_zone}"
                          + (f" rec={rec.written}f drop={rec.dropped}" if rec else ""))
                else:
                    print(f"  [{loop_n}] no_tag lost={ob.lost_frames} "
                          f"vid={video.fps:.0f} vis={vision.fps:.0f}")
                last_log = time.time()

      
            if show_video and time.time() - last_disp> DISP_INTERVAL:
                last_disp = time.time()
                if det['frame'] is not None:
                    cv2.imshow("Tello Tracker", draw_hud(det['frame'], state))
                key = cv2.waitKey(1) & 0xFF
            else:
                k = poll_key_headless()
                key = ord(k) if k else 0

            if key == ord('q'):
                print("[in] q -> land"); break
            if key == ord(' '):
                paused = not paused
                if not motors_off:
                    try: tello.send_rc_control(0,0,0,0)
                    except Exception: pass
                print(f"[in] {'PAUSED' if paused else 'RESUMED'}")
            if key == 27:
                if not motors_off:
                    try: tello.emergency()
                    except Exception: pass
                print("[in] esc -> emergency"); break

            if time.time() - last_bat_check >5.0:
                bat = safe_battery(tello)
                if bat is not None:
                    cur_bat = bat
                    if bat < LOW_BATTERY_LAND:
                        print(f"[warn] battery low bro turn off({bat}%), landing"); break
                last_bat_check = time.time()

            elapsed = time.time() -t0
            if elapsed < DT: time.sleep(D - elapsed)

    except KeyboardInterrupt:
        print("\n[interrupt]")
    finally:
        vision.stop()
        video.stop()
        if rec: rec.stop()
        if not motors_off:
            try:#tryyyyy
                tello.send_rc_control(0,0,0,0)
                time.sleep(0.5)
                tello.land()
            except Exception as e:
                print(f"[finally] land: {e}")
        try: tello.streamoff()
        except Exception: pass
        if show_video: cv2.destroyAllWindows()
        print("[done]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--motors-off", action="store_true")
    ap.add_argument("--no-yaw",  action="store_true")
    ap.add_argument("--display",  action="store_true",
                    help="show windowdefault off)")
    ap.add_argument("--record",     default=None, metavar="FILE",
                    help="write annotated video.avi recommended)")
    ap.add_argument("--hover-dist", type=float, default=HOVER_DIST)
    ap.add_argument("--cmd-scale", type=int,   default=CMD_SCALE)
    ap.add_argument("--model",      default=DEFAULT_MODEL_FILE)
    ap.add_argument("--vecnorm",   default=DEFAULT_VN_FILE)
    args = ap.parse_args()

    if not HAVE_APRILTAG:
        print("[err] pupil-apriltags not installed"); return

    if args.dry_run: dry_run(args)
    else: fly(args)

if __name__ == "__main__":
    main()
