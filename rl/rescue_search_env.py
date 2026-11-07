"""Task-level 2-D rescue search. NOT a quadruped dynamics or life detector.

Maps are configuration-space grids (0.4 m cells). Compass and grid odometry
are ideal. Policy sees ray observations, observed-map frontier guidance and
line-of-sight synthetic cues, never hidden victim coordinates. Ground truth
is used only for reward, termination and post-hoc evaluation.
"""
from collections import deque
import hashlib
import gymnasium as gym
import numpy as np

SCHEMA = 'rescue-observed-search-v1'
SCENARIOS = ('corridor', 'rubble', 'dropout')
DIRECTIONS = np.array([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)], int)
CUE_RANGE = 3
CUE_RAY_FRACTIONS = np.linspace(0., 1., 12)[:, None]


class RescueSearchEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, scenario='rubble', map_seeds=range(100,132), seed=0, max_steps=256,
                 *, rubble_count=19, corridor_gap=3):
        super().__init__()
        if scenario not in SCENARIOS:
            raise ValueError('unknown scenario')
        self.scenario = scenario
        self.map_seeds = tuple(map(int, map_seeds))
        if not self.map_seeds or max_steps < 1:
            raise ValueError('map seeds and positive horizon required')
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        if rubble_count < 0 or not 2 <= corridor_gap <= 5:
            raise ValueError('invalid obstacle count or corridor gap')
        self.rubble_count = rubble_count
        self.corridor_gap = corridor_gap
        self.size = 20
        self.action_space = gym.spaces.Discrete(9)  # 8 compass moves or stationary scan
        self.observation_space = gym.spaces.Box(-1., 1., (39,), np.float32)
        self._cache = {}

    def _neighbors(self, p, free):
        for i, d in enumerate(DIRECTIONS):
            q = p + d
            if not (0 <= q[0] < self.size and 0 <= q[1] < self.size) or not free[tuple(q)]:
                continue
            if d[0] and d[1] and (not free[p[0]+d[0],p[1]] or not free[p[0],p[1]+d[1]]):
                continue
            yield i, q

    def _make_map(self, seed):
        key = (self.scenario,seed)
        if key in self._cache:
            return self._cache[key]
        rng = np.random.default_rng(seed)
        blocked = np.zeros((self.size,self.size), bool)
        blocked[[0,-1],:] = True
        blocked[:,[0,-1]] = True
        if self.scenario == 'corridor':
            for x in (6,12):
                gap = int(rng.integers(3,15))
                blocked[x,1:-1] = True
                blocked[x,gap:gap+self.corridor_gap] = False
        else:
            for _ in range(self.rubble_count):
                x,y = rng.integers(3,self.size-2,size=2)
                blocked[x:x+int(rng.integers(1,3)),y:y+int(rng.integers(1,3))] = True
        blocked[1:5,1:5] = False
        free = ~blocked
        visited = {(2,2)}
        queue = deque([np.array((2,2))])
        while queue:
            p = queue.popleft()
            for _,q in self._neighbors(p,free):
                t = tuple(q)
                if t not in visited:
                    visited.add(t)
                    queue.append(q)
        choices = np.array(sorted(p for p in visited if sum(p)>11))
        if len(choices)<4:
            raise RuntimeError('generated map has insufficient reachable area')
        targets = choices[rng.choice(len(choices),size=4,replace=False)]
        # Three synthetic survivor cues and one nuisance cue; not a classifier.
        result = (blocked,targets,np.array([True,True,True,False]))
        self._cache[key] = result
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        self.map_seed = int(options.get('map_seed',self.rng.choice(self.map_seeds)))
        self.blocked,self.targets,self.real = self._make_map(self.map_seed)
        self.pos = self._start_position()
        self.known = np.zeros_like(self.blocked,dtype=np.int8)
        self.visits = np.zeros_like(self.blocked,dtype=np.int32)
        self.visits[tuple(self.pos)] = 1
        self.scanned = np.zeros(len(self.targets),bool)
        self.t = self.collisions = self.interventions = self.motion_violations = 0
        self.candidate_events = []
        self.ranges = np.ones(8,np.float32)
        self.sensor_valid = self.link_valid = True
        self._observe_map()
        return self._obs(),self._info()

    def _start_position(self):
        return np.array([2,2])

    def _observe_map(self):
        self.known[tuple(self.pos)] = 1
        for i,d in enumerate(DIRECTIONS):
            self.ranges[i] = 1.
            for k in range(1,7):
                q = self.pos+d*k
                if np.any(q<0) or np.any(q>=self.size):
                    self.ranges[i] = (k-1)/6
                    break
                self.known[tuple(q)] = -1 if self.blocked[tuple(q)] else 1
                if self.blocked[tuple(q)]:
                    self.ranges[i] = (k-1)/6
                    break

    def _visible(self):
        if not self.sensor_valid:
            return []
        ids = []
        for i,q in enumerate(self.targets):
            delta = q-self.pos
            if self.scanned[i] or np.linalg.norm(delta)>CUE_RANGE:
                continue
            cells = np.rint(self.pos+CUE_RAY_FRACTIONS*delta).astype(int)
            if not self.blocked[cells[:,0],cells[:,1]].any():
                ids.append(i)
        return ids

    def _reachable_routes(self, free, *, start=None):
        start = self.pos if start is None else start
        seen = {tuple(start)}
        queue = deque([(start,-1,0)])
        while queue:
            p,first,distance = queue.popleft()
            yield p,first,distance
            for i,q in self._neighbors(p,free):
                if tuple(q) not in seen:
                    seen.add(tuple(q))
                    queue.append((q,i if first<0 else first,distance+1))

    def _frontier(self):
        # BFS only on observed free space, without hidden occupancy.
        free = self.known==1
        for p,first,distance in self._reachable_routes(free):
            if distance:
                for d in DIRECTIONS[::2]:
                    q = p+d
                    if 0<=q[0]<self.size and 0<=q[1]<self.size and self.known[tuple(q)]==0:
                        return first,distance
        return self._frontier_fallback(free)

    def _frontier_fallback(self, free):
        # Revisit the least visited known free neighbor.
        options = [(self.visits[tuple(q)],i) for i,q in self._neighbors(self.pos,free)]
        return (min(options)[1],1) if options else (-1,0)

    def _obs(self):
        adjacent,visited = [],[]
        for d in DIRECTIONS:
            q = np.clip(self.pos+d,0,self.size-1)
            adjacent.append(self.known[tuple(q)])
            visited.append(min(self.visits[tuple(q)]/5.,1.))
        hint,distance = self._frontier()
        frontier = np.zeros(8)
        if hint>=0:
            frontier[hint] = 1.
        cue = np.zeros(3)
        visible = self._visible()
        if visible:
            i = min(visible,key=lambda i:np.linalg.norm(self.targets[i]-self.pos))
            cue[:2] = (self.targets[i]-self.pos)/3.
            cue[2] = 1.
        return np.asarray([*self.ranges,*adjacent,*visited,*frontier,*cue,
                           float(self.sensor_valid),float(self.link_valid),
                           min(distance/30.,1.),self.t/self.max_steps],np.float32)

    def _scan_visible(self):
        reward = 0.
        for i in self._visible():
            self.scanned[i] = True
            reward += 8. if self.real[i] else .5
            self.candidate_events.append(dict(step=self.t,grid=self.targets[i].tolist(),
                cue='synthetic_proxy',status='awaiting_human_review'))
        return reward

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('invalid rescue action')
        previous_position = self.pos.copy()
        self.t += 1
        self.link_valid, self.sensor_valid = self._sensor_status()
        reward = -.02
        collision = False
        before = np.count_nonzero(self.known)
        hint,_ = self._frontier()
        if not self.link_valid:
            # Remote-control authority expires; automatically hold position.
            self.interventions += int(action < 8)
        elif action<8 and self.sensor_valid:
            q = self.pos+DIRECTIONS[action]
            legal = {i for i,_ in self._neighbors(self.pos,~self.blocked)}
            if action in legal:
                self.pos = q
                reward += .08*int(action==hint)  # observed-frontier shaping
                reward += .05*int(self.visits[tuple(q)]==0)
                reward -= .03*min(self.visits[tuple(q)],4)
                self.visits[tuple(q)] += 1
            else:
                self.collisions += 1
                collision = True
                reward -= 1.
        if self.sensor_valid:
            self._observe_map()
        self.motion_violations += int(not self.link_valid and not np.array_equal(previous_position,self.pos))
        reward += .025*(np.count_nonzero(self.known)-before)
        if action==8 and self.link_valid and self.sensor_valid:
            reward += self._scan_visible()
        success = self._task_success()
        if success:
            reward += 10.
        info = self._info()
        info['collision'] = collision
        return self._obs(),float(reward),success,bool(self.t>=self.max_steps and not success),info

    def _sensor_status(self):
        return (not (self.scenario=='dropout' and 50<=self.t%100<56),
                not (self.scenario=='dropout' and 20<=self.t%70<24))

    def _task_success(self):
        return bool(self.scanned[self.real].all())

    def _info(self):
        return dict(map_seed=self.map_seed,scenario=self.scenario,
                    success=self._task_success(),found=int(self.scanned[self.real].sum()),
                    candidates=int(self.scanned.sum()),collisions=self.collisions,
                    coverage=float(np.mean(self.known!=0)),interventions=self.interventions,
                    unsafe_motion=self.motion_violations,steps=self.t,
                    map_hash=hashlib.sha256(self.blocked.tobytes()).hexdigest())

    def baseline_action(self):
        if not self.link_valid or not self.sensor_valid or self._visible():
            return 8
        hint,_ = self._frontier()
        return hint if hint>=0 else 8

    def snapshot(self):
        """Post-hoc replay. Does not include hidden targets or hidden obstacles."""
        return dict(step=self.t,position=self.pos.tolist(),known=self.known.tolist(),
                    events=list(self.candidate_events),sensor_valid=self.sensor_valid,
                    link_valid=self.link_valid,**self._info())
