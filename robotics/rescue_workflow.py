"""Human review and authority state machine, for simulation only.

Names/roles are audit labels, not authenticated identities. No actuator output.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import math
from .rescue import RescueInterlock, RescueState


class RescueMission:
    def __init__(self):
        self.mode='idle'
        self.candidates={}
        self.events=[]
        self.gate=RescueInterlock()
        self.cut_enabled=False

    def _event(self,action,actor,**data):
        event=dict(sequence=len(self.events)+1,time=datetime.now(timezone.utc).isoformat(),
                   action=action,actor=actor,**data)
        self.events.append(event)

    def dispatch(self,action,actor='operator',**data):
        if not isinstance(actor,str) or not actor.strip() or len(actor)>80:
            raise ValueError('named operator required')
        # Any new command revokes the previous instantaneous tool-enable result.
        self.cut_enabled=False
        if action=='start':
            if self.mode!='idle':
                raise ValueError('mission already started')
            self.mode='autonomy'
        elif action=='candidate':
            if self.mode not in ('autonomy','manual','review'):
                raise ValueError('mission must be active to record a candidate')
            cid=str(data.get('id',''))
            if not cid or cid in self.candidates:
                raise ValueError('new candidate ID required')
            confidence=float(data.get('confidence',.5))
            if not math.isfinite(confidence) or not 0<=confidence<=1:
                raise ValueError('invalid synthetic cue score')
            source=str(data.get('source','simulation'))
            self.candidates[cid]=dict(id=cid,status='pending',source=source,
                synthetic_score=confidence,sha256=hashlib.sha256(source.encode()).hexdigest(),
                life_confirmed=False)
            self.mode='review'
        elif action=='review':
            if self.mode!='review':
                raise ValueError('review mode required')
            cid=str(data.get('id',''))
            candidate=self.candidates.get(cid)
            if not candidate or candidate['status']!='pending':
                raise ValueError('pending evidence required')
            if data.get('decision') not in ('recheck','reject','accept_cue'):
                raise ValueError('invalid human review decision')
            decision=data['decision']
            if decision=='recheck':
                candidate['recheck_requested_by']=actor
            else:
                candidate['status']='reviewed_cue' if decision=='accept_cue' else 'rejected'
                candidate['reviewer']=actor
                # Accepting a cue is NOT diagnosing life or authorizing cutting.
                self.mode='review' if any(c['status']=='pending' for c in self.candidates.values()) else 'paused'
        elif action in ('pause','manual'):
            if self.mode=='idle' or self.mode=='estop':
                raise ValueError('active non-estopped mission required')
            self.mode='paused' if action=='pause' else 'manual'
        elif action=='resume':
            if self.mode not in ('paused','manual') or self.gate.latched:
                raise ValueError('clear latch and pause/review before resuming')
            if any(c['status']=='pending' for c in self.candidates.values()):
                raise ValueError('human review is still pending')
            self.mode='autonomy'
        elif action in ('disconnect','estop'):
            self.gate.latched=True
            self.mode='estop' if action=='estop' else 'paused'
        elif action=='rearm':
            if self.mode not in ('paused','estop'):
                raise ValueError('stop before rearming')
            state=self._state(data.get('state',{}))
            if data.get('confirmed') is not True or not self.gate.rearm(state,operator_confirmed=True):
                raise ValueError('manual confirmation, released hold, live heartbeat and locked tool required')
            self.mode='paused'
        elif action=='tool_check':
            if self.mode not in ('paused','manual') or self.gate.latched:
                raise ValueError('pause robot and clear latch before tool checks')
            cid=str(data.get('id',''))
            if self.candidates.get(cid,{}).get('status')!='reviewed_cue':
                raise ValueError('a human-reviewed cue is required')
            if any(c['status']=='pending' for c in self.candidates.values()):
                raise ValueError('complete pending human reviews before tool checks')
            state=self._state(data.get('state',{}))
            result=self.gate.evaluate(state,cut_requested=True)
            self.cut_enabled=result['cut_enabled']
            self._event(action,actor,candidate=cid,input=asdict(state),result=result,
                        notice='Software permission check only; no relay or tool is connected')
            return dict(**self.snapshot(),tool_result=result)
        else:
            raise ValueError('unknown mission action')
        self._event(action,actor,details=data)
        return self.snapshot()

    @staticmethod
    def _state(raw):
        if not isinstance(raw,dict):
            raise ValueError('state object required')
        allowed=RescueState.__dataclass_fields__
        if set(raw)-set(allowed):
            raise ValueError('unknown tool-state field')
        state=RescueState(**raw)
        for field in ('heartbeat_age_s','gas_age_s'):
            value=getattr(state,field)
            if not isinstance(value,(float,int)) or isinstance(value,bool) or not math.isfinite(value) or value<0:
                raise ValueError('finite nonnegative sensor ages required')
        for field in set(allowed)-{'heartbeat_age_s','gas_age_s'}:
            if type(getattr(state,field)) is not bool:
                raise ValueError('boolean interlock fields required')
        return state

    def snapshot(self):
        return dict(mode=self.mode,latched=self.gate.latched,
                    candidates=list(self.candidates.values()),events=self.events[-100:],
                    cut_enabled=self.cut_enabled,hardware_connected=False,
                    authority='local demonstration; role labels are not authentication')
