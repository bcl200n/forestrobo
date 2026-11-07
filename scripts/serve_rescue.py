"""Loopback-only rescue development demo API. No robot hardware endpoint."""
import argparse
from datetime import datetime
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from urllib.parse import urlparse,parse_qs

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from robotics.rescue_workflow import RescueMission
from robotics.rescue_selection import SCENARIOS,selection,arm_preview
from robotics.manipulator import CATALOG,chain

mission=RescueMission()
lock=threading.Lock()


def read_json(path,default=None):
    try:return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return default


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--run',default='rl/models/rescue_search_20261106_v1')
    parser.add_argument('--open-browser',action='store_true')
    args=parser.parse_args()
    run=(ROOT/args.run).resolve()
    if not run.is_relative_to(ROOT/'rl/models'):
        parser.error('run must be inside rl/models')
    audit=ROOT/'ops'/('rescue-events-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.jsonl')
    allowed_origins={f'http://127.0.0.1:{args.port}',f'http://localhost:{args.port}',
                     'http://127.0.0.1:8765','http://localhost:8765'}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):
            super().__init__(*a,directory=str(ROOT/'web'),**kw)

        def end_headers(self):
            origin=self.headers.get('Origin')
            if origin in allowed_origins:
                self.send_header('Access-Control-Allow-Origin',origin)
                self.send_header('Vary','Origin')
            super().end_headers()

        def send_json(self,value,status=200):
            body=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            if self.headers.get('Origin') not in allowed_origins:
                return self.send_json({'error':'local demo origin required'},403)
            self.send_response(204)
            self.send_header('Access-Control-Allow-Methods','GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers','Content-Type')
            self.end_headers()

        def do_GET(self):
            parsed=urlparse(self.path)
            if parsed.path=='/api/rescue/catalog':
                return self.send_json(dict(scenarios=SCENARIOS,catalog=CATALOG))
            if parsed.path=='/api/rescue/status':
                return self.send_json(dict(training=read_json(run/'status.json',{'status':'not_started'}),
                    baseline=read_json(run/'baseline.json'),evaluation=read_json(run/'evaluation.json'),
                    validation=read_json(run/'validation.json'),protocol=read_json(run/'protocol.json'),
                    run=str(run.relative_to(ROOT)).replace('\\','/')))
            if parsed.path=='/api/rescue/replay':
                label=parse_qs(parsed.query).get('controller',['baseline'])[0]
                if label not in ('baseline','evaluation','validation'):
                    return self.send_json({'error':'unknown controller'},400)
                replay=read_json(run/f'{label}-replay.json')
                return self.send_json(replay if replay else {'error':'replay not available'},200 if replay else 404)
            if parsed.path=='/api/rescue/mission':
                with lock:return self.send_json(mission.snapshot())
            if parsed.path.startswith('/api/'):
                return self.send_json({'error':'unknown route'},404)
            return super().do_GET()

        def do_POST(self):
            if self.headers.get('Origin') and self.headers['Origin'] not in allowed_origins:
                return self.send_json({'error':'local demo origin required'},403)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=32768:
                    raise ValueError('invalid payload length')
                if self.headers.get_content_type()!='application/json':
                    raise ValueError('JSON required')
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):
                    raise ValueError('JSON object required')
                path=urlparse(self.path).path
                if path=='/api/rescue/select':
                    return self.send_json(selection(payload['scenario'],float(payload.get('object_mass_kg',0))))
                if path=='/api/rescue/arm':
                    result=arm_preview(payload['scenario'],payload['target_m'],
                        float(payload.get('object_mass_kg',0)),payload.get('reaction_n',[0,0,0]))
                    _,origins,_,_=chain(result['ik']['joints_rad'],result['selection']['arm_id'])
                    result['joint_origins_m']=origins.tolist()
                    return self.send_json(result)
                if path=='/api/rescue/mission':
                    with lock:
                        before=len(mission.events)
                        if payload.get('action')=='import_replay':
                            label=payload.get('controller','validation')
                            if label not in ('validation','evaluation'):
                                raise ValueError('a learned-policy replay is required')
                            replay=read_json(run/f'{label}-replay.json')
                            if not replay or not replay.get('frames'):
                                raise ValueError('learned replay not available yet')
                            if mission.mode not in ('autonomy','manual','review'):
                                raise ValueError('start or resume the mission before importing evidence')
                            imported=0
                            for index,event in enumerate(replay['frames'][-1]['events']):
                                source=json.dumps(dict(controller=label,map_seed=replay['map_seed'],event=event),sort_keys=True)
                                cid=f"RL-{label}-{replay['map_seed']}-{hashlib.sha256(source.encode()).hexdigest()[:8]}"
                                if cid in mission.candidates:
                                    continue
                                mission.dispatch('candidate',actor=payload.get('actor','operator'),
                                    id=cid,source=source,confidence=.5)
                                imported+=1
                            result=dict(**mission.snapshot(),imported=imported)
                        else:
                            result=mission.dispatch(**payload)
                        with audit.open('a',encoding='utf-8') as f:
                            for event in mission.events[before:]:
                                f.write(json.dumps(event,ensure_ascii=False)+'\n')
                        return self.send_json(result)
                return self.send_json({'error':'unknown route'},404)
            except (ValueError,KeyError,TypeError) as exc:
                return self.send_json({'error':str(exc)},400)

    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Rescue demo: http://127.0.0.1:{args.port}/rescue.html',flush=True)
    print(f'Training: {run}',flush=True)
    if args.open_browser:
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{args.port}/rescue.html')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
