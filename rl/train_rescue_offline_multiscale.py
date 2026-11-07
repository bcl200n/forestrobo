"""Offline rescue-search pilot with action cloning and a learned subgoal."""
import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO

from rescue_search_env import SCENARIOS
from train_rescue_scan_wait import ScanWaitEnv, WAIT, metrics
from train_rescue_search import evaluate, now, write_json


SCHEMA = 'rescue-offline-subgoal-v1'
OBSERVATIONS = 39
ACTIONS = 10
HORIZON = 8


class OfflineSearchEnv(ScanWaitEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, split=True, **kwargs)

    def baseline_action(self):
        if not self.sensor_valid or not self.link_valid:
            return WAIT
        return super().baseline_action()

    def _info(self):
        info = super()._info()
        info['first_candidate_step'] = self.candidate_events[0]['step'] if self.candidate_events else None
        return info


def future_offsets(positions, horizon):
    """Relative endpoint in an individual episode, normalized by the horizon."""
    steps = np.arange(len(positions)-1)
    future = np.minimum(steps+horizon, len(positions)-1)
    return (positions[future]-positions[steps]).astype(np.float32)/horizon


def collect(seeds, horizon, *, environment_class=OfflineSearchEnv):
    observations, actions, offsets, positions, rewards, episodes = [], [], [], [], [], []
    boundaries = [0]
    for scenario in SCENARIOS:
        env = environment_class(scenario=scenario, map_seeds=seeds)
        try:
            for seed in seeds:
                obs, info = env.reset(seed=seed, options={'map_seed':seed})
                path = [env.pos.copy()]
                for _ in range(env.max_steps):
                    action = env.baseline_action()
                    observations.append(obs.copy())
                    actions.append(action)
                    obs, reward, term, trunc, info = env.step(action)
                    rewards.append(reward)
                    path.append(env.pos.copy())
                    if term or trunc:
                        break
                path = np.asarray(path, dtype=np.int16)
                positions.extend(path[:-1])
                offsets.extend(future_offsets(path, horizon))
                boundaries.append(len(actions))
                episodes.append(dict(info, final_position=path[-1].tolist(), terminated=term, truncated=trunc))
        finally:
            env.close()
    arrays = dict(observation=np.asarray(observations, dtype=np.float32),
                  action=np.asarray(actions, dtype=np.int64),
                  goal_offset=np.asarray(offsets, dtype=np.float32),
                  position=np.asarray(positions, dtype=np.int16),
                  reward=np.asarray(rewards, dtype=np.float32),
                  episode_boundaries=np.asarray(boundaries, dtype=np.int64))
    return arrays,episodes


class SearchPolicy(nn.Module):
    def __init__(self, multiscale, observation_size=OBSERVATIONS):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(observation_size,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU())
        self.goal = nn.Linear(64,2) if multiscale else None
        self.action = nn.Linear(66 if multiscale else 64,ACTIONS)

    def forward(self, observations):
        features = self.encoder(observations)
        goal = torch.tanh(self.goal(features)) if self.goal is not None else None
        if goal is not None:
            features = torch.cat((features,goal),dim=-1)
        return self.action(features),goal

    @torch.inference_mode()
    def predict(self, observation, deterministic=True):
        features = torch.as_tensor(observation,dtype=torch.float32)
        logits,_ = self(features)
        return logits.argmax(dim=-1).numpy(),None


def tensors(data):
    return tuple(torch.from_numpy(data[field]) for field in ('observation','action','goal_offset'))


def fit(training, validation, multiscale, seed, epochs, checkpoint, *, schema=SCHEMA):
    torch.manual_seed(seed)
    observation_size = training['observation'].shape[1]
    model = SearchPolicy(multiscale, observation_size)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
    train_x, train_action, train_goal = tensors(training)
    valid_x, valid_action, valid_goal = tensors(validation)
    generator = torch.Generator().manual_seed(seed+100)
    selected_loss = float('inf')
    selected = None
    curves = []
    started = perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        batches = torch.randperm(len(train_action),generator=generator).split(256)
        for batch in batches:
            logits,goal = model(train_x[batch])
            loss = nn.functional.cross_entropy(logits,train_action[batch])
            if goal is not None:
                loss = loss + nn.functional.mse_loss(goal,train_goal[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits,goal = model(valid_x)
            action_loss = float(nn.functional.cross_entropy(logits,valid_action))
            accuracy = float((logits.argmax(-1)==valid_action).float().mean())
            goal_error = float(nn.functional.mse_loss(goal,valid_goal)) if goal is not None else None
        curves.append(dict(epoch=epoch,validation_action_loss=action_loss,
                           validation_accuracy=accuracy,validation_goal_mse=goal_error))
        if action_loss < selected_loss:
            selected_loss = action_loss
            selected = (epoch,{key:value.detach().clone() for key,value in model.state_dict().items()})
    seconds = perf_counter()-started
    selected_epoch,weights = selected
    model.load_state_dict(weights)
    torch.save(dict(schema=schema,multiscale=multiscale,observation_size=observation_size,
                    state_dict=weights),checkpoint)
    return model,dict(training_seconds=seconds,selected_epoch=selected_epoch,history=curves,
                      parameters=sum(p.numel() for p in model.parameters()))


def trial(model,seeds,output,label, *, environment_class=OfflineSearchEnv, schema=SCHEMA, horizon=256):
    started = perf_counter()
    report = evaluate(model,seeds,output,label,environment_class=environment_class,schema=schema)
    rows = report['episodes']
    result = metrics(report, horizon=horizon)
    result['mean_first_candidate_cost'] = float(np.mean([
        row['first_candidate_step'] if row['first_candidate_step'] is not None else horizon for row in rows]))
    result['mean_steps'] = float(np.mean([row['steps'] for row in rows]))
    result['evaluation_seconds'] = perf_counter()-started
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-maps',type=int,default=64,help='Number of map seeds per scenario')
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--model-seeds',type=int,nargs='+',default=[31,32,33])
    parser.add_argument('--prior-ppo',type=Path)
    args = parser.parse_args()
    if not 1 <= args.train_maps <= 96 or args.epochs < 1 or len(set(args.model_seeds)) != len(args.model_seeds):
        parser.error('require 1–96 training seeds, positive epochs and unique model seeds')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    split = dict(train=list(range(32000,32000+args.train_maps)),
                 validation=list(range(32100,32108)),test=list(range(32200,32216)))
    sources = ('train_rescue_offline_multiscale.py','train_rescue_scan_wait.py','train_rescue_guided.py',
               'train_rescue_search.py','rescue_search_env.py')
    protocol = dict(schema=SCHEMA,scenario_names=SCENARIOS,seeds=split,model_seeds=args.model_seeds,
                    epochs=args.epochs,subgoal_horizon_steps=HORIZON,device='cpu',
                    teacher='observed-map frontier with explicit wait during outages',
                    selection='minimum validation action cross entropy',
                    scope='2-D synthetic cue search; shared observed-action guard',
                    sources={name:sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources})
    write_json(args.output/'protocol.json',protocol)

    def status(state, **fields):
        write_json(args.output/'status.json',dict(status=state,schema=SCHEMA,updated_at=now(),**fields))

    data,collection,results = {},{},{}
    try:
        for label in ('train','validation'):
            status('collecting',partition=label,collection=collection)
            started = perf_counter()
            data[label],episodes = collect(split[label],HORIZON)
            seconds = perf_counter()-started
            np.savez_compressed(args.output/f'{label}.npz',**data[label])
            write_json(args.output/f'{label}-episodes.json',episodes)
            collection[label] = dict(episodes=len(episodes),steps=len(data[label]['action']),
                                     successful=sum(row['success'] for row in episodes),seconds=seconds)
        status('evaluating',controller='frontier',collection=collection)
        results['frontier'] = trial(None,split['test'],args.output,'frontier')
        if args.prior_ppo:
            prior = PPO.load(args.prior_ppo,device='cpu')
            if prior.observation_space.shape!=(OBSERVATIONS,) or prior.action_space.n!=ACTIONS:
                raise ValueError('prior policy must use the current observation and split action spaces')
            results['prior_ppo'] = trial(prior,split['test'],args.output,'prior_ppo')
        for label,multiscale in [('single_step',False),('two_scale',True)]:
            results[label] = []
            for seed in args.model_seeds:
                name = f'{label}-{seed}'
                status('training',controller=name,collection=collection,results=results)
                model,training = fit(data['train'],data['validation'],multiscale,seed,args.epochs,
                                     args.output/f'{name}.pt')
                status('evaluating',controller=name,collection=collection,results=results)
                score = trial(model,split['test'],args.output,name)
                write_json(args.output/f'{name}-training.json',training)
                results[label].append(dict(seed=seed,training_seconds=training['training_seconds'],
                                           parameters=training['parameters'],**score))
        status('complete',collection=collection,results=results)
        print(dict(collection=collection,results=results),flush=True)
    except BaseException as exc:
        status('failed',error=repr(exc),collection=collection,results=results)
        raise


if __name__ == '__main__':
    main()
