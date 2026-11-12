"""JSON-lines TextWorld service running under AcceRL's existing interpreter.

Stdout is protocol-only. Slime does not need a TextWorld installation.
"""
import contextlib
import json
import sys


def main():
    with contextlib.redirect_stdout(sys.stderr):
        import textworld
        import textworld.gym
        from accerl_agent.agent_textworld import (
            parse_model_action, format_textworld_user_content,
            format_textworld_illegal_action_feedback, make_textworld_request_infos,
        )
    env = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    if request['op'] == 'reset':
                        if env is not None:
                            env.close()
                        env_id = textworld.gym.register_game(request['game'], request_infos=make_textworld_request_infos(), max_episode_steps=40)
                        env = textworld.gym.make(env_id)
                        if hasattr(env, 'seed'):
                            env.seed(request.get('seed', 42))
                        obs, infos = env.reset()
                        score = float(infos.get('score', 0))
                        attempts = 0
                        environment_steps = 0
                        feedback = format_textworld_user_content(obs, infos)
                        result = {'content': feedback, 'reward': 0, 'reason': None, 'env_step': False}
                    elif request['op'] == 'step':
                        if env is None:
                            raise ValueError('reset required')
                        attempts += 1
                        action = parse_model_action(request['text'], infos.get('admissible_commands', []))
                        reason = None
                        if action.action is None:
                            reward = -0.1
                            feedback = format_textworld_illegal_action_feedback(action.normalized or request['text'].strip(), obs, infos)
                        else:
                            obs, new_score, done, infos = env.step(action.action)
                            environment_steps += 1
                            new_score = float(infos.get('score', new_score))
                            reward = new_score - score + float(infos.get('won', False)) - 0.5 * float(infos.get('lost', False))
                            score = new_score
                            feedback = format_textworld_user_content(obs, infos)
                            if infos.get('won'): reason = 'won'
                            elif infos.get('lost'): reason = 'lost'
                            elif done:
                                reason = 'step_limit' if environment_steps >= 40 else 'environment_done_without_terminal_signal'
                        if reason is None and attempts >= 40:
                            reason = 'step_limit'
                        result = {'content': feedback, 'reward': reward, 'reason': reason, 'env_step': action.action is not None}
                    else:
                        raise ValueError('Unknown operation')
                    result.update(score=score, max_score=float(infos.get('max_score', 0)), won=bool(infos.get('won', False)))
                print(json.dumps(result), flush=True)
            except Exception as error:
                print(json.dumps({'error': f'{type(error).__name__}: {error}'}), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
