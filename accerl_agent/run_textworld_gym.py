import textworld
import textworld.gym


GAME_FILE = "data/tw_games/coin_l1_seed0.z8"


def pick_action(infos):
    """
    最简单的规则 agent：
    1. 如果官方给了 winning policy，就直接用第一步；
    2. 否则优先 take coin；
    3. 否则优先 go xxx；
    4. 否则 look。
    """

    # policy_commands 是官方给的 oracle 路径，调试时可以用。
    policy = infos.get("policy_commands", [])
    if policy:
        return policy[0]

    admissible = infos.get("admissible_commands", [])
    if not admissible:
        return "look"

    for cmd in admissible:
        if "take coin" in cmd:
            return cmd

    for cmd in admissible:
        if cmd.startswith("go "):
            return cmd

    if "look" in admissible:
        return "look"

    return admissible[0]


def main():
    request_infos = textworld.EnvInfos(
        objective=True,
        inventory=True,
        admissible_commands=True,
        policy_commands=True,   # 调试用，正式训练不要用
        score=True,
        max_score=True,
        won=True,
        lost=True,
        moves=True,
    )

    env_id = textworld.gym.register_game(
        GAME_FILE,
        request_infos=request_infos,
        max_episode_steps=20,
    )

    env = textworld.gym.make(env_id)

    obs, infos = env.reset()

    print("=" * 80)
    print("Initial observation:")
    print(obs)

    print("\nObjective:")
    print(infos.get("objective"))

    print("\nAdmissible commands:")
    for cmd in infos.get("admissible_commands", []):
        print("-", cmd)

    done = False

    for step in range(20):
        action = pick_action(infos)

        print("\n" + "=" * 80)
        print(f"Step {step}")
        print("Action:", action)

        obs, score, done, infos = env.step(action)

        print("\nObservation:")
        print(obs)

        print("\nScore:", score)
        print("Done:", done)
        print("Won:", infos.get("won"))
        print("Lost:", infos.get("lost"))

        print("\nAdmissible commands:")
        for cmd in infos.get("admissible_commands", []):
            print("-", cmd)

        if done:
            break

    env.close()


if __name__ == "__main__":
    main()