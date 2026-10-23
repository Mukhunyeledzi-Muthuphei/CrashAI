import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from crashai import controllers as ctl
from crashai.env import CrashAvoidEnv
from crashai.evaluation import run_episode
from crashai.scenarios import SCENARIOS


@pytest.mark.parametrize("controls", ["full", "pedal", "brakes_only", "driver"])
def test_gym_api(controls):
    check_env(CrashAvoidEnv(controls), skip_render_check=True)


def test_takeover_is_identical_across_controllers():
    """Fair comparison: same seed -> same moment and state at takeover for every control layout."""
    states = []
    for controls in ("full", "pedal", "brakes_only", "driver"):
        env = CrashAvoidEnv(controls)
        env.reset(seed=123, options={"scenario": "cut_in"})
        states.append((env.t, env.car.S.copy()))
    for t, S in states[1:]:
        assert t == states[0][0]
        np.testing.assert_allclose(S, states[0][1])


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenarios_trigger_near_threshold(scenario):
    env = CrashAvoidEnv("full")
    for seed in range(10):
        _, info = env.reset(seed=seed, options={"scenario": scenario})
        assert info["ttc"] < env.ttc_trigger or abs(env.car.sideslip) > 0.1


def test_braking_beats_doing_nothing():
    crashes = {}
    for name, controls, c in [("driver", "driver", ctl.DriverOnly()), ("brake", "full", ctl.BrakeStraight())]:
        env = CrashAvoidEnv(controls)
        n = 0
        for seed in range(40):
            info = run_episode(env, c, seed, "stopped")
            n += info["outcome"] in ("crash", "guardrail")
        crashes[name] = n
    assert crashes["brake"] < crashes["driver"]
