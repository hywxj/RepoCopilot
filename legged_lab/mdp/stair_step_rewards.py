"""Rewards from the measured, depth-derived single-step supervisor."""


def stair_step_metric(env, metric: str, quadratic_error: bool = False, max_normalized_error: float = 9.):
    value = env.step_reward_metrics[metric]
    if not quadratic_error:
        return value
    if metric not in ("feet_reference", "body_reference") or max_normalized_error <= 0:
        raise ValueError("Quadratic tracking requires a reference metric and a positive cap.")
    # Penalize inverse proximity; drifting far away must
    # remain costly instead of saturating to a near-zero tracking reward.
    return -(value.clamp_min(1.e-6).reciprocal()-1.).clamp(0., max_normalized_error)


def single_step_failure(env):
    return (env.reset_buf & ~env.step_success_event).float() / env.step_dt
