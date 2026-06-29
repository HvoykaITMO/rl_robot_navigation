from utils import constants as c


def get_action_id(throttle, steering, brake):
    for idx, action in enumerate(c.DISCRETE_ACTIONS):
        if (
            action["throttle"] == throttle
            and action["steering"] == steering
            and action["brake"] == brake
        ):
            return idx
    raise ValueError(f"Action not found: {throttle=}, {steering=}, {brake=}")