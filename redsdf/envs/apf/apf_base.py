class ApfBase:
    def __init__(self, kinematics, max_action):
        self.kinematics = kinematics
        self.max_action = max_action

    def update(self, joint_position, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass
