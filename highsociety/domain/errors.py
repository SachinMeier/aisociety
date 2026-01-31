class RuleViolation(Exception):
    pass


class InvalidAction(RuleViolation):
    pass


class InvalidState(RuleViolation):
    pass
