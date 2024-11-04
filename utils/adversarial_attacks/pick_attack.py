from utils.adversarial_attacks.attacks import Attacks


class AttackName:
    nominal = "nominal"
    pgd = "pgd"
    jetfool = "jetfool"
    minimizer = "minimizer"


def pick_attack(attack: str = None, *args, **kwargs):
    match attack:
        case AttackName.nominal:
            return Attacks(*args, **kwargs).nominal
        case AttackName.pgd:
            return Attacks(*args, **kwargs).pgd
        case AttackName.jetfool:
            return Attacks(*args, **kwargs).jetfool
        case AttackName.minimizer:
            return Attacks(*args, **kwargs).minimizer
        case _:
            raise NotImplementedError
