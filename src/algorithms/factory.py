from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.sgsac import SGSAC
from algorithms.SCPL0 import SCPL0
from algorithms.SCPL0r import SCPL0r
from algorithms.SCPL import SCPL
from algorithms.SCPLr import SCPLr
from rl_generalization.SCPL.src.algorithms.SCPL import SGSAC83
from rl_generalization.SCPL.src.algorithms.SCPL0r import SGSAC84

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA,
    "sgsac": SGSAC,
    "scpl": SCPL,
    "scpl0": SCPL0,
    "scplr": SCPLr,
    "scpl0r": SCPL0r,
	
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
