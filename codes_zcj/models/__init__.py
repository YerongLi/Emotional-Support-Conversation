
from models.strat_blenderbot_small import Model as strat_blenderbot_small
from models.vanilla_blenderbot_small import Model as vanilla_blenderbot_small

from models.strat_dialogpt import Model as strat_dialogpt
from models.vanilla_dialogpt import Model as vanilla_dialogpt
from models.vanilla_dialogpt_neg import Model as vanilla_dialogpt_neg

models = {
    
    'vanilla_blenderbot_small': vanilla_blenderbot_small,
    'strat_blenderbot_small': strat_blenderbot_small,
    
    'vanilla_dialogpt': vanilla_dialogpt,
    'vanilla_dialogpt_neg': vanilla_dialogpt_neg,
    'strat_dialogpt': strat_dialogpt,
}