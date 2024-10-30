def get_task(name):
    # if name == 'game24':
    #     from src.tot.tasks.game24 import Game24Task
    #     return Game24Task()
    # elif name == 'text':
    #     from tot.tasks.text import TextTask
    #     return TextTask()
    # elif name == 'crosswords':
    #     from tot.tasks.crosswords import MiniCrosswordsTask
    #     return MiniCrosswordsTask()
    if name == 'gsm8k':
        from tasks.gsm8k import GSM8KTask
        return GSM8KTask()
    # if name == 'strategyqa': 
    #     from tasks.strategyqa import StrategyQA
    #     return StrategyQA()
    else:
        raise NotImplementedError