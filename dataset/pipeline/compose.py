
""" Compose multi processing steps sequentially
"""

from yacs.config import CfgNode

from ..build import PIPELINES


@PIPELINES.register_module()
class Compose:
    """ Compose multiple processing methods sequentially. """
    def __init__(self, config):
        """
        Args:
            config (CfgNode):
            - pipelines (Seq[CfgNode|Callable)]): sequence of pipeline objects or
                config dict to be composed.
        """
        pipelines = config.pipelines
        assert isinstance(pipelines, CfgNode)

        self.pipelines = []
        for p_name, p_cfg in pipelines.items():
            pipeline_cls = PIPELINES.get(p_name)
            self.pipelines.append(pipeline_cls(p_cfg))

    def __call__(self, raw_ex, tgt_ex=None):
        """ call function to apply pipelines sequentially.
        Args:
            raw_ex (dict): the raw_ex directly loaded from file, each `pipeline` extract
                the needed information from it.
            tgt_ex (dict): each `pipeline` put the processed information into it.
                - 'code': Code
                - 'summary': Summary
        Returns:
            tgt_ex (dict): updated tgt_ex
        """
        for p in self.pipelines:
            tgt_ex = p(raw_ex, tgt_ex)
            if tgt_ex is None:
                return None
        return tgt_ex

    def __repr__(self):
        format_info = f'{self.__class__.__name__} ('
        for p in self.pipelines:
            str_ = p.__repr__()
            if 'Compose(' in str_:
                str_ = str_.replace('\n', '\n    ')
            format_info += f'\n    {str_}'
        format_info += '\n)'
        return format_info
