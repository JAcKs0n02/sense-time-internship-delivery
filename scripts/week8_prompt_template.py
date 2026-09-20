"""Scoped OpenCompass 0.5.3 meta-template renderer for Week 8 benchmarks.

Only placeholders in the template are interpreted. Inserted dataset text is
literal, so mathematical braces and backslashes survive rendering unchanged.
This module does not patch OpenCompass globals or support legacy string/label
prompt templates. Its supported path is five-shot generation with a standalone begin ICE marker.
"""
import copy
import re

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.registry import ICL_PROMPT_TEMPLATES
from opencompass.utils.prompt import PromptList

_PLACEHOLDER = re.compile(r'\{([^{}]+)\}')


def format_once(template, values):
    return _PLACEHOLDER.sub(
        lambda match: str(values[match[1]]) if match[1] in values else match[0],
        template,
    )


class NonRecursivePromptList(PromptList):
    def format(self, **kwargs):
        result = NonRecursivePromptList()
        for item in self:
            if isinstance(item, dict):
                value = copy.deepcopy(item)
                if 'prompt' in value:
                    value['prompt'] = format_once(value['prompt'], kwargs)
            else:
                value = format_once(item, kwargs)
            result.append(value)
        return result

    def replace(self, src, dst):
        # The base class returns PromptList, which would restore unsafe format.
        return NonRecursivePromptList(super().replace(src, dst))


@ICL_PROMPT_TEMPLATES.register_module()
class NonRecursivePromptTemplate(PromptTemplate):
    def __init__(self, template, ice_token=None, sep_token=None):
        super().__init__(template, ice_token=ice_token, sep_token=sep_token)
        if self.prompt_type != 'meta' or 'round' not in template:
            raise ValueError('Week 8 requires a meta template with a round')
        if sep_token is not None:
            raise ValueError('Week 8 meta template does not support sep_token')
        if ice_token is not None:
            if template.get('begin') != ice_token:
                raise ValueError('Week 8 meta template requires a standalone begin ICE marker')
            if ice_token in repr({k: v for k, v in template.items() if k != 'begin'}):
                raise ValueError('Week 8 meta template ICE marker must occur only in begin')

    def _encode_template(self, prompt_template, ice):
        return NonRecursivePromptList(super()._encode_template(prompt_template, ice))

    def generate_label_prompt_item(self, *args, **kwargs):
        raise ValueError('Week 8 meta template supports generation with separate ICE only')

    def generate_item(self, entry, output_field=None,
                      output_field_replace_token='', ice_field_replace_token=''):
        values = copy.deepcopy(entry)
        if output_field is not None:
            values[output_field] = output_field_replace_token
        result = NonRecursivePromptList()
        for item in self._encode_template(self.template, ice=False):
            # Splice already-rendered examples at the original marker. Never
            # search inserted data for markers or format examples a second time.
            if self.ice_token is not None and isinstance(item, str) and item == self.ice_token:
                if isinstance(ice_field_replace_token, PromptList):
                    result.extend(copy.deepcopy(ice_field_replace_token))
                elif isinstance(ice_field_replace_token, str):
                    result.append(ice_field_replace_token)
                else:
                    raise TypeError('ICE must be a string or PromptList')
            else:
                result.extend(NonRecursivePromptList([item]).format(**values))
        return result
