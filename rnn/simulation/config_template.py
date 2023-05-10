
from config_manager import config_field
from config_manager import config_template

class ConfigTemplate:

    _simulation_template = config_template.Template(
        fields=[
            config_field.Field(
                name='sim_number',
                types=[type(None), int],
            ),
        ],
        level=['simulation'],
    )
    _data_template = config_template.Template(
        fields=[
            config_field.Field(
                name='datadir',
                types=[str, type(None)],
            ),
            config_field.Field(
                name='datafile',
                types=[str, type(None)],
            ),
        ],
        level=['data'],
    )

    _neurons_template = config_template.Template(
        fields=[
            config_field.Field(
                name='p_recurrent',
                types=[float],
                requirements=[
                    lambda x: x >= 0
                ],
            ),
            config_field.Field(
                name='n1',
                types=[int],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='tau',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='g1',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='gin',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='gout',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='noise',
                types=[float],
                requirements=[
                    lambda x: x >= 0
                ],
            ),
        ],
        level=['neurons'],
    )

    _regularization_template = config_template.Template(
        fields=[
            config_field.Field(
                name='alpha1',
                types=[float],
                requirements=[
                    lambda x: x >= 0
                ],
            ),
            config_field.Field(
                name='gamma1',
                types=[float],
                requirements=[
                    lambda x: x >= 0
                ],
            ),
            config_field.Field(
                name='beta1',
                types=[float],
                requirements=[
                    lambda x: x >= 0
                ],
            ),
            config_field.Field(
                name='clipgrad',
                types=[float],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='ccareg',
                types=[bool, type(None)],
            ),
            config_field.Field(
                name='pcas_file',
                types=[str, type(None)],
            ),
            config_field.Field(
                name='delta',
                types=[float],
                requirements=[
                    lambda x: x >= 0
                ],
            ),
            config_field.Field(
                name='ccareg_type',
                types=[str, type(None)],
                requirements=[
                    lambda x: x in [None, 'norm', 'sum_squared', 'sum']
                ],
            ),
            config_field.Field(
                name='ccareg_components_start',
                types=[int, type(None)],
            ),
            config_field.Field(
                name='ccareg_components_end',
                types=[int, type(None)],
            ),
            config_field.Field(
                name='ccareg_start_trial',
                types=[int, type(None)],
            ),
            config_field.Field(
                name='rel_start',
                types=[int, type(None)],
            ),
            config_field.Field(
                name='rel_end',
                types=[int, type(None)],
            ),
        ],
        level=['regularization'],
    )

    _training_template = config_template.Template(
        fields=[
            config_field.Field(
                name='optimizer',
                types=[str],
                requirements=[
                    lambda x: x in ['Adam', 'FORCE']
                ],
                default = 'Adam'
            ),
            config_field.Field(
                name='batch_size',
                types=[int],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='training_trials',
                types=[int],
                requirements=[
                    lambda x: x > 0
                ],
            ),
            config_field.Field(
                name='lr',
                types=[float],
            ),
        ],
        level=['training'],
    )

    _logging_template = config_template.Template(
        fields=[
            config_field.Field(
                name='log_model',
                types=[bool],
            ),
            config_field.Field(
                name='log_interval',
                types=[int, type(None)],
            ),
            config_field.Field(
                name='log_epochs',
                types=[list, type(None)],
            ),
        ],
        level=['logging'],
    )

    base_config_template = config_template.Template(
        fields=[
            config_field.Field(
                name='outdir',
                types=[str, type(None)],
            ),
            config_field.Field(
                name='gpu_id',
                types=[int],
            ),
            config_field.Field(
                name='seed',
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name='timestamp',
                types=[str, type(None)],
            ),
        ],
        nested_templates=[
            _simulation_template,
            _data_template,
            _neurons_template,
            _regularization_template,
            _training_template,
            _logging_template,
        ],
    )