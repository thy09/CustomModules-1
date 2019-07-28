import os

from tools.amlservice_scaffold.amlservice_pipeline import Module, PipelineStep, run_pipeline
from azureml.core import RunConfiguration, Workspace
from azureml.core.environment import DEFAULT_GPU_IMAGE


MODULE_SPECS_FOLDER = 'module_specs'


def spec_file_path(spec_file_name):
    return os.path.join(MODULE_SPECS_FOLDER, spec_file_name)


def get_workspace(name, subscription_id, resource_group):
    return Workspace.get(
        name=name,
        subscription_id=subscription_id,
        resource_group=resource_group
    )


def get_run_config(comp, compute_name, use_gpu=False):
    # if comp.image:
    #     run_config = RunConfiguration()
    #     run_config.environment.docker.base_image = comp.image
    # else:
    run_config = RunConfiguration(conda_dependencies=comp.conda_dependencies)
    run_config.target = compute_name
    run_config.environment.docker.enabled = True
    if use_gpu:
        run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE
        run_config.environment.docker.gpu_support = True

    return run_config


def create_pipeline_steps(compute_name):
    # Load module spec from yaml file
    score = Module(
        spec_file_path=spec_file_path('score.yaml'),
        source_directory='script',
    )

    # Run config setting
    run_config_score = get_run_config(score, compute_name, use_gpu=True)

    # Assign parameters

    # Convert to a list of PipelineStep, which can be ran by AML Service
    pipeline_step_list = [
        PipelineStep(score, run_config=run_config_score)
    ]

    return pipeline_step_list


if __name__ == '__main__':
    workspace = get_workspace(
        name="chjinche-test-service",
        subscription_id="e9b2ec51-5c94-4fa8-809a-dc1e695e4896",
        resource_group="chjinche"
    )
    compute_name = 'gpu-compute0'
    pipeline_steps = create_pipeline_steps(compute_name)
    run_pipeline(steps=pipeline_steps, experiment_name='Object-Detection', workspace=workspace)
