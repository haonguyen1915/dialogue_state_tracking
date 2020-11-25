from comet.lib import get_cfd, load_yaml, configure_logger
import click
import os
import yaml
from dst.dst_leaner import DialogueStateTrackingLearner

PRJ_DIR = os.getcwd()
SYSTEM_DIR = get_cfd(1)

configure_logger()


@click.group()
def entry_point():
    pass


@click.command()
@click.option('--config', '-c',
              required=True,
              default=f'{SYSTEM_DIR}/configure/config.yaml')
def train(config):
    configure = load_yaml(config)
    learner = DialogueStateTrackingLearner(configure, mode='train')
    learner.train()


@click.command()
@click.option('--config', '-c',
              required=True,
              default=f'{SYSTEM_DIR}/configure/config.yaml')
@click.option('--test_data', '-c',
              required=True,
              default=f'{SYSTEM_DIR}/configure/config.yaml')
def test(test_data, config):
    configure = load_yaml(config)
    learner = DialogueStateTrackingLearner(configure, mode='test')
    learner.train()


entry_point.add_command(train)
entry_point.add_command(test)

if __name__ == "__main__":
    train()
