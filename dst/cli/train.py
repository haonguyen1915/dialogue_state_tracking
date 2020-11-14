import click
import os


def execute_cmd(cmd):
    os.system(cmd)


@click.group()
def train():
    pass


@train.command()
@click.option("--name", "-n", "name",
              type=str,
              show_default=True,
              default=f"comet_{DOMAIN}".lower(),
              help="The data id")
@click.option("--description", '-d', "description",
              type=str,
              show_default=True,
              default="where to save model for comet",
              help="The description for model")
@click.option("--status", "status",
              type=str,
              show_default=True,
              default="public",
              help="The description for model")
def create(name, description, status):
    if not name:
        _logger.error(f"Create a new model container, need NAME DOMAIN "
                      f"provided> You can configure in environment file")
        exit()
    print_title("Creating a dataset container")
    create_resources(name, description, status, "dataset")
