import click

@click.group()
@click.option('--debug', default=False, is_flag=True, help='Debug mode.')
def cli(debug):
    """
    DELLve benchmark command line interface.

    Type 'dellve COMMAND --help to see help for commands listed below.
    """

@cli.command('start', short_help='Starts benchmark service.')
@click.option('--config', default='dellve.config.yaml',
    help='Configuration file name.', type=click.File('rb'))
@click.option('--username', prompt=True)
@click.option('--password', prompt=True, hide_input=True)
def start(config, username, password):
    """
    Starts DELLve benchmark background service.
    """
    click.echo('Starting benchmark service')
    click.echo('Note: this actually needs to be implemented though...')
    raise NotImplementedError()

@cli.command('stop', short_help='Stops benchmark service.')
def stop():
    """
    Stops DELLve benchmark background service.
    """
    click.echo('Stopping benchmark service')
    click.echo('Note: this actually needs to be implemented though...')
    raise NotImplementedError()

if __name__ == '__main__':
    cli()
