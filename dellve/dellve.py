import click
import config
import pick
import service

@click.group()
@click.option('--config-file', 'config_file', default='dellve.config.yaml',
    help='Configuration file name.', type=click.File('r'))
def cli(config_file):
    """
    DELLve benchmark command line interface.

    Type 'dellve COMMAND --help to see help for commands listed below.
    """
    print 'Loading configuration file...'
    config.load(config_file) # load DELLve configuration

@cli.command('start', short_help='Starts the benchmark service.')
@click.option('--debug', default=False, is_flag=True, help='Debug mode.')
@click.option('--username', prompt=True)
@click.option('--password', prompt=True, hide_input=True)
def start(debug, username, password):
    """
    Starts DELLve benchmark background service.
    """
    click.echo('Starting benchmark service...')
    service.DELLveService(debug).start() # start DELLve daemon service

@cli.command('stop', short_help='Stops the benchmark service.')
def stop():
    """
    Stops DELLve benchmark background service.
    """
    click.echo('Stopping benchmark service...')
    service.DELLveService().stop()

@cli.command('status', short_help='Gets the status of benchmark service.')
def status():
    """
    Gets the status DELLve benchmark background service.
    """
    service.DELLveService().status()

@cli.command('run', short_help='Run the benchmarks.')
@click.option('--all', default=False, is_flag=True, help='Run all benchmarks.')
def run(all):
    """Runs the user specified benchmarks"""
    title = '\n'.join([
        'Please select benchmarks to run:',
        '',
        'Press UP and DOWN arrow keys to navigate',
        'Press SPACE to select benchmarks',
        'Press ENTER to proceede'
    ])

    options = [
        'DELLveShallow',
        'DELLveDeep',
        'DELLveDeep++',
        'DELLveUltraDeepPremium'
    ]

    selected = pick.pick(options, title, indicator='+', multi_select=True)

    from worker import DELLveTooDeepException

    raise DELLveTooDeepException('Please contact Joseph Shalabi for tech support.')

if __name__ == '__main__':
    cli()
