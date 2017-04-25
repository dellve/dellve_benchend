import api
import benchmark
import click
import config
import daemon
import os
import pick
import shutil
import signal
import sys
import template
import time
import tqdm
import util

@click.group()
@click.option('--config-file', 'config_file',
              default=config.get('config-file'),
              help='Configuration file name.', type=click.File('r'))
def cli(config_file):
    """DELLve benchmark command line interface.

    Type 'dellve COMMAND --help' to see help for commands listed below.
    """
    config.load(config_file)  # load DELLve configuration


@cli.command('dir', short_help='Show application directory.')
def dir():
    """Show DELLve application directory path.
    """
    click.echo(config.get('app-dir'))

@cli.command('config', short_help='Edit the configuration file.')
def _config():
    """Opens DELLve configuration file in default editor.
    """
    # Helper function that appends application directory to path
    file_path = lambda p: os.path.join(config.get('app-dir'), p)
    config_file = file_path('config.json')
    if not os.path.exists(config_file):
        shutil.copy(file_path('default.config.json'), config_file)
    # Open file in default editor
    click.edit(filename=config_file)

def validate_server(ctx, param, value):
    # TODO: make sure value[0] is a valid host name
    # TODO: make sure value[1] is a valid port number
    # TODO: make sure server with host name and port number is up!

    return {'host': value[0], 'port': value[1]}

@cli.command('ls', short_help='List installed benchmarks.')
@click.option('--server', '-s', 'server', callback=validate_server,
              default=(config.get('http-host'), config.get('http-port')),
              help='Host and port of remote server API.',
              metavar='<HOST> <PORT>', type=(str, int))
def ls(server):
    """Lists installed DELLve benchmarks on the local machine or on a remote
    server.

    If HOST and PORT are not specified, benchmarks installed on the local
    machine are listed (Default values for HOST and PORT are specified in
    config file).

    We can also list benchmarks installed on a remote server by specifying the
    HOST and PORT the server's Dellve API is available on.
    """

    # Overwrite config values
    config.set('http-host', server['host'])
    config.set('http-port', server['port'])

    # Get list of benchmarks from API
    benchmarks = util.api_get('benchmark',
        err_msg='Unable to query installed benchmarks').json()

    # Show list of benchmarks in pretty format
    click.echo('\n'.join(['    ' + b['name'] for b in benchmarks]))


@cli.command('start', short_help='Start the benchmark service.')
@click.option('--no-detach', 'no_detach', default=False,
              help='Start without detaching background daemon.',
              is_flag=True)
def start(no_detach):
    """Starts DELLve benchmark background service.
    """
    # Delegate task to daemon
    daemon.Daemon(no_detach).do_action('start')

@cli.command('status', short_help='Get the status of benchmark service.')
def status():
    """Gets the status DELLve benchmark background service.
    """
    # Delegate task to daemon
    daemon.Daemon().do_action('status')

@cli.command('stop', short_help='Stop the benchmark service.')
def stop():
    """Stops DELLve benchmark background service.
    """
    # Delegate task to daemon
    daemon.Daemon().do_action('stop')

@cli.command('restart', short_help='Restart the benchmark service.')
def restart():
    """Restarts DELLve benchmark background service.
    """
    # Delegate task to daemon
    daemon.Daemon().do_action('restart')


@cli.command('run', short_help='Runs the benchmarks either locally or remotely.')
@click.option('--server', '-s', 'server', callback=validate_server,
              default=(config.get('http-host'), config.get('http-port')),
              help='Host and port of remote server API.',
              metavar='<HOST> <PORT>',
              type=(str, int))
def run(server):
    """Runs the specified benchmarks.

    If HOST and PORT are not specified, benchmarks are run on the local machine
    (Default values for HOST and PORT are specified in config file).

    We can also run benchmarks on a remote server by specifying the HOST and PORT
    the server's API is available on.
    """

    # Overwrite config values
    config.set('http-host', server['host'])
    config.set('http-port', server['port'])

    # Get list of benchmarks from API
    benchmarks = util.api_get('benchmark',
        err_msg='Unable to query installed benchmarks').json()

    # Ensure there're some benchmarks to be run
    if (len(benchmarks) < 1):
        click.echo('Please, install at least one benchmark plugin.', err=True)
        return

    # Select which benchmarks to run
    options = [b['name'] for b in benchmarks]
    title = '\n'.join([
        'Please select benchmarks to run:',
        '',
        'Press UP and DOWN arrow keys to navigate',
        'Press SPACE to select benchmarks',
        'Press ENTER to proceede'
    ])
    picked = pick.pick(options, title, indicator='+', multi_select=True)

    # Get list of selected benchmarks
    benchmarks = [benchmarks[index] for name, index in picked]

    # Create run state
    run_benchmark = {}

    # Helper function for getting benchmark progress
    get_progress = lambda: util.api_get('benchmark/progress',
        err_msg='Unable to query benchmark progress').json()

    # Define OS signal handlers
    def handler(signum, frame):
        if run_benchmark:
            _id = run_benchmark['id']
            for b in benchmarks:
                if b['id'] == _id:
                    name = b['name']
            if util.api_post('benchmark/%d/stop', _id).ok:
                click.echo('Stopping %s benchmark...OK' % name)
                click.echo(''.join(get_progress()['output']))
            else:
                click.echo('Stopping %s benchmark...FAILED' % name)
        sys.exit(signum) # TODO: set proper exit code

    # Register SIGINT& SIGTERM handlers
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # Run benchmarks
    for b in benchmarks:

        # Update run state (this is used in signal handler)
        run_benchmark.update({'id': b['id']})

        # Start benchmark through HTTP API
        util.api_post('benchmark/%s/start', b['id'],
            err_msg='Unable to start %s benchmark' % b['name'])

        # Get initial progress
        progress_res = get_progress()

        # Create progress bar and run benchmark
        with tqdm.tqdm(desc=b['name'], total=100) as progress_bar:
            old_progress = 0
            while (progress_res['running']):
                progress_res = get_progress()
                new_progress = progress_res['progress']
                progress_bar.update(new_progress - old_progress)
                old_progress = new_progress
                time.sleep(0.01)

        # Echo out benchmark's output
        # map(click.echo, ''.join(progress_res['output']))
        click.echo(''.join(progress_res['output']))


@cli.command('new', short_help='Create a new benchmark.')
def new():
    """Generates the boilerplate code for a new DELLve-benchmark plugin
    """
    dir_name = click.prompt('Please enter a directory name', default='./')
    package_name = click.prompt('Please enter a Python package name')
    benchmark_name = click.prompt('Please enter a unique benchmark class name')
    template.Template().render(dir_name, package_name, benchmark_name)
