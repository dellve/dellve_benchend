import api
import benchmark
import click
import config
import daemon
import os
import pick
import requests
import shutil
import signal
import sys
import template
import time
import tqdm

@click.group()
@click.option('--config-file', 'config_file',
              default=config.get('config-file'),
              help='Configuration file name.', type=click.File('r'))
def cli(config_file):
    """DELLve benchmark command line interface.

    Type 'dellve COMMAND --help' to see help for commands listed below.
    """
    config.load(config_file)  # load DELLve configuration


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

def validate_api(ctx, param, value):
    # TODO
    # Validate values

    try:
        api = 'http://{}:{}'.format(value[0], value[1])
        r = requests.get('{}/benchmark'.format(api))
        return api

    except requests.ConnectionError:
        raise click.BadParameter('Could not connect to {}'.format(api))


@cli.command('ls', short_help='List installed benchmarks.')
@click.option('--server', '-s', 'api', help='Host and port of remote server API.',
              type=(str, int), metavar='<HOST> <PORT>', callback=validate_api,
              default=(config.get('http-host'), config.get('http-port')))
def ls(api):
    """Lists installed DELLve benchmarks on the local machine or on a remote server.

    If HOST and PORT are not specified, benchmarks installed on the local
    machine are listed (Default values for HOST and PORT are specified in
    config file).

    We can also list benchmarks installed on a remote server by specifying the
    HOST and PORT the server's Dellve API is available on.
    """

    # Load benchmarks
    benchmarks = requests_get_with_error('{}/benchmark'.format(api),
                             'Unable to query installed benchmarks')

    print '\n'.join(['    ' + b['name'] for b in benchmarks])


@cli.command('start', short_help='Start the benchmark service.')
@click.option('--debug', 'debug', default=False,
              help='Turn on debug mode.', is_flag=True)
def start(debug):
    """Starts DELLve benchmark background service.
    """
    click.echo('Starting benchmark service...')
    daemon.Daemon(debug=debug).do_action('start')


@cli.command('status', short_help='Get the status of benchmark service.')
def status():
    """Gets the status DELLve benchmark background service.
    """
    click.echo('Getting benchmark status...')
    daemon.Daemon().do_action('status')


@cli.command('stop', short_help='Stop the benchmark service.')
def stop():
    """Stops DELLve benchmark background service.
    """
    click.echo('Stopping benchmark service...')
    daemon.Daemon().do_action('stop')

@cli.command('run', short_help='Runs the benchmarks either locally or remotely.')
@click.option('--server', '-s', 'api', help='Host and port of remote server API.',
              type=(str, int), metavar='<HOST> <PORT>', callback=validate_api,
              default=(config.get('http-host'), config.get('http-port')))
def run(api):
    """Runs the specified benchmarks.

    If HOST and PORT are not specified, benchmarks are run on the local machine
    (Default values for HOST and PORT are specified in config file).

    We can also run benchmarks on a remote server by specifying the HOST and PORT
    the server's API is available on.
    """

    # Load benchmarks
    benchmarks = requests_get_with_error('{}/benchmark'.format(api),
                             'Unable to query installed benchmarks')

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

    # Get benchmark IDs for selected benchmarks
    picked_idx = [int(index) for name, index in picked]
    picked_idx.sort()
    benchmark_ids = [benchmarks[i]['id'] for i in picked_idx]

    # Run benchmarks sequentially
    for bid in benchmark_ids:
        # Start benchmark
        r_start = requests.get('{api}/benchmark/{bid}/start'.format(api=api, bid=bid))

        if (r_start.ok):
            # Ensure SIGINT/SIGTERM on CLI will also terminate running benchmark
            def handler(signum, frame):
                print 'Received signal: {}'.format(signum)
                print 'Stopping benchmark...'
                r_stop = requests.get('{api}/benchmark/{bid}/stop'.format(api=api, bid=bid))
                if (r_stop.ok):
                    print 'OK'

                sys.exit(signum) # TODO: set proper exit code

            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)

            # See if benchmark is running
            progress = requests_get_with_error('{api}/benchmark/progress'.format(api=api),
                                   'Unable to query benchmark progress')

            # Create progress bar
            with tqdm.tqdm(desc=progress['name'], total=100) as progress_bar:
                old_progress = 0
                while (progress['running']):
                    progress = requests_get_with_error('{api}/benchmark/progress'.format(api=api),
                                           'Unable to query benchmark progress')
                    new_progress = progress['progress']
                    progress_bar.update(new_progress - old_progress)
                    old_progress = new_progress
                    time.sleep(0.01)

            # TODO: get console output through run_detail
            # print ''.join(benchmark.output)

        else:
            click.echo('Unable to start benchmark(s)', err=True)
            break

@cli.command('new', short_help='Create a new benchmark.')
def new():
    """Generates the boilerplate code for a new DELLve-benchmark plugin
    """
    dir_name = click.prompt('Please enter a directory name', default='./')
    package_name = click.prompt('Please enter a Python package name')
    benchmark_name = click.prompt('Please enter a unique benchmark class name')
    template.Template().render(dir_name, package_name, benchmark_name)

def requests_get_with_error(url, err_msg):
    r = requests.get(url)
    if (r.ok):
        return r.json()

    else:
        click.echo(err_msg, err=True)
        sys.exit(-1)
